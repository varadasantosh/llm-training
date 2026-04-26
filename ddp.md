---
layout: distill
title: Data Parallelism - From Vanilla DDP to ZeRO
description: Understanding Distributed Data Parallel training and memory optimization techniques
date: 2026-03-15
permalink: /ddp/

authors:
  - name: Varada Santosh
    affiliations:
      name:

toc:
  - name: Introduction
    subsections:
      - name: Time Factor
      - name: Memory Factor
      - name: Parallelism Techniques
      - name: The Coordination Problem
      - name: NCCL — The Communication Foundation
  - name: Vanilla DDP
    subsections:
      - name: How DDP Works
      - name: DDP Step by Step 
      - name: The Redundancy Problem
  - name: NCCL Operations
    subsections:
      - name: AllReduce
      - name: ReduceScatter
      - name: AllGather
  - name: ZeRO Stage 1
    subsections:
      - name: Optimizer State Partitioning
  - name: ZeRO Stage 2
    subsections:
      - name: Gradient Partitioning
  - name: ZeRO Stage 3
    subsections:
      - name: Parameter Partitioning

# bibliography: references.bib

_styles: >
  .ddp-callout {
    padding: 12px 16px;
    margin: 10px 0;
    border-radius: 0 6px 6px 0;
    border-left: 4px solid;
  }
  .ddp-callout.compute {
    background: rgba(220, 53, 69, 0.07);
    border-color: #dc3545;
  }
  .ddp-callout.mem-static {
    background: rgba(230, 119, 0, 0.07);
    border-color: #e67700;
  }
  .ddp-callout.mem-dynamic {
    background: rgba(47, 158, 68, 0.07);
    border-color: #2f9e44;
  }
  .ddp-callout strong {
    display: block;
    margin-bottom: 4px;
  }
  .goal-statement {
    text-align: center;
    font-weight: 700;
    padding: 14px 24px;
    margin: 24px 0;
    border-top: 2px solid var(--global-theme-color);
    border-bottom: 2px solid var(--global-theme-color);
  }
  .factor-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 6px;
  }
  .tag-time { background: rgba(220, 53, 69, 0.12); color: #c92a2a; }
  .tag-memory { background: rgba(230, 119, 0, 0.12); color: #d9480f; }
  .step-tag {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 3px 12px 3px 3px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(25, 113, 194, 0.09);
    color: #1971c2;
    margin-bottom: 8px;
  }
  .step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #1971c2;
    color: #fff;
    font-size: 0.72rem;
    font-weight: 700;
    flex-shrink: 0;
  }
  .ddp-note {
    background: var(--global-code-bg-color, #f8f9fa);
    border-left: 3px solid var(--global-divider-color, #ccc);
    padding: 10px 14px;
    margin: 16px 0;
    border-radius: 0 4px 4px 0;
    font-size: 0.85rem;
  }
  .ddp-note .note-label {
    display: block;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--global-text-color-light, #888);
    margin-bottom: 4px;
    font-weight: 600;
  }
---

## Introduction

In the [previous section]({{ '/' | relative_url }}), we established two fundamental constraints that make training frontier models on a single GPU impossible — time and memory.

> **Compute constraint** At $3.8 \times 10^{25}$ FLOPs,
> training Llama 3 405B on a single H100 (67 TFLOPS FP32) would take ~18,000 years even at 100% > utilisation.

> **Static Memory constraint:** Parameters, Gradients, and Optimizer States for Llama 3 8B 
> require ~144 GB — nearly double the H100's 80 GB VRAM.

> **Dynamic Memory constraint:** Even for a single sequence, Activations 
> add another 38 GB (with FlashAttention) to 294 GB (without). 
> A realistic micro-batch pushes total activation memory to ~1.2 TB — 
> more than 15× what a single GPU can hold.

These are two distinct problems that require two distinct ways of thinking about the solution.

### Time Factor

<span class="factor-tag tag-time">Time</span>

Though the scale of the problem is very large(18,000 years) in this context , time constraints are a familiar problem in engineering. The standard solution is parallelism divide the work across multiple workers, each handling a subset of the problem simultaneously.

The same principle applies here. Training on 15 trillion tokens means processing an enormous amount of data. If we split that data across multiple GPUs — each GPU seeing a different subset — and run the forward and backward passes simultaneously, training time scales down proportionally with the number of GPUs. This is the core idea behind **Data Parallelism**.

### Memory Factor

<span class="factor-tag tag-memory">Memory</span>

The memory constraint is harder. Adding more GPUs does not automatically solve a memory problem — we need to carefully manage what each GPU holds.

Setting activations aside and focusing on the static components — Parameters, Gradients, and Optimizer States — Llama 3 8B alone requires 144 GB. A single H100 has 80 GB. Even ignoring activations entirely, the model's training state does not fit on one GPU. We cannot simply replicate the model across GPUs; we need to orchestrate how its components are distributed across them.

### Parallelism Techniques

These constraints are addressed through a family of techniques — **Data Parallelism, Tensor Parallelism, Pipeline Parallelism, Context Parallelism, and Expert Parallelism** — each targeting a different bottleneck.

### The Coordination Problem

Splitting data across GPUs introduces an immediate challenge. Each GPU sees a different subset of the data, so after the backward pass each GPU has computed different gradients. If every GPU then updates its own parameters independently, the model copies diverge — by the next iteration, GPU 0 and GPU 1 are no longer training the same model.

<div class="goal-statement">Our goal is to train one model, not N independent models.</div>

To achieve this, GPUs cannot work in isolation — they need to communicate and synchronize at specific points during training. How efficiently they do this determines how well the parallelism strategy scales.

### NCCL — The Communication Foundation

Every parallelism technique listed above relies on a common communication layer. For NVIDIA GPUs, that layer is [NCCL](https://developer.nvidia.com/nccl) — the NVIDIA Collective Communications Library. NCCL provides the core routines for moving data across GPUs, whether they share the same node or are spread across multiple nodes over a network. PyTorch, DeepSpeed, and Megatron-LM all build on top of NCCL.

<div class="ddp-note">
  <span class="note-label">Further Reading</span>
  The Llama 3 team extended NCCL into <strong>NCCLX</strong>, optimizing collective operations for their specific network topology across large GPU clusters. See Section 3.3.3 (Collective Communication) of the Llama 3 <a href="https://arxiv.org/pdf/2407.21783">paper</a> for details.
</div>

NCCL exposes a set of collective operations — each designed for a specific communication pattern. We will introduce them as they are needed, but here is the full set we will encounter across all parallelism techniques:

| Operation | What it does | First appears in |
|---|---|---|
| Broadcast | Send from one GPU to all others | DDP initialization |
| AllReduce | Sum/average across all GPUs, result to all | DDP gradient sync |
| ReduceScatter | Reduce then distribute different shards | ZeRO-1/2/3 |
| AllGather | Collect shards from all GPUs, result to all | ZeRO-1/2/3 |
| Send/Recv | Point-to-point between two GPUs | Pipeline Parallelism |

In practice, frontier teams do not rely on a single technique — they combine multiple dimensions simultaneously. The Llama 3 paper (Section 3.3.2) describes how Meta used four dimensions of parallelism — Tensor, Pipeline, Context, and Data — across 16,000 GPUs to train the 405B model. That combined approach is where this series is headed.

We start with the simplest — **Distributed Data Parallel training** — and build from there.


## Vanilla DDP

Distributed Data Parallelism - The core idea is straightforward: keep the model identical on every GPU, but split the training data across them into shards, number of shards are determined by number of GPUs available for training, DDP has one pre-requisite that is the complete model must fit on a single GPU. Parameters, Gradients, and Optimizer States all need to reside on each GPU simultaneously. For Llama 3 8B that means 144 GB per GPU — already beyond a single H100. This is DDP's fundamental limitation, and it is exactly what ZeRO is designed to solve.

For now, let's understand how DDP works when the model does fit — and why it is such a useful starting point.


### How DDP Works

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555;">Step</th>
      <th style="padding:10px 12px; border:2px dashed #555;">What Happens</th>
      <th style="padding:10px 12px; border:2px dashed #555;">NCCL Operation</th>
    </tr>
</thead>
<tbody>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>1. Initialize</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Model parameters broadcast from rank 0 to all GPUs. Gradients and optimizer states are initialized to zero.</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>Broadcast from rank 0</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>2. Data Split</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Global batch is divided into micro-batches, one per GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None (data loader handles this)</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>3. Forward Pass</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU computes forward pass on its micro-batch independently</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>4. Backward Pass</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU computes gradients for its micro-batch</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>5. Gradient Sync</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Gradients are averaged across all GPUs</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>AllReduce</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>6. Optimizer Step</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU updates its local parameters using averaged gradients</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 1: Vanilla DDP Training Steps</figcaption>

</figure>

> Steps 2–6 repeat every iteration until convergence

### DDP Step by Step

<span class="step-tag"><span class="step-num">1</span>Initialization</span>

At the start of training, one GPU — rank 0 — holds the initial model. Its parameters are broadcast to every other GPU using NCCL's Broadcast operation. From this point forward, every GPU holds an identical copy of the model.

<span class="step-tag"><span class="step-num">2</span>Data Split and Forward Pass</span>

The global batch is divided into micro-batches — one per GPU. Each GPU runs its forward pass independently on its own micro-batch, computing its own loss. No communication is needed here — the GPUs work in complete isolation.

<span class="step-tag"><span class="step-num">3</span>Backward Pass and the Divergence Problem</span>

Each GPU runs its backward pass independently, computing gradients based on what it saw. This is where the problem appears. GPU 0 saw sequences 0–127. GPU 1 saw sequences 128–255. Each GPU computed different gradients — and if each GPU now updates its own parameters independently, the model copies diverge. By the next iteration, GPU 0 and GPU 1 are no longer training the same model.

<div class="goal-statement">Our goal is one model, not N independent models.</div>

<span class="step-tag"><span class="step-num">4</span>Gradient Sync — AllReduce</span>

The solution is to synchronize gradients before the optimizer step. This is what AllReduce does — every GPU participates simultaneously. AllReduce happens in two phases:

<d-aside>
  <b>NCCL AllReduce — how it works</b>
  <pre style="font-size:0.72rem; margin-top:6px; line-height:1.5;">ncclResult_t ncclAllReduce(
  const void*    sendbuff,  // each GPU's local gradients
  void*          recvbuff,  // averaged gradients (output)
  size_t         count,     // number of gradient elements
  ncclDataType_t datatype,  // e.g. ncclFloat32
  ncclRedOp_t    op,        // ncclSum → then divide by N
  ncclComm_t     comm,      // communicator (which GPUs participate)
  cudaStream_t   stream     // CUDA stream to run on
);</pre>
  <p style="font-size:0.78rem; margin:8px 0 4px;"><b>What happens under the hood:</b></p>
  <ol style="font-size:0.75rem; margin:0; padding-left:1.2em; line-height:1.6;">
    <li>Every GPU writes its gradients into <code>sendbuff</code></li>
    <li>NCCL performs a ring-based reduction across all GPUs
      <ul style="margin:2px 0; padding-left:1em;">
        <li>each GPU passes values to its neighbour in a ring</li>
        <li>values are summed as they travel around the ring</li>
      </ul>
    </li>
    <li>The sum is divided by N (number of GPUs)</li>
    <li>Every GPU receives the averaged gradient in <code>recvbuff</code></li>
  </ol>
  <p style="font-size:0.75rem; margin:8px 0 0; border-top:1px solid var(--global-divider-color,#ddd); padding-top:6px;">Ring-based AllReduce avoids one GPU becoming a bottleneck — communication cost is O(N) not O(N²)</p>
</d-aside>

**ReduceScatter** — each GPU sends its gradients around the ring. As gradients travel, they are summed. At the end, each GPU holds the reduced sum for only one shard of the gradients — not the full tensor.

{% include figure.liquid path="assets/img/llm-training/ddp/Vanilla-DDP-Before-AllReduce.svg" class="img-medium" caption="Figure 1: ReduceScatter — Gradients reduced and distributed as shards across GPUs" %}

**AllGather** — each GPU then broadcasts its reduced shard to every other GPU. At the end, every GPU holds the complete averaged gradient tensor.

{% include figure.liquid path="assets/img/llm-training/ddp/Vanilla-DDP-After-AllReduce.svg" class="img-medium" caption="Figure 2: AllGather — Each GPU broadcasts its reduced shard so all GPUs receive complete averaged gradients" %}

Together these two operations achieve a full AllReduce without any single GPU becoming a bottleneck.

Once AllGather completes, every GPU holds the complete averaged gradient. Every GPU runs an identical optimizer step. Every GPU begins the next iteration with identical parameters. The model stays in sync.

<span class="step-tag"><span class="step-num">5</span>Optimizer Step</span>

With identical averaged gradients on every GPU, each GPU runs its optimizer step locally — no communication needed. Because every GPU started with the same gradients and runs the same update rule, parameters and optimizer states remain perfectly consistent across all GPUs.

### The Redundancy Problem

DDP is effective at solving the time problem. By splitting the batch across N GPUs and running forward and backward passes simultaneously, training throughput approaches linear scaling — but every GPU still holds a complete copy of all static memory.

But look carefully at what every GPU is holding:

| Component | Precision | Memory per GPU | Replicated? |
|---|---|---|---|
| Parameters — working copy | BF16 | $2\phi$ bytes | ✓ Every GPU |
| Parameters — master copy | FP32 | $4\phi$ bytes | ✓ Every GPU |
| Gradients | FP32 | $4\phi$ bytes | ✓ Every GPU |
| Optimizer State $m_t$ | FP32 | $4\phi$ bytes | ✓ Every GPU |
| Optimizer State $v_t$ | FP32 | $4\phi$ bytes | ✓ Every GPU |
| **Total** | | $18\phi$ **bytes** | **✓ Every GPU** |

Every single byte of static memory is duplicated across every GPU. With 8 GPUs the cluster holds 1.15 TB of static memory — when 144 GB would logically suffice. The other 1.0 TB is pure redundancy.

DDP solves the time problem. It does nothing for the memory problem.

## NCCL Operations

Before we look at how ZeRO solves the memory problem, we need to understand the three collective operations it relies on. We used AllReduce in Vanilla DDP — ZeRO replaces it with a more granular pair of operations. Understanding each one precisely is what makes ZeRO's design legible.

All three operations follow the same ring topology. GPUs are arranged in a logical ring. Each GPU only communicates with its two neighbors — its left neighbor sends to it, it sends to its right neighbor. This avoids any single GPU becoming a bottleneck and keeps communication costs predictable as the number of GPUs grows.

### AllReduce

AllReduce takes a tensor that exists on every GPU, applies a reduction (sum or average) across all copies, and writes the result back to every GPU. After AllReduce, every GPU holds the same reduced tensor.

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
  <tr>
    <th style="padding:10px 12px; border:2px dashed #555;">Phase</th>
    <th style="padding:10px 12px; border:2px dashed #555;">What happens</th>
    <th style="padding:10px 12px; border:2px dashed #555;">Result</th>
  </tr>
</thead>
<tbody>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <td style="padding:10px 12px; border:2px dashed #555;"><strong>ReduceScatter</strong></td>
    <td style="padding:10px 12px; border:2px dashed #555;">Each GPU splits its tensor into N shards and sends them around the ring. Each GPU accumulates the sum for one shard.</td>
    <td style="padding:10px 12px; border:2px dashed #555;">Each GPU holds the fully reduced sum for 1/N of the tensor</td>
  </tr>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555;"><strong>AllGather</strong></td>
    <td style="padding:10px 12px; border:2px dashed #555;">Each GPU broadcasts its reduced shard around the ring. Every GPU receives all N shards.</td>
    <td style="padding:10px 12px; border:2px dashed #555;">Every GPU holds the complete reduced tensor</td>
  </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 2: AllReduce as two phases — ReduceScatter followed by AllGather</figcaption>
</figure>

Total data transferred per GPU: $2 \times \frac{(N-1)}{N} \times \text{tensor\_size}$. For large N this approaches $2 \times \text{tensor\_size}$, and crucially this cost does not grow with N — adding more GPUs does not increase the per-GPU communication volume.

This is exactly what Vanilla DDP uses for gradient synchronization.

### ReduceScatter

ReduceScatter is the first half of AllReduce, but it is also used independently in ZeRO. It takes a tensor from every GPU, reduces (sums) them, and distributes the result so each GPU ends up with a different shard of the reduced tensor.

**Input:** Every GPU holds the full tensor (e.g., all gradients)  
**Output:** GPU $i$ holds only shard $i$ of the reduced tensor

{% include figure.liquid path="assets/img/llm-training/ddp/ReduceScatter.svg" class="img-medium" caption="Figure 3: ReduceScatter — N GPUs each contribute a full tensor; each GPU receives the reduced sum for its assigned shard" %}

ZeRO Stage 2 uses ReduceScatter instead of AllReduce for gradient synchronization. Rather than every GPU computing and storing the full averaged gradient, each GPU only receives the shard it is responsible for. The gradients that each GPU does not own are never materialized — they are reduced in-flight and discarded.

### AllGather

AllGather is the second half of AllReduce, and the most frequently used operation in ZeRO Stages 1, 2, and 3. It takes a different shard from each GPU and assembles the complete tensor on every GPU.

**Input:** GPU $i$ holds shard $i$ (different on every GPU)  
**Output:** Every GPU holds the complete tensor (all shards concatenated)

{% include figure.liquid path="assets/img/llm-training/ddp/AllGather.svg" class="img-medium" caption="Figure 4: AllGather — Each GPU contributes its unique shard; every GPU receives the complete assembled tensor" %}

Total data transferred per GPU: $\frac{(N-1)}{N} \times \text{tensor\_size}$, again independent of N.

ZeRO uses AllGather to reconstruct parameters on-demand. Since Stage 3 shards the model parameters themselves across GPUs, parameters must be gathered before each forward and backward pass, then discarded immediately after. AllGather is what makes this reconstruction possible without a dedicated parameter server.

---

These three operations are the entire communication vocabulary of ZeRO. Every stage is a different choice of when to use each one — and what to discard afterward.

## ZeRO Stage 1

ZeRO (Zero Redundancy Optimizer) is a family of memory optimization techniques introduced in the [ZeRO paper](https://arxiv.org/abs/1910.02054) by the DeepSpeed team at Microsoft. The core insight is that in Vanilla DDP, every GPU holds a complete copy of optimizer states, gradients, and parameters — but for a given optimizer step, each parameter only needs to be updated once. The redundancy is not necessary for correctness; it is simply how DDP was designed.

ZeRO eliminates this redundancy progressively across three stages. Each stage shards a different component across GPUs.

### Optimizer State Partitioning

<span class="factor-tag tag-memory">Memory</span>

ZeRO Stage 1 targets the largest single source of static memory: optimizer states. For Adam, this is the first moment $m_t$ and second moment $v_t$ — together $8\phi$ bytes, or 44% of the total $18\phi$ bytes.

**The idea:** Instead of every GPU holding the full optimizer state for all parameters, partition the optimizer states across GPUs. GPU $i$ is responsible for updating only the $1/N$ slice of parameters assigned to it.

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
  <tr>
    <th style="padding:10px 12px; border:2px dashed #555;">Component</th>
    <th style="padding:10px 12px; border:2px dashed #555;">Vanilla DDP (per GPU)</th>
    <th style="padding:10px 12px; border:2px dashed #555;">ZeRO Stage 1 (per GPU)</th>
  </tr>
</thead>
<tbody>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <td style="padding:10px 12px; border:2px dashed #555;">Parameters (BF16 working)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$2\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$2\phi$ bytes</td>
  </tr>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555;">Parameters (FP32 master)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
  </tr>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <td style="padding:10px 12px; border:2px dashed #555;">Gradients (FP32)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi$ bytes</td>
  </tr>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555;">Optimizer State $m_t$ (FP32)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
  </tr>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <td style="padding:10px 12px; border:2px dashed #555;">Optimizer State $v_t$ (FP32)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
  </tr>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555;"><strong>Total</strong></td>
    <td style="padding:10px 12px; border:2px dashed #555;">$18\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$\left(6 + \frac{12}{N}\right)\phi$ bytes</td>
  </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 3: ZeRO Stage 1 — Optimizer states and FP32 master copy partitioned across N GPUs</figcaption>
</figure>

With 8 GPUs: $6 + 12/8 = 7.5\phi$ bytes per GPU — a **2.4× reduction** from $18\phi$.

**How training proceeds with Stage 1:**

<span class="step-tag"><span class="step-num">1</span>Forward and Backward Pass</span>

Identical to Vanilla DDP. Every GPU holds the full parameters and computes its forward and backward pass on its micro-batch independently.

<span class="step-tag"><span class="step-num">2</span>Gradient Sync — ReduceScatter</span>

Instead of AllReduce, Stage 1 uses ReduceScatter. Each GPU ends up with the averaged gradients for only the parameters it owns. The gradients for parameters owned by other GPUs are reduced and discarded — never stored in full.

<span class="step-tag"><span class="step-num">3</span>Optimizer Step — Local</span>

Each GPU runs its optimizer step using only its gradient shard and its local optimizer state shard. No cross-GPU communication is needed here.

<span class="step-tag"><span class="step-num">4</span>Parameter Sync — AllGather</span>

After the optimizer step, each GPU has updated its shard of the FP32 master parameters. An AllGather broadcasts each shard to every GPU, so all GPUs end with the complete updated parameters for the next iteration.

**The trade-off:** Stage 1 adds one AllGather per step (to reconstruct parameters after the optimizer step). The ReduceScatter replaces the AllReduce (same total communication volume). So Stage 1 has the same communication cost as Vanilla DDP while significantly reducing memory.

## ZeRO Stage 2

### Gradient Partitioning

<span class="factor-tag tag-memory">Memory</span>

ZeRO Stage 2 extends Stage 1 by also partitioning gradients. In Stage 1, every GPU still stores the full gradient tensor ($4\phi$ bytes) even though each GPU only needs the gradients for the parameters it owns. Stage 2 fixes this.

**The idea:** Since each GPU only runs the optimizer step for its parameter shard, it only needs the gradient shard for those parameters. All other gradients can be discarded immediately after ReduceScatter completes.

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
  <tr>
    <th style="padding:10px 12px; border:2px dashed #555;">Component</th>
    <th style="padding:10px 12px; border:2px dashed #555;">ZeRO Stage 1 (per GPU)</th>
    <th style="padding:10px 12px; border:2px dashed #555;">ZeRO Stage 2 (per GPU)</th>
  </tr>
</thead>
<tbody>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <td style="padding:10px 12px; border:2px dashed #555;">Parameters (BF16 working)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$2\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$2\phi$ bytes</td>
  </tr>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555;">Parameters (FP32 master)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
  </tr>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <td style="padding:10px 12px; border:2px dashed #555;">Gradients (FP32)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
  </tr>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555;">Optimizer State $m_t$ (FP32)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
  </tr>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <td style="padding:10px 12px; border:2px dashed #555;">Optimizer State $v_t$ (FP32)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
  </tr>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555;"><strong>Total</strong></td>
    <td style="padding:10px 12px; border:2px dashed #555;">$\left(6 + \frac{12}{N}\right)\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$\left(2 + \frac{16}{N}\right)\phi$ bytes</td>
  </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 4: ZeRO Stage 2 — Gradients also partitioned; each GPU stores only what it needs for its optimizer step</figcaption>
</figure>

With 8 GPUs: $2 + 16/8 = 4\phi$ bytes per GPU — a **4.5× reduction** from $18\phi$, and **1.9×** improvement over Stage 1.

**How gradient partitioning works during the backward pass:**

Rather than accumulating the full gradient buffer and then running ReduceScatter, Stage 2 integrates ReduceScatter into the backward pass itself. As each layer's gradients are computed, they are immediately reduced and sent to the owning GPU. By the time the backward pass completes, each GPU holds only its gradient shard — the full gradient tensor was never assembled in memory.

This is the key implementation insight: ReduceScatter is pipelined with computation, so the communication and computation overlap. The memory saving is not just at the end of the backward pass — the peak gradient memory during the pass is also reduced.

The communication pattern remains unchanged from Stage 1: ReduceScatter for gradient sync, AllGather after the optimizer step to reconstruct full parameters.

## ZeRO Stage 3

### Parameter Partitioning

<span class="factor-tag tag-memory">Memory</span>

ZeRO Stage 3 completes the sharding by also partitioning the model parameters themselves. Stages 1 and 2 left every GPU holding the full BF16 working copy of all parameters ($2\phi$ bytes) — the dominant remaining cost. Stage 3 shards this across GPUs as well.

**The idea:** Each GPU only holds $1/N$ of the BF16 working parameters at rest. Parameters are reconstructed via AllGather before each layer's forward or backward pass, then immediately discarded.

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
  <tr>
    <th style="padding:10px 12px; border:2px dashed #555;">Component</th>
    <th style="padding:10px 12px; border:2px dashed #555;">ZeRO Stage 2 (per GPU)</th>
    <th style="padding:10px 12px; border:2px dashed #555;">ZeRO Stage 3 (per GPU)</th>
  </tr>
</thead>
<tbody>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <td style="padding:10px 12px; border:2px dashed #555;">Parameters (BF16 working)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$2\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$2\phi/N$ bytes</td>
  </tr>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555;">Parameters (FP32 master)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
  </tr>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <td style="padding:10px 12px; border:2px dashed #555;">Gradients (FP32)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
  </tr>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555;">Optimizer State $m_t$ (FP32)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
  </tr>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <td style="padding:10px 12px; border:2px dashed #555;">Optimizer State $v_t$ (FP32)</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi/N$ bytes</td>
  </tr>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555;"><strong>Total</strong></td>
    <td style="padding:10px 12px; border:2px dashed #555;">$\left(2 + \frac{16}{N}\right)\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">$\frac{18\phi}{N}$ bytes</td>
  </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 5: ZeRO Stage 3 — All components partitioned; memory scales linearly with N GPUs</figcaption>
</figure>

With 8 GPUs: $18\phi/8 = 2.25\phi$ bytes per GPU — an **8× reduction** from Vanilla DDP. For Llama 3 8B, that is 18 GB per GPU instead of 144 GB.

This is the theoretical maximum: every byte of static memory is shared across GPUs. The cluster's aggregate memory now matches what a single GPU would need for DDP, with no redundancy.

**The cost: increased communication**

Stage 3 exchanges memory savings for additional AllGather calls. Every forward pass through a layer requires an AllGather to reconstruct that layer's parameters. After the forward pass, the gathered parameters are freed. The backward pass requires another AllGather for each layer. In total, Stage 3 performs roughly $3\times$ the AllGather volume of Stage 2.

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
  <tr>
    <th style="padding:10px 12px; border:2px dashed #555;">Stage</th>
    <th style="padding:10px 12px; border:2px dashed #555;">Memory per GPU (N=8)</th>
    <th style="padding:10px 12px; border:2px dashed #555;">Reduction vs DDP</th>
    <th style="padding:10px 12px; border:2px dashed #555;">Communication vs DDP</th>
  </tr>
</thead>
<tbody>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <td style="padding:10px 12px; border:2px dashed #555;"><strong>Vanilla DDP</strong></td>
    <td style="padding:10px 12px; border:2px dashed #555;">$18\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">1×</td>
    <td style="padding:10px 12px; border:2px dashed #555;">1× (AllReduce)</td>
  </tr>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555;"><strong>ZeRO Stage 1</strong></td>
    <td style="padding:10px 12px; border:2px dashed #555;">$7.5\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">2.4×</td>
    <td style="padding:10px 12px; border:2px dashed #555;">1× (RS + AG)</td>
  </tr>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <td style="padding:10px 12px; border:2px dashed #555;"><strong>ZeRO Stage 2</strong></td>
    <td style="padding:10px 12px; border:2px dashed #555;">$4\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">4.5×</td>
    <td style="padding:10px 12px; border:2px dashed #555;">1× (RS + AG)</td>
  </tr>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555;"><strong>ZeRO Stage 3</strong></td>
    <td style="padding:10px 12px; border:2px dashed #555;">$2.25\phi$ bytes</td>
    <td style="padding:10px 12px; border:2px dashed #555;">8×</td>
    <td style="padding:10px 12px; border:2px dashed #555;">~1.5× (extra AG per layer)</td>
  </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 6: ZeRO stages compared — memory savings vs. communication overhead</figcaption>
</figure>

In practice, whether Stage 3's extra communication is acceptable depends on the interconnect bandwidth between GPUs. On NVLink (within a single node), the bandwidth is high enough that the extra AllGathers are rarely a bottleneck. Across nodes over InfiniBand, the increased communication volume can slow training — which is why practitioners often combine Stage 2 within nodes and Stage 3 across nodes, or use ZeRO with activation checkpointing to stay within Stage 2.

<div class="ddp-note">
  <span class="note-label">Further Reading</span>
  ZeRO is described in detail in <a href="https://arxiv.org/abs/1910.02054">ZeRO: Memory Optimizations Toward Training Trillion Parameter Models</a> (Rajbhandari et al., 2020). DeepSpeed's <a href="https://www.deepspeed.ai/tutorials/zero/">ZeRO tutorial</a> covers practical configuration for all three stages.
</div>

Data parallelism — both Vanilla DDP and ZeRO — addresses the time constraint and partially the memory constraint. But ZeRO still requires that activations fit on each GPU during the forward pass. For very large models or long sequences, activations remain the dominant memory pressure. That is where tensor parallelism enters — splitting not the data, but the model itself, across GPUs.

<div style="display:flex; justify-content:space-between; margin-top:48px; padding-top:20px; border-top:1px solid var(--global-divider-color,#dee2e6);">
  <a href="{{ '/' | relative_url }}" style="font-size:0.9rem; font-weight:600;">← Part 1: Memory &amp; The Case for Parallelism</a>
  <a href="{{ '/tensor-parallelism/' | relative_url }}" style="font-size:0.9rem; font-weight:600;">Next: Tensor Parallelism →</a>
</div>