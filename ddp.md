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
      - name: Broadcast
      - name: AllReduce
      - name: ReduceScatter
      - name: AllGather
  - name: Vanilla DDP
    subsections:
      - name: How DDP Works
      - name: DDP Step by Step 
      - name: DDP Limitations
  - name: ZeRO-1 Sharding Optimizer States
    subsections:
      - name: Optimizer State Partitioning
  - name: ZeRO-2 Sharding Gradients
    subsections:
      - name: Gradient Partitioning
  - name: ZeRO-3 Sharding Parameters
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

Though the scale of the problem is very large(18,000 years) in this context , time constraints are a familiar problem in engineering. The standard solution is parallelism - divide the work across multiple workers, each handling a subset of the problem simultaneously.

The same principle applies here. Training on 15 trillion tokens means processing an enormous amount of data. If we split that data across multiple GPUs — each GPU seeing a different subset — and run the forward and backward passes simultaneously, training time scales down proportionally with the number of GPUs. This is the core idea behind **Data Parallelism**.

### Memory Factor

<span class="factor-tag tag-memory">Memory</span>

The memory constraint is harder. Adding more GPUs does not automatically solve a memory problem — we need to carefully manage what each GPU holds.

Setting activations aside and focusing on the static components — Parameters, Gradients, and Optimizer States — Llama 3 8B alone requires 144 GB. A single H100 has 80 GB. Even ignoring activations entirely, the model's training state does not fit on one GPU. We cannot simply replicate the model across GPUs; we need to orchestrate how its components are distributed across them.

### Parallelism Techniques

These constraints are addressed through a family of techniques — **Data Parallelism, Tensor Parallelism (TP), Pipeline Parallelism (PP), Context Parallelism (CP), and Expert Parallelism (EP)** — each targeting a different bottleneck.

### The Coordination Problem

Splitting data across GPUs introduces an immediate challenge. Each GPU sees a different subset of the data, so after the backward pass each GPU has computed different gradients. If every GPU then updates its own parameters independently, the model copies diverge — by the next iteration, GPU 0 and GPU 1 are no longer training the same model.

<div class="goal-statement">Our goal is to train one model, not N independent models.</div>

To achieve this, GPUs cannot work in isolation — they need to communicate and synchronize at specific points during training. This coordination requirement appears everywhere in distributed computing — multiple processes writing to a shared database need locking to prevent conflicts, multiple servers handling web requests need load balancing to share work evenly, multiple threads in a program need synchronization primitives to stay in step.

Distributed LLM training is the same class of problem. Multiple GPUs computing gradients from different data need to agree on a single averaged value before updating parameters. Multiple GPUs holding different parameter shards need to share them before the forward pass can run.

NCCL provides the synchronization primitives for this — purpose-built for GPU-to-GPU communication at scale.

### NCCL — The Communication Foundation

Every parallelism technique — DDP, ZeRO-1/2/3, Tensor, Pipeline, Context, and Expert Parallelism — requires GPUs to communicate at specific points during training. 

The form that communication takes differs by technique. After every backward pass, gradients must be coordinated across GPUs — either fully synchronized via AllReduce in DDP, or averaged and distributed as shards via ReduceScatter in ZeRO stages. After every optimizer 
step, parameters must be reconstructed. In ZeRO-3, parameters must be fetched layer by layer before every forward and backward pass.

All communication operations need a foundation to run on. For NVIDIA GPUs, that foundation is [NCCL](https://developer.nvidia.com/nccl) — the NVIDIA Collective Communications Library. NCCL provides the core routines for moving data across GPUs, whether they share the same node connected via NVLink, or spread across multiple nodes connected over InfiniBand. NCCL is not hardware-agnostic — it runs on NVIDIA GPUs only.

This is where PyTorch's abstraction layer becomes important. Each accelerator vendor provides its own communication library — **RCCL for AMD**, **oneCCL for Intel XPUs**, proprietary libraries for Google TPUs. PyTorch's **torch.distributed** sits above all of them, selecting the appropriate backend automatically based on the hardware available. ML researchers write **torch.distributed** or **torch.DistributeDataParallel** code once and it runs across different hardware without modification.

In this blog we solely focus on NCCL — since Llama-3 was trained on NVIDIA H100 GPUs — but similar communication patterns apply across backends.


<div class="ddp-note">
  <span class="note-label">Further Reading</span>
  The Llama 3 team extended NCCL into <strong>NCCLX</strong>, optimizing collective operations for their specific network topology across large GPU clusters. See Section 3.3.3 (Collective Communication) of the Llama 3 <a href="https://arxiv.org/pdf/2407.21783">paper</a> for details.
</div>

NCCL exposes a set of collective operations — each designed for a specific communication pattern. These operations are at the core of the distributed training process

| Operation | What it does |  appears in |
|---|---|---|
| Broadcast | Send from one GPU to all others | DDP initialization |
| AllReduce | Sum/average across all GPUs, result to all | DDP gradient sync, TP|
| ReduceScatter | Reduce then distribute different shards | ZeRO-1/2/3, Sequence Parallelism |
| AllGather | Collect shards from all GPUs, result to all | ZeRO-1/2/3 |
| Send/Recv | Point-to-point between two GPUs | Pipeline Parallelism, Context Parallelism |

### Broadcast
Broadcast sends data from one GPU — rank 0 — to all other GPUs. This is used once at the start of DDP training to ensure every GPU begins with identical model parameters.

A naive implementation would send from rank 0 directly to all other GPUs — but this makes rank 0 a bottleneck. Data transfer is limited by rank 0's bandwidth, and the time scales linearly with the number of GPUs.

NCCL avoids this by building a logical tree over the flat network topology:

```
  Rank 0 (root)
  ├── Rank 1
  │   ├── Rank 3
  │   └── Rank 4
  └── Rank 2
      ├── Rank 5
      └── Rank 6
```
Each node forwards data to its children simultaneously. Communication time scales with O(log N) depth of the tree rather than O(N). Within a single node, NVLink is used for fast intra-node transfers. Across nodes, InfiniBand handles the inter-node hops.

**Used in:** DDP initialization — once per training run.

### AllReduce

AllReduce takes a tensor that exists on every GPU, applies a reduction (sum or average) across all copies, and writes the result back to every GPU. After AllReduce, every GPU holds the same reduced tensor.

Before AllReduce (4 GPUs, gradient sync):
  - GPU-0: $\nabla\text{W}_{i}^0$
  - GPU-1: $\nabla\text{W}_{i}^1$    
  - GPU-2: $\nabla\text{W}_{i}^2$   
  - GPU-3: $\nabla\text{W}_{i}^3$

After AllReduce:
All GPUS: $(\nabla\text{W}_i^0 + \nabla\text{W}_i^1 +  \nabla\text{W}_i^2 + \nabla\text{W}_i^3)/4$

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
    <td style="padding:10px 12px; border:2px dashed #555;">Each GPU holds the fully reduced average for 1/N of the tensor</td>
  </tr>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555;"><strong>AllGather</strong></td>
    <td style="padding:10px 12px; border:2px dashed #555;">Each GPU broadcasts its reduced shard around the ring. Every GPU receives all N shards.</td>
    <td style="padding:10px 12px; border:2px dashed #555;">Every GPU holds the complete reduced tensor</td>
  </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">AllReduce as Two Phases — ReduceScatter followed by AllGather</figcaption>
</figure>

All NCCL communication operations can be implemented using different topologies, the ring-based topology used in PyTorch and other distributed frameworks ensures every GPU is active simultaneously — no single GPU becomes a bottleneck. Communication cost is linear in data size and independent of the number of GPUs.

**Used in:** DDP gradient synchronization — once per training iteration.

### ReduceScatter

ReduceScatter is the first half of AllReduce, in ZeRO stages it is used independently without the subsequent AllGather. It takes a tensor from every GPU, reduces (averages) them and distributes the result so each GPU ends up with a different shard of the reduced tensor.

**Before ReduceScatter:** Every GPU holds the full tensor (e.g., all gradients)

  - GPU-0: $\nabla\text{W}_1^0$ , $\nabla\text{W}_2^0$ , $\nabla\text{W}_3^0$, $\nabla\text{W}_4^0$
  - GPU-1: $\nabla\text{W}_1^1$ , $\nabla\text{W}_2^1$ , $\nabla\text{W}_3^1$, $\nabla\text{W}_4^1$ 
  - GPU-2: $\nabla\text{W}_1^2$ , $\nabla\text{W}_2^2$ , $\nabla\text{W}_3^2$, $\nabla\text{W}_4^2$
  - GPU-3: $\nabla\text{W}_1^3$ , $\nabla\text{W}_2^3$ , $\nabla\text{W}_3^3$, $\nabla\text{W}_4^3$


**After ReduceScatter:** GPU $i$ holds only shard $i$ of the reduced tensor

  - GPU-0: averaged($\nabla \text{W}_1$) - globally averaged, shard 0
  - GPU-1: averaged($\nabla \text{W}_2$) - globally averaged, shard 1
  - GPU-2: averaged($\nabla \text{W}_3$) - globally averaged, shard 2
  - GPU-3: averaged($\nabla \text{W}_4$) - globally averaged, shard 3



<!-- Figure 3: ReduceScatter diagram - TODO: Add image -->

ZeRO Stage 2 uses ReduceScatter instead of AllReduce for gradient synchronization. Rather than every GPU computing and storing the full averaged gradient, each GPU only receives the shard it is responsible for. The gradients that each GPU does not own are never materialized — they are reduced in-flight and discarded. ZeRO Stage 1 also uses ReduceScatter — it decomposes AllReduce into ReduceScatter + optimizer step + AllGather.


### AllGather

AllGather is the second half of AllReduce, and the most frequently used operation in ZeRO Stages 1, 2, and 3. It takes a different shard from each GPU and assembles the complete tensor on every GPU.

Before AllGather (4 GPUs, after optimizer step):
  - GPU-0: updated W₁ shard
  - GPU-1: updated W₂ shard
  - GPU-2: updated W₃ shard
  - GPU-3: updated W₄ shard

After AllGather:
  All GPUs: [W₁, W₂, W₃, W₄] 

**Input:** GPU $i$ holds shard $i$ (different on every GPU)  
**Output:** Every GPU holds the complete tensor (all shards concatenated)

<!-- Figure 4: AllGather diagram - TODO: Add image -->

Total data transferred per GPU: $\frac{(N-1)}{N} \times \text{tensor\_size}$, again independent of N.

ZeRO uses AllGather to reconstruct parameters on-demand. Since ZeRO-3 shards the model parameters themselves across GPUs, parameters must be gathered layer by layer before each forward and backward pass, then discarded immediately after. Each GPU contributes its shard and receives the complete model, all without any central coordinator.


### Send/Recv

Send/Recv enables direct point-to-point communication between two specific GPUs — GPU A sends data, GPU B receives it. Unlike the collective operations above, Send/Recv does not involve all GPUs simultaneously.

This is the primitive used in Pipeline Parallelism, where activations flow forward from one pipeline stage to the next, and gradients flow backward — one stage at a time. We will cover this in detail in the Pipeline Parallelism section.

**Used in:** Pipeline Parallelism — every micro-batch.

In practice, frontier teams do not rely on a single technique — they combine multiple dimensions simultaneously. The Llama 3 paper (Section 3.3.2) describes how Meta used four dimensions of parallelism — Tensor, Pipeline, Context, and Data — across 16,000 GPUs to train the 405B model. That combined approach is where this series is headed.

We start with the simplest — **Distributed Data Parallel training** — and build from there.


## Vanilla DDP

Distributed Data Parallelism - The core idea is straightforward: keep the model identical on every GPU, but split the training data across them into shards, number of shards are determined by number of GPUs available for training, DDP has one pre-requisite that is the complete model must fit on single GPU. Parameters, Gradients, and Optimizer States all need to reside on each GPU simultaneously. For Llama 3 8B that means 144 GB per GPU — already beyond a single H100. This is DDP's fundamental limitation, and it is exactly what ZeRO is designed to solve.

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
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>1.Initialize</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Model parameters broadcast from rank 0 to all GPUs. Gradients and optimizer states are initialized to zero.</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>Broadcast from rank 0</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>2.Data Split</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Global batch is divided into micro-batches, one per GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None (data loader handles this)</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>3.Forward Pass</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU computes forward pass on its micro-batch independently</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>4.Backward Pass</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU computes gradients for its micro-batch</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>5.Gradient Sync</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Gradients are averaged across all GPUs</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>AllReduce</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>6.Optimizer Step</strong></td>
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
  const void*    sendbuff,  // each GPUs local gradients
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

### DDP Limitations

DDP is effective at solving the time problem. By splitting the batch across N GPUs and running forward and backward passes simultaneously, training throughput approaches linear scaling — but one major drawback with DDP is that model should fit on single GPU, which is not the case with large scale models, entire model can't fit on a single GPU, even though a model can fit on single GPU we are copying all parameters, gradients, optimizer states to each of the GPU which causes redundancy

But looking carefully at what every GPU is holding, it is evident that static memory is duplicated across every GPU. With 8 GPUs the cluster holds 1.15 TB of static memory. This shows redundant memory issue. Next few sections focus on reducing the redundancy by using ZeRO-1, ZeRO-2, ZeRO-3

<figure id="table-2-ddp-memory">
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555;">Component</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Memory per GPU</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Share of Total</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Replicated?</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Parameters BF16 (working copy)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$2\phi$ bytes</td>
      <td style="padding:10px 12px; border:2px dashed #555;">11%</td>
      <td style="padding:10px 12px; border:2px dashed #555;">✓ Every GPU</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Parameters FP32 (master copy)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$4\phi$ bytes</td>
      <td style="padding:10px 12px; border:2px dashed #555;">22%</td>
      <td style="padding:10px 12px; border:2px dashed #555;">✓ Every GPU</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Gradients FP32</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$4\phi$ bytes</td>
      <td style="padding:10px 12px; border:2px dashed #555;">22%</td>
      <td style="padding:10px 12px; border:2px dashed #555;">✓ Every GPU</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Optimizer State $m_t$, $v_t$ FP32</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$8\phi$ bytes</td>
      <td style="padding:10px 12px; border:2px dashed #555;">44%</td>
      <td style="padding:10px 12px; border:2px dashed #555;">✓ Every GPU</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>Total</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>$18\phi$ bytes</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>✓ Every GPU</strong></td>
    </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 2: DDP Memory Requirements per GPU</figcaption>
</figure>


## ZeRO-1 Sharding Optimizer States

Vanilla DDP shards data across GPUs — each GPU sees a different slice of the training data. ZeRO extends this by also sharding the model state itself: parameters, gradients, and optimizer states. 

Vanilla DDP addressed the time problem effectively but the memory problem remained untouched — for models like Llama 3 8B whose static components alone require 144 GB, these are replicated across GPUs. With 8 GPUs the cluster holds 1.15 TB of static memory when 144GB would be sufficient.

Looking at memory requirements of the model components from [table](#table-2-ddp-memory). The largest single component is optimizer states. Adam's m and v vectors consume $8\phi$ bytes per GPU — 44% of total static memory — fully replicated across every GPU despite each GPU only needing the states for the parameters it owns.

ZeRO-1 targets optimizer states first for two reasons: they are the largest component, and sharding them has no impact on the forward or backward pass — optimizer states are only read and written during the optimizer step. This makes them the safest component to shard without changing any other part of the training loop.

### The Communication Pattern

In vanilla DDP, gradient synchronization uses AllReduce — a single operation that averages gradients and returns the full averaged tensor to every GPU.

ZeRO-1  decomposes **AllReduce** into its two constituent operations, ReduceScatter and AllGather, and inserts the optimizer step between them. This modification is what enables the memory saving: each GPU only needs to retain the gradient shard it is responsible for, discarding the rest before the optimizer step.

**Training path:**
> Forward Pass → Backward Pass → ReduceScatter → Local Optimizer Step → AllGather

<span class="step-tag"><span class="step-num">1</span>ReduceScatter</span>

Each GPU computes its local gradients during the backward pass. ReduceScatter then averages these gradients across all GPUs — but instead of returning the full averaged tensor to everyone, it distributes different shards to different GPUs:

**Before ReduceScatter (4 GPUs):** <br>

  GPU 0: [$\nabla \text{W}_0^{\text{local}}$, $\nabla \text{W}_1^{\text{local}}$, $\nabla \text{W}_2^{\text{local}}$, $\nabla \text{W}_3^{\text{local}}$] <br>
  GPU 1: [$\nabla \text{W}_0^{\text{local}}$, $\nabla \text{W}_1^{\text{local}}$, $\nabla \text{W}_2^{\text{local}}$, $\nabla \text{W}_3^{\text{local}}$] <br>
  GPU 2: [$\nabla \text{W}_0^{\text{local}}$, $\nabla \text{W}_1^{\text{local}}$, $\nabla \text{W}_2^{\text{local}}$, $\nabla \text{W}_3^{\text{local}}$] <br>
  GPU 3: [$\nabla \text{W}_0^{\text{local}}$, $\nabla \text{W}_1^{\text{local}}$, $\nabla \text{W}_2^{\text{local}}$, $\nabla \text{W}_3^{\text{local}}$] <br>

**After ReduceScatter:** <br>

  GPU 0: averaged($\nabla \text{W}_0$) ← globally averaged, just for shard 0 <br>
  GPU 1: averaged($\nabla \text{W}_1$) ← globally averaged, just for shard 1 <br>
  GPU 2: averaged($\nabla \text{W}_2$) ← globally averaged, just for shard 2 <br>
  GPU 3: averaged($\nabla \text{W}_3$) ← globally averaged, just for shard 3 <br>

Each GPU receives the globally averaged gradient — just for its own shard. The averaging is numerically identical to what AllReduce would have produced.

<span class="step-tag"><span class="step-num">2</span>Local Optimizer Step</span>

Each GPU now has exactly what it needs — the globally averaged gradient for its parameter shard and its local optimizer states for that same shard. It updates its parameter shard locally. 
No communication needed.

<span class="step-tag"><span class="step-num">3</span>AllGather</span>

After the optimizer step, each GPU holds an updated version of its parameter shard. But every GPU needs the complete updated model for the next forward pass. AllGather operation collects each GPU's updated shard to reconstruct the full parameter set on every GPU.

After optimizer step: <br>
GPU 0: updated $\text{W}_0$ shard <br>
GPU 1: updated $\text{W}_1$ shard <br>
GPU 2: updated $\text{W}_2$ shard <br>
GPU 3: updated $\text{W}_3$ shard <br>

After AllGather: <br>
All GPUs: complete updated [ W₀ , W₁ , W₂ , W₃ ]

<div style="width: 100%; margin: 24px 0;">
{% include figure.liquid path="assets/img/llm-training/ddp/ZeRO-Stage-1-Pipeline.svg" class="img-fluid w-100" zoomable=true caption="Figure 3: ZeRO Stage-1 Pipeline" %}
</div>

### Example Workflow

For clarity, consider a simplified model with 4 parameter matrices W₁, W₂, W₃, W₄, we use 4 GPUs. Llama 3 uses no bias terms so we omit bias here.

 1. All GPUs hold identical Parameters w₁,w₂,w₃,w₄ -> Fully replicated on all GPUs.

 2. Optimizer state shards (unique per GPU):

    - GPU-0: (m₁, v₁)  ← owns optimizer states for W₁
    - GPU-1: (m₂, v₂)  ← owns optimizer states for W₂
    - GPU-2: (m₃, v₃)  ← owns optimizer states for W₃
    - GPU-3: (m₄, v₄)  ← owns optimizer states for W₄

3. Each GPU processes its data shard independently.

   - GPU-0: X₀ → forward(W₁,W₂,W₃,W₄) → Ŷ₀ → Loss₀
   - GPU-1: X₁ → forward(W₁,W₂,W₃,W₄) → Ŷ₁ → Loss₁
   - GPU-2: X₂ → forward(W₁,W₂,W₃,W₄) → Ŷ₂ → Loss₂
   - GPU-3: X₃ → forward(W₁,W₂,W₃,W₄) → Ŷ₃ → Loss₃

4. Every GPU computes local gradients for **all** parameters.
   - GPU-0: [∇W₁⁰, ∇W₂⁰, ∇W₃⁰, ∇W₄⁰]  ← based on X₀
   - GPU-1: [∇W₁¹, ∇W₂¹, ∇W₃¹, ∇W₄¹]  ← based on X₁
   - GPU-2: [∇W₁², ∇W₂², ∇W₃², ∇W₄²]  ← based on X₂
   - GPU-3: [∇W₁³, ∇W₂³, ∇W₃³, ∇W₄³]  ← based on X₃

5. Each GPU has different gradients — the model will diverge if these are not synchronized.
   ReduceScatter averages gradients across all GPUs and distributes each shard to its responsible GPU:

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0;">

<div>
<h4 style="text-align:center; margin-bottom:12px; color:#555;">Before ReduceScatter</h4>
<p style="text-align:center; font-size:12px; color:#888; margin-bottom:12px;">Each GPU holds local gradients for all parameters</p>
<table style="width:100%; border-collapse:collapse; font-size:12px; border:2px solid #6c757d;">
<thead>
  <tr style="background:#6c757d; color:white;">
    <th style="padding:8px; border:1px solid #6c757d;">GPU</th>
    <th style="padding:8px; border:1px solid #6c757d;">$\nabla\text{W}_1$</th>
    <th style="padding:8px; border:1px solid #6c757d;">$\nabla\text{W}_2$</th>
    <th style="padding:8px; border:1px solid #6c757d;">$\nabla\text{W}_3$</th>
    <th style="padding:8px; border:1px solid #6c757d;">$\nabla\text{W}_4$</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-0</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_1^0$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_2^0$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_3^0$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_4^0$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-1</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_1^1$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_2^1$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_3^1$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_4^1$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-2</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_1^2$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_2^2$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_3^2$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_4^2$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-3</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_1^3$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_2^3$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_3^3$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_4^3$</td>
  </tr>
</tbody>
</table>
<p style="text-align:center; font-size:11px; color:#666; margin-top:8px;">Buffer: $4\phi$ per GPU — fully replicated</p>
</div>

<div>
<h4 style="text-align:center; margin-bottom:12px; color:#1971c2;">After ReduceScatter</h4>
<p style="text-align:center; font-size:12px; color:#888; margin-bottom:12px;">Each GPU receives only its averaged shard</p>
<table style="width:100%; border-collapse:collapse; font-size:12px; border:2px solid #1971c2;">
<thead>
  <tr style="background:#1971c2; color:white;">
    <th style="padding:8px; border:1px solid #1971c2;">GPU</th>
    <th style="padding:8px; border:1px solid #1971c2;">Owned Shard</th>
    <th style="padding:8px; border:1px solid #1971c2;">Averaging Formula</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-0</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#d4edda; font-weight:600;">$\bar{\nabla}\text{W}_1$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-size:11px;">$\bar{\nabla}\text{W}_1 = \frac{1}{N}\sum_{i=0}^{N-1}\nabla\text{W}_1^i$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-1</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#d4edda; font-weight:600;">$\bar{\nabla}\text{W}_2$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-size:11px;">$\bar{\nabla}\text{W}_2 = \frac{1}{N}\sum_{i=0}^{N-1}\nabla\text{W}_2^i$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-2</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#d4edda; font-weight:600;">$\bar{\nabla}\text{W}_3$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-size:11px;">$\bar{\nabla}\text{W}_3 = \frac{1}{N}\sum_{i=0}^{N-1}\nabla\text{W}_3^i$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-3</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#d4edda; font-weight:600;">$\bar{\nabla}\text{W}_4$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-size:11px;">$\bar{\nabla}\text{W}_4 = \frac{1}{N}\sum_{i=0}^{N-1}\nabla\text{W}_4^i$</td>
  </tr>
</tbody>
</table>
<p style="text-align:center; font-size:11px; color:#1971c2; margin-top:8px; font-weight:600;">Buffer: $4\phi$ per GPU (ZeRO-1 retains full buffer)</p>
</div>

</div>

<p style="text-align:center; margin:16px 0; font-size:13px;">
<span style="display:inline-block; width:12px; height:12px; background:#fff3cd; border:1px solid #ddd; margin-right:4px; vertical-align:middle;"></span> Local gradient (pre-averaging)
&nbsp;&nbsp;
<span style="display:inline-block; width:12px; height:12px; background:#d4edda; border:1px solid #ddd; margin-right:4px; vertical-align:middle;"></span> Globally averaged gradient (retained)
</p>

6. Each GPU receives the globally averaged gradient — identical to what AllReduce would have produced — just for its own shard. Each GPU now has two components required to update parameters for next iteration

   - The globally averaged gradient for its shard
   - Its local optimizer states for that same shard

   GPU-0 updates W₁ using its local (m₁, v₁) and $\nabla \text{W}_1{_avg}$. Each GPU applies the same Adam update rule to its own shard — the process is identical, only the parameters differ.

<d-aside>
  <b>Adam update — GPU-0 updating W₁</b>
  <pre style="font-size:0.75rem; margin-top:6px; line-height:1.6;">m₁ = β₁·m₁ + (1-β₁)·∇W₁_avg   ← 1st moment
v₁ = β₂·v₁ + (1-β₂)·(∇W₁_avg)² ← 2nd moment

m̂₁ = m₁ / (1-β₁ᵗ)  ← bias correction
v̂₁ = v₁ / (1-β₂ᵗ)

W₁ = W₁ - η·m̂₁/(√v̂₁+ε)  ← parameter update</pre>
  <p style="font-size:0.75rem; margin:6px 0 0;">$\beta_1$, $\beta_2$ — moment decay rates; $\eta$ — learning rate; $\epsilon$ — numerical stability constant. Each GPU runs this identically for its own shard.</p>
</d-aside>

7. AllGather - After the optimizer step each GPU holds only its updated parameter shard. AllGather reconstructs the complete model:

   Before AllGather:
   - GPU-0: updated $\text{W}_1$  
   - GPU-1: updated $\text{W}_2$
   - GPU-2: updated $\text{W}_3$  
   - GPU-3: updated $\text{W}_4$
  
   After AllGather:
   All GPUs: [updated W₁, updated W₂, updated W₃, updated W₄]

The final parameters are identical across GPUs. Each GPU applies the same Adam update rule to its own optimizer state shard — the mechanics are identical, only the slice of the model differs. The model does not diverge.

Critically, each GPU carries its optimizer state shard forward across iterations. GPU-0 always owns and updates $(m_1, v_1)$; those states accumulate momentum across every training step just as they would in DDP — the only difference is that no single GPU ever holds the full optimizer state simultaneously.

### Training Steps
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
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>1.Initialize</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Model parameters broadcast from rank 0 to all GPUs. Gradients initialized to zero.</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>Broadcast from rank 0</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>2.Data Split</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Global batch is divided into micro-batches, one per GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None (data loader handles this)</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>3.Initialize Optimizer State Shards</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU initializes only its assigned optimizer state shard to zero</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>4.Forward Pass</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU computes forward pass on its micro-batch independently</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>5.Backward Pass</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU locally computes gradients for its micro-batch</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>6.Gradient Sync</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Gradients averaged across all GPUs; each GPU receives only its assigned gradient shard</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>ReduceScatter</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>7.Optimizer Step</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU updates its parameter shard using its local gradient and optimizer state shards</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>8.Gather Updated Params</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">AllGather — every GPU contributes its updated parameters and receives the updated parameters from other GPUs to reconstruct model </td>
      <td style="padding:10px 12px; border:2px dashed #555;">AllGather</td>
    </tr>
    
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 3: ZeRO Stage-1 Training Steps</figcaption>
</figure>

### Memory Comparison: DDP vs ZeRO-1

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:13px; border:2px dashed #555;">
<thead>
    <tr>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:left;" rowspan="2">Component</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#dc3545; color:white;" colspan="2">DDP</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#1971c2; color:white;" colspan="2">ZeRO-1</th>
    </tr>
    <tr>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Memory</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Distribution</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Memory</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Distribution</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Parameters BF16</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$2\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$2\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Parameters FP32</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Gradients FP32</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Optimizer States $(m_t, v_t)$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$8\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600; color:#1971c2;">$8\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">Sharded</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8); font-weight:600;">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>Total per GPU</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;" colspan="2"><strong>$18\phi$</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;" colspan="2"><strong>$(10 + 8/N)\phi$</strong></td>
    </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 4: Memory Distribution Comparison — DDP vs ZeRO-1</figcaption>
</figure>

<p style="text-align:center; font-size:12px; margin-top:8px;">
<span style="display:inline-block; width:12px; height:12px; background:#f8d7da; border:1px solid #ddd; margin-right:4px; vertical-align:middle;"></span> Replicated (redundant across all GPUs)
&nbsp;&nbsp;
<span style="display:inline-block; width:12px; height:12px; background:#d4edda; border:1px solid #ddd; margin-right:4px; vertical-align:middle;"></span> Sharded (distributed across N GPUs)
</p>

### Memory Savings for Llama 3 8B (8 GPUs)

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <th style="padding:12px; border:2px dashed #555; text-align:left;">Approach</th>
      <th style="padding:12px; border:2px dashed #555; text-align:center;">Formula</th>
      <th style="padding:12px; border:2px dashed #555; text-align:center;">Memory/GPU</th>
      <th style="padding:12px; border:2px dashed #555; text-align:center;">Reduction</th>
      <th style="padding:12px; border:2px dashed #555; text-align:center;">Freed/GPU</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:12px; border:2px dashed #555; font-weight:600; color:#dc3545;">DDP</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center;">$18\phi$</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center; font-weight:600;">144 GB</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center;">—</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center;">—</td>
    </tr>
    <tr style="background:rgba(25, 113, 194, 0.05);">
      <td style="padding:12px; border:2px dashed #555; font-weight:600; color:#1971c2;">ZeRO-1</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center;">$(10 + 8/8)\phi = 11\phi$</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center; font-weight:600;">88 GB</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center; color:#1971c2; font-weight:600;">38%</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center;">56 GB</td>
    </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 5: Memory Savings — DDP vs ZeRO-1 for Llama 3 8B ($\phi$ = 8B, N = 8 GPUs)</figcaption>
</figure>

## ZeRO-2 Sharding Gradients

While ZeRO-1 shards optimizer states, ZeRO-2 goes further by also sharding gradients. After ReduceScatter in ZeRO-1, each GPU's gradient buffer stays allocated at full size ($4\phi$) — but only $4\phi/N$ contains meaningful data (the owned averaged gradient shard). The remaining $4\phi - 4\phi/N$ sits allocated but unused until the next iteration begins. ZeRO-2 deallocates this unused portion immediately after ReduceScatter, so each GPU retains only the gradient shard it needs for its local optimizer step.

The communication pattern is identical to ZeRO-1 — ReduceScatter followed by AllGather — and the training loop is unchanged. The only difference is one deallocation call after ReduceScatter. For the full example workflow and training steps, refer to the [ZeRO-1 section](#zero-1-sharding-optimizer-states).

**Training path:**
> Forward Pass → Backward Pass → ReduceScatter → Discard Non-Owned Gradients → Local Optimizer Step → AllGather

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0;">

<div>
<h4 style="text-align:center; margin-bottom:12px; color:#555;">Before ReduceScatter</h4>
<p style="text-align:center; font-size:12px; color:#888; margin-bottom:12px;">Each GPU holds local gradients for all parameters</p>
<table style="width:100%; border-collapse:collapse; font-size:12px; border:2px solid #6c757d;">
<thead>
  <tr style="background:#6c757d; color:white;">
    <th style="padding:8px; border:1px solid #6c757d;">GPU</th>
    <th style="padding:8px; border:1px solid #6c757d;">$\nabla\text{W}_1$</th>
    <th style="padding:8px; border:1px solid #6c757d;">$\nabla\text{W}_2$</th>
    <th style="padding:8px; border:1px solid #6c757d;">$\nabla\text{W}_3$</th>
    <th style="padding:8px; border:1px solid #6c757d;">$\nabla\text{W}_4$</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-0</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_1^0$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_2^0$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_3^0$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_4^0$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-1</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_1^1$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_2^1$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_3^1$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_4^1$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-2</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_1^2$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_2^2$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_3^2$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_4^2$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-3</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_1^3$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_2^3$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_3^3$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla\text{W}_4^3$</td>
  </tr>
</tbody>
</table>
<p style="text-align:center; font-size:11px; color:#666; margin-top:8px;">Buffer: $4\phi$ per GPU — fully replicated</p>
</div>

<div>
<h4 style="text-align:center; margin-bottom:12px; color:#2f9e44;">After ReduceScatter + Deallocation</h4>
<p style="text-align:center; font-size:12px; color:#888; margin-bottom:12px;">Each GPU keeps only its averaged shard</p>
<table style="width:100%; border-collapse:collapse; font-size:12px; border:2px solid #2f9e44;">
<thead>
  <tr style="background:#2f9e44; color:white;">
    <th style="padding:8px; border:1px solid #2f9e44;">GPU</th>
    <th style="padding:8px; border:1px solid #2f9e44;">Owned Shard</th>
    <th style="padding:8px; border:1px solid #2f9e44;">Averaging Formula</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-0</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#d4edda; font-weight:600;">$\bar{\nabla}\text{W}_1$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-size:11px;">$\bar{\nabla}\text{W}_1 = \frac{1}{N}\sum_{i=0}^{N-1}\nabla\text{W}_1^i$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-1</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#d4edda; font-weight:600;">$\bar{\nabla}\text{W}_2$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-size:11px;">$\bar{\nabla}\text{W}_2 = \frac{1}{N}\sum_{i=0}^{N-1}\nabla\text{W}_2^i$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-2</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#d4edda; font-weight:600;">$\bar{\nabla}\text{W}_3$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-size:11px;">$\bar{\nabla}\text{W}_3 = \frac{1}{N}\sum_{i=0}^{N-1}\nabla\text{W}_3^i$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-3</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#d4edda; font-weight:600;">$\bar{\nabla}\text{W}_4$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-size:11px;">$\bar{\nabla}\text{W}_4 = \frac{1}{N}\sum_{i=0}^{N-1}\nabla\text{W}_4^i$</td>
  </tr>
</tbody>
</table>
<p style="text-align:center; font-size:11px; color:#2f9e44; margin-top:8px; font-weight:600;">Buffer: $4\phi/N$ per GPU — $(N-1)/N$ freed</p>
</div>

</div>

<p style="text-align:center; margin:16px 0; font-size:13px;">
<span style="display:inline-block; width:12px; height:12px; background:#fff3cd; border:1px solid #ddd; margin-right:4px; vertical-align:middle;"></span> Local gradient (pre-averaging)
&nbsp;&nbsp;
<span style="display:inline-block; width:12px; height:12px; background:#d4edda; border:1px solid #ddd; margin-right:4px; vertical-align:middle;"></span> Globally averaged gradient (retained)
</p>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0;">

<div>
<h4 style="text-align:center; margin-bottom:12px; color: var(--global-theme-color);">After ReduceScatter — ZeRO-1</h4>
<table style="width:100%; border-collapse:collapse; font-size:13px; border:2px solid #1971c2;">
<thead>
  <tr style="background:#1971c2; color:white;">
    <th style="padding:8px; border:1px solid #1971c2;">GPU</th>
    <th style="padding:8px; border:1px solid #1971c2;">Shard 0</th>
    <th style="padding:8px; border:1px solid #1971c2;">Shard 1</th>
    <th style="padding:8px; border:1px solid #1971c2;">Shard 2</th>
    <th style="padding:8px; border:1px solid #1971c2;">Shard 3</th>
    <th style="padding:8px; border:1px solid #1971c2;">Buffer</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-0</td>
    <td style="padding:8px; border:1px solid #ddd; background:#d4edda; text-align:center;">$\bar{\nabla}\text{W}_1$</td>
    <td style="padding:8px; border:1px solid #ddd; background:#f8d7da; text-align:center; color:#999;">unused</td>
    <td style="padding:8px; border:1px solid #ddd; background:#f8d7da; text-align:center; color:#999;">unused</td>
    <td style="padding:8px; border:1px solid #ddd; background:#f8d7da; text-align:center; color:#999;">unused</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center;">$4\phi$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-1</td>
    <td style="padding:8px; border:1px solid #ddd; background:#f8d7da; text-align:center; color:#999;">unused</td>
    <td style="padding:8px; border:1px solid #ddd; background:#d4edda; text-align:center;">$\bar{\nabla}\text{W}_2$</td>
    <td style="padding:8px; border:1px solid #ddd; background:#f8d7da; text-align:center; color:#999;">unused</td>
    <td style="padding:8px; border:1px solid #ddd; background:#f8d7da; text-align:center; color:#999;">unused</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center;">$4\phi$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-2</td>
    <td style="padding:8px; border:1px solid #ddd; background:#f8d7da; text-align:center; color:#999;">unused</td>
    <td style="padding:8px; border:1px solid #ddd; background:#f8d7da; text-align:center; color:#999;">unused</td>
    <td style="padding:8px; border:1px solid #ddd; background:#d4edda; text-align:center;">$\bar{\nabla}\text{W}_3$</td>
    <td style="padding:8px; border:1px solid #ddd; background:#f8d7da; text-align:center; color:#999;">unused</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center;">$4\phi$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-3</td>
    <td style="padding:8px; border:1px solid #ddd; background:#f8d7da; text-align:center; color:#999;">unused</td>
    <td style="padding:8px; border:1px solid #ddd; background:#f8d7da; text-align:center; color:#999;">unused</td>
    <td style="padding:8px; border:1px solid #ddd; background:#f8d7da; text-align:center; color:#999;">unused</td>
    <td style="padding:8px; border:1px solid #ddd; background:#d4edda; text-align:center;">$\bar{\nabla}\text{W}_4$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center;">$4\phi$</td>
  </tr>
</tbody>
</table>
<p style="text-align:center; font-size:12px; color:#666; margin-top:8px;">Full buffer allocated — $(N-1)/N$ freed per GPU</p>
</div>

<div>
<h4 style="text-align:center; margin-bottom:12px; color: var(--global-theme-color);">After ReduceScatter — ZeRO-2</h4>
<table style="width:100%; border-collapse:collapse; font-size:13px; border:2px solid #2f9e44;">
<thead>
  <tr style="background:#2f9e44; color:white;">
    <th style="padding:8px; border:1px solid #2f9e44;">GPU</th>
    <th style="padding:8px; border:1px solid #2f9e44;">Owned Shard</th>
    <th style="padding:8px; border:1px solid #2f9e44;">Buffer</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-0</td>
    <td style="padding:8px; border:1px solid #ddd; background:#d4edda; text-align:center;">$\bar{\nabla}\text{W}_1$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-weight:600; color:#2f9e44;">$4\phi/N$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-1</td>
    <td style="padding:8px; border:1px solid #ddd; background:#d4edda; text-align:center;">$\bar{\nabla}\text{W}_2$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-weight:600; color:#2f9e44;">$4\phi/N$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-2</td>
    <td style="padding:8px; border:1px solid #ddd; background:#d4edda; text-align:center;">$\bar{\nabla}\text{W}_3$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-weight:600; color:#2f9e44;">$4\phi/N$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-3</td>
    <td style="padding:8px; border:1px solid #ddd; background:#d4edda; text-align:center;">$\bar{\nabla}\text{W}_4$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-weight:600; color:#2f9e44;">$4\phi/N$</td>
  </tr>
</tbody>
</table>
<p style="text-align:center; font-size:12px; color:#666; margin-top:8px;">Non-owned shards deallocated — $(N-1)/N$ memory freed</p>
</div>

</div>

<p style="text-align:center; font-size:13px; margin-top:16px;"><em>$\bar{\nabla}\text{W}_i$ denotes the globally averaged gradient for shard $i$</em></p>


### Memory Comparison: DDP vs ZeRO-1 vs ZeRO-2

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:13px; border:2px dashed #555;">
<thead>
    <tr>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:left;" rowspan="2">Component</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#dc3545; color:white;" colspan="2">DDP</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#1971c2; color:white;" colspan="2">ZeRO-1</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#2f9e44; color:white;" colspan="2">ZeRO-2</th>
    </tr>
    <tr>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Memory</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Distribution</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Memory</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Distribution</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Memory</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Distribution</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Parameters BF16</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$2\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$2\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$2\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Parameters FP32</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Gradients FP32</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600; color:#2f9e44;">$4\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">Sharded</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Optimizer States $(m_t, v_t)$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$8\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600; color:#1971c2;">$8\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">Sharded</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600; color:#2f9e44;">$8\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">Sharded</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8); font-weight:600;">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>Total per GPU</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;" colspan="2"><strong>$18\phi$</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;" colspan="2"><strong>$(10 + 8/N)\phi$</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;" colspan="2"><strong>$(6 + 12/N)\phi$</strong></td>
    </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 6: Memory Distribution Comparison — DDP vs ZeRO-1 vs ZeRO-2</figcaption>
</figure>

<p style="text-align:center; font-size:12px; margin-top:8px;">
<span style="display:inline-block; width:12px; height:12px; background:#f8d7da; border:1px solid #ddd; margin-right:4px; vertical-align:middle;"></span> Replicated (redundant across all GPUs)
&nbsp;&nbsp;
<span style="display:inline-block; width:12px; height:12px; background:#d4edda; border:1px solid #ddd; margin-right:4px; vertical-align:middle;"></span> Sharded (distributed across N GPUs)
</p>

### Memory Savings for Llama 3 8B (8 GPUs)

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <th style="padding:12px; border:2px dashed #555; text-align:left;">Approach</th>
      <th style="padding:12px; border:2px dashed #555; text-align:center;">Formula</th>
      <th style="padding:12px; border:2px dashed #555; text-align:center;">Memory/GPU</th>
      <th style="padding:12px; border:2px dashed #555; text-align:center;">Reduction</th>
      <th style="padding:12px; border:2px dashed #555; text-align:center;">Freed/GPU</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:12px; border:2px dashed #555; font-weight:600; color:#dc3545;">DDP</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center;">$18\phi$</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center; font-weight:600;">144 GB</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center;">—</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center;">—</td>
    </tr>
    <tr>
      <td style="padding:12px; border:2px dashed #555; font-weight:600; color:#1971c2;">ZeRO-1</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center;">$(10 + 8/8)\phi = 11\phi$</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center; font-weight:600;">88 GB</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center; color:#1971c2; font-weight:600;">38%</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center;">56 GB</td>
    </tr>
    <tr style="background:rgba(47, 158, 68, 0.05);">
      <td style="padding:12px; border:2px dashed #555; font-weight:600; color:#2f9e44;">ZeRO-2</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center;">$(6 + 12/8)\phi = 7.5\phi$</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center; font-weight:600;">60 GB</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center; color:#2f9e44; font-weight:600;">58%</td>
      <td style="padding:12px; border:2px dashed #555; text-align:center;">84 GB</td>
    </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 7: Memory Savings for Llama 3 8B ($\phi$ = 8B parameters, N = 8 GPUs)</figcaption>
</figure>


> ZeRO-2 saves memory by $\frac{4(N-1)}{N}\phi$ Bytes by sharding optimizer states & gradients without extra communication overhead

## ZeRO-3 Sharding Parameters

ZeRO-1 shards optimizer states; ZeRO-2 additionally shards gradients. One component remains fully replicated on every GPU after both stages: the parameter tensors — $2\phi$ bytes of BF16 working copy and $4\phi$ bytes of FP32 master copy per device. ZeRO-3 eliminates this final redundancy by partitioning parameters across N GPUs, so each GPU owns and stores only $\phi/N$ parameters. Since the optimizer step updates only owned parameters — and gradients and optimizer states are already sharded across GPUs — no full parameter copy is needed between iterations.

### Communication Pattern

ZeRO-3 changes the communication pattern significantly compared to ZeRO-1 and ZeRO-2.

In ZeRO-1 and ZeRO-2, parameters were fully replicated — no AllGather was needed during the forward or backward pass. AllGather only appeared once per iteration to synchronize updated parameters.

In ZeRO-3, parameters are sharded. Every GPU holds only its own parameter shard. To compute the forward pass, each layer's parameters must be gathered from all GPUs, used, then immediately freed. The same happens during the backward pass. This means AllGather happens 
per layer, twice per iteration — once forward, once backward.

This is ZeRO-3's fundamental trade-off: memory is traded for communication.

**Training path:**

 → Forward Pass (per-layer AllGather) → Backward Pass (per-layer AllGather) → ReduceScatter → Local Optimizer Step


<span class="step-tag"><span class="step-num">1</span>Forward Pass — Per-Layer AllGather</span>

Each GPU holds only its parameter shard. To compute the forward pass, parameters must be reconstructed layer-by-layer. For each layer, AllGather collects the parameter shard from its owner GPU and broadcasts it to all GPUs. Once the layer computation completes, non-owned parameters are immediately discarded to free memory. This gather-compute-discard cycle repeats for every layer in the model.

<span class="step-tag"><span class="step-num">2</span>Backward Pass — Per-Layer AllGather</span>

The backward pass follows the same pattern in reverse layer order. For each layer, AllGather reconstructs the full parameters, gradients are computed locally, then non-owned parameters are discarded. After the backward pass completes, each GPU holds local gradients for all parameters — but only its own parameter shard.

<span class="step-tag"><span class="step-num">3</span>ReduceScatter — Gradient Synchronization</span>

Each GPU computed gradients based on different data, so gradients must be synchronized. ReduceScatter averages gradients across all GPUs and distributes each shard to its responsible GPU. After ReduceScatter, GPU $i$ holds only the globally averaged gradient for shard $i$ — the same result as AllReduce, but without materializing the full gradient tensor on any single GPU.

<span class="step-tag"><span class="step-num">4</span>Local Optimizer Step</span>

Each GPU now has exactly what it needs: its parameter shard, the averaged gradient for that shard, and its optimizer states for that shard. It applies the Adam update locally — no communication required. The parameter shard is updated in place.

<span class="step-tag"><span class="step-num">5</span>End of Iteration — No Final AllGather</span>

Unlike ZeRO-1 and ZeRO-2, there is no AllGather at the end of the iteration. Parameters remain sharded. The full model is never materialized simultaneously on any GPU. The next iteration begins with Step 1, where AllGather reconstructs parameters on-demand during the forward pass.

> Repeat Steps 1–5 until model convergence

### Training Steps
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
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>1.Initialize</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU initializes its own parameter shard locally</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>None</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>2.Data Split</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Global batch is divided into micro-batches, one per GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None (data loader handles this)</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>3.Initialize Optimizer State Shards</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU initializes only its assigned optimizer state shard to zero</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>4.Forward Pass</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU gathers each layer's parameters on demand, computes the forward pass, then discards non-owned parameters</td>
      <td style="padding:10px 12px; border:2px dashed #555;">AllGather Parameters (per layer)</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>5.Backward Pass</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Same per-layer gather pattern in reverse; local gradients computed for each layer, then non-owned parameters discarded</td>
      <td style="padding:10px 12px; border:2px dashed #555;">AllGather Parameters (per layer)</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>6.Gradient Sync</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Gradients averaged across all GPUs; each GPU receives only its assigned gradient shard</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>ReduceScatter</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>7.Optimizer Step</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU updates its parameter shard using its local gradient and optimizer state shards</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 8: ZeRO Stage-3 Training Steps</figcaption>
</figure>


### Example Workflow

For clarity, consider a simplified model with 4 parameter matrices W₁, W₂, W₃, W₄ distributed across 4 GPUs. Llama 3 uses no bias terms so we omit bias here.

<div class="ddp-note">
  <span class="note-label">Key Difference from ZeRO-1/2</span>
  In ZeRO-3, parameters are <strong>sharded </strong>. Each GPU holds only 1/N of the model parameters permanently. To compute forward or backward passes, parameters must be gathered <strong>per-layer</strong>, used, then immediately discarded. This is ZeRO-3's fundamental trade-off: memory savings in exchange for increased communication.
</div>

**1. Parameter Sharding (Initialization)**

Unlike ZeRO-1/2 where all GPUs hold complete parameters, ZeRO-3 partitions parameters across GPUs:

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0;">

<div>
<h4 style="text-align:center; margin-bottom:12px; color:#dc3545;">ZeRO-1/2: Replicated Parameters</h4>
<table style="width:100%; border-collapse:collapse; font-size:12px; border:2px solid #dc3545;">
<thead>
  <tr style="background:#dc3545; color:white;">
    <th style="padding:8px; border:1px solid #dc3545;">GPU</th>
    <th style="padding:8px; border:1px solid #dc3545;">Parameters Held</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-0</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#f8d7da;">W₁, W₂, W₃, W₄</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-1</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#f8d7da;">W₁, W₂, W₃, W₄</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-2</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#f8d7da;">W₁, W₂, W₃, W₄</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-3</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#f8d7da;">W₁, W₂, W₃, W₄</td>
  </tr>
</tbody>
</table>
<p style="text-align:center; font-size:11px; color:#dc3545; margin-top:8px;">Memory: $6\phi$ per GPU (BF16 + FP32)</p>
</div>

<div>
<h4 style="text-align:center; margin-bottom:12px; color:#9c36b5;">ZeRO-3: Sharded Parameters</h4>
<table style="width:100%; border-collapse:collapse; font-size:12px; border:2px solid #9c36b5;">
<thead>
  <tr style="background:#9c36b5; color:white;">
    <th style="padding:8px; border:1px solid #9c36b5;">GPU</th>
    <th style="padding:8px; border:1px solid #9c36b5;">Parameters Held</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-0</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#e9d5f5; font-weight:600;">W₁ only</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-1</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#e9d5f5; font-weight:600;">W₂ only</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-2</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#e9d5f5; font-weight:600;">W₃ only</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-3</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#e9d5f5; font-weight:600;">W₄ only</td>
  </tr>
</tbody>
</table>
<p style="text-align:center; font-size:11px; color:#9c36b5; margin-top:8px; font-weight:600;">Memory: $6\phi/N$ per GPU — $(N-1)/N$ reduction</p>
</div>

</div>

**2. Optimizer State Sharding (same as ZeRO-1/2)**

Each GPU initializes optimizer states only for its owned parameter shard:

- GPU-0: $(m_1, v_1)$ ← optimizer states for W₁
- GPU-1: $(m_2, v_2)$ ← optimizer states for W₂
- GPU-2: $(m_3, v_3)$ ← optimizer states for W₃
- GPU-3: $(m_4, v_4)$ ← optimizer states for W₄

**3. Forward Pass — Per-Layer AllGather**

This is where ZeRO-3 differs fundamentally. Each GPU needs the full model to compute forward pass, but parameters are sharded. The solution: gather parameters **layer-by-layer**, compute, then discard.

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:13px; border:2px solid #9c36b5;">
<thead>
  <tr style="background:#9c36b5; color:white;">
    <th style="padding:10px; border:1px solid #9c36b5;">Layer</th>
    <th style="padding:10px; border:1px solid #9c36b5;">Operation</th>
    <th style="padding:10px; border:1px solid #9c36b5;">Memory State</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="padding:10px; border:1px solid #ddd; font-weight:600;">Layer 1 (W₁)</td>
    <td style="padding:10px; border:1px solid #ddd;">AllGather W₁ from GPU-0 → compute → discard non-owned</td>
    <td style="padding:10px; border:1px solid #ddd; font-size:11px;">Temporary: full W₁ on all GPUs</td>
  </tr>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <td style="padding:10px; border:1px solid #ddd; font-weight:600;">Layer 2 (W₂)</td>
    <td style="padding:10px; border:1px solid #ddd;">AllGather W₂ from GPU-1 → compute → discard non-owned</td>
    <td style="padding:10px; border:1px solid #ddd; font-size:11px;">Temporary: full W₂ on all GPUs</td>
  </tr>
  <tr>
    <td style="padding:10px; border:1px solid #ddd; font-weight:600;">Layer 3 (W₃)</td>
    <td style="padding:10px; border:1px solid #ddd;">AllGather W₃ from GPU-2 → compute → discard non-owned</td>
    <td style="padding:10px; border:1px solid #ddd; font-size:11px;">Temporary: full W₃ on all GPUs</td>
  </tr>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <td style="padding:10px; border:1px solid #ddd; font-weight:600;">Layer 4 (W₄)</td>
    <td style="padding:10px; border:1px solid #ddd;">AllGather W₄ from GPU-3 → compute → discard non-owned</td>
    <td style="padding:10px; border:1px solid #ddd; font-size:11px;">Temporary: full W₄ on all GPUs</td>
  </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:13px; color:#666; margin-top:8px;">Per-layer AllGather during forward pass — parameters gathered on-demand, then discarded</figcaption>
</figure>

After forward pass completes:
- GPU-0: X₀ → Ŷ₀ → Loss₀ (holds only W₁)
- GPU-1: X₁ → Ŷ₁ → Loss₁ (holds only W₂)
- GPU-2: X₂ → Ŷ₂ → Loss₂ (holds only W₃)
- GPU-3: X₃ → Ŷ₃ → Loss₃ (holds only W₄)

**4. Backward Pass — Per-Layer AllGather (Again)**

The backward pass requires the same per-layer AllGather pattern. For each layer, parameters are gathered, gradients computed, then non-owned parameters discarded:

- For each layer $i$ (in reverse order):
  1. **AllGather** $W_i$ from its owner GPU
  2. **Compute** local gradients $\nabla W_i^{\text{local}}$
  3. **Discard** non-owned copy of $W_i$

After backward pass, each GPU holds local gradients for **all** parameters:
- GPU-0: $[\nabla W_1^0, \nabla W_2^0, \nabla W_3^0, \nabla W_4^0]$ ← based on X₀
- GPU-1: $[\nabla W_1^1, \nabla W_2^1, \nabla W_3^1, \nabla W_4^1]$ ← based on X₁
- GPU-2: $[\nabla W_1^2, \nabla W_2^2, \nabla W_3^2, \nabla W_4^2]$ ← based on X₂
- GPU-3: $[\nabla W_1^3, \nabla W_2^3, \nabla W_3^3, \nabla W_4^3]$ ← based on X₃

**5. ReduceScatter — Gradient Synchronization**

Each GPU has different gradients — the model will diverge if these are not synchronized. ReduceScatter averages gradients across all GPUs and distributes each shard to its responsible GPU:

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0;">

<div>
<h4 style="text-align:center; margin-bottom:12px; color:#555;">Before ReduceScatter</h4>
<p style="text-align:center; font-size:12px; color:#888; margin-bottom:12px;">Each GPU holds local gradients for all parameters</p>
<table style="width:100%; border-collapse:collapse; font-size:12px; border:2px solid #6c757d;">
<thead>
  <tr style="background:#6c757d; color:white;">
    <th style="padding:8px; border:1px solid #6c757d;">GPU</th>
    <th style="padding:8px; border:1px solid #6c757d;">$\nabla W_1$</th>
    <th style="padding:8px; border:1px solid #6c757d;">$\nabla W_2$</th>
    <th style="padding:8px; border:1px solid #6c757d;">$\nabla W_3$</th>
    <th style="padding:8px; border:1px solid #6c757d;">$\nabla W_4$</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-0</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_1^0$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_2^0$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_3^0$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_4^0$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-1</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_1^1$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_2^1$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_3^1$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_4^1$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-2</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_1^2$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_2^2$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_3^2$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_4^2$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-3</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_1^3$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_2^3$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_3^3$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#fff3cd;">$\nabla W_4^3$</td>
  </tr>
</tbody>
</table>
<p style="text-align:center; font-size:11px; color:#666; margin-top:8px;">Buffer: $4\phi$ per GPU</p>
</div>

<div>
<h4 style="text-align:center; margin-bottom:12px; color:#9c36b5;">After ReduceScatter</h4>
<p style="text-align:center; font-size:12px; color:#888; margin-bottom:12px;">Each GPU receives only its averaged shard</p>
<table style="width:100%; border-collapse:collapse; font-size:12px; border:2px solid #9c36b5;">
<thead>
  <tr style="background:#9c36b5; color:white;">
    <th style="padding:8px; border:1px solid #9c36b5;">GPU</th>
    <th style="padding:8px; border:1px solid #9c36b5;">Owned Gradient</th>
    <th style="padding:8px; border:1px solid #9c36b5;">Formula</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-0</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#e9d5f5; font-weight:600;">$\bar{\nabla}W_1$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-size:11px;">$\frac{1}{N}\sum_{i=0}^{N-1}\nabla W_1^i$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-1</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#e9d5f5; font-weight:600;">$\bar{\nabla}W_2$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-size:11px;">$\frac{1}{N}\sum_{i=0}^{N-1}\nabla W_2^i$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-2</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#e9d5f5; font-weight:600;">$\bar{\nabla}W_3$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-size:11px;">$\frac{1}{N}\sum_{i=0}^{N-1}\nabla W_3^i$</td>
  </tr>
  <tr>
    <td style="padding:8px; border:1px solid #ddd; font-weight:600;">GPU-3</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; background:#e9d5f5; font-weight:600;">$\bar{\nabla}W_4$</td>
    <td style="padding:8px; border:1px solid #ddd; text-align:center; font-size:11px;">$\frac{1}{N}\sum_{i=0}^{N-1}\nabla W_4^i$</td>
  </tr>
</tbody>
</table>
<p style="text-align:center; font-size:11px; color:#9c36b5; margin-top:8px; font-weight:600;">Buffer: $4\phi/N$ per GPU — non-owned gradients discarded</p>
</div>

</div>

<p style="text-align:center; margin:16px 0; font-size:13px;">
<span style="display:inline-block; width:12px; height:12px; background:#fff3cd; border:1px solid #ddd; margin-right:4px; vertical-align:middle;"></span> Local gradient (pre-averaging)
&nbsp;&nbsp;
<span style="display:inline-block; width:12px; height:12px; background:#e9d5f5; border:1px solid #ddd; margin-right:4px; vertical-align:middle;"></span> Globally averaged gradient (retained)
</p>

**6. Local Optimizer Step**

Each GPU now has exactly what it needs to update its parameter shard:
- The globally averaged gradient for its shard ($\bar{\nabla}W_i$)
- Its local optimizer states for that shard ($(m_i, v_i)$)
- Its local parameter shard ($W_i$)

<d-aside>
  <b>Adam update — GPU-0 updating W₁</b>
  <pre style="font-size:0.75rem; margin-top:6px; line-height:1.6;">m₁ = β₁·m₁ + (1-β₁)·∇̄W₁   ← 1st moment
v₁ = β₂·v₁ + (1-β₂)·(∇̄W₁)² ← 2nd moment

m̂₁ = m₁ / (1-β₁ᵗ)  ← bias correction
v̂₁ = v₁ / (1-β₂ᵗ)

W₁ = W₁ - η·m̂₁/(√v̂₁+ε)  ← parameter update</pre>
  <p style="font-size:0.75rem; margin:6px 0 0;">$\beta_1$, $\beta_2$ — moment decay rates; $\eta$ — learning rate; $\epsilon$ — numerical stability constant. Each GPU runs this identically for its own shard.</p>
</d-aside>

**7. End of Iteration — Parameters Remain Sharded**

<div class="ddp-note">
  <span class="note-label">Critical Difference from ZeRO-1/2</span>
  Unlike ZeRO-1/2, there is <strong>no final AllGather</strong> to synchronize parameters. Each GPU keeps only its updated parameter shard. The full model is reconstructed on-demand during the next iteration's forward pass via per-layer AllGather operations.
</div>

After optimizer step:
- GPU-0: updated W₁ (holds only W₁)
- GPU-1: updated W₂ (holds only W₂)
- GPU-2: updated W₃ (holds only W₃)
- GPU-3: updated W₄ (holds only W₄)

The iteration is complete. Parameters remain sharded. The next iteration begins with Step 4 (Forward Pass), where AllGather reconstructs parameters layer-by-layer as needed.

### Communication Cost Trade-off

ZeRO-3's memory savings come at the cost of increased communication:

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:13px; border:2px dashed #555;">
<thead>
  <tr style="background:var(--global-code-bg-color, #f8f8f8);">
    <th style="padding:10px 12px; border:2px dashed #555; text-align:left;">Stage</th>
    <th style="padding:10px 12px; border:2px dashed #555; text-align:center;">AllGather Calls/Iteration</th>
    <th style="padding:10px 12px; border:2px dashed #555; text-align:center;">ReduceScatter Calls/Iteration</th>
    <th style="padding:10px 12px; border:2px dashed #555; text-align:center;">Total Volume</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#1971c2;">ZeRO-1</td>
    <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">1 (after optimizer)</td>
    <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">1 (gradient sync)</td>
    <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$2\phi$</td>
  </tr>
  <tr>
    <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#2f9e44;">ZeRO-2</td>
    <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">1 (after optimizer)</td>
    <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">1 (gradient sync)</td>
    <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$2\phi$</td>
  </tr>
  <tr style="background:rgba(156, 54, 181, 0.05);">
    <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#9c36b5;">ZeRO-3</td>
    <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600;">2L (L per forward + L per backward)</td>
    <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">1 (gradient sync)</td>
    <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600;">$3\phi$</td>
  </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:13px; color:#666; margin-top:8px;">ZeRO-3 trades memory for communication — performs 2L AllGather calls per iteration — each communicating $\phi/L$ bytes (one layer's parameters). Across all 2L calls this totals $2\phi$ bytes, plus $\phi$ for ReduceScatter — giving $3\phi$ total versus $2\phi$ for ZeRO-1/2.</figcaption>
</figure>

### Memory Comparison: DDP vs ZeRO-1 vs ZeRO-2 vs ZeRO-3

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:13px; border:2px dashed #555;">
<thead>
    <tr>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:left;" rowspan="2">Component</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#dc3545; color:white;" colspan="2">DDP</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#1971c2; color:white;" colspan="2">ZeRO-1</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#2f9e44; color:white;" colspan="2">ZeRO-2</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#9c36b5; color:white;" colspan="2">ZeRO-3</th>
    </tr>
    <tr>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Memory</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Distribution</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Memory</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Distribution</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Memory</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Distribution</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Memory</th>
      <th style="padding:8px; border:2px dashed #555; text-align:center; font-size:11px;">Distribution</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Parameters BF16</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$2\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$2\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$2\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$2\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">Sharded</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Parameters FP32</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">Sharded</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Gradients FP32</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600; color:#2f9e44;">$4\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">Sharded</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600; color:#2f9e44;">$4\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">Sharded</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Optimizer States $(m_t, v_t)$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$8\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">Replicated</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600; color:#1971c2;">$8\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">Sharded</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600; color:#2f9e44;">$8\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">Sharded</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600; color:#2f9e44;">$8\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">Sharded</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8); font-weight:600;">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>Total per GPU</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;" colspan="2"><strong>$18\phi$</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;" colspan="2"><strong>$(10 + 8/N)\phi$</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;" colspan="2"><strong>$(6 + 12/N)\phi$</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;" colspan="2"><strong>$(18/N)\phi$</strong></td>
    </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 9: Memory Distribution Comparison — DDP vs ZeRO-1 vs ZeRO-2 vs ZeRO-3</figcaption>
</figure>


## Summary

Training large language models faces two fundamental constraints: **time** and **memory**. A single GPU cannot hold the entire model state — parameters, gradients, and optimizer states for Llama 3 8B alone require 144 GB, nearly double an H100's 80 GB VRAM. Parallelism techniques address both constraints by distributing computation and model state across multiple GPUs.

### The Communication Foundation

Splitting data and model state across GPUs introduces a coordination challenge: without synchronized communication, each GPU would train a divergent copy of the model. NVIDIA's NCCL (NVIDIA Collective Communications Library) provides the communication primitives that keep GPUs synchronized:

| Operation | Purpose | Used In |
|-----------|---------|---------|
| **Broadcast** | Send data from one GPU to all others | DDP initialization |
| **AllReduce** | Sum/average across GPUs, result to all | DDP gradient sync |
| **ReduceScatter** | Reduce then distribute different shards | ZeRO-1/2/3 |
| **AllGather** | Collect shards from all GPUs, result to all | ZeRO-1/2/3 |

### Evolution from DDP to ZeRO

This section covered Data Parallelism and its memory-optimized extensions through ZeRO (Zero Redundancy Optimizer). Each stage progressively eliminates redundancy:

### Vanilla DDP

The complete model is replicated on every GPU. The global batch is split into micro-batches — one per GPU. Each GPU runs its forward and backward pass independently, then 
gradients are synchronized across all GPUs before the optimizer step.

> Training Path: Forward Pass → Backward Pass → ReduceScatter Gradients → Local Optimizer Step

**The limitation:** Every component — parameters, gradients, and optimizer states — is fully replicated across every GPU. With 8 GPUs the cluster holds 8 × 144 GB = 1.15 TB of static 
memory when 144 GB would logically suffice.

### ZeRO Stage-1 — Shard Optimizer States

Each GPU initializes and maintains only 1/N of the optimizer states. The gradient synchronization mechanism changes from AllReduce to ReduceScatter — each GPU receives only the averaged gradient shard for its assigned parameters. After the local optimizer step, AllGather reconstructs the full parameter set for the next forward pass.

> Training Path:Forward Pass → Backward Pass → ReduceScatter Gradients → Local Optimizer 
> Step → AllGather Parameters


### ZeRO Stage-2 — Shard Gradients

Identical to ZeRO-1 with one addition: after ReduceScatter, non-owned gradient buffer space is immediately deallocated. Each GPU retains only the gradient shard it needs for its 
local optimizer step — reducing gradient memory from $4\phi$ to $4\phi/N$ per GPU.

Training Path:
> Forward Pass → Backward Pass → ReduceScatter Gradients → Discard non-owned gradient shards
> → Local Optimizer Step → AllGather Parameters

### ZeRO Stage-3 — Shard Parameters

All three components are now sharded — parameters, gradients, and optimizer states. Each GPU holds only 1/N of everything at rest. This breaks DDP's fundamental requirement — the model no longer needs to fit on a single GPU.

The forward and backward passes require per-layer AllGather operations — parameters are fetched layer by layer, used, then immediately freed. This gather-compute-discard cycle 
repeats for every layer in both passes.

> Training Path: AllGather Parameters (per layer) → Forward Pass → AllGather Parameters (per layer) → 
> Backward Pass → ReduceScatter Gradients → Local Optimizer Step

**The trade-off:** Parameters are no longer replicated — but AllGather now happens 2L times per iteration instead of once.

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:13px; border:2px dashed #555;">
<thead>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <th style="padding:10px 12px; border:2px dashed #555; text-align:left;">Strategy</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:left;">What's Sharded</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:left;">Training Path</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#dc3545;">Vanilla DDP</td>
      <td style="padding:10px 12px; border:2px dashed #555;">Data only (model fully replicated)</td>
      <td style="padding:10px 12px; border:2px dashed #555; font-size:12px;">Forward → Backward → <strong>AllReduce</strong> Gradients → Optimizer</td>
    </tr>
    <tr style="background:rgba(25, 113, 194, 0.03);">
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#1971c2;">ZeRO-1</td>
      <td style="padding:10px 12px; border:2px dashed #555;">Data + Optimizer States</td>
      <td style="padding:10px 12px; border:2px dashed #555; font-size:12px;">Forward → Backward → <strong>ReduceScatter</strong> Gradients → Optimizer → <strong>AllGather</strong> Params</td>
    </tr>
    <tr style="background:rgba(47, 158, 68, 0.03);">
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#2f9e44;">ZeRO-2</td>
      <td style="padding:10px 12px; border:2px dashed #555;">Data + Optimizer States + Gradients</td>
      <td style="padding:10px 12px; border:2px dashed #555; font-size:12px;">Forward → Backward → <strong>ReduceScatter</strong> Gradients → Optimizer → <strong>AllGather</strong> Params</td>
    </tr>
    <tr style="background:rgba(156, 54, 181, 0.03);">
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#9c36b5;">ZeRO-3</td>
      <td style="padding:10px 12px; border:2px dashed #555;">Data + Optimizer States + Gradients + Parameters</td>
      <td style="padding:10px 12px; border:2px dashed #555; font-size:12px;"><strong>AllGather</strong> Params (per layer) → Forward → <strong>AllGather</strong> Params (per layer) → Backward → <strong>ReduceScatter</strong> Gradients → Optimizer</td>
    </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:13px; color:#666; margin-top:8px;">Table: Training path evolution from DDP to ZeRO-3</figcaption>
</figure>

### Memory Footprint Comparison

Each ZeRO stage reduces per-GPU memory by eliminating redundant copies:

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:13px; border:2px dashed #555;">
<thead>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <th style="padding:10px 12px; border:2px dashed #555; text-align:left;">Strategy</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center;">Parameters</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center;">Gradients</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center;">Optimizer States</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center;">Total per GPU</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#dc3545;">Vanilla DDP</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">$6\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">$8\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600;">$18\phi$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#1971c2;">ZeRO-1</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">$6\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">$8\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600;">$(10 + 8/N)\phi$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#2f9e44;">ZeRO-2</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#f8d7da;">$6\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">$4\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">$8\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600;">$(6 + 12/N)\phi$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#9c36b5;">ZeRO-3</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">$6\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">$4\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; background:#d4edda;">$8\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600;">$18\phi/N$</td>
    </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:13px; color:#666; margin-top:8px;">$\phi$ = number of parameters, N = number of GPUs. <span style="background:#f8d7da; padding:2px 6px; border-radius:3px;">Replicated</span> <span style="background:#d4edda; padding:2px 6px; border-radius:3px;">Sharded</span></figcaption>
</figure>

### Communication Cost Comparison

Memory savings come with communication trade-offs:

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:13px; border:2px dashed #555;">
<thead>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <th style="padding:10px 12px; border:2px dashed #555; text-align:left;">Strategy</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center;">ReduceScatter</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center;">AllGather</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center;">Total Volume/Iteration</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#dc3545;">Vanilla DDP</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">1× (gradients)</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">1× (gradients)</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$2\phi$ (AllReduce)</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#1971c2;">ZeRO-1</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">1× (gradients)</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">1× (parameters)</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$2\phi$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#2f9e44;">ZeRO-2</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">1× (gradients)</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">1× (parameters)</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">$2\phi$</td>
    </tr>
    <tr style="background:rgba(156, 54, 181, 0.03);">
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#9c36b5;">ZeRO-3</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">1× (gradients)</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600;">2L× (params per layer)</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; font-weight:600;">$3\phi$</td>
    </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:13px; color:#666; margin-top:8px;">L = number of layers. ZeRO-3 trades memory for communication — 2L AllGather calls (forward + backward) vs 1 in ZeRO-1/2.</figcaption>
</figure>

### Llama 3 8B: Concrete Memory Numbers (8 GPUs)

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:13px; border:2px dashed #555;">
<thead>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <th style="padding:10px 12px; border:2px dashed #555; text-align:left;">Strategy</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center;">Memory per GPU</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center;">Reduction vs DDP</th>
      <th style="padding:10px 12px; border:2px dashed #555; text-align:center;">Fits on H100 (80GB)?</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#dc3545;">Vanilla DDP</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">144 GB</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">—</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; color:#dc3545; font-weight:600;">✗ No</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#1971c2;">ZeRO-1</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">88 GB</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">38%</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; color:#dc3545; font-weight:600;">✗ No</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#2f9e44;">ZeRO-2</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">60 GB</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">58%</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; color:#2f9e44; font-weight:600;">✓ Yes</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555; font-weight:600; color:#9c36b5;">ZeRO-3</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">18 GB</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center;">87%</td>
      <td style="padding:10px 12px; border:2px dashed #555; text-align:center; color:#2f9e44; font-weight:600;">✓ Yes</td>
    </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:13px; color:#666; margin-top:8px;">Llama 3 8B ($\phi$ = 8B parameters) on 8 GPUs. Memory shown is static only (excludes activations).</figcaption>
</figure>

### Key Takeaways

1. **Vanilla DDP** replicates everything — fast but memory-inefficient. Works only when the full model fits on a single GPU.

2. **ZeRO-1** shards optimizer states (44% of memory) with no extra communication cost. First choice when DDP doesn't fit.

3. **ZeRO-2** additionally shards gradients by deallocating non-owned shards after ReduceScatter. Same communication as ZeRO-1.

4. **ZeRO-3** shards parameters themselves, enabling models that don't fit even with ZeRO-2. Trade-off: 2L AllGather calls per iteration (one per layer for forward and backward).

The choice between stages depends on model size and available GPU memory. For Llama 3 8B on 8× H100s, ZeRO-2 provides sufficient memory savings with minimal communication overhead. For larger models like Llama 3 405B, ZeRO-3 becomes necessary.

