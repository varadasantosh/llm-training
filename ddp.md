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
      - name: DDP Limitations
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

NCCL exposes a set of collective operations — each designed for a specific communication pattern. These operations are core of the distributted training process, frameworks like PyTorch create warppers around these libraries to abstract communication frameworks, this is the core software engineering practice, if there are no abstractions provided by PyTorch, we need to implement seperate logic for each ML accelerators (NVIDIA GPU, AMD GPU, Google TPU, Apple , Intel XPU etc...), this allows us to use `torch.distributed` or `torch.nn.DistributedDataParallel` without worrying about accelerator being used, while it is important for performance and other aspects, ML researcher can solely focus on algorithm and architecture. 

| Operation | What it does | First appears in |
|---|---|---|
| Broadcast | Send from one GPU to all others | DDP initialization |
| AllReduce | Sum/average across all GPUs, result to all | DDP gradient sync |
| ReduceScatter | Reduce then distribute different shards | ZeRO-1/2/3 |
| AllGather | Collect shards from all GPUs, result to all | ZeRO-1/2/3 |
| Send/Recv | Point-to-point between two GPUs | Pipeline Parallelism |

**Braodcast**
Broadcast operation - A single GPU sends the same copy of data or tensors to all GPU , generally a GPU with Rank-0 send tensors to all other Ranks, this means data transfer is limited by bandwidth of Rank-0 GPU, also GPU-0 becomes bottle neck. NCCL tries to reduce this dependency on single GPU by creating logical tree structure on top of flat network topology.

This is generally used in DDP to place replicate model on ALL GPUs before training process starts.

**Reduce Scatter**

**All Gather**

**All Reduce**

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

DDP is effective at solving the time problem. By splitting the batch across N GPUs and running forward and backward passes simultaneously, training throughput approaches linear scaling — but one major drawback with DDP is that model should fit on single GPU, which is not the case with large scale models , entire model can't fit on a single GPU, even though a model can fit on single GPU we are copying all parmeters, gradients, optimizers states to each of the GPU which causes redundancy

But looking carefully at what every GPU is holding, it is evident that static memory is duplicated across every GPU. With 8 GPUs the cluster holds 1.15 TB of static memory. This shows redundant memory issue. Next few sections focus on reducing the redundacy by using ZeRO-I, ZeRO-II, ZeRO-III

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
      <td style="padding:10px 12px; border:2px dashed #555;">Optimizer State $m_t$, $v_t FP32$</td>
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


## ZeRO Stage-1 - Sharding Optimizer States

Vanilla DDP shards data across GPUs — each GPU sees a different slice of the training data. ZeRO extends this by also sharding the model state itself: parameters, gradients, and optimizer states. 

Vanilla DDP addressed the time problem effectively but the memory problem remained untouched — for models like Llama 3 8B whose static components alone require 144 GB, these are replicated across GPUs. With 8 GPUs the cluster holds 1.15 TB of static memory when 144GB would be sufficient.

Looking at memory requirements of the model components from [table](#table-2-ddp-memory). The largest single component is optimizer states. Adam's m and v vectors consume $8\phi$ bytes per GPU — 44% of total static memory — fully replicated across every GPU despite each GPU only needing the states for the parameters it owns.

ZeRO Stage-1 targets optimizer states first for two reasons: they are the largest component, and sharding them has no impact on the forward or backward pass — optimizer states are only read and written during the optimizer step. This makes them the safest component to shard without changing any other part of the training loop.

### The Communication Pattern

In vanilla DDP, gradient synchronization uses AllReduce — a single operation that averages gradients and returns the full averaged tensor to every GPU.

ZeRO Stage-1  decomposes **AllReduce** it into its two constituent operations, ReduceScatter and AllGather, and inserts the optimizer step between them. This modification is what enables the memory saving: each GPU only needs to retain the gradient shard it is responsible for, discarding the rest before the optimizer step.

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

After the optimizer step, each GPU holds an updated version of its parameter shard. But every GPU needs the complete updated model for the next forward pass. AllGather broadcasts 
each GPU's updated shard to all other GPUs, reconstructing the full parameter set on every GPU.

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

   Before ReduceScatter — every GPU holds all local gradients:

   - GPU-0: [∇W₁⁰, ∇W₂⁰, ∇W₃⁰, ∇W₄⁰]
   - GPU-1: [∇W₁¹, ∇W₂¹, ∇W₃¹, ∇W₄¹]
   - GPU-2: [∇W₁², ∇W₂², ∇W₃², ∇W₄²]
   - GPU-3: [∇W₁³, ∇W₂³, ∇W₃³, ∇W₄³]

   After ReduceScatter — each GPU holds only its averaged shard:
   - GPU-0: ∇W₁_avg = (∇W₁⁰ + ∇W₁¹ + ∇W₁² + ∇W₁³) / 4
   - GPU-1: ∇W₂_avg = (∇W₂⁰ + ∇W₂¹ + ∇W₂² + ∇W₂³) / 4
   - GPU-2: ∇W₃_avg = (∇W₃⁰ + ∇W₃¹ + ∇W₃² + ∇W₃³) / 4
   - GPU-3: ∇W₄_avg = (∇W₄⁰ + ∇W₄¹ + ∇W₄² + ∇W₄³) / 4

6. Each GPU receives the globally averaged gradient — identical to what AllReduce would have produced — just for its own shard. Each GPU now has two components required to update parameters for next iteration

   - The globally averaged gradient for its shard
   - Its local optimizer states for that same shard

   GPU-0 updates W₁ using its local (m₁, v₁) and ∇W₁_avg. Each GPU applies the same Adam update rule to its own shard — the mechanics are identical, only the parameters differ.

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
   - GPU-0: updated W₁  
   - GPU-1: updated W₂
   - GPU-2: updated W₃  
   - GPU-3: updated W₄
  
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
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>1. Initialize</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Model parameters broadcast from rank 0 to all GPUs. Gradients initialized to zero.</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>Broadcast from rank 0</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>2. Data Split</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Global batch is divided into micro-batches, one per GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None (data loader handles this)</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>3. Initialize Optimizer State Shards</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU initializes only its assigned optimizer state shard to zero</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>4. Forward Pass</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU computes forward pass on its micro-batch independently</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>5. Backward Pass</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU locally computes gradients for its micro-batch</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>6. Gradient Sync</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Gradients averaged across all GPUs; each GPU receives only its assigned gradient shard</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>ReduceScatter</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>7. Optimizer Step</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU updates its parameter shard using its local gradient and optimizer state shards</td>
      <td style="padding:10px 12px; border:2px dashed #555;">None</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>8. Gather Updated Params</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Updated parameter shards broadcast to all GPUs; full model reconstructed</td>
      <td style="padding:10px 12px; border:2px dashed #555;">AllGather</td>
    </tr>
    
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 3: ZeRO Stage-1 Training Steps</figcaption>
</figure>

### Memory Requirements
Table below captures the memory requirements for ZeRO Stage-1, memory required for storing optimizer states is now reduced by the factor of number of GPUs

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555;">Component</th>
      <th style="padding:10px 12px; border:2px dashed #555;">ZeRO-1</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Replicated?</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Parameters BF16 (working copy)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$2\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">✓ Every GPU</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Parameters FP32 (master copy) </td>
      <td style="padding:10px 12px; border:2px dashed #555;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">✓ Every GPU</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Gradients FP32</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$4\phi$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">✓ Every GPU</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Optimizer State $m_t$ ,$v_t$ FP32 </td>
      <td style="padding:10px 12px; border:2px dashed #555;">$8\phi/N$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">✗ Sharded across N GPUs</td>
    </tr>
    
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>Total</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>$(10 + 8/N)\phi$</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;"></td>
    </tr>
</tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 4: ZeRO Stage-1 Memory Requirements</figcaption>
</figure>

**Memory Calculation for Llama 3 8B (8 GPUs):**
- DDP: $18\phi$ = 144 GB per GPU
- ZeRO-1: $(10 + 8/8)\phi = 11\phi$ ≈ 88 GB per GPU
- **Saving: 38% reduction — 56 GB freed per GPU**

## ZeRO-2 - Sharding Gradients

While ZeRO-1 shards optimizer states, ZeRO-2 further optimizes memory by sharding the gradients themselves. ZeRO-1 reduces the memory footprint by approximately 38%, but each GPU still holds a full set of gradients during the backward pass.

ZeRO-2 exploits a key insight from ZeRO-1: After ReduceScatter in ZeRO-1, each GPU's gradient buffer remains allocated at full size ($4\phi$) - but only the owned shard contains 
meaningful data (average gradients required for training). The remaining $4\phi$ - $4\phi/N$ sits allocated but unused until zero_grad() clears it. GPU memory is scarce resource, holding $4\phi$ when only $4\phi/N$ is needed is wasting resources. After ReduceScatter ZeRO-2 immediately deallocates memory of unused gradients buffer space. Each GPU retains only the gradient shard it needs for the local optimizer step — eliminating the redundancy
 

The communication pattern, for ZeRO-2 is identical to ZeRO-1 — ReduceScatter followed by AllGather. The difference lies in immediately freeing non-owned gradient buffer space after ReduceScatter.

In ZeRO-1, the Reduce-Scatter averages the relevant gradient shard but leaves the remaining redundant gradients in place, occupying memory . ZeRO-2 deallocate the non-owned shards. This changes gradients from redundant component into a sharded component. Only difference is freeing up the gradient shard memory but the communication pattern in forward and backward passes are identical to ZeRO-1 — refer to that [section](#zero-stage-1---sharding-optimizer-states) for walkthrough of example workflow and training steps.

**Before ReduceScatter:**

  GPU-0:  $\nabla \text{W}_1^0$, $\nabla \text{W}_2^0$, $\nabla \text{W}_3^0$, $\nabla \text{W}_4^0$ 

  GPU-1:  $\nabla \text{W}_1^1$, $\nabla \text{W}_2^1$, $\nabla \text{W}_3^1$, $\nabla \text{W}_4^1$ 

  GPU-2:  $\nabla \text{W}_1^2$, $\nabla \text{W}_2^2$, $\nabla \text{W}_3^2$, $\nabla \text{W}_4^2$ 

  GPU-3:  $\nabla \text{W}_1^3$, $\nabla \text{W}_2^3$, $\nabla \text{W}_3^3$, $\nabla \text{W}_4^3$ 

**After ReduceScatter:**

  GPU-0: $\nabla \text{W}_1\text{_avg}$ = ($\nabla \text{W}_1^0 + \nabla \text{W}_1^1 + \nabla \text{W}_1^2 + \nabla \text{W}_1^3$) / 4

  GPU-1: $\nabla \text{W}_2\text{_avg}$ = ($\nabla \text{W}_2^0 + \nabla \text{W}_2^1 + \nabla \text{W}_2^2 + \nabla \text{W}_2^3$) / 4

  GPU-2: $\nabla \text{W}_3\text{_avg}$ = ($\nabla \text{W}_3^0 + \nabla \text{W}_3^1 + \nabla \text{W}_3^2 + \nabla \text{W}_3^3$) / 4

  GPU-3: $\nabla \text{W}_4\text{_avg}$ = ($\nabla \text{W}_4^0 + \nabla \text{W}_4^1 + \nabla \text{W}_4^2 + \nabla \text{W}_4^3$) / 4


  GPU-0 keeps: $\nabla \text{W}_1 \text{_avg}$ frees: $\nabla \text{W}_2\text{_avg}$, $\nabla \text{W}_3\text{_avg}$, $\nabla \text{W}_4\text{_avg}$

  GPU-1 keeps: $\nabla \text{W}_2 \text{_avg}$ frees: $\nabla \text{W}_1\text{_avg}$, $\nabla \text{W}_3\text{_avg}$, $\nabla \text{W}_4\text{_avg}$

  GPU-2 keeps: $\nabla \text{W}_3\text{_avg}$ frees: $\nabla \text{W}_1\text{_avg}$, $\nabla \text{W}_2\text{_avg}$, $\nabla \text{W}_4\text{_avg}$

  GPU-3 keeps: $\nabla \text{W}_4\text{_avg}$ frees: $\nabla \text{W}_1\text{_avg}$, $\nabla \text{W}_2\text{_avg}$, $\nabla \text{W}_3\text{_avg}$

After ReduceScatter (ZeRO-1):

  GPU-0: $\text{averaged}(\nabla \text{W}_0)$| Unused | Unused | Unused  | full $4\phi$ buffer |
  GPU-1: Unused | $\text{averaged}(\nabla \text{W}_1)$ | Unused | Unused | full $4\phi$ buffer |
  GPU-2: Unused | Unused | $\text{averaged}(\nabla \text{W}_2)$ | Unused | full $4\phi$ buffer |
  GPU-3: Unused | Unused | Unused | $\text{averaged}(\nabla \text{W}_3)$ | full $4\phi$ buffer |

After ReduceScatter (ZeRO-2):

  GPU-0: $\text{averaged}(\nabla \text{W}_0)$  ← buffer reduced to $4\phi/N$ <br>
  GPU-1: $\text{averaged}(\nabla \text{W}_1)$  ← buffer reduced to $4\phi/N$ <br>
  GPU-2: $\text{averaged}(\nabla \text{W}_2)$  ← buffer reduced to $4\phi/N$ <br>
  GPU-3: $\text{averaged}(\nabla \text{W}_3)$  ← buffer reduced to $4\phi/N$ <br>




**Training path:**
> Forward Pass → Backward Pass → ReduceScatter → Discard unused Gradients  → Local Optimizer Step → AllGather



## ZeRO Stage-3


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
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 6: AllReduce as Two Phases — ReduceScatter followed by AllGather</figcaption>
</figure>

Total data transferred per GPU: $2 \times \frac{(N-1)}{N} \times \text{tensor\_size}$. For large N this approaches $2 \times \text{tensor\_size}$, and crucially this cost does not grow with N — adding more GPUs does not increase the per-GPU communication volume.

This is exactly what Vanilla DDP uses for gradient synchronization.

### ReduceScatter

ReduceScatter is the first half of AllReduce, but it is also used independently in ZeRO. It takes a tensor from every GPU, reduces (sums) them, and distributes the result so each GPU ends up with a different shard of the reduced tensor.

**Input:** Every GPU holds the full tensor (e.g., all gradients)  
**Output:** GPU $i$ holds only shard $i$ of the reduced tensor

<!-- Figure 3: ReduceScatter diagram - TODO: Add image -->

ZeRO Stage 2 uses ReduceScatter instead of AllReduce for gradient synchronization. Rather than every GPU computing and storing the full averaged gradient, each GPU only receives the shard it is responsible for. The gradients that each GPU does not own are never materialized — they are reduced in-flight and discarded.

### AllGather

AllGather is the second half of AllReduce, and the most frequently used operation in ZeRO Stages 1, 2, and 3. It takes a different shard from each GPU and assembles the complete tensor on every GPU.

**Input:** GPU $i$ holds shard $i$ (different on every GPU)  
**Output:** Every GPU holds the complete tensor (all shards concatenated)

<!-- Figure 4: AllGather diagram - TODO: Add image -->

Total data transferred per GPU: $\frac{(N-1)}{N} \times \text{tensor\_size}$, again independent of N.

ZeRO uses AllGather to reconstruct parameters on-demand. Since Stage 3 shards the model parameters themselves across GPUs, parameters must be gathered before each forward and backward pass, then discarded immediately after. AllGather is what makes this reconstruction possible without a dedicated parameter server.

---

These three operations are the entire communication vocabulary of ZeRO. Every stage is a different choice of when to use each one — and what to discard afterward.