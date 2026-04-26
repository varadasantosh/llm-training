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
      - name: The Redundancy Problem
  - name: NCCL Operations
    subsections:
      - name: AllReduce
      - name: AllGather
      - name: ReduceScatter
  - name: ZeRO Stage 1
    subsections:
      - name: Optimizer State Partitioning
  - name: ZeRO Stage 2
    subsections:
      - name: Gradient Partitioning
  - name: ZeRO Stage 3
    subsections:
      - name: Parameter Partitioning

series_nav:
  title: "LLM Training Parallelism"
  prev:
    label: "Part 1: Memory & The Case for Parallelism"
    url: "/"
  next:
    label: "Part 3: Tensor Parallelism"
    url: "/tensor-parallelism/"

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

<div style="display:flex; justify-content:space-between; align-items:center; margin-top:48px; padding-top:20px; border-top:1px solid var(--global-divider-color,#dee2e6); font-size:0.9rem;">
  <a href="{{ '/' | relative_url }}" style="font-weight:600;">← Part 1: Memory &amp; The Case for Parallelism</a>
  <span style="color:var(--global-text-color-light,#aaa);">Part 3: Tensor Parallelism (coming soon)</span>
</div>


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
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>1. Initialize</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Model parameters broadcast from rank 0 to all GPUs. Gradients and optimizer states  initialized to zero.</td>
      <td style="padding:10px 12px; border:2px dashed #555;">Broadcast from rank 0</td>
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
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Steps 2-6 repeat every iteration until convergence</figcaption>
</figure>

### DDP Step by Step
At the start of training, one GPU — rank 0 — holds the initial model. Its parameters are broadcast to every other GPU using NCCL's Broadcast operation. From this point 
forward, every GPU holds an identical copy of the model.

**Data Split and Forward Pass**

The global batch is divided into micro-batches — one per GPU. Each GPU runs its forward pass independently on its own micro-batch, computing its own loss. No communication 
is needed here — the GPUs work in complete isolation.

**Backward Pass and the Divergence Problem**

Each GPU runs its backward pass independently, computing gradients based on what it saw. This is where the problem appears. GPU 0 saw sequences 0-127. GPU 1 saw sequences 128-255. Each GPU computed different gradients — and if each GPU now updates its own parameters independently,the model copies diverge. By the next iteration, GPU 0 and GPU 1 are no longer training the same model.

Our goal is one model, not N independent models.

**Gradient Sync — AllReduce**

The solution is to synchronize gradients before the optimizer step. This is what AllReduce does.

AllReduce is a collective operation — every GPU participates simultaneously:

```
1. Every GPU sends its gradients
2. NCCL computes the average across all GPUs
3. Every GPU receives the averaged gradient
```

Before AllReduce:   GPU 0: $\abla\text{W}_0$,  GPU 1: $\abla\text{W}_1$,  GPU 2: $\abla \text{W}_2$
After AllReduce:    All GPUs: ($\abla\text{W}_0 + abla\text{W}_1  + abla\text{W}_2 ) / 3

After AllReduce every GPU has identical gradients. Every GPU runs an identical optimizer step. Every GPU arrives at the next iteration with identical parameters. The model stays in sync.

**Optimizer Step**

With identical averaged gradients on every GPU, each GPU runs its optimizer step locally — no communication needed. Because every GPU started with the same gradients and runs 
the same update rule, Parameters and Optimizer States remain perfectly consistent across all GPUs.

### What DDP Gives Us

DDP is effective at solving the time problem. By splitting the batch across N GPUs and running forward and backward passes simultaneously, training throughput scales nearly 
linearly with the number of GPUs.

But look carefully at what every GPU is holding:

| Component | Precision | Memory per GPU | Replicated? |
|---|---|---|---|
| Parameters — working copy | BF16 | 2$\phi$ bytes | ✓ Every GPU |
| Parameters — master copy | FP32 | 4$\phi$ bytes | ✓ Every GPU |
| Gradients | FP32 | 4$\phi$ bytes | ✓ Every GPU |
| Optimizer State m | FP32 | 4$\phi$ bytes | ✓ Every GPU |
| Optimizer State v | FP32 | 4$\phi$ bytes | ✓ Every GPU |
| **Total** | | **18$\phi$ bytes** | **✓ Every GPU** |


Every single byte of static memory is duplicated across every GPU. With 8 GPUs the cluster holds 1.15 TB of static memory — when 144 GB would logically suffice. The other 1.0 TB is pure redundancy.

DDP solves the time problem. It does nothing for the memory problem.