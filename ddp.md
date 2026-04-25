---
layout: distill
title: Data Parallelism - From Vanilla DDP to ZeRO
description: Understanding Distributed Data Parallel training and memory optimization techniques
date: 2026-03-15
permalink: /ddp/
series: llm-training

authors:
  - name: Varada Santosh
    affiliations:
      name:

toc:
  - name: Introduction
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

# bibliography: references.bib
---

## Introduction

In the [previous section](/), we established why a single GPU cannot train frontier LLMs. The numbers were stark:

- **Static memory** (Parameters + Gradients + Optimizer States) for Llama 3 8B requires **~144 GB** — nearly double the H100's 80 GB VRAM
- **Dynamic memory** (Activations) adds another **~38-294 GB** depending on FlashAttention usage
- A realistic micro-batch pushes total memory to **~1.3 TB** — more than 16× what a single GPU can hold

The solution is parallelism — distributing the workload across multiple GPUs. But parallelism comes in different flavors, each addressing different bottlenecks:

| Parallelism Type | What it Distributes | Primary Benefit |
|------------------|---------------------|-----------------|
| **Data Parallelism** | Training data (batches) | Faster training via larger effective batch sizes |
| **Tensor Parallelism** | Individual layers/operations | Enables larger models per layer |
| **Pipeline Parallelism** | Model layers across GPUs | Enables deeper models |
| **Sequence Parallelism** | Sequence dimension | Handles longer contexts |

This section focuses on **Data Parallelism** — the most intuitive and widely-used approach. We'll start with vanilla Distributed Data Parallel (DDP), understand its limitations, and then explore how ZeRO (Zero Redundancy Optimizer) progressively eliminates memory redundancy across three stages.

---

## Vanilla DDP

### How DDP Works

Distributed Data Parallel (DDP) is the simplest form of parallelism. The core idea: **replicate the entire model on every GPU, but split the training data**.

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555;">Step</th>
      <th style="padding:10px 12px; border:2px dashed #555;">What Happens</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Communication</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>1. Initialize</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Each GPU gets a complete copy of the model (parameters, gradients, optimizer states)</td>
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
</figure>

The key insight: since all GPUs start with identical parameters and receive identical averaged gradients, they remain synchronized after each optimizer step — no explicit parameter synchronization needed.

```python
# PyTorch DDP in 4 lines
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = DDP(model.to(device), device_ids=[local_rank])
# Training loop remains unchanged — DDP handles gradient sync automatically
```

### The Redundancy Problem

Vanilla DDP solves the **compute** problem — N GPUs process N× more data per step. But it does nothing for **memory**. Every GPU still holds:

| Component | Per-GPU Memory | Redundancy |
|-----------|----------------|------------|
| Parameters (BF16) | 2φ bytes | N copies |
| Parameters (FP32 master) | 4φ bytes | N copies |
| Gradients (FP32) | 4φ bytes | N copies |
| Optimizer States (FP32) | 8φ bytes | N copies |
| **Total Static** | **18φ bytes** | **N copies** |

For Llama 3 8B with φ = 8.03B parameters across 8 GPUs:
- **Total memory used:** 8 × 144 GB = **1,152 GB**
- **Unique data stored:** 144 GB
- **Redundancy:** 8× (7/8 of memory is wasted on duplicates)

This is the fundamental limitation of vanilla DDP: **memory scales with model size, not with GPU count**. Adding more GPUs speeds up training but doesn't let you train larger models.

> **The question becomes:** Can we keep the compute benefits of data parallelism while eliminating this memory redundancy?

This is exactly what ZeRO addresses — but first, we need to understand the communication primitives that make it possible.

---

## NCCL Operations

NVIDIA Collective Communications Library (NCCL) provides the building blocks for GPU-to-GPU communication. Understanding these primitives is essential for grasping how ZeRO works.

### AllReduce

**Purpose:** Combine values from all GPUs and distribute the result back to all GPUs.

**Operation:** Each GPU contributes a tensor → tensors are summed (or averaged) → every GPU receives the final result.

```
Before AllReduce:
  GPU 0: [1, 2, 3]
  GPU 1: [4, 5, 6]
  GPU 2: [7, 8, 9]

After AllReduce (sum):
  GPU 0: [12, 15, 18]
  GPU 1: [12, 15, 18]
  GPU 2: [12, 15, 18]
```

**Used in:** Vanilla DDP gradient synchronization

**Communication cost:** 2(N-1)/N × data_size ≈ 2 × data_size for large N

---

### AllGather

**Purpose:** Collect data from all GPUs and distribute the complete collection to all GPUs.

**Operation:** Each GPU contributes a tensor → all tensors are concatenated → every GPU receives the full concatenated result.

```
Before AllGather:
  GPU 0: [A]
  GPU 1: [B]
  GPU 2: [C]

After AllGather:
  GPU 0: [A, B, C]
  GPU 1: [A, B, C]
  GPU 2: [A, B, C]
```

**Used in:** ZeRO-3 parameter gathering before forward/backward pass

**Communication cost:** (N-1)/N × data_size ≈ data_size for large N

---

### ReduceScatter

**Purpose:** Reduce data across GPUs and scatter different portions to different GPUs.

**Operation:** Tensors are summed across GPUs → result is split → each GPU receives one chunk.

```
Before ReduceScatter:
  GPU 0: [1, 2, 3]
  GPU 1: [4, 5, 6]
  GPU 2: [7, 8, 9]

After ReduceScatter (sum, then scatter):
  GPU 0: [12]      (sum of first elements)
  GPU 1: [15]      (sum of second elements)
  GPU 2: [18]      (sum of third elements)
```

**Used in:** ZeRO-2/3 gradient reduction and partitioning

**Communication cost:** (N-1)/N × data_size ≈ data_size for large N

---

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555;">Operation</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Input → Output</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Communication Cost</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Used In</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>AllReduce</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">N tensors → N identical reduced tensors</td>
      <td style="padding:10px 12px; border:2px dashed #555;">~2 × data_size</td>
      <td style="padding:10px 12px; border:2px dashed #555;">Vanilla DDP</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>AllGather</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">N chunks → N complete tensors</td>
      <td style="padding:10px 12px; border:2px dashed #555;">~1 × data_size</td>
      <td style="padding:10px 12px; border:2px dashed #555;">ZeRO-3</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>ReduceScatter</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">N tensors → N different reduced chunks</td>
      <td style="padding:10px 12px; border:2px dashed #555;">~1 × data_size</td>
      <td style="padding:10px 12px; border:2px dashed #555;">ZeRO-2, ZeRO-3</td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 2: NCCL Collective Operations Summary</figcaption>
</figure>

> **Key insight:** AllReduce = ReduceScatter + AllGather. ZeRO exploits this decomposition to eliminate redundancy.

---

## ZeRO Stage 1

### Optimizer State Partitioning

ZeRO Stage 1 targets the largest memory consumer: **optimizer states**.

Recall from the memory breakdown:
- Optimizer states (Adam's m and v): **8φ bytes** (44% of static memory)
- Parameters + Gradients: **10φ bytes** (56% of static memory)

**The insight:** During the optimizer step, each parameter only needs its own optimizer state. There's no reason for GPU 0 to hold optimizer states for parameters that GPU 1 will update.

**ZeRO-1 approach:** Partition optimizer states across GPUs. Each GPU only stores optimizer states for 1/N of the parameters.

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555;">Component</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Vanilla DDP</th>
      <th style="padding:10px 12px; border:2px dashed #555;">ZeRO-1</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Parameters (BF16 + FP32)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">6φ per GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">6φ per GPU</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Gradients (FP32)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">4φ per GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">4φ per GPU</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;">Optimizer States (FP32)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">8φ per GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>8φ/N per GPU</strong></td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f0f0f0);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>Total per GPU</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>18φ</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>10φ + 8φ/N</strong></td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 3: Memory Comparison — Vanilla DDP vs ZeRO-1</figcaption>
</figure>

**For Llama 3 8B with 8 GPUs:**
- Vanilla DDP: 18 × 8.03B = **144 GB per GPU**
- ZeRO-1: (10 + 8/8) × 8.03B = 11 × 8.03B = **88 GB per GPU**

**Memory reduction:** 39% — still doesn't fit on an 80GB H100, but significant progress.

**Communication overhead:** After the optimizer step, updated parameters must be broadcast to all GPUs. This adds an AllGather operation but the cost is manageable.

---

## ZeRO Stage 2

### Gradient Partitioning

ZeRO Stage 2 extends the partitioning to **gradients**.

**The insight:** Gradients are only needed for the optimizer step. If GPU 0 only updates 1/N of the parameters, it only needs gradients for those parameters.

**ZeRO-2 approach:** 
1. During backward pass, compute gradients normally
2. Use **ReduceScatter** instead of AllReduce — each GPU receives reduced gradients only for its partition
3. Discard gradients for parameters this GPU doesn't own

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555;">Component</th>
      <th style="padding:10px 12px; border:2px dashed #555;">ZeRO-1</th>
      <th style="padding:10px 12px; border:2px dashed #555;">ZeRO-2</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Parameters (BF16 + FP32)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">6φ per GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">6φ per GPU</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;">Gradients (FP32)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">4φ per GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>4φ/N per GPU</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Optimizer States (FP32)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">8φ/N per GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">8φ/N per GPU</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f0f0f0);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>Total per GPU</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>10φ + 8φ/N</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>6φ + 12φ/N</strong></td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 4: Memory Comparison — ZeRO-1 vs ZeRO-2</figcaption>
</figure>

**For Llama 3 8B with 8 GPUs:**
- ZeRO-1: 11 × 8.03B = **88 GB per GPU**
- ZeRO-2: (6 + 12/8) × 8.03B = 7.5 × 8.03B = **60 GB per GPU**

**Memory reduction:** 58% from vanilla DDP — now fits on an 80GB H100!

**Communication:** ReduceScatter has the same cost as AllReduce's reduce phase, so no additional overhead.

---

## ZeRO Stage 3

### Parameter Partitioning

ZeRO Stage 3 completes the picture by partitioning **parameters** themselves.

**The insight:** Parameters are only needed during forward and backward passes. If we can gather them just-in-time and discard them immediately after, we don't need to store the full model on any single GPU.

**ZeRO-3 approach:**
1. Each GPU stores only 1/N of the parameters
2. Before each layer's forward pass: **AllGather** to reconstruct full parameters
3. Compute forward pass
4. Discard gathered parameters (keep only owned partition)
5. Repeat for backward pass

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555;">Component</th>
      <th style="padding:10px 12px; border:2px dashed #555;">ZeRO-2</th>
      <th style="padding:10px 12px; border:2px dashed #555;">ZeRO-3</th>
    </tr>
</thead>
<tbody>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;">Parameters (BF16 + FP32)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">6φ per GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>6φ/N per GPU</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Gradients (FP32)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">4φ/N per GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">4φ/N per GPU</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Optimizer States (FP32)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">8φ/N per GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">8φ/N per GPU</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f0f0f0);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>Total per GPU</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>6φ + 12φ/N</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>18φ/N</strong></td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 5: Memory Comparison — ZeRO-2 vs ZeRO-3</figcaption>
</figure>

**For Llama 3 8B with 8 GPUs:**
- ZeRO-2: 7.5 × 8.03B = **60 GB per GPU**
- ZeRO-3: 18/8 × 8.03B = 2.25 × 8.03B = **18 GB per GPU**

**Memory reduction:** 87.5% from vanilla DDP — dramatic improvement!

**The tradeoff:** Communication increases significantly. Each layer requires an AllGather before forward and another before backward. For a 32-layer model, that's 64 additional AllGather operations per training step.

---

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555;">Stage</th>
      <th style="padding:10px 12px; border:2px dashed #555;">What's Partitioned</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Memory per GPU</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Llama 3 8B (8 GPUs)</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Communication</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>Vanilla DDP</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Nothing</td>
      <td style="padding:10px 12px; border:2px dashed #555;">18φ</td>
      <td style="padding:10px 12px; border:2px dashed #555;">144 GB</td>
      <td style="padding:10px 12px; border:2px dashed #555;">AllReduce (gradients)</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>ZeRO-1</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">Optimizer States</td>
      <td style="padding:10px 12px; border:2px dashed #555;">10φ + 8φ/N</td>
      <td style="padding:10px 12px; border:2px dashed #555;">88 GB</td>
      <td style="padding:10px 12px; border:2px dashed #555;">+ AllGather (params)</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>ZeRO-2</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">+ Gradients</td>
      <td style="padding:10px 12px; border:2px dashed #555;">6φ + 12φ/N</td>
      <td style="padding:10px 12px; border:2px dashed #555;">60 GB</td>
      <td style="padding:10px 12px; border:2px dashed #555;">ReduceScatter (gradients)</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>ZeRO-3</strong></td>
      <td style="padding:10px 12px; border:2px dashed #555;">+ Parameters</td>
      <td style="padding:10px 12px; border:2px dashed #555;">18φ/N</td>
      <td style="padding:10px 12px; border:2px dashed #555;">18 GB</td>
      <td style="padding:10px 12px; border:2px dashed #555;">+ AllGather per layer</td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 6: ZeRO Stages Summary — Memory vs Communication Tradeoff</figcaption>
</figure>

> **The fundamental tradeoff:** ZeRO trades memory for communication. As you move from Stage 1 → 2 → 3, memory requirements drop linearly with GPU count, but communication overhead increases. The right choice depends on your hardware topology (NVLink bandwidth, network speed) and model size.

---

*Next section: Tensor Parallelism — when even ZeRO-3 isn't enough and you need to split individual layers across GPUs.*