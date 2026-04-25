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
---

## Introduction

In the [previous section]({{ '/' | relative_url }}), we established limiting factors for training frontier model using GPU, one factor being time & the other factor is memory,

 
> **Compute alone makes a single GPU infeasible.** At $3.8 \times 10^{25}$ FLOPs,
> training Llama 3 405B on a single H100 (67 TFLOPS FP32) would take ~18,000 years even at 100% > utilisation.

> **Static Memory:** Parameters, Gradients, and Optimizer 
> States for Llama 3 8B require ~144 GB — nearly double 
> the H100's 80 GB VRAM.

> **Dynamic Memory:** Even for a single sequence, Activations 
> add another 38 GB (with FlashAttention) to 312 GB (without). 
> A realistic micro-batch pushes total memory to ~1.2 TB — 
> more than 15× what a single GPU can hold.

These are two distinct problems and they require two distinct ways of thinking about the solution.

**Time Factor:-** though the scale of the problem is very large(18,000 years) in this context , But time constraints are a familiar problem in engineering. The standard solution is parallelism: divide the work across multiple workers, threads, or processes, each handling a subset of the problem simultaneously.

The same principle applies here. Training on 15 trillion tokens means processing an enormous amount of data. If we split that data across multiple GPUs — each GPU seeing a 
different subset — and run the forward and backward passes simultaneously, training time scales down proportionally with the number of GPUs. This is the core idea behind Data Parallelism.

**Memory Factor:-**  The memory constraint is harder. Adding more GPUs doesn't 
automatically solve a memory problem — we need to carefully manage what each GPU holds.

Let's set activations aside for now and focus on the static components — Parameters, Gradients, and Optimizer States. For Llama 3 8B these alone require 144 GB. A single H100 
has 80 GB. Even ignoring activations entirely, the model's training state does not fit on one GPU. This means we cannot simply copy the model to multiple GPUs we need to carefully orchestrate how the model's components are distributed across them. 


**Parallelism Techniques**

Enter Parallelism techniques - These constraints are addressed through a family of 
Parallelism techniques — **Data Parallelism,Tensor Parallelism, Context Parallelism, Pipeline Parallelism, Expert Parallelism**.

For large scale models single approach is not sufficient, In practice, teams combine multiple techniques simultaneously to maximize GPU utilization and minimize training time, section (3.3.2-Parallelism for Model Scaling) of Llama-3 [paper](https://arxiv.org/pdf/2407.21783) ddescribes how Meta used 4 dimensions of parallelism — Tensor, Pipeline, Context, and Data — to train the model series across 16,000 GPUs.

We will work through each technique in turn, starting with the simplest — Data Parallelism — and building toward the combined approach that makes frontier training possible.

**The Coordination Problem**

Splitting data across GPUs introduces an immediate challenge. Each GPU sees a different subset of the data, so after the backward pass each GPU has computed different gradients. If every GPU then updates its own parameters independently, the model copies diverge — by the next iteration, GPU 0 and GPU 1 are no longer training the same model.

Our goal is to train one model, not N independent models.

To achieve this, GPUs cannot work in isolation — they need to communicate and synchronize at specific points during training. How efficiently they do this determines how well the parallelism strategy scales.

**NCCL — The Communication Foundation**

Every parallelism technique in this blog relies on a common communication layer. For NVIDIA GPUs, that layer is [NCCL](https://developer.nvidia.com/nccl) — the NVIDIA Collective Communications Library.

NCCL provides the core routines for moving data across GPUs, whether they share the same node or are spread across multiple nodes connected over a network. Rather than each framework implementing its own communication primitives, PyTorch, DeepSpeed, and Megatron-LM all build on top of NCCL

The Llama-3 team extended NCCL into NCCLX, optimizing collective operations specifically for their network topology across large GPU clusters. For a detailed treatment of the collective communication operations used in Llama-3 training, see Section 3.3.3-Collective Communication of the Llama-3 [paper](https://arxiv.org/pdf/2407.21783).

NCCL exposes a set of collective operations — each one designed for a specific communication pattern. We will introduce them as they are needed, but here is the full set we will encounter across the parallelism techniques:

| Operation | What it does | First appears in |
|---|---|---|
| Broadcast | Send from one GPU to all others | DDP initialization |
| AllReduce | Sum/average across all GPUs, result to all | DDP gradient sync |
| ReduceScatter | Reduce then distribute different shards | ZeRO-1/2/3 |
| AllGather | Collect shards from all GPUs, result to all | ZeRO-1/2/3 |
| Send/Recv | Point-to-point between two GPUs | Pipeline Parallelism |

We start with the simplest approach — Distributed Data Parallel training — and build from there.
