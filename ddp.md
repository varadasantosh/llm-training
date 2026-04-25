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
automatically solve a memory problem — it only helps if we carefully manage what each GPU holds.

Let's set activations aside for now and focus on the static components — Parameters, Gradients, and Optimizer States. For Llama 3 8B these alone require 144 GB. A single H100 
has 80 GB. Even ignoring activations entirely, the model's training state does not fit on one GPU. This means we cannot simply copy the model to multiple GPUs we need to carefully orchestrate how the model's components are distributed across them.

Solution to these problems can be implemented using different Parallelism techniques like **Data Parallelism,Tensor Parallelism, Context Parallelism, Pipeline Parallelism, Expert Parallelism** etc, for large scale models one approach is not sufficient most of the times, they employ combination of different techniques to achieve higher GPU utilization and faster training times,  section (3.3.2-Parallelism for Model Scaling) of Llama-3 [paper](https://arxiv.org/pdf/2407.21783) describes 4 dimensions of Parallelism were used to train the model series, we will start with simplest approach of **Data Parallelelism** in this section and continue to build on this.

Every parallelism techniques needs splitting data and model across GPU's. Here is where things get interesting. If we split the training data across multiple GPUs — each GPU seeing a different subset — each GPU will compute different gradients after its backward pass. If every GPU then updates its own parameters independently, the model copies diverge. By the next iteration, GPU 0 and GPU 1 are no longer training the same model.

**Our goal is to train one model, not N independent models.**

To achieve this, GPUs cannot work in complete isolation — they need to communicate and coordinate at specific points during training, these communication frameworks or libraries are crucial for parallelism techniques - for NVIDIA GPU's [NCCL](https://developer.nvidia.com/nccl) is underlying library that provides routines for moving data across GPU's in single node or 

GPU's present across nodes, NCCL itself deserves its own topic but here we will cover basic operations used across all of these Parallelism techniques. Llama-3  team has extended the NCCL library and created NCCLX to customize the routines to optimize them for their infrastructure and network topology, please refer to section(3.3.3-Collective Communication) from Lalam-3 [paper](https://arxiv.org/pdf/2407.21783) 
