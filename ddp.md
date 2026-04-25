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
    training Llama 3 405B on a single H100 (67 TFLOPS FP32) would take ~18,000 years even at 100% utilisation.

> **Static memory** (Parameters + Gradients + Optimizer States) for Llama 3 8B requires **~144 GB** — nearly double the H100's 80 GB VRAM

> **Dynamic memory** considering hypotehical one sequence per batch Activations adds another **~38-294 GB** depending on FlashAttention usage. A realistic micro-batch pushes total memory to **~1.2 TB** — more than 15× what a single GPU can hold

**Time Factor:-** though the scale of the problem is very large(18,000 years) in this context , this is a common problem across different domains & software engineering ,for problems related to time constraints engineers employ solutions like multiple workers, threads, processes etc... to divide the work between them thus reducing time taken to process the entire dataset.

Applying this solution in our context means to train a large language model on a corpus of 15T tokens, we need to split this training data into different buckets with each GPU having access to a bucket or subset of data, trainining process involves forward pass & backward pass during backward pass we need to update gradients across batches and iterations, when we train a model on different GPU's with access to different datasets , in the absence of co-ordination between GPU's this would end up in producing multiple models each having different parameters(weights & bias). Our objective is to train single model rather than multiple models. to achieve training of single model with multiple GPU's being trained on different subsets of data we clearly need communication & co-ordniation between them, we will look into this in next sections.kkkkl 

**Memory Factor:-**  The problem with time constraint can be addressed with multiple GPU's, but for a memory constraint this is not straight forward, for a model to be trained on GPU it should be present on GPU & when we refer to a model in the context of training it involves both static & dynamic components, like mentioned in earlier sections activations occupy large portion of memory. there are different solutions to handle activations for now let us keep them out of the equation, consider placing only static components Parameters + Gradients + Optimizer States on single GPU, due to sheer size of **144 GB** in the case of 8B model even this is not possible to place the model on single GPU, these static components need to be distributed across multiple GPU's, here again each GPU has access to subset of training data , to update the weights which are distributed across multiple GPU's it require certain steps to co-ordinate and communicate between them

Whether we only distribute data between GPU
