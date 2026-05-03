---
layout: distill
title: Tensor Parallelism
description: Understanding Tensor Parallelism for training large language models
date: 2026-03-16
permalink: /tp/

authors:
  - name: Varada Santosh
    affiliations:   
      name:

toc:
  - name: Introduction

series_nav:
  title: "LLM Training Parallelism"
  prev:
    label: "Part 2: Data Parallelism & ZeRO"
    url: "/ddp/"

# bibliography: references.bib
---

## Introduction

The previous section covered Data Parallelism and ZeRO — techniques that distribute data and model state across GPUs. ZeRO-3 in particular achieves 87.5% memory reduction by sharding parameters, gradients, and optimizer states across 
all GPUs in the cluster. But there is one component ZeRO does not address — Activations.

### The Activation Problem
Activations are the intermediate outputs produced at every layer during the forward pass. The backward pass needs them to compute gradients — so they must all be held in memory simultaneously until backpropagation completes. Unlike static components whose memory is fixed by model architecture, activation memory scales with three variables:

- **Batch Size** — more sequences means more activations
- **Sequence Length** — longer sequences means larger activations 
  per layer
- **Model depth** — more transformer blocks means more layers 
  storing activations

For Llama 3 8B with S=8192 and B=1, activation memory per block reaches ~9.78 GB without FlashAttention — and ~38 GB across all 32 blocks. For a realistic micro-batch of 32 sequences, this exceeds 1 TB.

### Activation Recomputation — Partial Relief

One approach is to not store activations at all during the forward pass — recompute them on demand during the backward pass instead. This trades memory for compute:

Without recomputation:
Store all activations → high memory, no extra compute (Not feasible solution due to limitations on GPU VRAM)

With recomputation (gradient checkpointing):
Store only checkpoints → low memory, ~33% extra compute

For medium-scale models this works well. But for 405B models trained across thousands of GPUs, even recomputed activations create pressure — the recomputation itself requires the full parameter set to be present for each layer.

### Why Tensor Parallelism

ZeRO shards model state — but each GPU still runs the complete computation for its micro-batch. Every matrix multiplication happens in full on each GPU.

Tensor Parallelism takes a different approach — instead of distributing what is stored, it distributes the computation itself. Individual weight matrices are split across GPUs, and the matrix multiplications that power the forward and backward pass are executed in parallel across those GPUs simultaneously.

This means:
- Each GPU holds only a fraction of each weight matrix
- The forward pass computation is genuinely parallelized across GPUs — not just replicated
- Activation memory is naturally distributed since each GPU only computes part of each layer's output

This is what makes Tensor Parallelism essential for models like Llama 3 405B — and why Llama-3 combined it with Data Parallelism, Pipeline Parallelism, and Context Parallelism as four simultaneous dimensions of parallelism across 16,000 GPUs.

### Core Principle

Tensor Parallelism relies on two mathematical properties of matrix multiplication **distributive** & **associative**

        ```
        distributive  -  A(X₁|X)₂ = (A.X₁|A.X₂)  
        associative   -  A*(B*C) = (A*B)*C
        ```
Matrix multiplication always involves two matrices **A: M*K, B:K*N** , dimesions  M & N are outer dimensions and K is referred as inner dimension

> M: denotes Rows of Matrix A - Outer dimension
> N: deontes Columsn of Matrix B - Outer dimension
> K: denotes Columns of Matrix A & Rows of B - Inner dimension

General matrix multiplication in ML training is of the form ```Y= $X.W^T$``` , X = Input data , W= Weight matrix. Weigth matrices can be split column-wise or row-wise. Following above notation splitting across inner dimension implies splitting the weight matrix row-wise, splitting across outer dimension implies two possibilities - to split input data **X** across rows or split weight matrix **W** across columns, each has different implications on communication between GPUs which directly impacts LLM training process. Let us look into these scenarios individually to understand the implications

1. Split Inner dimensions:

    - X - Shape M*K
    - W - Shape K*N
    - Splitting across Kth dimension, If we only split the weight matrix across
      rows, this change shape of W to be of shape K/2*N across 2 GPUs, while shape of X stays M*K, this prevents matrix multiplication due to mismatch in dimensions of X & W. 
    - Splitting columns of X and rows of weight matrix , this reshapes both X and W. X -> M*K/2 , W -> K/2*N
    - Both GPUs produce M*N but they are partial results since each only worked on partial weights , AllReduce is required to gather results from multiple GPU


2. Split Outer dimensions:

    - X - Shape M*K
    - W - Shape K*N
    - Splitting across Outer dimension has two possibilities 1. Split X matrix row-wise(M dimension) or split W matrix column-wise(N dimension).
    - Splitting W matrix column-wise reshapes W to K*N/2 and X stays same M*K
    - Splitting X matrix row-wise reshapes X to M/2*K AND W stays same K*N
    - Splitting X row-wise across M dimension leaves us weight matrix without splitting, which does not give us much benefit 
    - Splitting W matrix column-wise and X.W results in shape M*N/2 on two GPUs,
      requires AllGather to reconstruct all values from GPUs


Looking at both one that is more beneficial is Splitting across Kth dimension, does both splitting X:Input across columns and W:Weights across rows, reshapes X:M*K/2 & W:K/2*N , X*W results in M*N and AllReduce is required for aggregating results from both GPUs.










 though matrix multiplication is an associative operation, floating point operations are not associative, to maintain consistency between iterations and model convergence a mixed precision training is used for training Large Language Models. two copies of weights are used one with BF16/FP16 which does not gaurantee associativity , to make sure the accuracy of precision it requires us to maitain another master copy of parameters in FP32, the weights used with BF16 gives a benefit of Activations being produced in BF16/FP16 rather than FP32, this help reduce activation memory size and split the tensors across GPUs
