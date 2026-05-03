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

Tensor Parallelism relies on two key properties of  matrix multiplication **distributive** & **associative**

**1. Distributive Property**
    
    ```A·(B + C) = A·B + A·C```

In ML training context a weight matrix can be partitioned and each partition multiplied independently — partial results combine to produce the exact correct answer.

   X.W = X.(W₁+W₂) = X.W₁ + X.W₂

Here Weight matrix W is partitioned into W₁ & W₂ across 2 GPUs, each 
partition multiplied independently — partial results combine to produce the exact correct answer

**2. Associative Property**
    
    ```A·(B·C) = (A·B)·C```

This property allows re-ordering matrix mulitplications without changing the result - Output of split layer on one GPU feeds correctly into next split layer on same GPU without inter-layer communication.

Together these properties mean matrix multiplications can be partitioned across GPUs producing partital results, we need a communication operation like **AllReduce** or **AllGather** to arrive at final result 