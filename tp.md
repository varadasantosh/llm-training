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

The earlier sections of the series was focused on understanding constraints in training frontier LLMs. Solution to these constraints includes distributed training implemented using Parallelism techniques. Majority scenarios include more than one Parallelism techniques. Llama-3 series of models trained using 4 types of Parallelism Data Parallelism(DP), Tensor Parallelism(TP), Context Parallelism(CP) & Pipeline Parallelism(PP). Previous section was focused on distributed training using DDP & ZeRO. DDP & Zero was an initial attempt towards optimizing memory & reducing training time. 

DDP & ZeRO split input data into micro batches & shard static components of a model (optimizer states -> gradients -> parameters) across GPUs, while discussing about contributors to model memory, apart from static components model also has dynmaice component - activations which is variable and occupies large portion of GPU memory, we kept discussion of activations for later point of time, it is now time to discuss how to handle activations. How to opitmize memory required for storing activations.

**Recomputing Activations**
Training loop is combination of forward pass + backward pass , forward pass through different layers creats intermediate activations, activations are required for calculating gradients during backward pass, one block in transformer architecture has multiple components (RMS Norm, MHA or GQA, MLP), each component has multiple layers and activations are present at every layer, size of these activations scales with number of layers in each block and number of transformer blocks, batch size and sequence length, static components itself account for double the size of the H100 memory of VRAM. Though DDP along with ZeRO-3 solves memory problem for small- medium scale models by sharding data, parameters, gradients and optimizer states, it does not solve the problem for large scale models like 405B and does not address the constraints posed by activations. Activation recomputations address this by re-computing activations rather than storing them until backward pass , here computing power to recompute acticvationrs is traded with memory for storing them, while recomputing activations solve the issues with medium size models, large scale models can't still fit activations on single GPU, this is where TP helps in solving the problem

