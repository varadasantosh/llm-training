---
layout: distill
title: LLM Training using Parallelism Techniques
description: A comprehensive guide to parallelism techniques for training large language models
date: 2026-03-14
permalink: /

authors:
  - name: Varada Santosh
    affiliations:
      name:

toc:
  - name: Introduction
  - name: The Training Loop-A Quick Refresher

# bibliography: references.bib
---

## Introduction

Technology has a pattern — every few decades, evolution in technology challenges status quo and redraws the boundaries of the society, Mechanization didn't just make farming faster, it triggered the Industrial Revolution.The internet didn't 
just make communication cheaper, it shrunk the entire world. These 
weren't incremental improvements, they were resets.

Machine Learning is goig through one such moments right now.

<!-- Add timeline of ML evolution -->

**What are Large Language Models**
Large Language Models are token factories, we know how a factory takes raw materials, runs them through a series of machines and processes, and produces something useful at the end, LLMs do exactly that. Tokens go in, get processed through layers of computation across GPU machines, and tokens come out.

What comes out depends entirely on what goes in. Feed it tokens from a patient's health record, it returns a diagnosis summary. Feed it a pull request, it gives back a code review. Feed it a bank's financial report, it produce the key risks and bottlenecks . Give a text describing the scene and imagination output can be an image or set of images . Same factory completely different products.

Before LLMs, every task needed its own model. A translation model. A summarization model. A sentiment classifier. Each one trained for one job,  LLMs change the paradigm, trained on raw text at internet scale, that just... works across all of them. 

**So how does this factory actually learn?**

For an LLM to understand language and context across different domains, it needs to be exposed to enormous amounts of information during training — and this information comes in the form of tokens. We're talking trillions of them.

As the model processes these tokens, it adjusts its internal parameters — billions of them — to better understand each token in the context it appears in. That is really the whole point of training: keep adjusting the parameters until the model genuinely 
understands the input it's seeing.

Getting there isn't a single step either. LLMs go through several training phases — Pre-Training on massive datasets, followed by Post-Training refinements like Reinforcement Learning from Human Feedback (RLHF) and more. Each phase deserves its own deep dive, and we'll save those for another time.

What this blog focuses on is a challenge that sits underneath all of those phases — how do you even run this training at scale? Trillions of tokens, billions of parameters, thousands of GPUs. No single machine can hold a model this large, let alone train it in any reasonable amount of time.

## The Training Loop-A Quick Refresher

Though pytorch training loop is familiar to all of us let's refresh the process of forward and backward pass for one more time to lay foundation for the later stages

    {% highlight python %}
        def train_step(model, optimizer, batch):
            for batch in dataloader:
                input,target = batch  
                output = model(input)
                loss = loss_fn(output,target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    {% endhighlight %}

**Pytorch Training Loop**
- model(input) - Forward Pas
- loss.backward() - Backward Pass , Calculate Gradients w.r.to inputs & activations
- optimizer.step() - Update optimizer states (Valid for Optimizers like Adam)
- optimizer.zero_grad() - Reset Gradients to Zero before next batch
   
{% include figure.liquid path="assets/img/llm-training/section-2/forward-pass.png" class="img-small" caption="Figure: Forward Pass" %}

\begin{equation*}
\mathbf{A}_1 = \mathbf{X} \cdot \mathbf{W}_1^\top + \mathbf{b}_1 
\end{equation*}

\begin{equation*}
\hat{\mathbf{Y}} = \mathbf{A}_1 \cdot \mathbf{W}_2^\top + \mathbf{b}_2
\end{equation*}
