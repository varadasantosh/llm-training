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

The pytorch training loop is familiar to all of us let's walk through it once more to build the foundation for later stages.

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
   

The forward pass moves in one direction — input X enters Layer 1, gets multiplied with that layer's weights, a bias is added, and the result becomes activation A₁. That activation becomes the input to Layer 2, where the same operation happens again, giving us our final prediction Ŷ. This pattern holds for any number of layers — the output of layer n is always the input to layer n+1. To complete a forward pass, each layer only needs two things: its Parameters (w, b) and the incoming Activations.

<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:15px;">
  <thead>
    <tr style="border-bottom:2px solid var(--global-divider-color, #ddd); text-align:left;">
      <th style="padding:10px 12px;">Layer</th>
      <th style="padding:10px 12px;">Equation</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid var(--global-divider-color, #eee);">
      <td style="padding:10px 12px; white-space:nowrap;">Layer 1</td>
      <td style="padding:10px 12px;">$\mathbf{A}_1 = \mathbf{X} \cdot \mathbf{W}_1^\top + \mathbf{b}_1$</td>
    </tr>
    <tr style="border-bottom:1px solid var(--global-divider-color, #eee);">
      <td style="padding:10px 12px; white-space:nowrap;">Layer 2 (Output)</td>
      <td style="padding:10px 12px;">$\hat{\mathbf{Y}} = \mathbf{A}_1 \cdot \mathbf{W}_2^\top + \mathbf{b}_2$</td>
    </tr>
  </tbody>
</table>

{% include figure.liquid path="assets/img/llm-training/section-2/forward-pass.png" class="img-small" caption="Figure-1: Forward Pass" %}

Once loss_fn gives us the loss, it acts as a compass — telling the model 
how wrong it was, which direction to correct itself. The backward pass uses this signal to update the parameters of every layer in the network.

How much each parameter should change is determined by three things:Gradients, Optimizer States, and Learning Rate. Learning Rate is a hyperparameter we'll dedicate separate discussion to — for now, let's focus on Gradients and Optimizer States, since these are computed and stored every single training step.

Figure 2 shows exactly what's flowing in and out of each layer during 
backward pass. Let's look at the equations behind it.

Backward pass involves two types of Gradient calculations

1. Gradients w.r.t Parameters(Weights) - Necessary for calculating Optimizer States & Updating Weights for the respective Layer
2. Gradients w.r.t Activations - Required for passing them as input to Preceeding Layer - Preceeding Layers uses them to calculate the Gradients

<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:15px;">
  <thead>
    <tr style="border-bottom:2px solid var(--global-divider-color, #ddd); text-align:left;">
      <th style="padding:10px 12px;">Step</th>
      <th style="padding:10px 12px;">Description</th>
      <th style="padding:10px 12px;">Equation</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid var(--global-divider-color, #eee); background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px;" colspan="3"><strong>Loss</strong></td>
    </tr>
    <tr style="border-bottom:1px solid var(--global-divider-color, #eee);">
      <td style="padding:10px 12px; white-space:nowrap;">Loss gradient</td>
      <td style="padding:10px 12px;">Gradient of loss w.r.t output</td>
      <td style="padding:10px 12px;">$\frac{\nabla L}{\nabla \hat{Y}}$</td>
    </tr>
    <tr style="border-bottom:1px solid var(--global-divider-color, #eee); background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px;" colspan="3"><strong>Layer 2 — Gradients</strong></td>
    </tr>
    <tr style="border-bottom:1px solid var(--global-divider-color, #eee);">
      <td style="padding:10px 12px; white-space:nowrap;">w.r.t Weights $W_2$</td>
      <td style="padding:10px 12px;">For updating weights</td>
      <td style="padding:10px 12px;">$\frac{\nabla L}{\nabla W_2} = \frac{\nabla L}{\nabla \hat{Y}} \cdot A_1^\top$</td>
    </tr>
    <tr style="border-bottom:1px solid var(--global-divider-color, #eee);">
      <td style="padding:10px 12px; white-space:nowrap;">w.r.t Activations $A_1$</td>
      <td style="padding:10px 12px;">Passed to preceeding layer</td>
      <td style="padding:10px 12px;">$\frac{\nabla L}{\nabla A_1} = \frac{\nabla L}{\nabla \hat{Y}} \cdot W_2$</td>
    </tr>
    <tr style="border-bottom:1px solid var(--global-divider-color, #eee); background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px;" colspan="3"><strong>Layer 1 — Gradients</strong></td>
    </tr>
    <tr style="border-bottom:1px solid var(--global-divider-color, #eee);">
      <td style="padding:10px 12px; white-space:nowrap;">w.r.t Weights $W_1$</td>
      <td style="padding:10px 12px;">For updating weights</td>
      <td style="padding:10px 12px;">$\frac{\nabla L}{\nabla W_1} = \frac{\nabla L}{\nabla A_1} \cdot X^\top$</td>
    </tr>
    <tr style="border-bottom:1px solid var(--global-divider-color, #eee);">
      <td style="padding:10px 12px; white-space:nowrap;">w.r.t Activations $X$</td>
      <td style="padding:10px 12px;">Backpropagated to input</td>
      <td style="padding:10px 12px;">$\frac{\nabla L}{\nabla X} = \frac{\nabla L}{\nabla A_1} \cdot W_1$</td>
    </tr>
    <tr style="border-bottom:1px solid var(--global-divider-color, #eee); background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px;" colspan="3"><strong>Parameter Update (Adam Optimizer)</strong></td>
    </tr>
    <tr style="border-bottom:1px solid var(--global-divider-color, #eee);">
      <td style="padding:10px 12px; white-space:nowrap;">1st moment</td>
      <td style="padding:10px 12px;">Exponential moving avg of gradients</td>
      <td style="padding:10px 12px;">$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \frac{\nabla L}{\nabla W}$</td>
    </tr>
    <tr style="border-bottom:1px solid var(--global-divider-color, #eee);">
      <td style="padding:10px 12px; white-space:nowrap;">2nd moment</td>
      <td style="padding:10px 12px;">Exponential moving avg of squared gradients</td>
      <td style="padding:10px 12px;">$v_t = \beta_2 v_{t-1} + (1 - \beta_2) \left(\frac{\nabla L}{\nabla W}\right)^2$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; white-space:nowrap;">Weight update</td>
      <td style="padding:10px 12px;">Adaptive learning rate step</td>
      <td style="padding:10px 12px;">$W_{t+1} = W_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}$</td>
    </tr>
  </tbody>
</table>



{% include figure.liquid path="assets/img/llm-training/section-2/backward-pass.png" class="img-small" caption="Figure-2: Backward Pass" %}