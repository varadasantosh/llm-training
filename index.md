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
  - name: Why One GPU is Never Enough
    subsections:
      - name: What actually lives on that GPU?
      - name: Parameters
      - name: Activations
      - name: Putting it All Together

# bibliography: references.bib
---

## Introduction

Technology has a pattern — every few decades, evolution in technology challenges status quo and redraws the boundaries of the society, Mechanization didn't just make farming faster, it triggered the Industrial Revolution.The internet didn't 
just make communication cheaper, it shrunk the entire world. These 
weren't incremental improvements, they were resets.

Machine Learning is going through one such moment right now.

<!-- Add timeline of ML evolution -->

**What are Large Language Models**
Large Language Models are token factories,Much like a traditional factory takes raw materials, runs them through a series of machines, and produces a finished product,LLMs do exactly that with information. Whether the input is text, images, or video, everything is converted into tokens.These tokens are passed as input & processed through layers of computation across GPU machines, and useful tokens come out the other side.

But not all factories produce the same quality of goods. What comes out depends on three pillars: the raw materials you feed in, the design of the factory itself, and how well the manufacturing process is executed. For LLMs, these translate directly to **Data Preparation**, **Model Architecture**, and the **Training Process**.

  {% include figure.liquid path="assets/img/llm-training/section-1/model-ingredients.svg" class="img-medium" caption="Figure-1: Model Ingredients" %}

The utility of the factory depends entirely on the input. Feed it a patient's health records, and it returns a diagnostic summary. Feed it a pull request, and it provides a code review. Feed it a financial report, and it identifies risks and bottlenecks. It is the same factory producing completely different products based on the prompt.

However, running a token factory at this scale is not easy to achieve.Frontier models evolving rapidly consist of hundreds of billions of parameters trained on trillions of tokens. We quickly hit a wall where a single GPU simply cannot hold the workload.

**Analogy**: building a house for a few people is a straightforward task. Building a city for millions to live comfortably requires entirely different levels of planning, infrastructure, and communication. Model training faces the same leap in complexity. This blog explores how Parallelism techniques provide the infrastructure necessary to build these "cities" of intelligence.

**So how does this factory actually learn?**

For an LLM to understand language and context across different domains, it needs to be exposed to enormous amounts of information during training — and this information comes in the form of tokens. We're talking trillions of them.

As the model processes these tokens, it adjusts its internal parameters — billions of them — to better understand each token in the context it appears in. That is really the whole point of training: keep adjusting the parameters until the model genuinely 
understands the input it's seeing.

Getting there isn't a single step either. LLMs go through several training phases — Pre-Training on massive datasets, followed by Post-Training refinements like Reinforcement Learning from Human Feedback (RLHF) and more. Each phase deserves its own deep dive, and we'll save those for another time.

What this blog focuses on is a challenge that sits underneath all of those phases — how do you even run this training at scale? Trillions of tokens, billions of parameters, thousands of GPUs. No single machine can hold a model this large, let alone train it in any reasonable amount of time.

## The Training Loop-A Quick Refresher

The PyTorch training loop is something most of us know well. But before diving into parallelism techniques, a clear understanding of how the forward and backward passes work creates a baseline for everything that follows — hence it is worth revisiting the mechanics before moving forward.

{% highlight python %}
  def train_step(model, optimizer, dataloader):
      for batch in dataloader:
          input,target = batch  
          output = model(input)
          loss = loss_fn(output,target)
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
{% endhighlight %}

   

**Forward Pass** : input X enters Layer 1, gets multiplied with that layer's weights, a bias is added, and the result becomes activation A₁. That activation becomes the input to Layer 2, where the same operation happens again, giving us our final prediction Ŷ. This pattern holds for any number of layers — the output of layer n-1 is always the input to layer n. To complete a forward pass, each layer only needs two things: its Parameters (w, b) and the incoming Activations.

<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:15px; border:2px dashed #555;">
  <thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555;">Layer</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Equation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:10px 12px; white-space:nowrap; border:2px dashed #555;">Layer 1</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$\mathbf{A}_1 = \mathbf{X} \cdot \mathbf{W}_1^\top + \mathbf{b}_1$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; white-space:nowrap; border:2px dashed #555;">Layer 2 (Output)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$\hat{\mathbf{Y}} = \mathbf{A}_1 \cdot \mathbf{W}_2^\top + \mathbf{b}_2$</td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 1: Forward Pass: Equations</figcaption>
</figure>

<figure id="fig-forward-pass">
{% include figure.liquid path="assets/img/llm-training/section-2/forward-pass.svg" class="img-medium" caption="Figure-2: Forward Pass" %}

</figure>
**Backward Pass:** Once **loss_fn** gives us the loss, it acts as a compass — telling the model 
how wrong it was, which direction to correct itself. The backward pass uses this signal to update the parameters of every layer in the network.

How much each parameter should change is determined by three things:Gradients, Optimizer States, and Learning Rate. Learning Rate is a hyperparameter we'll dedicate separate discussion to — for now, let's focus on Gradients and Optimizer States, since these are computed and stored every single training step.

<a href="#fig-backward-pass">Figure 3</a> shows exactly what's flowing in and out of each layer during 
backward pass. Let's look at the equations behind it.


<d-aside>
  <b>Backward pass involves two types of Gradient calculations:</b>

  <ol>
    <li><b>w.r.t Parameters (Weights)</b> — needed for calculating 
    Optimizer States and updating the layer's weights</li>
    <li><b>w.r.t Activations</b> — passed as input to the preceding 
    layer to continue the gradient chain</li>
  </ol>
</d-aside>

<figure id="fig-backward-pass">
{% include figure.liquid path="assets/img/llm-training/section-2/backward-pass.svg" class="img-medium" caption="Figure-3: Backward Pass" %}
</figure>

<figure id="table-backward-pass">
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:15px; border:2px dashed #555;">
  <thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555;">Step</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Description</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Equation</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;" colspan="3"><strong>Loss</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; white-space:nowrap; border:2px dashed #555;">Loss gradient</td>
      <td style="padding:10px 12px; border:2px dashed #555;">Gradient of loss w.r.t output</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$\frac{\nabla L}{\nabla \hat{Y}}$</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;" colspan="3"><strong>Layer 2 — Gradients</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; white-space:nowrap; border:2px dashed #555;">w.r.t Weights $W_2$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">For updating weights</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$\frac{\nabla L}{\nabla W_2} = \frac{\nabla L}{\nabla \hat{Y}} \cdot A_1^\top$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; white-space:nowrap; border:2px dashed #555;">w.r.t Activations $A_1$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">Passed to preceding layer</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$\frac{\nabla L}{\nabla A_1} = \frac{\nabla L}{\nabla \hat{Y}} \cdot W_2$</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;" colspan="3"><strong>Layer 1 — Gradients</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; white-space:nowrap; border:2px dashed #555;">w.r.t Weights $W_1$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">For updating weights</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$\frac{\nabla L}{\nabla W_1} = \frac{\nabla L}{\nabla A_1} \cdot X^\top$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; white-space:nowrap; border:2px dashed #555;">w.r.t Activations $X$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">Backpropagated to input</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$\frac{\nabla L}{\nabla X} = \frac{\nabla L}{\nabla A_1} \cdot W_1$</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;" colspan="3"><strong>Parameter Update (Adam Optimizer)</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; white-space:nowrap; border:2px dashed #555;">1st moment</td>
      <td style="padding:10px 12px; border:2px dashed #555;">Exponential moving avg of gradients</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \frac{\nabla L}{\nabla W}$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; white-space:nowrap; border:2px dashed #555;">2nd moment</td>
      <td style="padding:10px 12px; border:2px dashed #555;">Exponential moving avg of squared gradients</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$v_t = \beta_2 v_{t-1} + (1 - \beta_2) \left(\frac{\nabla L}{\nabla W}\right)^2$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; white-space:nowrap; border:2px dashed #555;">Weight update</td>
      <td style="padding:10px 12px; border:2px dashed #555;">Adaptive learning rate step</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$W_{t+1} = W_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}$</td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 2: Backward Pass Gradient Calculations and Parameter Updates</figcaption>
</figure>



<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; table-layout:fixed; border:2px dashed #555;">
  <colgroup>
    <col style="width:35%;">
    <col style="width:18%;">
    <col style="width:47%;">
  </colgroup>
  <thead>
    <tr style="text-align:left;">
      <th style="padding:8px 12px; border:2px dashed #555;">Code</th>
      <th style="padding:8px 12px; text-align:center; border:2px dashed #555;">Phase</th>
      <th style="padding:8px 12px; border:2px dashed #555;">What happens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:8px 12px; font-family:monospace; white-space:nowrap; border:2px dashed #555;"><code>output = model(input)</code></td>
      <td style="padding:8px 12px; text-align:center; border:2px dashed #555;"><span style="display:inline-block; min-width:110px; background:#e7f0fa; color:#1971c2; padding:3px 10px; border-radius:4px; font-weight:600; font-size:12px; text-align:center;">Forward Pass</span></td>
      <td style="padding:8px 12px; border:2px dashed #555;">Input flows through layers, producing activations and final prediction</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; font-family:monospace; white-space:nowrap; border:2px dashed #555;"><code>loss = loss_fn(output, target)</code></td>
      <td style="padding:8px 12px; text-align:center; border:2px dashed #555;"><span style="display:inline-block; min-width:110px; background:#e7f0fa; color:#1971c2; padding:3px 10px; border-radius:4px; font-weight:600; font-size:12px; text-align:center;">Forward Pass</span></td>
      <td style="padding:8px 12px; border:2px dashed #555;">Compare prediction with target to compute scalar loss</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; font-family:monospace; white-space:nowrap; border:2px dashed #555;"><code>loss.backward()</code></td>
      <td style="padding:8px 12px; text-align:center; border:2px dashed #555;"><span style="display:inline-block; min-width:110px; background:#fce4ec; color:#c2255c; padding:3px 10px; border-radius:4px; font-weight:600; font-size:12px; text-align:center;">Backward Pass</span></td>
      <td style="padding:8px 12px; border:2px dashed #555;">Compute gradients w.r.t weights and activations for every layer</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; font-family:monospace; white-space:nowrap; border:2px dashed #555;"><code>optimizer.step()</code></td>
      <td style="padding:8px 12px; text-align:center; border:2px dashed #555;"><span style="display:inline-block; min-width:110px; background:#efe8e5; color:#846358; padding:3px 10px; border-radius:4px; font-weight:600; font-size:12px; text-align:center;">Optimizer Step</span></td>
      <td style="padding:8px 12px; border:2px dashed #555;">Update optimizer states (e.g. Adam moments) and adjust weights</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; font-family:monospace; white-space:nowrap; border:2px dashed #555;"><code>optimizer.zero_grad()</code></td>
      <td style="padding:8px 12px; text-align:center; border:2px dashed #555;"><span style="display:inline-block; min-width:110px; background:#f0f0f0; color:#555; padding:3px 10px; border-radius:4px; font-weight:600; font-size:12px; text-align:center;">Cleanup</span></td>
      <td style="padding:8px 12px; border:2px dashed #555;">Reset gradients to zero before the next batch</td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 3: Pytorch Training Loop Steps </figcaption>
</figure>

## Why One GPU is Never Enough

Now that we have refreshed our memory on the basic PyTorch training loop, we are ready to understand the requirements for training a frontier Large Language Model. Training an LLM is a fundamentally different engineering problem compared to earlier models.

To understand why, let's start with something concrete. According to the "Llama 3 Herd of Models" [paper](https://arxiv.org/pdf/2407.21783), Llama 3 405B had a training budget of $3.8 \times 10^{25}$ FLOPs.

If we tried to train this model on a high-end CPU running at 11.37 TFLOPS, it would take approximately 105,900 years (see <a href="#table-gpu-vs-cpu">Table 4</a>). Even on a single NVIDIA H100 GPU — one of the most powerful GPUs available at the time of training Llama-3, running at 67 TFLOPS (FP32) — the time comes down to around 17,972 years. 

We are making two major assumptions here: (1) that a 405B model could fit on a single GPU (it can't), and (2) that we could achieve 100% of peak throughput. In practice, achieving peak throughput is nearly impossible — even with hand-crafted kernels like FlashAttention and meticulous infrastructure planning, 40–50% of peak throughput is considered successful.

This is why scaling laws dictate that for architectures of this magnitude, CPUs are no longer a viable choice. GPU architectures are purpose-built for the massive, parallel mathematical computations required at scale. This became evident in the history of CNNs when researchers used GPUs to train AlexNet; today, for LLMs, the GPU's role is irreplaceable.

But even if we had infinite time, there's another fundamental constraint: **memory**.

<figure>
<table id="table-gpu-vs-cpu" style="width:100%; border-collapse:collapse; margin:24px 0; font-size:15px; border:2px dashed #555;">
  <thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555;">Hardware</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Peak Throughput</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Time Calculation</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Time to Train Llama-3 405B</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:10px 12px; border:2px dashed #555;" colspan="4"><strong>Training Budget:</strong> $3.8 \times 10^{25}$ FLOPs &nbsp;|&nbsp; <strong>Formula:</strong> $T = \frac{\text{Training Budget}}{\text{Peak Throughput}}$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">NVIDIA H100 GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">67 TFLOPS ($67 \times 10^{12}$ FLOPS)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$\frac{3.8 \times 10^{25}}{67 \times 10^{12}} = 5.67 \times 10^{11}$ sec</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>~17,972 Years</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">High-End CPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">~11.37 TFLOPS (11.37 \times 10^{12}$ FLOPS)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$\frac{3.8 \times 10^{25}}{11.37 \times 10^{12}} = 3.342 \times 10^{12}$ sec</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>~105,900 Years</strong></td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 4: GPU vs CPU Training Time Comparison for Llama-3 405B</figcaption>
</figure>

### What actually lives on that GPU?

Let's go back to that assumption we made earlier — that a 405B model could fit on a single GPU. To understand why that's a problem, we need to understand what a model actually is.

A model isn't just its weights. When we call model.to(device), we're not just moving parameters onto the GPU — we're moving an entire graph of interconnected components: Parameters for the forward pass, Gradients for every parameter, Optimizer States 
like Adam's m and v vectors, Activations from the forward pass held in memory for the backward pass, and temporary buffers for intermediate calculations. On top of these, input and output tensors need to sit on the same device as the model. All of that 
has to live in VRAM simultaneously.

Looking more closely at these components, they fall into two categories — Static and Dynamic. Static components like Parameters, Gradients, and Optimizer States are determined purely by model architecture and remain fixed regardless of how you configure your training run. Dynamic components — primarily Activations — scale directly with Batch Size and Sequence Length, and can exceed all static components combined at large scales.

The table below captures the key configurations that drive memory requirements across both categories. In the sections that follow, we'll work through the memory formula for each of them — starting with static components where the math is straightforward, then moving to Activations which require a different approach.


<figure>
<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; border:2px dashed #555;">
<thead>
    <tr style="text-align:left;">
      <th style="padding:8px 12px; border:2px dashed #555; width:30%;">Category</th>
      <th style="padding:8px 12px; border:2px dashed #555;">Configuration</th>
      <th style="padding:8px 12px; border:2px dashed #555; width:35%;">Description</th>
    </tr>
</thead>
<tbody>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:8px 12px; border:2px dashed #555;" colspan="3"><strong>Memory Components</strong></td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;" rowspan="3"><strong>Static</strong><br><span style="font-size:0.8rem; color:#666;">(Fixed by architecture)</span></td>
      <td style="padding:8px 12px; border:2px dashed #555;">Parameters (w, b)</td>
      <td style="padding:8px 12px; border:2px dashed #555;">Model weights and biases</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Gradients ($\nabla$)</td>
      <td style="padding:8px 12px; border:2px dashed #555;">Computed during backward pass</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Optimizer States (m, v)</td>
      <td style="padding:8px 12px; border:2px dashed #555;">Adam's 1st & 2nd moments</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;"><strong>Dynamic</strong><br><span style="font-size:0.8rem; color:#666;">(Scales with data)</span></td>
      <td style="padding:8px 12px; border:2px dashed #555;">Activations</td>
      <td style="padding:8px 12px; border:2px dashed #555;">Intermediate outputs held for backward pass</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:8px 12px; border:2px dashed #555;" colspan="3"><strong>Model Architecture</strong></td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;" rowspan="6"><strong>Architecture</strong><br><span style="font-size:0.8rem; color:#666;">(Defines model structure)</span></td>
      <td style="padding:8px 12px; border:2px dashed #555;">Vocabulary Size (V)</td>
      <td style="padding:8px 12px; border:2px dashed #555;">Number of unique tokens</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Hidden Dimension (H)</td>
      <td style="padding:8px 12px; border:2px dashed #555;">Token embedding size</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Transformer Blocks (L)</td>
      <td style="padding:8px 12px; border:2px dashed #555;">Number of layers</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Attention Type</td>
      <td style="padding:8px 12px; border:2px dashed #555;">MHA, GQA, or MLA</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">FFN Type</td>
      <td style="padding:8px 12px; border:2px dashed #555;">Dense FFN or MoE</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Experts (if MoE)</td>
      <td style="padding:8px 12px; border:2px dashed #555;">Number of expert networks</td>
    </tr>
    <tr style="background:var(--global-code-bg-color, #f8f8f8);">
      <td style="padding:8px 12px; border:2px dashed #555;" colspan="3"><strong>Training Configuration</strong></td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;" rowspan="3"><strong>Data & Precision</strong><br><span style="font-size:0.8rem; color:#666;">(Training choices)</span></td>
      <td style="padding:8px 12px; border:2px dashed #555;">Batch Size (B)</td>
      <td style="padding:8px 12px; border:2px dashed #555;">Sequences per training step</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Sequence Length (S)</td>
      <td style="padding:8px 12px; border:2px dashed #555;">Context window size</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Numerical Precision</td>
      <td style="padding:8px 12px; border:2px dashed #555;">FP32, BF16, FP16 for each component</td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 5: Configurations Influencing GPU Memory Requirements</figcaption>
</figure>

### Parameters

In order to understand how much memory is consumed by parameters, gradients, optimizer
states, and activations, we first need to walk through every learnable component in
the model. Each configuration in Table 5 above influences the final parameter count — and
with small changes to those knobs, many of the open-source LLMs you know emerge.
We will not cover every variant here; instead we focus on two reference architectures:
the classic encoder-decoder Transformer from Attention Is All You Need<d-cite key="vaswani2017attention"></d-cite>
(treated as a decoder-only stack for a fair comparison) and the Llama 3 decoder-only
architecture.<d-cite key="dubey2024llama3"></d-cite>
For a gallery of OSS LLM architectures and how their configurations differ, see the
OSS LLM Architecture Gallery section.

Steps common to both architectures

- Tokenisation — each sentence in the training corpus is split into tokens.
- Vocabulary — the set of unique tokens across the entire corpus, of size $$V$$.
- Token Embeddings — each token index is mapped to a dense vector of dimension
- $$H$$ via a learnable table of shape $$(V \times H)$$.
- Positional information — how position is encoded differs between the two architectures and is discussed below.
- $$L \times$$ Transformer Block — the core repeated unit.
- Final normalisation.
- LM Head — projects the hidden state of the last token back to vocabulary logits.


<figure id="table-arch-comparison">
<table class="arch-table">
  <thead>
    <tr>
      <th style="width:18%"><span class="comp">COMPONENT</span></th>
      <th style="width:41%">
        <div class="col-header-classic">CLASSIC TRANSFORMER</div>
        <div class="sub">MHA · FFN · LAYERNORM · WITH BIAS</div>
      </th>
      <th style="width:41%">
        <div class="col-header-llama">LLAMA 3</div>
        <div class="sub">GQA · SWIGLU · RMSNORM · NO BIAS</div>
      </th>
    </tr>
  </thead>
  <tbody>

    <!-- EMBEDDING SECTION -->
    <tr><td colspan="3" class="section-header">EMBEDDING · ONCE</td></tr>
    <tr>
      <td class="comp">W_E</td>
      <td><div class="main">V × H</div></td>
      <td><div class="main">V × H</div></td>
    </tr>
    <tr>
      <td><span class="comp">Position</span></td>
      <td><div class="main">Sinusoidal</div><div class="sub">0 params · pre-block</div></td>
      <td><div class="main">RoPE</div><div class="sub">0 params · inside attention</div></td>
    </tr>

    <!-- PER LAYER SECTION -->
    <tr><td colspan="3" class="section-header">PER TRANSFORMER LAYER (× L)</td></tr>
    <tr>
      <td><span class="comp">Norm</span><div class="sub">pre-attention</div></td>
      <td><div class="main">LayerNorm 2H</div><div class="sub">scale γ + shift β · post-norm</div></td>
      <td><div class="main">RMSNorm H</div><div class="sub">scale γ only · pre-norm</div></td>
    </tr>
    <tr>
      <td class="comp">W_Q</td>
      <td>
        <div class="main">H × H + H</div>
        <div class="sub">Weights: H*H</div>
        <div class="sub">Bias: H</div>
      </td>
      <td>
        <div class="main">H × H </div>
        <div class="sub">Weights: H*H</div>
      </td>
    </tr>
    <tr>
      <td class="comp">W_K, W_V</td>
      <td>
        <div class="main">2 × (H × H + H)</div>
        <div class="sub">Weights: H * H</div>
        <div class="sub">Bias: H</div>
      </td>
      <td>
        <div class="main">2 × [H × (g/n)H]</div>
        <div class="sub">n = number of Q heads</div>
        <div class="sub">g = number of KV heads</div>
      </td>
    </tr>
    <tr>
      <td class="comp">W_O</td>
      <td>
        <div class="main">H × H + H</div>
        <div class="sub">Weights: H * H</div>
        <div class="sub">Bias: H</div>
      </td>
      <td>
        <div class="main">H × H</div>
        <div class="sub">Weights: H * H</div>
      </td>
    </tr>
    <tr>
      <td><span class="comp">Norm</span><div class="sub">pre-MLP</div></td>
      <td><div class="main">LayerNorm 2H</div></td>
      <td><div class="main">RMSNorm H</div></td>
    </tr>
    <tr>
      <td class="comp">MLP / FFN</td>
      <td>
        <div class="main">2 × (H × 4H + 4H)</div>
        <div class="sub">Up Projection: h ⇒ 4h (weights: h*4h, bias: 4h)</div>
        <div class="sub">Down Projection 4h ⇒ h (weights: 4h*h, bias: h)</div>
      </td>
      <td>
        <div class="main">3 × (H × f)</div>
        <div class="sub">$\text{MLP}(x) = ( \text{SiLU}(xW_g) \cdot xW_u ) W_d$</div>
        <div class="sub">Gate Projection (Wg): H * f</div>
        <div class="sub">Up Projection (Wu): H * f</div>
        <div class="sub">Down Projection (Wd): H * f</div>
        <div class="note"><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py#L214" target="_blank">f ≈ 3.5*h (SwiGLU Gate/Up/Down)</a></div>
      </td>
    </tr>

    <!-- OUTPUT SECTION -->
    <tr><td colspan="3" class="section-header">OUTPUT · ONCE</td></tr>
    <tr>
      <td class="comp">Final Norm</td>
      <td><div class="main">LayerNorm 2H</div></td>
      <td><div class="main">RMSNorm H</div></td>
    </tr>
    <tr>
      <td class="comp">LM Head</td>
      <td><div class="main">Tied to W_E</div><div class="sub">0 extra params</div></td>
      <td><div class="main">H × V</div><div class="sub">Separate matrix</div></td>
    </tr>
    <tr>
      <td><span class="comp">TOTAL</span><div class="sub">ESTIMATE</div></td>
      <td style="font-size:0.75rem">VH + L(12H² + 13H) + 2H</td>
      <td style="font-size:0.75rem">VH + L[(2+2g/n)H² + 3Hf + 2H] + H + VH</td>
    </tr>

  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 6: Classic Transformer vs Llama 3 Architecture Comparison</figcaption>
</figure>


##### Worked example — Llama 3 8B

We use the published configuration from HuggingFace
([config.json](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json),
[params.json](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/original/params.json)).

```
V = 128,256   (vocab_size)
H = 4,096     (hidden_size)
L = 32        (num_hidden_layers)
n = 32        (num_attention_heads)
g = 8         (num_key_value_heads)
f = 14,336    (intermediate_size)
d = H/n = 128 (per-head dimension)
```

<aside>
  <strong>Why f = 14,336?</strong><br>
  Llama 3 derives the FFN width as
  int(ffn_dim_multiplier × int(8H/3)) rounded up to the nearest
  multiple_of = 1,024. With ffn_dim_multiplier = 1.3 this gives
  int(1.3 × 10,922) =  14,198  rounded up to nearest multiple of 1024 → <strong>14,336</strong>.
</aside>

##### 1 — Token embedding  *(once)*

$$W_E = V \times H = 128{,}256 \times 4{,}096 = \textbf{525,336,576} \approx 525\text{ M}$$

---

##### 2 — Transformer layer  *(× 32)*

**Pre-attention RMSNorm**

$$\text{RMSNorm} = H = 4{,}096$$

**Attention**

```
W_Q = H × H         = 4,096 × 4,096          = 16,777,216
W_K = H × (g·d)     = 4,096 × (8 × 128)      =  4,194,304
W_V = H × (g·d)     = 4,096 × 1,024          =  4,194,304
W_O = H × H         = 4,096 × 4,096          = 16,777,216
                                        ─────────────────
                     Attention subtotal       = 41,943,040  ≈ 41.9 M
```

> **GQA saving:** W_K and W_V are each 4,194,304 vs 16,777,216 in MHA —
> exactly **4× smaller** because $$g/n = 8/32 = \tfrac{1}{4}$$.

**Pre-MLP RMSNorm**

$$\text{RMSNorm} = H = 4{,}096$$

**SwiGLU MLP**

```
W_gate = H × f      = 4,096 × 14,336         = 58,720,256
W_up   = H × f      = 4,096 × 14,336         = 58,720,256
W_down = f × H      = 14,336 × 4,096         = 58,720,256
                                        ─────────────────
                     MLP subtotal             = 176,160,768  ≈ 176.2 M
```

**Per-layer total**

$$
\underbrace{8{,}192}_{2\times\text{RMSNorm}}
+ \underbrace{41{,}943{,}040}_{\text{attention}}
+ \underbrace{176{,}160{,}768}_{\text{MLP}}
= \textbf{218,112,000} \approx 218\text{ M}
$$

**All 32 layers**

$$32 \times 218{,}112{,}000 = \textbf{6,979,584,000} \approx 6.98\text{ B}$$

---

##### 3 — Output  *(once)*

```
Final RMSNorm = H                              =          4,096
W_out = H × V   = 4,096 × 128,256             =    525,336,576  ≈ 525 M
```

> W_out is *not* weight-tied to W_E in Llama 3 — both contribute independently.
> <a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json#L22" target="_blank">See Llama 3 8B config.json</a>


##### 4 — Total

| Component | Parameters |
| :--- | ---: |
| Token embedding W_E | 525,336,576 |
| 32 × transformer layer | 6,979,584,000 |
| Final RMSNorm | 4,096 |
| LM head W_out | 525,336,576 |
| **Total** | **8,030,261,248 ≈ 8.03 B** |


---

Now that we have $$ \phi \approx 8.03\text{ B}$$, we can calculate the exact 
memory cost for each training component. Parameters in FP32 occupy 4 bytes each — in BF16/FP16 they occupy 2 bytes. HuggingFace stores Llama-3 weights in BF16 for inference [hugging-face](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json#L23). 

Inference only requires a forward pass — no gradients, no optimizer updates — so storing parameters in a single precision works fine. Training is a different story.

During training we need to run both forward and backward passes, compute gradients, and update parameters using optimizer states. If we store everything in BF16, we get a real speed advantage — GPU architectures from Volta onwards include Tensor Cores that deliver significantly higher throughput for matrix multiplications. Later architectures from A100 added native BF16/FP16 support via Tensor Cores. However, using BF16 everywhere we are trading accuracy for throughput .

Gradient values can be very small, and BF16's limited precision rounds small values to zero — losing the update entirely. The same applies to optimizer states, where Adam's moment estimates need to track subtle changes across millions of steps. Lost precision here 
leads directly to training instability and poor convergence.

The solution is mixed precision — use BF16 where speed matters, FP32 where precision matters. The steps below show how this plays out across one training step:


1. **Forward Pass** — parameters are kept in BF16; fast Tensor Core matrix
     multiplications run in BF16, producing BF16 activations and outputs.
2. **Backward Pass** — gradients are initially computed in BF16, then
     accumulated and reduced in FP32 to preserve precision on small values.
3. **Optimizer Step** — FP32 gradients and FP32 optimizer states (Adam's
     $m_t$, $v_t$) are used to compute the parameter update entirely in FP32.
4. **Parameter Cast** — the updated FP32 master weights are cast back to BF16,
     ready for the next forward pass.


| Component| Precision| Bytes| 
| :---| ---: | ---:|
| Parameters($$\phi$$) working copy|  BF16 | 2 Bytes |
| Parameters($$\phi$$) master copy| FP32 | 4 Bytes |
| Gradients Acc| FP32 | 4 Bytes|
| Optimizer States| FP32 | 4 Bytes |
   
  
Since Gradients and Optimizer States scale directly with parameter count, we can express their memory cost in terms of $$\phi$$:


> $\text{Parameters — working copy} \quad \rightarrow \quad 2\phi \text{ bytes (BF16)}$
>
> $\text{Parameters — master copy} \quad \rightarrow \quad 4\phi \text{ bytes (FP32)}$
>
> $\text{Gradients } (\nabla W) = \phi \quad \rightarrow \quad 4\phi \text{ bytes (FP32)}$
>
> $\text{Optimizer States } (m, v) = 2\phi \quad \rightarrow \quad 8\phi \text{ bytes (FP32)}$
>
> ───────────────────────────────────
>
> $\text{Total Static Memory} \quad = \quad 18\phi \text{ bytes}$

This sums up to **18 bytes per parameter** for mixed precision training with Adam Optimizer For Llama-3 8B with $$\phi \approx 8.03\text{B}$$ parameters, static components alone require approximately **144GB** — nearly double the H100's 80GB VRAM, 144GB memory is required before a single activation is stored.

Next we look at Activations — the dynamic component whose memory cost changes with every training configuration (**Batch Size, Sequence Length**).

### Activations

Regardless of architecture variant, every transformer-based LLM shares the same
fundamental structure: input tokens flow through an embedding layer, then through
L repeated transformer blocks (each containing Attention and an MLP/FFN), and
finally through an output projection. At each step, the output of one operation
becomes the input to the next — these intermediate outputs are called **Activations**. 

Activations are not discarded after the forward pass. The backward pass requires
the activation produced at each layer to compute the gradient for that layer's
weights (see <a href="#table-backward-pass">Table 2</a>). This means every
activation from the entire forward pass must stay in memory until backpropagation
finishes.

This is what separates activations from the static components. Parameters,
Gradients, and Optimizer States are fixed by model architecture — their memory
cost is constant for a given model. Activations scale with **Batch Size** and
**Sequence Length**, and at large scales they easily dwarf the static footprint.
The largest share of activation memory sits inside the transformer blocks —
the table below breaks this down operation by operation for one Llama 3 block.


##### Per-Block Activation Memory Breakdown

<figure id="table-activation-memory">
<table class="arch-table">
  <thead>
    <tr>
      <th style="width:10%"><span class="comp">Step</span></th>
      <th style="width:25%"><span class="comp">Operation</span></th>
      <th style="width:35%"><span class="comp">Shape</span></th>
      <th style="width:30%"><span class="comp">Memory (Bytes)</span></th>
    </tr>
  </thead>
  <tbody>
    <!-- ATTENTION SECTION -->
    <tr><td colspan="4" class="section-header">ATTENTION</td></tr>
    
    <tr>
      <td class="comp">1.a</td>
      <td><div class="main">Block Input/Residual(Pre-Attn)</div></td>
      <td><div class="main">$(B, S, H)$</div></td>
      <td><div class="main">$2 \cdot BSH$</div></td>
    </tr>
    <tr>
      <td class="comp">1.b</td>
      <td><div class="main">RMSNorm (Pre-Attn)</div></td>
      <td><div class="main">$(B, S, H)$</div></td>
      <td><div class="main">$2 \cdot BSH$</div></td>
    </tr>
    
    <tr>
      <td class="comp">2.a</td>
      <td><div class="main">Q Projection</div></td>
      <td><div class="main">$(B, S, H)$</div></td>
      <td><div class="main">$2 \cdot BSH$</div></td>
    </tr>
    <tr>
      <td class="comp">2.b</td>
      <td><div class="main">K Projection</div></td>
      <td><div class="main">$(B, S, g/n \cdot H)$</div></td>
      <td><div class="main">$0.5 \cdot BSH$</div></td>
    </tr>
    <tr>
      <td class="comp">2.c</td>
      <td><div class="main">V Projection</div></td>
      <td><div class="main">$(B, S, g/n \cdot H)$</div></td>
      <td><div class="main">$0.5 \cdot BSH$</div></td>
    </tr>
    <tr>
      <td class="comp">2.d</td>
      <td><div class="main">Score ($QK^T$)</div></td>
      <td><div class="main">$(B, n, S, S)$</div></td>
      <td><div class="main">$2 \cdot BnS^2$</div></td>
    </tr>
    <tr>
      <td class="comp">2.e</td>
      <td><div class="main">Softmax</div></td>
      <td><div class="main">$(B, n, S, S)$</div></td>
      <td><div class="main">$2 \cdot BnS^2$</div></td>
    </tr>
    <tr>
      <td class="comp">2.f</td>
      <td><div class="main">Attn Output</div></td>
      <td><div class="main">$(B, S, H)$</div></td>
      <td><div class="main">$2 \cdot BSH$</div></td>
    </tr>
    <tr>
      <td class="comp">2.g</td>
      <td><div class="main">O Projection</div></td>
      <td><div class="main">$(B, S, H)$</div></td>
      <td><div class="main">$2 \cdot BSH$</div></td>
    </tr>

    <!-- MLP SECTION -->
    <tr><td colspan="4" class="section-header">MLP / FFN</td></tr>
    
    <tr>
      <td class="comp">3.a</td>
      <td><div class="main">Block Input/ Residual(Pre-MLP)</div></td>
      <td><div class="main">$(B, S, H)$</div></td>
      <td><div class="main">$2 \cdot BSH$</div></td>
    </tr>
    
    <tr>
      <td class="comp">3.b</td>
      <td><div class="main">RMSNorm (Pre-MLP)</div></td>
      <td><div class="main">$(B, S, H)$</div></td>
      <td><div class="main">$2 \cdot BSH$</div></td>
    </tr>
    
    <tr>
      <td class="comp">4.a</td>
      <td><div class="main">Gate Proj</div></td>
      <td><div class="main">$(B, S, f)$</div></td>
      <td><div class="main">$2 \cdot BSf$</div></td>
    </tr>
    <tr>
      <td class="comp">4.b</td>
      <td><div class="main">Up Proj</div></td>
      <td><div class="main">$(B, S, f)$</div></td>
      <td><div class="main">$2 \cdot BSf$</div></td>
    </tr>
    <tr>
      <td class="comp">4.c</td>
      <td><div class="main">Down Proj</div></td>
      <td><div class="main">$(B, S, H)$</div></td>
      <td><div class="main">$2 \cdot BSH$</div></td>
    </tr>
     <tr>
      <td class="comp">4.d</td>
      <td><div class="main">SiLU Output</div></td>
      <td><div class="main">$(B, S, f)$</div></td>
      <td><div class="main">$2 \cdot BSf$</div></td>
    </tr>
    <tr><td colspan="4" class="section-header">TOTAL PER BLOCK</td></tr>
    <tr>
      <td colspan="4" style="padding:16px 12px;">
        <div style="text-align:center; font-size:0.9rem;">
          <strong style="font-size:1rem; color:#c040c0;">$\text{Total} = 17 \cdot BSH + 4 \cdot BnS^2 + 6 \cdot BSf$</strong>
        </div>
        <div style="margin-top:12px; font-size:0.75rem; color:#666; line-height:1.6;">
          <strong>Breakdown:</strong><br>
          <span style="color:#888;">Attention:</span> 2BSH (Residual) + 2BSH (RMSNorm) + 2BSH (Q) + 0.5BSH (K) + 0.5BSH (V) + 2BnS² (Score) + 2BnS² (Softmax) + 2BSH (Attn Out) + 2BSH (O Proj)<br>
          <span style="color:#888;">MLP:</span> 2BSH (Residual) + 2BSH (RMSNorm) + 2BSf (Gate) + 2BSf (Up) + 2BSf (SiLU) + 2BSH (Down)
        </div>
      </td>
    </tr>

  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 7: Per-Block Activation Memory Breakdown for Llama 3 (8B)</figcaption>
</figure>

<div style="display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin:16px 0; font-size:0.8rem;">
  <span style="background:#f0f0f0; padding:4px 10px; border-radius:4px;"><strong>B</strong> = Batch Size</span>
  <span style="background:#f0f0f0; padding:4px 10px; border-radius:4px;"><strong>S</strong> = Sequence Length</span>
  <span style="background:#f0f0f0; padding:4px 10px; border-radius:4px;"><strong>H</strong> = Hidden Dimension</span>
  <span style="background:#f0f0f0; padding:4px 10px; border-radius:4px;"><strong>L</strong> = Transformer Blocks</span>
  <span style="background:#f0f0f0; padding:4px 10px; border-radius:4px;"><strong>n</strong> = Q Heads</span>
  <span style="background:#f0f0f0; padding:4px 10px; border-radius:4px;"><strong>g</strong> = KV Heads</span>
  <span style="background:#f0f0f0; padding:4px 10px; border-radius:4px;"><strong>f</strong> = FFN Intermediate Size</span>
</div>

> **FlashAttention Optimization:** Llama 3 uses [FlashAttention](https://arxiv.org/abs/2205.14135), which computes attention in tiles without materializing the full S×S score matrix. Steps 2.d (Score) and 2.e (Softmax) are never stored — it uses online softmax calculations to reduce memory for attention scores & softmax from O(S²) to O(S).

The [Llama 3 paper (Section 3.4)](https://arxiv.org/pdf/2407.21783#section.3.4) details the pre-training recipe used for training the Llama 3 series of models. For Llama 3 405B, they use a **progressive batch size schedule**, Similar recipes are used for the 8B and 70B models.

| Training Phase | Global Batch Size | Sequence Length | Training Tokens |
|----------------|-------------------|-----------------|-----------------|
| Initial | 4M tokens | 4,096 | 0 → 252M |
| Phase 2 | 8M tokens | 8,192 | 252M → 2.87T |
| Final | 16M tokens | 8,192 | 2.87T → end |



**Example Calculation for Llama-3 8B:**

while the sequence length is gradually increased, for simplicity let us consider that Llama-3 8B uses configurations **S=8192, H=4096, f=14,336**. Considering a batch of single sequence **(B=1)** with BF16/FP16 precision: -  without **FlashAttention** Activation memory per transformer block ≈ **9.78GB**. For L=32 blocks: **32 × 9.78GB = 312.96 GB** -  with **FlashAttention:** this reduces to approximately **~37GB** (eliminating the O(S²) attention score storage)

```
Term 1: 17 × 8192 × 4096  =  570,425,344 bytes   ≈  0.53 GB
Term 2: 4 × 32 × 8192²    =  8,589,934,592 bytes ≈  8.59 GB
Term 3: 6 × 8192 × 14336  =  704,643,072 bytes   ≈  0.65 GB
─────────────────────────────────────────────────────────────
Per block total            ≈  9.78 GB
× 32 blocks                ≈  312.96 GB  (baseline, no FA)


Term 1: 17 × 8192 × 4096  =  57,04,25,344 bytes  ≈  0.53 GB
Term 2: 4 × 32 × 8192     =  1,048,576 bytes    ≈  0.001 GB
Term 3: 6 × 8192 × 14336  =  704,643,072 bytes  ≈  0.66 GB
─────────────────────────────────────────────────────────────
Per block total            ≈  1.19 GB
× 32 blocks                ≈  38.10 GB  (FA)

```

| Configuration | Per Block | Total (×32)  |
|----------------|-------------------|-----------------|
| W/o Flash Attention | 9.78 GB| 312.96 GB | 
| Flash Attention |  1.19 GB| 38.10 GB |



### Putting it All Together

Below <a href="#llama-3-config" >table </a>(from the Llama 3 [paper](https://arxiv.org/pdf/2407.21783)) shows how architecture choices scale across model sizes. From 8B to 405B, the number of layers grows from 32 to 126, the hidden dimension from 4,096 to 16,384, and the FFN intermediate size from 14,336 to 53,248.

Each of these numbers is a direct multiplier on memory. More layers means more
weight matrices and more activations to hold during the forward pass. A larger
hidden dimension makes every one of those matrices wider. These factors do not
add — they compound. A model that is 2× deeper *and* 4× wider does not cost 6×
more memory; it costs closer to 8×.

Architecture is only one side of the memory equation. Dynamic memory — activations — is driven by the data side: how long each sequence is and how many sequences are processed simultaneously.

Deep learning models are data-hungry. The more capable we want a model to be,
the more data it must see during training. The Llama series makes this trend
concrete: Llama 2 was trained on 1.8 trillion tokens, Llama 3 on 15 trillion,
and Llama 4 on approximately 30 trillion. More tokens means longer training runs,
larger batch sizes, and correspondingly larger activation footprints at every step.

<figure id="llama-3-config">
  {% include figure.liquid path="assets/img/llm-training/section-3/Llama-3-config.png" class="img-medium" caption="Figure-4: Llama 3 Configurations" %}
</figure>

To summarize what we have covered in this section:

- **Compute alone makes a single GPU infeasible.** At $3.8 \times 10^{25}$ FLOPs,
    training Llama 3 405B on a single H100 (67 TFLOPS FP32) would take ~18,000 year even at 100% utilisation.

- **Memory is the harder constraint.** GPU components fall into two categories:
  *static* (Parameters, Gradients, Optimizer States — fixed by architecture) and
  *dynamic* (Activations — scale with Batch Size and Sequence Length).

- **Static memory for Llama 3 8B: ~144 GB.** With mixed-precision Adam, each
  of the 8.03 B parameters requires 18 bytes, totalling ~144 GB — nearly double
  the H100's 80 GB VRAM before a single activation is stored.

- **Dynamic memory compounds the problem.** For a single sequence (B=1,
  S=8,192), activation memory per block is ~9.86 GB without FlashAttention and
  ~1.27 GB with it. Across 32 blocks that is **~315 GB** vs **~41 GB**.

- **A realistic micro-batch makes it far worse.** In practice, the micro-batch
  on one GPU is not a single sequence. With a global batch of 16 M tokens and
  16,384 sequences, each of thousands of GPUs processes multiple sequences.
  Even at a modest 32 sequences per GPU, activation memory with FlashAttention
  reaches **32 × 41 GB ≈ 1.3 TB** — more than 16× the H100's VRAM.

This is why no single GPU can train a frontier LLM. The following sections explore  how parallelism techniques distribute these memory and compute requirements across thousands of GPUs.

