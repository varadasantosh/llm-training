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

# bibliography: references.bib
---

## Introduction

Technology has a pattern — every few decades, evolution in technology challenges status quo and redraws the boundaries of the society, Mechanization didn't just make farming faster, it triggered the Industrial Revolution.The internet didn't 
just make communication cheaper, it shrunk the entire world. These 
weren't incremental improvements, they were resets.

Machine Learning is going through one such moments right now.

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

The PyTorch training loop is something most of us know well. But before diving into parallelism techniques, a clear understanding of how the forward and backward passes work creates a baseline for every thing that follows — hence it is worth revisiting the mechanics before moving forward.

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

   

**Forward Pass** : input X enters Layer 1, gets multiplied with that layer's weights, a bias is added, and the result becomes activation A₁. That activation becomes the input to Layer 2, where the same operation happens again, giving us our final prediction Ŷ. This pattern holds for any number of layers — the output of layer n-1 is always the input to layer n. To complete a forward pass, each layer only needs two things: its Parameters (w, b) and the incoming Activations.

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

<figure>
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
      <td style="padding:10px 12px; border:2px dashed #555;">Passed to preceeding layer</td>
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

## Why One GPU is Never Enough

Now that we have refreshed our memory on the basic PyTorch training loop, we are ready to understand the requirements for training a frontier Large Language Model. Training a  Large Language Model is fundamentally different engineering problem from earlier models came before them.

To understand why, let's start with something concrete. According to the "Llama 3 Herd of Models" [paper](https://arxiv.org/pdf/2407.21783), Llama 3 405B  had a training budget of $3.8 \times 10^{25}$ FLOPs.

If we tried to train this model on a high-end CPU running at 11.37 TFLOPS, it would take approximately 105,900 years (see <a href="#table-gpu-vs-cpu">Table 1</a>).Even on a single 
NVIDIA H100 GPU — one of the most powerful GPUs available at the time of training Llama-3  running at 67 TFLOPS (FP32), the time comes down to around 17,972 years. We are making two major assumptions here that a 405B model could even fit on a single GPU (it can't) and that we could achieve 100% of the peak throughput. In practice, achieving peak throughput is nearly impossible with hand-crafted kernels like FlashAttention and meticulous infrastructure planning achieving 40 - 50% of Peak Throughput is considered successful. 

This is why Scaling laws dictate that for architectures of this magnitude, CPUs are no longer a viable choice. GPU architectures are purpose-built for the massive, parallel mathematical computations required at scale. This became evident in the history of CNNs when researchers used GPUs to train AlexNet; today, for LLMs, the GPU's role is irreplaceable.

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
      <td style="padding:10px 12px; border:2px dashed #555;">67 TFLOPS ($34 \times 10^{12}$ FLOPS)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$\frac{3.8 \times 10^{25}}{67 \times 10^{12}} = 5.67 \times 10^{11}$ sec</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>~17,972 Years</strong></td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">High-End CPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">~11.37 TFLOPS ($5.36 \times 10^{12}$ FLOPS)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$\frac{3.8 \times 10^{25}}{11.37 \times 10^{12}} = 3.342 \times 10^{12}$ sec</td>
      <td style="padding:10px 12px; border:2px dashed #555;"><strong>~105,900 Years</strong></td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 1: GPU vs CPU Training Time Comparison for Llama-3 405B</figcaption>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;"> Training Budget: 3.8 × 10²⁵ FLOPs | Formula: T = Training Budget / Peak Throughput</figcaption>
</figure>

### What actually lives on that GPU?

Let's go back to that assumption we made earlier — that a 405B model could fit on a single GPU. To understand why that's a problem, we need to understand what a model actually is.

A model isn't just its weights. When we call **model.to(device)**, it is not just moving parameters onto the GPU we're moving an entire graph of interconnected components. Parameters required for forward pass, Gradients for every parameter, Optimizer States like Adam's m and v vectors, Activations from the forward pass that need to be held in memory for the backward pass and temporary buffers for intermediate calculations. All of that has 
to live in VRAM simultaneously. Along with model ,input and output tensors need to sit on the same device as the model . When we add it all up, the memory required for a single training step is significantly more than just the model size alone.

Training memory breaks down into these components — each of them competing for same VRAM or Off-chip RAM.

<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; table-layout:fixed; border:2px dashed #555;">
<thead>
    <tr style="text-align:left;">
      <th style="padding:8px 12px; border:2px dashed #555;">Parameters</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Parameters(w,b)</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Gradients($\nabla$)</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Optimizer States(m𝑡,v𝘵)</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Activations</td>
    </tr>
     <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Batch Size(B)</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Context Length(S)</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Vocabulary Size(V)</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Token Embedding Dimension(H)</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Number of Transformer Blocks(L)</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Type of Attention (MHA, GQA, MLA)</td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">FFN vs. MoE </td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Numerical Precision(Parameters, Gradients, Activations, Optimizer States)</td>
    </tr>
  </tbody>
</table>


### How model architecture amplifies the problem

Below <a href="#llama-3-config" >table </a> from Llama-3 [paper](https://arxiv.org/pdf/2407.21783) clearly demonstrates how model architecture influence size of the models. the model configurations for 8B, 70B, 405B variants says everything, layers go from 32 to 126, 
model dimension grows from 4,096 to 16,384, and FFN dimension jumps from 14,336 to 53,248. Every one of these numbers is a multiplier on your memory requirement. More layers means more weight matrices to store and more activations to hold in memory during the forward pass. Larger dimensions make each of those weight matrices bigger

 
<figure id="llama-3-config">
  {% include figure.liquid path="assets/img/llm-training/section-3/Llama-3-config.png" class="img-medium" caption="Figure-4: Llama 3 Configurations" %}
</figure>


The ultimate objective of training a models is to be used across globe, computation and scaling requirements for training and inference largely remains same except few differences but this is a topic for another day, current discussion requires us to be aware that inference requires only **Parameters** & **Activations** to complete <a href="#fig-forward-pass">forward pass</a>, along with this we also need to consider additional memory required for temporary buffers. considering the Llama-3 model variants 8B,70B, 405B. as earlier Llama-3 Herd of models were trained using H100 GPU, the HBM memory available on H100 is 80GB. let us do some math to calculate the memory required for Parameters alone with different precision types, Like many other models Llama models use mixed precision for training (**FP32** & **BF16**) , once the training  is finished we get access to parameters in **BF16** [ref](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct/blob/main/config.json#L34), models also support **INT8** and **INT4** quantization, numbers from below tables gives clear indication that only models of size 8B can fit on single H100 GPU while the 32GB mentioned is only for Parameters we still required memory for Activations & KVCache , rest of the models require multiple GPU's to even hold Parameters


<figure>
<table id="table-params-fp32-bf16" style="width:100%; border-collapse:collapse; margin:24px 0; font-size:15px; border:2px dashed #555;">
  <thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555;">Model Size (No. Parameters)</th>
      <th style="padding:10px 12px; border:2px dashed #555;">FP32</th>
      <th style="padding:10px 12px; border:2px dashed #555;">BF16</th>
      <th style="padding:10px 12px; border:2px dashed #555;">INT8</th>
      <th style="padding:10px 12px; border:2px dashed #555;">INT4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">8B</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$8B * 4 Bytes = 32 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$8B * 2 Bytes = 16 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$8B * 1 Bytes = 8 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$8B * 4 Bits  = 4 GB$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">70B</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$70B * 4 Bytes = 280 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$70B * 2 Bytes = 140 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$70B * 1 Bytes = 70 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$70B * 4 Bits  = 35 GB$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">405BB</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$405B * 4 Bytes = 1620 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$405B * 2 Bytes = 810 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$405B * 1 Bytes = 405 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$405B * 4 Bits  = 202.5 GB$</td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 1: GPU vs CPU Training Time Comparison for Llama-3 405B</figcaption>
</figure>



Deep Learning models are hungry — the more capable you want the model to be, the more 
data it needs to see during training, looking at family of Llama models this is very evident Llama-2 was trained on 1.8 Trillion Tokens, subsequent models Llama-3 & Llama-4 was trained on 15 Trillion & 30 Trillion Tokens respectively which translates to increase in batch size, number of batches, longer training runs & more compute.

The time constraint and the memory constraint together make one thing clear — we need to distribute this problem across many GPUs working together. That coordination is exactly what Parallelism techniques are designed to solve, and that's what the rest of this blog is about.