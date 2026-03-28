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
  - name: Why One GPU is Never Enough: Anatomy of a Frontier Model 

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

{% include figure.liquid path="assets/img/llm-training/section-2/forward-pass.svg" class="img-medium" caption="Figure-1: Forward Pass" %}

**Backward Pass:** Once **loss_fn** gives us the loss, it acts as a compass — telling the model 
how wrong it was, which direction to correct itself. The backward pass uses this signal to update the parameters of every layer in the network.

How much each parameter should change is determined by three things:Gradients, Optimizer States, and Learning Rate. Learning Rate is a hyperparameter we'll dedicate separate discussion to — for now, let's focus on Gradients and Optimizer States, since these are computed and stored every single training step.

Figure 2 shows exactly what's flowing in and out of each layer during 
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


{% include figure.liquid path="assets/img/llm-training/section-2/backward-pass.svg" class="img-medium" caption="Figure-2: Backward Pass" %}

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

## Why One GPU is Never Enough: Anatomy of a Frontier Model :

Now that we have refreshed our memory on the basic PyTorch training loop, we are ready to understand the requirements for training a frontier Large Language Model.

Scaling laws dictate that for architectures of this magnitude, CPUs are no longer a viable choice. GPU architectures are purpose-built for the massive, parallel mathematical computations required at scale. This became evident in the history of CNNs when researchers used GPUs to train AlexNet; today, for LLMs, the GPU's role is irreplaceable.

To put this in perspective, let’s look at the training budget for Llama 3 405B. According to the "Llama 3 Herd of Models" [paper](https://arxiv.org/pdf/2407.21783), Llama team considered a budge of $3.8 \times 10^{25}$ FLOPs.

If we consider high-performing CPU (approx. 5.36 TFLOPS) the time taken to train a model would arrive at 224,000 years (see <a href="#table-gpu-vs-cpu">Table 1</a>).  Even with a high-performing GPU a single high-performance NVIDIA H100 GPU (mentioned in Llama-3 paper) with peak throughput of 34 TFLOPS (FP32) time required to train **405B** model is **35,400 Years**, We make two major assumptions here: first, that a 405B model could even fit on a single GPU (it can't), and second, that we can utilize 100% of the peak throughput. In practice, achieving peak throughput is nearly impossible without hand-crafted GPU kernels like FlashAttention and meticulous infrastructure planning. This massive time requirement is our primary motivation for Parallelism.



<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:15px; border:2px dashed #555;">
  <thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555; width:40%;">Metric</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">1 TFLOPS</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$10^{12}$ FLOPS</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">H100 Peak Performance</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$34 \times 10^{12}$ FLOPS</td>
    </tr>
     <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Peak Performance of High Performing CPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$5.6 \times 10^{12}$ FLOPS</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Training Budget (Llama 405B)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$3.8 \times 10^{25}$ FLOPS</td>
    </tr>
  </tbody>
</table>

<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:15px; border:2px dashed #555;">
  <thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555; width:40%;">Step</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Calculation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Time (seconds)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$T = \frac{\text{Training Budget}}{\text{Peak Throughput}} = \frac{3.8 \times 10^{25}}{34 \times 10^{12}} = 1.12 \times 10^{12} \text{ seconds}$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">Time (years)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$T = \frac{1.12 \times 10^{12}}{3.156 \times 10^{7}} \approx \textbf{35,400 Years}$</td>
    </tr>
  </tbody>
</table>


<table id="table-gpu-vs-cpu" style="width:100%; border-collapse:collapse; margin:24px 0; font-size:15px; border:2px dashed #555;">
  <thead>
    <tr style="text-align:left;">
      <th style="padding:10px 12px; border:2px dashed #555; width:40%;">Hardware</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Peak Throughput</th>
      <th style="padding:10px 12px; border:2px dashed #555;">Time to Train Llama-3 405B </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">NVIDIA H100 GPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">34 TFLOPS (FP32)</td>
      <td style="padding:10px 12px; border:2px dashed #555;">~35,400 Years</td>
    </tr>
    <tr>
       <td style="padding:10px 12px; border:2px dashed #555;">High-End CPU</td>
      <td style="padding:10px 12px; border:2px dashed #555;">~5.36 TFLOPS</td>
      <td style="padding:10px 12px; border:2px dashed #555;">~224,400 Years</td>
    </tr>
  </tbody>
</table>


However, a model is not just a single entity. When we call **model.to(device)**, we are moving a complex graph of Parameters , Activations, Gradients, and Optimizer States into the limited VRAM of the GPU. From the original Transformer to modern frontier models, every architectural contributed to innovation—from RMSNorm and RoPE in Llama to Mixture of Experts (MoE) in Mixtral and MLA in DeepSeek — impacts this physical footprint of the memory required for model.

Below is a summary of the variables that influence our "Memory Bill." We will categorize them into Static Model States (the size of the "factory") and Dynamic Training States (the size of the "workload").

<table style="width:100%; border-collapse:collapse; margin:24px 0; font-size:14px; table-layout:fixed; border:2px dashed #555;">
<thead>
    <tr style="text-align:left;">
      <th style="padding:8px 12px; border:2px dashed #555;">Parameters</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Parameters(Weights & Biases)</td>
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
      <td style="padding:8px 12px; border:2px dashed #555;">Numerical Precision(Parameters, Gradients, Activations)</td>
    </tr>
  </tbody>
</table>
 
  {% include figure.liquid path="assets/img/llm-training/section-3/Llama-3-config.png" class="img-medium" caption="Figure-1: Llama 3 Configurations" %}
