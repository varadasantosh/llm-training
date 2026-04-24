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

Now that we have refreshed our memory on the basic PyTorch training loop, we are ready to understand the requirements for training a frontier Large Language Model. Training a  Large Language Model is fundamentally different engineering problem from earlier models came before them.

To understand why, let's start with something concrete. According to the "Llama 3 Herd of Models" [paper](https://arxiv.org/pdf/2407.21783), Llama 3 405B  had a training budget of $3.8 \times 10^{25}$ FLOPs.

If we tried to train this model on a high-end CPU running at 11.37 TFLOPS, it would take approximately 105,900 years (see <a href="#table-gpu-vs-cpu">Table 4</a>).Even on a single 
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
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 4: GPU vs CPU Training Time Comparison for Llama-3 405B</figcaption>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;"> Training Budget: 3.8 × 10²⁵ FLOPs | Formula: T = Training Budget / Peak Throughput</figcaption>
</figure>

### What actually lives on that GPU?

Let's go back to that assumption we made earlier — that a 405B model could fit on a single GPU. To understand why that's a problem, we need to understand what a model actually is.

A model isn't just its weights. When we call model.to(device), we're not just moving parameters onto the GPU — we're moving an entire graph of interconnected components: Parameters for the forward pass, Gradients for every parameter, Optimizer States 
like Adam's m and v vectors, Activations from the forward pass held in memory for the backward pass, and temporary buffers for intermediate calculations. On top of these, input and output tensors need to sit on the same device as the model. All of that 
has to live in VRAM simultaneously.

Looking more closely at these components, they fall into two categories — Static and Dynamic. Static components like Parameters, Gradients, and Optimizer States are determined purely by model architecture and remain fixed regardless of how you configure your training run. Dynamic components — primarily Activations — scale directly with Batch Size and Sequence Length, and can exceed all static components combined at large scales.

The table below captures the key configurations that drive memory requirements across both categories. In the sections that follow, we'll work through the memory formula for each of them — starting with static components where the math is straightforward, then 
moving to Activations which require a different approach.


<figure>
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
      <td style="padding:8px 12px; border:2px dashed #555;">Number of Expers in MoE </td>
    </tr>
    <tr>
      <td style="padding:8px 12px; border:2px dashed #555;">Numerical Precision(Parameters, Gradients, Activations, Optimizer States)</td>
    </tr>
  </tbody>
</table>
</figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 5: Configurations Influencing Model Size</figcaption>
</figure>

#### **Parameters**

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



<!-- <style>
.arch-table{width:100%;border-collapse:collapse;font-family:'JetBrains Mono', 'Courier New', monospace;font-size:0.82rem;border:1px dashed #bbb}
.arch-table td,.arch-table th{border:1px dashed #ccc;padding:9px 14px;vertical-align:top;background:#fff;color:#111}
.arch-table thead th{font-weight:600}
.arch-table .section-header{background:#f2f2f2;color:#a030b0;font-size:0.72rem;letter-spacing:.05em;font-weight:500;padding:5px 14px;border-top:1px dashed #bbb}
.arch-table .col-header-classic{color:#9030b0;font-size:0.72rem;font-weight:600}
.arch-table .col-header-llama{color:#1a50c8;font-size:0.72rem;font-weight:600}
.arch-table .main{font-weight:500;margin-bottom:2px}
.arch-table .sub{display: block; font-size: 0.72rem; opacity: 0.7; margin-top: 4px;}
.arch-table .note{font-size:0.68rem;color:#b030a0;margin-top:3px}
.arch-table .comp{font-weight:600}
</style> -->

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
        <div class="main">H × H + H</div>
        <div class="sub">Weights: H*H</div>
        <div class="sub">Bias: H</div>
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
        <div class="sub">Bias: H</div>
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
        <div class="formula">MLP(x) = (SiLU(xW<sub>g</sub>) · xW<sub>u</sub>)W<sub>d</sub></div>
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
  int(ffn_dim_multiplier × int(2H/3)) rounded up to the nearest
  multiple_of = 1,024. With ffn_dim_multiplier = 1.3 this gives
  int(1.3 × 2,730) = 3,549, × 4 = 14,196, rounded up → <strong>14,336</strong>.
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
memory cost for each training component. Parameters in FP32 occupy 4 bytes each — in BF16/FP16 they occupy 2 bytes. HuggingFace stores Llama-3 weights in BF16 for inference ref from [hugging-face](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json#L23). 

Inference only requires a forward pass — no gradients, no optimizer updates — so storing parameters in a single precision works fine. Training is a different story.

During training we need to run both forward and backward passes, compute gradients, and update parameters using optimizer states. If we store everything in BF16, we get a real speed advantage — GPU architectures from A100 onwards include Tensor Cores that 
deliver significantly higher throughput for BF16/FP16 matrix multiplications than FP32. But using BF16 everywhere we are trading accuracy with high throughput .

Gradient values can be very small, and BF16's limited precision rounds small values to zero — losing the update entirely. The same applies to optimizer states, where Adam's moment estimates need to track subtle changes across millions of steps. Lost precision here 
leads directly to training instability and poor convergence.

The solution is mixed precision — use BF16 where speed matters, FP32 where precision matters. The steps below show how this plays out across one training step:


1. Forward Pass
    Parameters (BF16) → fast matrix multiplications 
    using Tensor Cores → Output (BF16)
2. Backward Pass:
    Gradients computed in BF16 initially
    Gradients accumulated/reduced in FP32  ← precision is more important  
3. Optimizer Step
    FP32 gradients + FP32 optimizer states → parameter update (FP32)  
4. Parameter Cast
    Updated parameters cast back to BF16 ← ready for next forward pass



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

Next we look at Activations — the dynamic component whose memory cost changes with every training configuration (**Batch Size, Sequnece Length**).

#### **Activations**
There are various types of Large Language Models (LLM,VLLM,Difussion LLM),While specific design and Architecture vary by modality and its purpose(chat, completions).  the fundamental building block remains the Transformer Block consisting of Layer Norm → Attention → Layer Norm -> MLP. Inside the MLP, data passes through several layers, Zooming into the MLP either Feed Forward or MoE, we see data flowing through sequence of layers.  During the forward pass, the output of layer $L_{n-1}$ serves as the input for layer $L_n$; these intermediate outputs are known as Activations.

each producing an intermediate output before passing it forward. These intermediate outputs are Activations. During the forward pass, the output of layer n becomes the input to layer n+1 — that output is an activation. This happens at every layer, across every block, for every sample in the batch.

These activations are not just temporary values they are critical for the The backward pass to compute gradients during backpropagation, at every layer we need activations that were produced during the forward pass (see <a href="#table-backward-pass">Table 2</a>). This means all activations from the entire forward pass must be held in memory until backpropagation is complete. Unlike Parameters, Gradients, and Optimizer States whose memory cost is fixed by model architecture, Activations scale with both Batch Size and Sequence Length — which is what makes them the most unpredictable component of all that lives on GPU.

| Component | Classic Transformer (MHA) | Llama 3 (GQA / SwiGLU) |
| :--- | :--- | :--- |
| **Position Coding** | Sinusoidal (Fixed) | **RoPE** (Inside Attention) |
| **Normalization** | LayerNorm (Scale + Shift) | **RMSNorm** (Scale Only) |
| **Attention** | $H \times H$ (Q, K, V) | **$H \times H$ (Q), $g \times d_k$ (K, V)** |
| **MLP / FFN** | $2 \times (H \times 4H)$ | **$3 \times (H \times f)$** |
{: .table .table-sm .table-bordered .table-hover .text-center}


### Per-Block Activation Memory Breakdown


| Step | Operation | Shape | Llama 3 8B Dim | Memory (Bytes) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **RMSNorm (Attn)** | $(B, S, H)$ | $(B, S, 4096)$ | $2 \cdot BSH$ |
| **2.a** | **Q Projection** | $(B, S, H)$ | $(B, S, 4096)$ | $2 \cdot BSH$ |
| **2.b** | **K Projection** | $(B, S, g \cdot d_k)$ | $(B, S, 1024)$ | $0.5 \cdot BSH$ |
| **2.c** | **V Projection** | $(B, S, g \cdot d_k)$ | $(B, S, 1024)$ | $0.5 \cdot BSH$ |
| **2.d** | **Score ($QK^T$)** | $(B, N, S, S)$ | $(B, 32, S, S)$ | $2 \cdot BnS^2$ |
| **2.e** | **Softmax** | $(B, N, S, S)$ | $(B, 32, S, S)$ | $2 \cdot BnS^2$ |
| **2.f** | **Attn Output** | $(B, S, H)$ | $(B, S, 4096)$ | $2 \cdot BSH$ |
| **2.g** | **O Projection** | $(B, S, H)$ | $(B, S, 4096)$ | $2 \cdot BSH$ |
| **3** | **RMSNorm (MLP)** | $(B, S, H)$ | $(B, S, 4096)$ | $2 \cdot BSH$ |
| **4.a** | **Gate Proj** | $(B, S, f)$ | $(B, S, 14336)$ | $2 \cdot BSf$ |
| **4.b** | **Up Proj** | $(B, S, f)$ | $(B, S, 14336)$ | $2 \cdot BSf$ |
| **4.c** | **Down Proj** | $(B, S, H)$ | $(B, S, 4096)$ | $2 \cdot BSH$ |
{: .table .table-sm .table-bordered .table-hover .text-center}





### How model architecture amplifies the problem

Below <a href="#llama-3-config" >table </a> from Llama-3 [paper](https://arxiv.org/pdf/2407.21783) clearly demonstrates how model architecture influence size of the models & how these  configurations change across models of different sizes: layers go from 32 to 126, 
model dimension grows from 4,096 to 16,384, and FFN dimension jumps from 14,336 to 53,248. Every one of these numbers is a multiplier on your memory requirement. More layers means more weight matrices to store and more activations to hold in memory during the forward pass. Larger dimensions make each of those weight matrices bigger. Every configuration here directly or indirectly influences the memory required for training — and together they compound fast

 
<figure id="llama-3-config">
  {% include figure.liquid path="assets/img/llm-training/section-3/Llama-3-config.png" class="img-medium" caption="Figure-4: Llama 3 Configurations" %}
</figure>


### How Much Memory Are We Actually Talking About?

Before we calculate total training memory, it helps to understand what we're working with. Training uses mixed precision — FP32 and BF16 — and once training is complete, models are typically served in BF16. Models also support INT8 and INT4 quantization [ref](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct#use-with-bitsandbytes) for 
inference, which trades some accuracy for significant memory savings.

The H100 GPU has 80GB of HBM memory. Let's <a href="#table-params-memory">see</a> how the Llama-3 model variants stack up against that — for Parameters alone, across different precision types.


<figure>
<table id="table-params-memory" style="width:100%; border-collapse:collapse; margin:24px 0; font-size:15px; border:2px dashed #555;">
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
      <td style="padding:10px 12px; border:2px dashed #555;">$8B * 1 Byte = 8 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$8B * 4 Bits  = 4 GB$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">70B</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$70B * 4 Bytes = 280 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$70B * 2 Bytes = 140 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$70B * 1 Byte = 70 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$70B * 4 Bits  = 35 GB$</td>
    </tr>
    <tr>
      <td style="padding:10px 12px; border:2px dashed #555;">405B</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$405B * 4 Bytes = 1620 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$405B * 2 Bytes = 810 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$405B * 1 Byte = 405 GB$</td>
      <td style="padding:10px 12px; border:2px dashed #555;">$405B * 4 Bits  = 202.5 GB$</td>
    </tr>
  </tbody>
</table>
<figcaption style="text-align:center; font-size:14px; color:#666; margin-top:8px;">Table 7: Memory Required for Parameters across Precision Types</figcaption>
</figure>

The 70B model needs 140GB just for Parameters — nearly double the H100's entire 80GB. The 
405B model needs 810GB in BF16 and 1620GB in FP32,this is only for Parameters(Weights, Biases).

Training model needs multiple forward & backward passes which requires memory for **Gradients**, **Activations**, **Optimizer States** and additional memory buffers for staging intermediate calculations. The <a href="#table-params-memory">table</a> only captures memory required for Parameters For inference, this is the primary cost — but training needs significantly more on top of this. Of all the model sizes here, Only an 8B model can fit on GPU. This is because of distinction between inference & training. inference only needs Parameters, Activations, and KV Cache to complete a forward pass. Training needs all of that plus Gradients and Optimizer States — which is why training memory requirements are significantly heavier than inference. 

### The Data Side of the Problem
Beyond model architecture and Parameters required, memory requirements are directly influenced by data used for training. Deep Learning models are hungry — the more capable you want the model to be, the more data it needs to see during training, Llama models made this trend clear , Llama-2 was trained on 1.8 Trillion Tokens, subsequent models Llama-3 was trained on 15 Trillion Tokens and  Llama-4 on 30 Trillion Tokens ,which translates to increase in batch size, number of batches, longer training runs & more compute.

We looked at the training budget and time it would have taken to train a model using single GPU and memory required for storing the data and model components on single GPU , together they strengthens requirement for distributing the model and data across multiple GPU's. Parallelism techniques are all about resolving the constraints by distributing the model, data, and computation across multiple GPUs working together.