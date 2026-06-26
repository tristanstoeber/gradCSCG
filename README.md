# Learning Cognitive Maps from Visual Experience: A Differentiable VQ-VAE and Gradient-Trained Cloned HMM Pipeline

**Arash Nikzad**<sup>1</sup>, **Sasan Sarbishegi**, **Ali Dasmeh**<sup>2</sup>, **Muhammad Asif**<sup>3</sup>, **Parsa Gharavi**<sup>1</sup>, **Erik Husom**<sup>4</sup>, **Sagar Sen**<sup>4</sup>, **Andrew B. Lehr**<sup>5,6,7</sup>, **Olivier Penacchio**<sup>8</sup>, **Ana Clemente**<sup>3</sup>, **Tristan M. Stöber**<sup>1,6,7,9</sup>  

<sup>1</sup> Goethe University Frankfurt, Frankfurt, Germany  
<sup>2</sup> Max Planck Institute for Human Development, Berlin, Germany  
<sup>3</sup> Department of Cognitive Neuropsychology, Max Planck Institute for Empirical Aesthetics, Frankfurt, Germany  
<sup>4</sup> SINTEF, Oslo, Norway  
<sup>5</sup> Department of Neuro- and Sensory Physiology, University Medical Center Göttingen, Göttingen, Germany  
<sup>6</sup> Circulant Labs, Bensheim, Germany  
<sup>7</sup> Institute of Computer Science and Campus Institute Data Science, University Göttingen, Germany  
<sup>8</sup> Computer Science Department, Universitat Autònoma de Barcelona, Bellaterra, Spain  
<sup>9</sup> Epilepsy Center Frankfurt Rhine-Main, Department of Neurology, Goethe University Frankfurt, Frankfurt, Germany


## Abstract

How can an agent build a structured map of its world from nothing but a stream of raw images and its own movements, and know it has returned to a place when no two visits ever look the same? In neuroscience, the Clone-Structured Cognitive Graph (CSCG) answers a version of this question for the hippocampus: it learns an interpretable map by splitting look-alike observations into context-specific *clones*. But CSCG presupposes a small, *given* alphabet of discrete symbols and is trained by expectation–maximization, so on its own it can neither look at pixels nor compose with the neural networks that can. We remove this barrier by reformulating CSCG as a single differentiable, gradient-trained module — **gradCSCG** — and coupling it to a learned VQ-VAE perceptual front-end. A *soft-emission* forward pass lets the map-learning objective flow back into perception, while a set of loss-balancing mechanisms keeps the two modules from collapsing during joint training. We show, first, that gradient training reproduces CSCG's hallmark results on the original symbolic grid-worlds — recovering room topology from heavily aliased observations — and second, that the map survives the replacement of symbols by raw, never-repeating MNIST images: across four aliased environments the pipeline recovers the true adjacency graph with **100% edge recall** and **0.95–1.00** state-to-place purity, and reading the map off the agent's own decoded trajectory yields edge precision **0.95–1.00** and **F1 0.98–1.00**, directly from pixels. gradCSCG is thus a proof of principle that a normative model of the hippocampus can become a composable building block of modern deep learning.

**Keywords:** cognitive maps, clone-structured cognitive graph (CSCG), differentiable sequence models, vector-quantized representation learning, hippocampus, NeuroAI, topology recovery.

---

## 1. Introduction
Well-structured internal representations allow biological and artificial agents to pick shortcuts on routes never taken and to provide guidance in situations where trial-and-error learning would be fatal. Thus, understanding and reengineering the emergence of such representations is a fundamental research frontier both in neuroscience [1–5] and artificial intelligence (AI) [6–8].

The **Clone-Structured Cognitive Graph** (CSCG) algorithm [9, 10], a normative hippocampus model, explains how well-structured representations may emerge from sensory sequence learning. CSCG compresses a series of observation–action pairs into a higher-order representation of its environment. While technically an overcomplete hidden Markov model (HMM) with a fixed emission matrix, it learns to disambiguate contexts from aliased observations. Obeying the Markov property — i.e. any subsequent state depends only on the current state — CSCG is forced to represent a novel context with a new clone among its hidden nodes. The graph emerging from this cloning operation provides a condensed map of the environment and is suitable for planning, consolidation and abstraction. However, while elegantly creating well-structured representations in static and relatively small environments, it is unclear how to extend this approach to richer, perceptually complex observations.
The bottleneck is that CSCG is trained by Expectation–Maximization over a fixed, discrete observation alphabet, which prevents seamless composition with gradient-trained neural modules.

We resolve this by reimplementing the cloned HMM as a single differentiable, gradient-trained computation [11] in TensorFlow — a model we call **gradCSCG** — which lets us co-train it with a vector-quantized variational autoencoder (VQ-VAE). We demonstrate that this approach preserves CSCG's expressivity while enabling it to handle sensory variability and complexity in environments composed of MNIST digits.

**Contributions**

1. An **end-to-end pipeline** that maps raw images to a topological cognitive map by coupling a VQ-VAE to **gradCSCG**, a gradient-trained action-conditioned cloned HMM (Section 3).
2. A **differentiable, soft-emission formulation of gradCSCG**: the forward algorithm is implemented as a differentiable log-space computation, so the sequence objective can be trained by backpropagation and gradients reach the perceptual front-end (Sections 3.3–3.5).
3. **Loss-balancing for stable joint training** — length normalization, weight annealing, a diversity penalty, and anti-collapse safeguards (Section 3.6).
4. A formal, reusable **topology-recovery evaluation suite** (Section 3.10).
5. An empirical study on four MNIST grid-world environments with strong aliasing (Sections 4–5).

---

## 2. Related Work

We build directly on the Clone-Structured Cognitive Graph (CSCG) which is the foundation of this work rather than a point of comparison. Here we relate our pipeline to other efforts to *learn* cognitive maps and, in particular, to couple them with neural networks.

**CSCG toolkit.** A differentiable cloned-HMM forward pass with a soft-observation interface and encoder-gradient flow has also been developed in concurrent, independent open-source work [21], whose image experiment couples a convolutional network with a gradient-based CSCG for supervised digit classification. Relative to this work, our specific contributions are the learned VQ-VAE discretizer trained jointly with the sequence model, the loss-balancing that keeps that joint training stable, and — crucially — the use of this gradient-based setup to *actually recover and evaluate an environment's map*: we deduce the physical adjacency graph from the learned transitions and score it against the ground-truth topology (Section 5). Their image experiment, by contrast, uses the convolutional–CSCG coupling for supervised digit classification only; it does not attempt map or topology recovery at all, which is the central question our pipeline is built to answer.

**Neural sequence models for cognitive maps.** A complementary line learns maps from the latent codes of a neural predictor rather than from a cloned HMM. Dedieu et al. [22] train a transformer with discrete bottlenecks on next-observation prediction and read interpretable cognitive maps off its bottleneck indices for planning in partially observed environments. As in our pipeline, a neural representation is discretized and a map is recovered from observation–action streams; the difference is *where the map lives*. In their approach the cognitive map is decoded from the transformer's bottleneck codes as a separate, post-hoc analysis and handed to an external solver for planning, so the network's own working representation stays dense and is not itself an interpretable, directly usable map. In ours the clone-graph sequence model is co-trained with the perceptual discretizer and its transition matrix *is* the map: an intrinsically interpretable, on-line structure that the model computes with and can be queried directly for planning and for understanding the environment — not a map reconstructed from the network after the fact.

**Dynamic and expanding maps.** Closer to the open problems we raise in Section 7, de Tinguy et al. [23] grow a cognitive map online — dynamically expanding it over predicted poses within an active-inference agent — and benchmark against CSCG on grid environments. Their focus on a map that *expands* as the agent explores is precisely the dynamic-allocation capability that our fixed clone budget lacks; their setting, however, is symbolic and active-inference-based, whereas we learn the discretization from pixels by gradient descent. Together these works mark out the space our contribution occupies: a *gradient-trained* CSCG that learns its own perceptual vocabulary and recovers environment topology from images without supervision.

---

## 3. Materials and Methods

### 3.1 Problem formulation

An agent produces an episode of length $T$: observations $x_{1:T}$ with $x_t\in\mathcal{X}\subset\mathbb{R}^{H\times W\times C}$ and actions $a_{1:T-1}$ with $a_t\in\mathcal{A}=\{1,\dots,A\}$, where $a_t$ is taken between times $t$ and $t{+}1$. Each observation is emitted at an underlying physical place $g_t\in\mathcal{G}$; the environment has a true undirected adjacency graph $\mathcal{M}=(\mathcal{G},\mathcal{E})$. The places $g_t$ and edges $\mathcal{E}$ are used *only for evaluation* and are never seen during training. The goal is to learn, from $(x_{1:T},a_{1:T-1})$ alone, a latent model whose transition structure recovers $\mathcal{M}$.

The pipeline (Figure 1) has two modules: a VQ-VAE that maps each image to a discrete token, and an action-conditioned cloned HMM over those tokens whose latent graph is the learned map.

```mermaid
flowchart LR
    img["Image x_t"] --> enc["Encoder E_φ"]
    enc -->|"z_t"| vq["Codebook {e_k}"]
    vq -->|"q̃_t"| dec["Decoder D_ψ"]
    dec --> rec(["ℒ_rec"])
    enc --> soft["Soft posterior ρ_t over K codes"]
    soft -->|"log ρ_t"| hmm["gradCSCG forward (soft)"]
    hmm --> nll(["ℒ_gradCSCG"])
    rec -. grad .-> dec
    nll -. grad .-> hmm
    hmm -. grad .-> soft
    soft -. grad .-> enc
```

***Figure 1: The gradCSCG pipeline.*** *Solid arrows: forward computation. Dashed arrows: gradient flow. The encoder feeds both a reconstruction branch (hard quantization $\tilde q_t$ + decoder) and a sequence branch (soft codebook posterior $\rho_t$ + differentiable cloned-HMM forward pass). Because the gradCSCG likelihood is differentiable in $\rho_t$, the topological objective $\mathcal{L}_{\mathrm{gradCSCG}}$ shapes the encoder. The codebook itself is updated by EMA, not by gradients.*

### 3.2 Perceptual front-end: VQ-VAE

**Encoder and quantization.**
A convolutional encoder $E_\phi:\mathcal{X}\to\mathbb{R}^{D}$ maps each image to a latent $z_t=E_\phi(x_t)$. A codebook $\{e_k\}_{k=1}^{K}$, $e_k\in\mathbb{R}^{D}$, where $K$ denotes the number of discrete latent codes and $D$ is the dimensionality of each codebook embedding, defines a discrete token by nearest-neighbour assignment,

$$k_t=\arg\min_{k\in\{1,\dots,K\}}\;\lVert z_t-e_k\rVert_2^2, \qquad q_t=e_{k_t},$$

where the squared distance is computed as $\lVert z-e_k\rVert_2^2=\lVert z\rVert_2^2-2\langle z,e_k\rangle+\lVert e_k\rVert_2^2$. Gradients cross the non-differentiable $\arg\min$ by the straight-through estimator [18],

$$\tilde q_t = z_t + \mathrm{sg}(q_t-z_t),$$

where $\mathrm{sg}(\cdot)$ is the stop-gradient operator, so the forward value is $q_t$ while $\partial\tilde q_t/\partial z_t=I$. A decoder $D_\psi$ reconstructs $\hat x_t=D_\psi(\tilde q_t)$.

**Losses.**
Over a minibatch $\mathcal{B}$,

$$\mathcal{L}_{\mathrm{rec}} = \tfrac{1}{|\mathcal{B}|}\sum_{t} \lVert x_t-\hat x_t\rVert_2^2,$$

$$\mathcal{L}_{\mathrm{commit}} = \tfrac{1}{|\mathcal{B}|}\sum_{t} \lVert \mathrm{sg}(q_t)-z_t\rVert_2^2,$$

the second term pulling encoder outputs toward their assigned code.

**EMA codebook updates.**
The codebook is *not* trained by gradient descent. With decay $\gamma\in(0,1)$, per batch we accumulate, for every code $k$, a cluster size $n_k$ and a vector sum $m_k$,

$$\begin{aligned}
n_k &\leftarrow \gamma\,n_k+(1-\gamma)\sum_{t}\mathbb{1}[k_t=k],\\
m_k &\leftarrow \gamma\,m_k+(1-\gamma)\sum_{t}\mathbb{1}[k_t=k]\,z_t,
\end{aligned}$$

and set $e_k\leftarrow m_k/\hat n_k$ with Laplace-smoothed size

$$\hat n_k=\frac{n_k+\epsilon}{\left(\sum_{k'}n_{k'}\right)+K\epsilon}\left(\sum_{k'}n_{k'}\right).$$

**Soft codebook posterior.**
For the differentiable coupling (Section 3.5) the encoder also emits a temperature-controlled posterior over the codebook,

$$\log\rho_t(k) = \log\mathrm{softmax}_{k}\left(-\lVert z_t-e_k\rVert_2^2/\tau\right),$$

which is differentiable in $z_t$. As $\tau\to0$, $\rho_t$ concentrates on $k_t$ and this recovers the hard assignment.

### 3.3 Sequence model: action-conditioned cloned HMM

**State space and clone structure.**
Each token $k$ is assigned $C_k\ge1$ *clones* — latent states that all emit token $k$ but participate in different transition contexts. With a trailing *sink* state $\bot$, the state space is $\mathcal{S}=\{1,\dots,N\}$ with $N=1+\sum_{k=1}^{K}C_k$. The sink state is a special terminal state that does not correspond to any visual token; it is used to absorb probability mass at the end of a sequence and to make sequence termination explicit in the HMM formulation. A fixed map $\omega:\mathcal{S}\setminus\{\bot\}\to\{1,\dots,K\}$ gives the token each state emits; in the uniform case $C_k\equiv C$ and $\omega(s)=\lceil s/C\rceil$. Emissions are *deterministic*:

$$B_{s,o}=\mathbb{1}[\omega(s)=o],\qquad \log B_{s,o}=\begin{cases}0,&\omega(s)=o \\\\ -\infty,&\text{otherwise,}\end{cases}$$

and the sink emits no real token. Clones are exactly the mechanism that disambiguates aliasing: one token observed at two places is explained by two clones with distinct transition rows.

**Parameters.**
The model has initial-state logits $\pi\in\mathbb{R}^{N}$ and action-conditioned transition logits $\Theta\in\mathbb{R}^{A\times N\times N}$, yielding

$$\bar\pi=\mathrm{softmax}(\pi),\qquad T_{a,i,j}=\mathrm{softmax}_{j}\left(\Theta_{a,i,\cdot}\right).$$

All learning resides in $(\pi,\Theta)$; emissions are fixed.

**Forward likelihood.**
Writing $\mathop{\mathrm{logsumexp}}_i u_i=\log\sum_i e^{u_i}$, the log-forward messages for an episode $(o_{1:T},a_{1:T-1})$ obey

$$\begin{aligned}
\log\alpha_1(j) ={}& \log\bar\pi_j+\log B_{j,o_1},\\
\log\alpha_{t+1}(j) ={}& \mathop{\mathrm{logsumexp}}_{i}\left[\log\alpha_t(i)+\log T_{a_t,i,j}\right] + \log B_{j,o_{t+1}},
\end{aligned}$$

and the episode log-likelihood is $\ell(o_{1:T}\mid a_{1:T-1})=\mathop{\mathrm{logsumexp}}_{j}\log\alpha_T(j)$. The training loss is the mean negative log-likelihood (NLL)

$$\mathcal{L}_{\mathrm{gradCSCG}}=-\tfrac{1}{|\mathcal{B}|}\sum_{(o,a)\in\mathcal{B}}\ell(o_{1:T}\mid a_{1:T-1}).$$

### 3.4 Gradient-based training of the cloned HMM

Unlike the classical EM training of cloned HMMs, we evaluate the forward recursion and NLL loss as a single differentiable, log-space computational graph (a masked time recursion over padded minibatches) and optimize $(\pi,\Theta)$ directly by stochastic gradient descent with Adam [19]. Log-space arithmetic with $\mathop{\mathrm{logsumexp}}$ keeps the recursion numerically stable over long episodes. This gradient formulation is what makes the sequence model composable with a neural front-end.

### 3.5 Differentiable soft-emission coupling

To let the topological objective shape perception, we replace the hard emission term $\log B_{j,o_{t}}$ in the forward pass by the soft log-posterior of the token that state $j$ emits:

$$\begin{aligned}
\log\alpha^{s}_1(j) ={}& \log\bar\pi_j+\log\rho_1\left(\omega(j)\right),\\
\log\alpha^{s}_{t+1}(j) ={}& \mathop{\mathrm{logsumexp}}_{i}\left[\log\alpha^{s}_t(i)+\log T_{a_t,i,j}\right] + \log\rho_{t+1}\left(\omega(j)\right),
\end{aligned}$$

with the sink assigned $\log\rho_t(\bot)=-\infty$. The resulting log-likelihood $\ell^{s}$ is differentiable in $\rho_{1:T}$ and hence, through the soft posterior, in the encoder parameters $\phi$. Two properties hold by construction:

1. **Consistency.** If $\rho_t$ is the one-hot distribution on the observed token $o_t$, then $\log\rho_t(\omega(j))=\log B_{j,o_t}$ and the soft recursion reduces exactly to the hard forward pass. As $\tau\to0$ the soft pipeline thus recovers the hard pipeline.
2. **End-to-end differentiability.** Gradients of $\mathcal{L}_{\mathrm{gradCSCG}}$ propagate $\ell^{s}\!\to\!\rho\!\to\!z\!\to\!\phi$, so the encoder is trained, in part, to produce tokens that make the action-conditioned sequence *explainable*.

### 3.6 Joint objective and loss balancing

We train the model in three phases. First, the VQ-VAE is trained independently to initialize the visual encoder and decoder and obtain stable discrete representations. Second, the complete model, comprising both the VQ-VAE and gradCSCG, is trained jointly in an end-to-end manner with stochastic gradient descent. Third, the VQ-VAE is frozen and gradCSCG is fine-tuned using hard emissions, allowing the temporal model to refine its transition structure while operating on discrete visual assignments.

**Combined objective.**
At joint step $t$,

$$\mathcal{L}_{\mathrm{joint}} = \mathcal{L}_{\mathrm{rec}} + \beta\,\mathcal{L}_{\mathrm{commit}} + \lambda_t\,\widetilde{\mathcal{L}}_{\mathrm{gradCSCG}} + \alpha_{\mathrm{div}}\,\mathcal{L}_{\mathrm{div}}.$$

**Length normalization.**
The raw NLL grows as $O(T)$, which on long episodes dwarfs the $O(1)$ reconstruction term and collapses the codebook. We therefore use the per-step NLL

$$\widetilde{\mathcal{L}}_{\mathrm{gradCSCG}}=-\tfrac{1}{|\mathcal{B}|}\sum \tfrac{1}{T}\,\ell^{s}(o_{1:T}\mid a_{1:T-1}).$$

**Weight annealing.**
The sequence-loss weight is ramped linearly so the codebook stabilizes under reconstruction before topological pressure turns on:

$$\lambda_t=\lambda\cdot\min\left(1,\;t/T_{\mathrm{anneal}}\right).$$

**Diversity penalty.**
Let $\bar\rho(k)=\frac{1}{|\mathcal{B}|}\sum_t\rho_t(k)$ be mean codebook usage and $H(\bar\rho)=-\sum_k\bar\rho(k)\log\bar\rho(k)$ its entropy. The penalty

$$\mathcal{L}_{\mathrm{div}}=\log K-H(\bar\rho)\;\ge\;0$$

vanishes only at uniform usage and counteracts codebook collapse.

**Anti-collapse safeguards and finalization.**
During Phase 2 we monitor codebook perplexity and keep the highest-perplexity checkpoint; optional $\lambda$-throttling, rollback, and dead-code revival provide further protection. A short *finalization* phase then freezes the encoder and refines $(\pi,\Theta)$ on hard tokens with the unnormalized loss, optionally with a transition-entropy regularizer $-\eta\sum_{a,i}H(T_{a,i,\cdot})$ to sharpen transition rows. The following algorithm summarizes one joint step.

> **Algorithm — One joint training step**
>
> **Require:** image chunk $x_{1:T}$, actions $a_{1:T-1}$, weights $\beta,\lambda_t,\alpha_{\mathrm{div}}$, temperature $\tau$
>
> 1. $z_t \gets E_\phi(x_t)$ &nbsp; *(encode)*
> 2. $q_t,\tilde q_t \gets \text{quantize}(z_t)$; update codebook by EMA
> 3. $\hat x_t \gets D_\psi(\tilde q_t)$; &nbsp; $`\mathcal{L}_{\mathrm{rec}},\mathcal{L}_{\mathrm{commit}} \gets`$ reconstruction and commitment losses (Section 3.2)
> 4. $\log\rho_t \gets \log\mathrm{softmax}(-\lVert z_t-e_\cdot\rVert^2/\tau)$
> 5. $\ell^{s}\gets$ soft forward pass (Section 3.5)
> 6. $`\widetilde{\mathcal{L}}_{\mathrm{gradCSCG}}\gets -\ell^{s}/T`$; &nbsp; $`\mathcal{L}_{\mathrm{div}}\gets \log K-H(\bar\rho)`$
> 7. $\mathcal{L}_{\mathrm{joint}}\gets$ combined objective (Section 3.6)
> 8. update $(\phi,\psi,\pi,\Theta)$ with Adam on $\nabla\mathcal{L}_{\mathrm{joint}}$

### 3.7 Decoding

The maximum-a-posteriori state path is obtained by the Viterbi recursion [15]:

$$\begin{aligned}
\delta_1(j) &= \log\bar\pi_j+\log B_{j,o_1},\\
\delta_{t+1}(j) &= \max_{i}\left[\delta_t(i)+\log T_{a_t,i,j}\right] + \log B_{j,o_{t+1}},
\end{aligned}$$

with backpointers $\psi_{t+1}(j)=\arg\max_i\left[\delta_t(i)+\log T_{a_t,i,j}\right]$ and traceback $s^\star_T=\arg\max_j\delta_T(j)$, $s^\star_t=\psi_{t+1}(s^\star_{t+1})$, giving the most likely hidden state at each time $t$. The decoded path $s^\star_{1:T}$ is the basis of all evaluation.

### 3.8 Token compaction and clone allocation

In Phase 1, codes never emitted by the trained encoder are pruned and the alphabet is relabelled to the active set. Clone counts $C_k$ may be uniform or set per observation *by hand* (a fixed, manually chosen per-observation clone budget); the deterministic emission structure of Section 3.3 is unchanged. This is a static hyperparameter chosen before training, **not** a learned, on-demand allocation of clones.

### 3.9 Evaluation metrics

All metrics are computed from the decoded path $s^\star_{1:T}$ of a held-out episode together with the ground-truth places $g_{1:T}$ and edges $\mathcal{E}$ (used only here, never in training). Let $\mathcal{V}$ be the set of visited states and $n(s,g)=\sharp\{t:s^\star_t=s,\;g_t=g\}$.

**State-to-place assignment.**
Each visited state is mapped to its majority place, $\chi(s)=\arg\max_{g}n(s,g)$.

**Clone purity.**

$$\mathrm{Purity}=\frac{1}{|\mathcal{V}|}\sum_{s\in\mathcal{V}}\frac{\max_g n(s,g)}{\sum_g n(s,g)},$$

with the visit-weighted variant $\sum_s\max_g n(s,g)\,/\,T$. High purity means clone states correspond cleanly to single physical places.

**Projected map and edge F1.**
Latent transitions are projected onto a place graph by

$$W(g,g')=\max_{a}\;\max_{\substack{i,j\in\mathcal{V}\\ \chi(i)=g,\;\chi(j)=g'}}T_{a,i,j},$$

and thresholded into a learned edge set $\widehat{\mathcal{E}}(\eta)=\{(g,g'):g\neq g',\,W(g,g')>\eta\}$. With $\mathrm{tp}=\lvert\widehat{\mathcal{E}}\cap\mathcal{E}\rvert$, precision, recall and F1 are defined in the usual way. We report F1 over a threshold sweep $\eta\in\{0.01,0.05,0.1,0.2,0.3\}$.

**Viterbi-path map and edge F1.**
The projected map above thresholds *every* learned transition and therefore admits weak false edges. As a sharper read-out we keep only the transitions the decoded path actually takes: from $s^\star_{1:T}$ we tally consecutive state transitions, project each to a place edge $(\chi(s^\star_t),\chi(s^\star_{t+1}))$ with $\chi(s^\star_t)\neq\chi(s^\star_{t+1})$, and retain an edge once it is traversed in more than a small fraction of the episode (we use $0.2\%$ of the $T$ steps). Precision, recall and F1 against $\mathcal{E}$ are then computed as above. This Viterbi-path read-out is the "Map F1" reported in Table 3 and the Viterbi-path columns of Table 4 (Section 5.3).

**Action-next-cell accuracy.**
For every represented place $g$ and action $a$, the predicted next place $\arg\max_{g'}W_a(g,g')$ is compared with the environment's true successor $\mathrm{next}(g,a)$.

**Token–place entropies.**
From the empirical token/place co-occurrence we report $H(\text{token}\mid\text{place})$ and $H(\text{place}\mid\text{token})$; the former is small when perception is consistent, the latter reflects the (irreducible) aliasing of the environment.

### 3.10 Implementation and hyperparameters

The encoder is composed of three convolutional layers with strides \(2,2,1\), followed by global average pooling and a dense projection to $\mathbb{R}^{D}$; the decoder uses the corresponding mirrored architecture. The gradCSCG forward pass, soft forward pass, and training steps are implemented as compiled TensorFlow graphs. Viterbi decoding is performed outside the compiled graph during inference. Transition logits are initialized with a bias toward the sink state so probability mass is well-defined before training. Table 1 lists all hyperparameters.

**Table 1.** Hyperparameters. Ranges span the four environments of Section 4.

| Component / parameter | Symbol | Value |
|---|---|---|
| ***VQ-VAE*** | | |
| Codebook size | $K$ | 4–10 |
| Embedding dimension | $D$ | 32 |
| Encoder base width | — | 32 |
| Commitment weight | $\beta$ | 0.25 |
| EMA decay | $\gamma$ | 0.99 |
| EMA smoothing | $\epsilon$ | $10^{-5}$ |
| ***gradCSCG (cloned HMM)*** | | |
| Action alphabet | $A$ | 4 |
| Clones per token | $C_k$ | 4–20 |
| Latent states | $N$ | 21–151 |
| ***Joint training*** | | |
| Softmax temperature | $\tau$ | 1.0 |
| gradCSCG-loss weight | $\lambda$ | 1.0 |
| Anneal horizon | $T_{\mathrm{anneal}}$ | iters/4 |
| Diversity weight | $\alpha_{\mathrm{div}}$ | 0.1 |
| gradCSCG learning-rate factor | — | 100 |
| Chunk length | $T$ | 256 |
| Joint minibatch | $\lvert\mathcal{B}\rvert$ | 4 |
| Joint iterations | — | 2000–5000 |
| ***Finalization*** | | |
| Iterations | — | 1000–5000 |
| Transition-entropy weight | $\eta$ | $10^{-3}$ |
| Minibatch | — | 8 |
| ***Optimization and data*** | | |
| Optimizer | — | Adam [19] |
| VQ-VAE / joint learning rate | — | $3\times10^{-4}$ |
| Finalization learning rate | — | $10^{-2}$ |
| Episodes × steps | — | $(4\text{–}10)\times10{,}000$ |

---

## 4. Experimental Setup

### 4.1 Design: recovering a map when perception must be learned

Our experiments take place in a family of controlled navigation environments. Each environment is a set of discrete locations — *places* — connected into a graph that is the hidden ground-truth map; in this work the graph is a 2-D grid, so each place is a cell linked to its immediate neighbours. An agent explores by a **random walk**: at every step it occupies one place, receives an *observation* produced by that place, and takes one of four movement actions (up, down, left, right) that moves it to an adjacent place, with walls and boundaries sticky (an invalid move leaves it where it is). Everything the learner ever sees is this stream of alternating observations and actions; the agent's true location and the graph's adjacency are never exposed, and are kept only for evaluation. The task is to reconstruct the map — which places border which — from the stream alone.

This is exactly the setting in which the original CSCG was validated, with one simplification on the perception side: there, each place emits a *single discrete symbol* from a small alphabet, so the observation vocabulary is given in advance [9]. We keep everything else — the known ground-truth topology, the strong perceptual aliasing, and the hidden position — but replace the symbol at each place with a *raw image*, so the vocabulary is no longer given and must be learned from pixels. This lets us ask, on the very maps George et al. used, whether a cognitive map can still be recovered once perception is itself part of the learning problem.

Formally, write $v_t\in\mathcal{V}$ for the place visited at time $t$, $a_t\in\{\text{up},\text{down},\text{left},\text{right}\}$ for the action taken, and $x_t$ for the observation received; an episode is the resulting stream
$$
(x_1,a_1),(x_2,a_2),\dots,(x_T,a_T).
$$
Each cell $v$ carries a digit label $d(v)\in\{0,\dots,9\}$, and we use two observation models:

- **Symbolic** (the original-CSCG control [9]): the observation is the digit token itself, $x_t = d(v_t)$.
- **Image** (our benchmark): the observation is a freshly drawn MNIST image of that digit, $x_t \sim \mathcal{D}_{d(v_t)}$, where $\mathcal{D}_{d}$ is the empirical MNIST distribution for digit $d$.

Under the image model, repeated visits to one cell never produce identical pixels (non-stationary appearance), while distinct cells that share a digit produce visually similar observations (aliasing). The model must therefore turn high-dimensional, variable images into *stable* discrete tokens before action context can resolve the remaining aliasing — a strictly harder problem than the symbolic one, and a more realistic test of whether temporal-structure learning can be coupled to learned perception.

### 4.2 Environments

The four environments (Table 2; Figure 2) are MNIST analogues of the canonical demonstrations of the original CSCG [9], each isolating a different facet of map-learning under aliasing:

- **`aliased`** — a $4\times4$ room with four digit classes arranged so that each recurs four times. With only four distinct observations across sixteen places, appearance alone is almost uninformative; this mirrors George et al.'s room with four unique observations (Fig. 2a,b of [9]).
- **`corridors`** — a $5\times5$ layout whose interior walls carve narrow corridors joined by repeated digits, a walled-maze variant that stresses recovery through bottlenecks.
- **`room`** — a $6\times6$ room in which a ring of distinct border digits surrounds a $4\times4$ interior of a *single* repeated digit, so sixteen interior cells look identical. This mirrors George et al.'s uniform-interior room (Fig. 2c,d of [9]); the large aliased core is the hardest case for a fixed clone budget, and we use it to illustrate that the per-observation clone budget can simply be set by hand (Section 5.3).
- **`two_rooms`** — a $13\times9$ map of two offset rooms that share a $3\times3$ patch, so one local appearance occurs at two globally distinct places (a *confounder*). This mirrors the two overlapping rooms George et al. use to probe transitive inference (Fig. 2e,f of [9]); it is the largest and most aliased benchmark.

**Table 2.** Benchmark environments. $K$ is the number of discrete observation tokens (codebook size); "Places" counts walkable cells; the last column names the corresponding experiment in the original CSCG paper [9].

| Environment | Grid | Places | Tokens $K$ | Original-CSCG analogue [9] |
|---|:---:|:---:|:---:|---|
| `aliased` | $4\times4$ | 16 | 4 | room, four unique observations (Fig. 2a,b) |
| `corridors` | $5\times5$ | 19 | 6 | walled maze |
| `room` | $6\times6$ | 36 | 10 | uniform-interior room (Fig. 2c,d) |
| `two_rooms` | $13\times9$ | 87 | 10 | two overlapping rooms (Fig. 2e,f) |

![Ground-truth layouts of the four benchmark environments.](figures/environments.png)

*Figure 2: The four benchmark environments, each shown as its grid of per-cell digit classes (walls in grey). Cells that share a digit are perceptually aliased; in the image benchmark every visit to a cell returns a different MNIST sample of its digit. Note the uniform interior of `room` and the shared corner that links the two halves of `two_rooms`.*

### 4.3 Protocol

For each environment we collect action-conditioned uniform random walks — 4–10 episodes of 10,000 steps — and never expose the agent's position or the adjacency graph to the model; both are retained only to score a held-out episode with the metrics of Section 3.9. Each run executes the three-stage pipeline of Section 3.6 (VQ-VAE warm-up → joint training → finalization). During warm-up an optional, benchmark-only digit-classification loss may be applied to the encoder; since object classes are unknown in a general environment it is disableable, and the perceptual front-end learns to discretize the observations without it.

---

## 5. Results

We evaluate gradCSCG on the four grid-worlds of Section 4. Training consumes only the observation–action stream; the agent's position and the true adjacency graph are withheld and used solely to score a held-out episode (Section 3.9). The argument runs in three steps: gradient training reproduces the cognitive maps of the original, EM-trained CSCG (Section 5.1); that recovery survives the replacement of the given symbol by a raw, never-repeating image (Section 5.2); and the gradient-trained formulation affords further properties of practical value (Section 5.3). Table 3 summarises the headline metrics; the subsections establish each claim in turn.

**Table 3.** Main results across the four environments (image observations; `room` uses a hand-set per-observation clone budget). State-to-place purity is visit-weighted; map recall and F1 are read from the Viterbi-path graph (Section 5.3, Table 4), with recall 1.00 throughout. Entries are means over repeated runs; the seed-to-seed s.d. is $\le0.01$ for every environment except `two_rooms` ($\le0.03$).

| Environment | Grid | $K$ | States $N$ | Perplexity | Clone purity | State→place purity | Action acc. | Map recall | Map F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `aliased` | $4\times4$ | 4 | 21 | 4.00 | 0.98 | 0.98 | 1.00 | 1.00 | **1.00** |
| `corridors` | $5\times5$ | 6 | 31 | 5.71 | 0.86 | 0.98 | **1.00** | 1.00 | 0.98 |
| `room` (hand-set clones) | $6\times6$ | 10 | 57 | 6.56 | 0.83 | 0.95 | 0.97 | 1.00 | 0.98 |
| `two_rooms` | $13\times9$ | 10 | 151 | 9.46 | 0.88 | 0.98 | 0.99 | 1.00 | **1.00** |

### 5.1 Gradient descent recovers the cognitive maps of the original CSCG

**The learned latent graph contains the true map.** Trained only on observation–action walks, gradCSCG concentrates its transition probability on the edges that mirror the environment's physical adjacency, and map-edge **recall is 1.00** in every environment. This is the central result of the original CSCG [9], obtained here by back-propagation rather than expectation–maximization. We first reproduce it in the *original symbolic setting* of [9], in which each cell emits its integer observation directly with no images: trained by gradient descent, gradCSCG recovers the same maps George et al. obtain by EM — a square room laid out as a 2-D grid, a rectangular room whose uniform interior is distinguished only by its walls, and three disjoint rooms stitched into a single coherent graph (Figure 3, cf. Fig. 2 of [9]). The recovered transition graph reproduces the room topology, and the node colours (one per observation) show that perceptually identical cells are split into separate, correctly-placed clones. This isolates the change of training rule from the change of observation model; the same recovery then carries over to raw images (Section 5.2).

![Integer-observation reproduction of the original CSCG with gradCSCG.](figures/section51_integer_cscg.png)

*Figure 3: gradCSCG reproduces the original CSCG on integer (symbolic) grid-worlds, trained by gradient descent rather than EM. **Top row:** the ground-truth map of each environment, every cell (or place) coloured by its observation. **Bottom row:** the transition graph recovered by gradCSCG, with nodes coloured by the observation they emit and positioned by the learned connectivity — colours are matched cell↔node, so the correspondence is direct. **Left:** a square room recovered as a 2-D grid from a heavily aliased symbol stream (cf. George et al. [9], Fig. 2a,b). **Middle:** a rectangular room whose uniform interior — the large block of a single repeated observation — is disambiguated only by the surrounding walls (cf. Fig. 2c,d). **Right:** three pentagonal rooms with identical local observations, stitched into one coherent graph by transitive inference (cf. Fig. 2e,f). In every case the recovered graph matches the ground-truth topology.*

**Clones absorb the aliasing into place-specific states.** Recovering topology from aliased observations requires splitting one observation into context-specific latent states — the clone mechanism. Decoding a held-out episode and assigning each visited clone to its majority place, clones are almost perfectly place-specific: visit-weighted state-to-place purity is 0.95–1.00, every place is represented by at least one clone, and a typical place is covered by only 1.5–1.7 clones (Table 3; the clone-in-layout panels of Figure 5). Gradient training therefore reproduces the context-split, place-cell-like representations that make the original CSCG interpretable.

**The recovered map answers "where does this action lead?".** A map is useful only if it can be queried for the outcome of an action. Reading the most likely successor place for each (place, action) pair off the learned transitions and comparing it with the true successor, action-outcome accuracy is 0.97–1.00 (Table 3) — the graph is consistent enough to support the one-step, inference-based planning of [9].

### 5.2 The maps survive raw, non-stationary, aliased images

**A learned front-end supplies CSCG-grade tokens from pixels.** Where the original CSCG is handed clean symbols, our model must manufacture them. The VQ-VAE does so almost deterministically: a place emits a single token on 1.0–1.1 of its visits on average, the conditional entropy $H(\text{token}\mid\text{place})$ is only 0.09–0.18 nats, and codebook perplexity tracks the number of digit classes — even though that place is never seen twice. Reconstructions stay faithful throughout joint training (Figure 4), confirming that the topological objective sharpens rather than collapses the codebook.

**Topology recovery holds end-to-end and at scale.** Driving gradCSCG with these *learned* tokens — not the ground-truth digits — the full pixel-to-map pipeline still recovers every adjacency (recall 1.00) up to the largest, most aliased environment, `two_rooms` ($13\times9$, 87 places, 151 latent states), at state-to-place purity 0.98 and map F1 1.00 (Figure 5; Table 3). The map that George et al. recover from symbols, we recover from images.

![VQ-VAE inputs and reconstructions with assigned token ids.](figures/reconstructions.png)

*Figure 4: Perception is stable under joint training (`two_rooms`). Top: input MNIST observations; bottom: VQ-VAE reconstructions, each annotated with the discrete token assigned to it. Every visit to a place yields a different image, yet the assigned token is consistent.*

![gradCSCG on MNIST observations across the four rooms.](figures/section52_mnist_rooms.png)

*Figure 5: gradCSCG on raw MNIST observations, one row per environment. **Left:** ground-truth layout (cells coloured by digit class). **Middle:** the decoded clone states drawn at their assigned physical cells — perceptually identical cells (same colour, repeated across the grid) are split into separate clones that sit at the correct places. **Right:** the recovered transition graph, nodes drawn as the MNIST digit they emit. The topology is recovered from images in every room, up to the $13\times9$ `two_rooms`.*

### 5.3 Additional properties of the gradient-trained model

**A minor convenience: per-observation clone budgets.** The clone count is a hyperparameter that can be set per observation rather than uniformly. Giving the heavily-aliased interior digit of `room` more clones (20) than its rarely-repeated border digits (4) recovers the same map with 57 latent states instead of 201, at no cost to recall. This is a *static, hand-set* allocation fixed before training — not a learned, on-demand assignment of clones — so we note it only as a practical knob, not a contribution.

**Reading the map off the Viterbi path recovers it almost exactly.** Projecting *every* above-threshold latent transition into the place graph floods it with weak false edges, so the naive precision is low (0.16–0.44 at $\eta=0.01$; Table 4, left). But those weak edges are decoding noise, not routes the agent ever takes. We therefore keep only the place edges that the Viterbi-decoded path actually traverses more than a small fraction of the episode — **0.2% of the $T$ steps** (20 transitions at $T=10{,}000$). This is a *minimum-traffic* cutoff that asks for genuine, repeated movement rather than a single mis-step, and it scales with episode length instead of being a fixed magic number. It removes essentially all false edges while leaving recall untouched: precision rises to **0.95–1.00** and map F1 to **0.98–1.00** across all four environments, at recall 1.00 (Table 4). The read-out is robust to the exact fraction: because true adjacencies are traversed hundreds of times per episode and spurious ones only a handful, any cutoff from $\sim$0.1% to $\sim$0.5% of $T$ (10–50 transitions here) leaves recall at 1.00 while precision saturates by $\sim$0.2%.

**Table 4.** Map-edge scores from the projected transition graph (every edge with weight $>0.01$) versus the Viterbi-path graph (edges traversed in more than 0.2% of the $T$ steps, i.e. $>20$ at $T=10{,}000$). Recall is 1.00 in both columns; the Viterbi-path read-out removes the weak false edges that depress the projected-graph precision.

| Environment | Projected $P$ | Projected $F_1$ | Viterbi-path $P$ | Viterbi-path $R$ | Viterbi-path $F_1$ |
|---|:---:|:---:|:---:|:---:|:---:|
| `aliased` | 0.44 | 0.62 | **1.00** | 1.00 | **1.00** |
| `corridors` | 0.24 | 0.39 | **0.95** | 1.00 | **0.98** |
| `room` (hand-set clones) | 0.27 | 0.42 | **0.95** | 1.00 | **0.98** |
| `two_rooms` | 0.16 | 0.27 | **1.00** | 1.00 | **1.00** |

**Joint training is stable.** Through the joint phase the reconstruction loss stays flat while the length-normalized gradCSCG term falls once its weight $\lambda_t$ ramps in; the finalization phase then sharply reduces the gradCSCG NLL with the encoder frozen (Figure 6) — the behaviour the loss-balancing terms of Section 3.6 are designed to produce.

![Training curves for two_rooms.](figures/loss_curves.png)

*Figure 6: Training dynamics (`two_rooms`). VQ-VAE warm-up, the joint phase (reconstruction, commitment, length-normalized gradCSCG, and diversity terms), and the pure-gradCSCG finalization phase with the encoder frozen.*

---

## 6. Discussion

Our central result is a **proof of principle**: the Clone-Structured Cognitive Graph (CSCG), a normative account of how the hippocampus builds cognitive maps [9, 10], can be reformulated to live inside a gradient-trained pipeline. Re-deriving its forward algorithm as a differentiable, log-space computation lets the model be optimized by backpropagation rather than expectation–maximization, and on the original symbolic grid-worlds the gradient-trained model reproduces the hallmark behaviour of George et al. — recovering room topology from heavily aliased observations and splitting identical-looking places into context-specific clones (Section 5.1). Crucially, the reformulation also makes CSCG **composable**: co-trained with a VQ-VAE perceptual front-end, it recovers cognitive maps of pixel-based environments — where every place looks different on each visit and many places look alike — that the symbolic model cannot even ingest (Sections 5.2–5.3). The contribution is therefore less a new map-learning result than a change of *substrate*: CSCG becomes a module that can be wired into, and trained jointly with, neural networks.

This change of substrate is worth making because the two research traditions that bear on map learning each pay a price the other avoids. Interpretable cognitive-map models — CSCG [9, 10], the Tolman–Eichenbaum Machine [14], the successor/predictive map [3] — recover sparse, relational structure that supports planning and generalization, but they presuppose a discrete, *given* observation alphabet and so cannot, by themselves, look at pixels. Neural world models — the Dreamer family of recurrent state-space models [6] and the vector-quantized representation learners they build on [12] — consume raw perception readily, but their latent dynamics are dense and hard to read as a map. gradCSCG sits in the gap between them: it keeps CSCG's sparse, aliasing-resolving clone structure while delegating perception to a learned front-end. The enabling fact is not itself new — the HMM forward recursion is a differentiable computation [16], and neural emissions for HMMs have been studied in language modelling [17] — but applying it to the *action-augmented* cloned HMM, and letting the topological objective propagate back to shape perception, is what turns an ordinary encoder into a *sequence-aware* one.

That coupling is not free, and the way it fails is instructive. Trained naively, the joint objective collapses the codebook, because the sequence likelihood grows with episode length and swamps reconstruction; the loss-balancing of Section 3.6 — length normalization, weight annealing, and a diversity penalty — is what makes the two modules cooperate rather than compete, and in our experiments it is the difference between learning a map and learning a single token. We regard this as the main practical lesson for any future CSCG-plus-network hybrid: the perceptual and sequence objectives must be explicitly balanced, or the stronger one consumes the other.

---

## 7. Limitations and Outlook

**Limitations.** Two caveats bound the present results directly. *(i) Read-out threshold.* High precision requires reading the map off the Viterbi-decoded path with a minimum-traffic cutoff (Section 5.3); the cutoff is adaptive (a fixed fraction of episode length) and recall is insensitive to it, but a fully parameter-free read-out — e.g. stronger transition-entropy regularization or edge calibration — is still open. *(ii) Scope.* The environments are controlled, static, 2-D MNIST grid-worlds; we have not yet tested 3-D, partially observed, non-stationary, or visually richer settings.

**Outlook.** The deeper limitation is conceptual, and it points to where this work goes next. By construction, CSCG captures the *static, relational* skeleton of an environment — which places exist and how they connect — but **true behavioural flexibility needs more than a static map**. A realistic agent must also track the *dynamic* elements of a scene (objects that move, appear, or change), handle far more complex and higher-dimensional observations, and adapt *quickly* as the world changes — none of which a clone graph over fixed places is designed to do. We therefore see gradCSCG not as a complete agent but as the *structural* component of a larger system, with genuine flexibility coming from pairing it with complementary modules trained in the same differentiable framework. The most natural partner is a **recurrent network (RNN)**: where the clone graph holds the slow, stable map, an RNN can flexibly carry the fast-changing, dynamic context the map omits [6] — a division of labour that mirrors how real neural circuits combine stable spatial codes with rapidly updating population activity. Richer input — 3-D or egocentric vision — would in turn call for a heavier perceptual front-end that pre-processes raw views before they are discretized. Making CSCG gradient-trained is precisely what makes such hybrids buildable, since every part can then be optimized by the same backpropagation.

Many questions remain open. The most immediate is **dynamic clone recruitment**: here the clone budget is a static hyperparameter fixed before training (Section 5.3), whereas the cloning principle ultimately calls for *recruiting a new clone whenever a novel context appears*. A differentiable mechanism that grows or prunes clones on demand would let the model size itself to an environment's true complexity — and learn it faster and more incrementally — a natural next step now that training is gradient-based. Threshold-free map read-out, scaling to larger and partially observed worlds, and coupling the learned map to a planning or reinforcement-learning loop [6] are further directions.

---

## 8. Conclusion

We have shown that the Clone-Structured Cognitive Graph can be trained by gradient descent while faithfully reproducing the behaviour of its EM-trained original, and that, so trained, it can be grounded in a learned VQ-VAE front-end to recover cognitive maps of pixel-based environments. We close on what this is *for*. The hippocampus — of which CSCG is a computational model [9, 10] — does not work in isolation; it operates in close interaction with the entorhinal cortex, neocortex, and many other regions [14], each carrying out a different kind of computation, and the cognitive map is useful precisely because it is embedded in that larger system. We read our contribution in the same spirit: a gradient-trained CSCG is not a monolithic solution but a *composable* module, and rephrasing it in the language of deep learning is what lets it be wired together with the complementary machinery it needs — neural perception, recurrent tracking of dynamics, and planning. Because the formulation is differentiable, the benefit runs in both directions: the same module that gains learned perception for artificial agents also becomes a more scalable normative model for studying how the brain builds and uses such maps. Unlocking the power of CSCG, in brains and in machines alike, is therefore less about the module on its own than about a clever division of labour among specialized parts.

**Code and data availability.** The implementation, benchmark environments, training scripts and evaluation suite are available in the project repository.

---

## References

1. E. C. Tolman. Cognitive maps in rats and men. *Psychological Review*, 55(4):189-208, 1948.
2. J. O'Keefe and L. Nadel. *The Hippocampus as a Cognitive Map*. Clarendon Press, 1978.
3. K. L. Stachenfeld, M. M. Botvinick, and S. J. Gershman. The hippocampus as a predictive map. *Nature Neuroscience*, 20:1643-1653, 2017.
4. J. C. R. Whittington, D. McCaffary, J. J. W. Bakermans, and T. E. J. Behrens. How to build a cognitive map: insights from models of the hippocampal formation. *Nature Neuroscience*, 25:1257-1272, 2022.
5. D. Ha and J. Schmidhuber. World models. *arXiv:1803.10122*, 2018.
6. D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap. Mastering diverse control tasks through world models. *Nature*, 640(8059):647-653, 2025.
7. J. Pasukonis, T. Lillicrap, and D. Hafner. Evaluating long-term memory in 3D mazes. *arXiv:2210.13383*, 2022.
8. M. R. Samsami, A. Zholus, J. Rajendran, and S. Chandar. Mastering memory tasks with world models. In *International Conference on Learning Representations (ICLR)*, 2024.
9. D. George, R. V. Rikhye, N. Gothoskar, J. S. Guntupalli, A. Dedieu, and M. Lázaro-Gredilla. Clone-structured graph representations enable flexible learning and vicarious evaluation of cognitive maps. *Nature Communications*, 12:2392, 2021.
10. R. V. Raju, J. S. Guntupalli, G. Zhou, M. Lázaro-Gredilla, and D. George. Space is a latent sequence: a theory of the hippocampus. *Science Advances*, 10:eadm8470, 2024.
11. W. Sun, J. Winnubst, M. Natrajan, et al. Learning produces an orthogonalized state machine in the hippocampus. *Nature*, 640:165-175, 2025.
12. A. van den Oord, O. Vinyals, and K. Kavukcuoglu. Neural discrete representation learning. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.
13. A. Razavi, A. van den Oord, and O. Vinyals. Generating diverse high-fidelity images with VQ-VAE-2. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.
14. J. C. R. Whittington, T. H. Muller, S. Mark, G. Chen, C. Barry, N. Burgess, and T. E. J. Behrens. The Tolman-Eichenbaum Machine: unifying space and relational memory through generalization in the hippocampal formation. *Cell*, 183(5):1249-1263, 2020.
15. L. R. Rabiner. A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2):257-286, 1989.
16. J. Eisner. Inside-outside and forward-backward algorithms are just backprop. In *Proceedings of the Workshop on Structured Prediction for NLP*, pages 1-17, 2016.
17. K. M. Tran, Y. Bisk, A. Vaswani, D. Marcu, and K. Knight. Unsupervised neural hidden Markov models. In *Proceedings of the Workshop on Structured Prediction for NLP*, pages 63-71, 2016.
18. Y. Bengio, N. Léonard, and A. Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. *arXiv:1308.3432*, 2013.
19. D. P. Kingma and J. Ba. Adam: a method for stochastic optimization. In *International Conference on Learning Representations (ICLR)*, 2015.
20. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278-2324, 1998.
21. R. Young. cscg_toolkit: JAX/PyTorch and Julia implementations of Clone-Structured Cognitive Graphs. GitHub repository, https://github.com/SynapticSage/cscg_toolkit, 2025.
22. A. Dedieu, W. Lehrach, G. Zhou, D. George, and M. Lázaro-Gredilla. Learning cognitive maps from transformer representations for efficient planning in partially observed environments. *arXiv:2401.05946*, 2024.
23. D. de Tinguy, T. Verbelen, and B. Dhoedt. Learning dynamic cognitive map with autonomous navigation. *arXiv:2411.08447*, 2024.
