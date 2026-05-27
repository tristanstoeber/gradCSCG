# gradCSCG

**Learning cognitive maps from visual experience.** An end-to-end pipeline that
couples a VQ-VAE perceptual front-end with a gradient-trained Clone-Structured
Cognitive Graph (CSCG / cloned HMM) to recover the topological map of an
environment directly from images and actions.

## Start here

- **[`gradCSCG_experiments.ipynb`](gradCSCG_experiments.ipynb)** — runnable
  notebook: the method, a quick end-to-end demo, the full results across five
  environments, a live deep-dive on a shipped trained model, and self-contained
  reproduction recipes for each published run.

## Reproducing the published experiments

The five runs reported in the paper (`aliased`, `corridors`, `room_uniform`,
`room_dynamic`, `two_rooms`) can each be reproduced from a single code cell in
Section 7 of the notebook. Each cell uses the exact hyperparameters recorded
in that run's `summary.txt` and outputs the corresponding clone-purity and
Viterbi-path-graph figures. The training is full Phase 2 joint + pure-HMM
finalization on 10,000-step episodes and takes about an hour of CPU per run.

The committed [`results/`](results/) tree carries the trained-model artifacts
for `two_rooms`, plus the four per-run diagnostic figures
(`reconstructions.png`, `codebook_usage.png`, `clone_purity.png`,
`viterbi_graph.png`) and the `summary.txt` metrics for all five environments.

This repository is a research codebase for learning cognitive maps from visual
experience. It combines a VQ-VAE perceptual front-end with a gradient-trained
Cloned HMM / CSCG sequence model.

The short version:

```text
raw observations -> VQ-VAE "eye" -> discrete vocabulary -> GradientHMM/CSCG -> learned cognitive map
```

The current practical target is to make `VQVAEGradientHMM` work reliably on
MNIST grid-world benchmarks and recover the physical adjacency structure of
the environment. The eventual research target is a more general 3D
environment where objects/classes and ground-truth map labels are not known in
advance.

This README covers the research goal, the method, how to run the code, what the
plots and metrics mean, and the project roadmap.

## Research Goal

The long-term research agenda is to build a model that can see visual
observations, compress them into a reusable discrete vocabulary, and use
sequential structure to build a cognitive map of the environment.

The desired division of labor is:

- The VQ-VAE is the "eye" of the system.
- The VQ-VAE should encode visually or contextually equivalent observations
  into a shared vocabulary rather than fragmenting the same context into many
  unrelated tokens.
- The CSCG / GradientHMM should use action-conditioned temporal structure to
  disambiguate perceptual aliasing and recover the map.
- The learned graph should recover physical adjacency.
- The cognitive map should reflect the actual map structure.

The main current success criteria are:

- Low HMM negative log likelihood.
- High clone purity.
- High physical map edge F1.
- High action-next-cell accuracy.
- Reconstruction loss remains respectable, so the VQ-VAE does not collapse.
- The learned graph should recover exact physical adjacency on benchmark maps.

The current focus is not Phase 3 and not the RNN route. Those are useful
future directions, but the immediate priority is:

```text
VQ-VAE + GradientHMM/CSCG working robustly on many examples and scenarios.
```

## Repository Map

| Path | Purpose |
| --- | --- |
| `models/vqvae.py` | TensorFlow VQ-VAE and EMA vector quantizer. |
| `models/gradient_hmm.py` | Gradient-trained action-conditioned HMM / CSCG. |
| `models/vqvae_cscg.py` | Main wrapper: `VQVAEGradientHMM`. Training, joint training, save/load, decode, graph plotting. |

## Main Pipeline

### Stagewise Training

The default pipeline is:

1. Collect trajectories from MNIST grid world.
2. Train VQ-VAE on images.
3. Encode each image to one discrete codebook token.
4. Optionally compact unused tokens.
5. Train `GradientHMM` on token/action sequences.
6. Decode trajectories with Viterbi.
7. Compute clone purity and physical map recovery metrics.
8. Render visualizations.

Command:

```powershell
python -m examples.mnist_gridworld_demo --grid aliased
```

Quick smoke run:

```powershell
python -m examples.mnist_gridworld_demo --quick
```

### Joint Training

Joint mode uses soft codebook assignments as differentiable HMM emissions:

```text
loss = recon + commitment + lambda * HMM_NLL + diversity/regularization terms
```

Command:

```powershell
python -m examples.mnist_gridworld_demo --joint --grid aliased
```

Joint mode exists and is useful for experimentation, but the immediate
priority is still making the `VQVAEGradientHMM` route produce clean maps
robustly.

## Important Training Flags

Core flags:

| Flag | Meaning |
| --- | --- |
| `--grid` | Grid layout: `unique`, `aliased`, `corridors`, `example`, `empty`, `two_rooms`. |
| `--episodes` | Number of rollout episodes. |
| `--steps` | Steps per episode. |
| `--codebook-size` | VQ-VAE vocabulary size. |
| `--num-clones` | CSCG clones per token. |
| `--clone-counts` | Optional comma-separated nonuniform clone counts indexed by grid observation label in the MNIST demo. Entry 0 is for digit/label 0. Omit to keep the old uniform `--num-clones` behavior. |
| `--vqvae-epochs` | VQ-VAE warmup epochs. |
| `--hmm-iters` | HMM training iterations. |
| `--map-thresholds` | Comma-separated thresholds for map edge F1 sweep. |
| `--quick` | Tiny smoke settings. |
| `--no-save-model` | Do not save model and episode bundle. |

Dynamic clone allocation:

- By default, every observation receives the same number of clones from
  `--num-clones`, which preserves the original CSCG setup.
- To use fixed nonuniform clone counts, pass `--clone-counts`. In the MNIST
  demo, entry 0 is the clone count for grid observation label/digit 0, entry 1
  is for label/digit 1, and so on. These are not arbitrary VQ codebook ids.

```powershell
python -m examples.mnist_gridworld_demo --grid empty --clone-counts 20,4,4,4,4,4,4,4,4,4
```

After VQ-VAE warmup, the demo assigns each VQ token its majority digit label
and maps the requested clone count onto that token. In compacted stagewise
mode, those token counts are then remapped internally onto the active compacted
vocabulary. The HMM still uses deterministic CSCG emissions; only the number
of clones assigned to each observation changes.

Trajectory/data flags:

| Flag | Meaning |
| --- | --- |
| `--rollout-policy random` | Uniform random walk. |
| `--rollout-policy balanced_cell_action` | Bias collection toward under-sampled `(cell, action)` pairs. Useful for toy benchmarks. |
| `--rollout-policy coverage_cell_action` | Bias collection toward under-sampled `(cell, action)` pairs and globally under-visited cells. Useful when edge/corner cells converge slowly. |
| `--coverage-random-prob 0.05` | Random-action probability used only by `coverage_cell_action`, so coverage-driven rollouts keep some exploration noise. |

Benchmark-only VQ-VAE supervision:

| Flag | Meaning |
| --- | --- |
| `--vqvae-supervision-weight 0.1` | Auxiliary digit-class loss during VQ-VAE warmup/stagewise training. |
| `--no-vqvae-supervision` | Disable supervised VQ-VAE warmup. This is the long-term realistic setting. |

Joint/finalize flags:

| Flag | Meaning |
| --- | --- |
| `--joint` | Enable joint soft-emission training. |
| `--joint-lambda` | Weight on HMM NLL in joint training. |
| `--joint-temperature` | Soft codebook assignment temperature. |
| `--no-joint-normalize` | Disable per-step HMM NLL normalization. |
| `--joint-diversity` | Encourage broad codebook use. |
| `--freeze-vqvae-during-joint` | Freeze the VQ-VAE eye during joint phase. |
| `--finalize-hmm-iters` | Extra pure-HMM training after joint phase. |
| `--finalize-transition-entropy` | Positive penalty to sharpen transition rows. |

## Output Artifacts

The main demo writes plots and metadata under the run directory.

| Artifact | Meaning |
| --- | --- |
| `loss_curves.png` | VQ-VAE and HMM loss curves. |
| `reconstructions.png` | Input/reconstruction/token examples. |
| `codebook_usage.png` | Histogram of token usage. |
| `clone_purity.png` | Confusion between decoded clone states and ground-truth cells. |
| `viterbi_path_graph.png` | Count-thresholded graph of clone-state transitions actually taken by the decoded Viterbi path. |
| `model/` | Saved VQ-VAE + HMM model. |
| `episodes.npz` | Saved episode data for replay/visualization. |

## Metrics

### HMM NLL

Lower is better. Measures how well the HMM explains token/action sequences.
It is necessary but not sufficient: a low NLL model can still learn a poor map
if tokens collapse or transitions overfit.

### Clone Purity

Measures whether decoded clone states correspond cleanly to physical cells.
High clone purity means a clone state tends to be used for one place rather
than many places.

## Map Plotter

### `plot_viterbi_path_graph`

This plot uses only the decoded Viterbi state sequence. If the path contains a
transition from state `x` to state `y`, it adds one directed edge `x -> y`.
Repeated transitions are counted, and the edge is drawn only when its count is
greater than `--viterbi-count-threshold` (default `20`). Edge labels show the
counts. The rendering style matches the simple `transition_graph.png` view,
but the edge source is the actual decoded path rather than thresholded
transition probabilities.


## GPU Notes

The code can run on GPU if TensorFlow sees a GPU in the environment.

Practical expectations:

- VQ-VAE training benefits most from GPU.
- Batched HMM training may benefit, but less dramatically because sequence
  forward passes can be sequential and XLA/graph compilation overhead matters.
- Small toy runs may not be faster on GPU because overhead dominates.
- Larger image batches and longer VQ-VAE training are more likely to benefit.

Useful check:

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
