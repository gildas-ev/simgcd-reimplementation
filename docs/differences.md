# Differences with the official implementation

Compared against the [official SimGCD repository](https://github.com/CVMI-Lab/SimGCD).
For how each component works, see the [code walkthrough](walkthrough.md).

## Findings about the official code

#### ColorJitter is not used

`ColorJitter()` is called with no arguments. All torchvision defaults are 0, so the call is the
identity: no color augmentation is actually applied on CIFAR-100. I kept it as-is to match the
reported results; the effect of real color jitter was not tested.

#### Temperature mismatch between paper and code

The paper states τ_u = 0.07 and τ_c = 1.0, while the code swaps the two values.
I follow the code.

#### Entropy defined up to a constant

My entropy term in `losses/classification` differs from the official code by a constant log K,
which doesn't affect gradient descent. I followed the definition of entropy of my physics classes.
As a result, my total loss is shifted by a constant compared to the official one
(see the loss curve in the [README](../README.md#results)).

## Protocol

#### The seed also draws the split

In the official code, `get_cifar_100_datasets` draws the labelled/unlabelled split with a fixed
seed (`seed=0`), and `get_datasets` never overrides it: all official runs share the same split,
and only the training randomness changes. In my code, `config.seed` drives both the split and
training. The 5 seeds in the README therefore use 5 different splits, and the reported std
includes the variance of the split on top of the variance of training. This may explain part of the gap on New classes std
(3.4 against 2.5). Running the 5 seeds on the official split was not tested.

## Design choices

- **Prototypes.** Instead of reparametrizing the prototype weight matrix with the deprecated
  weight normalization, I store the prototypes unnormalized in an `nn.Parameter` of shape
  (100, 768) and normalize them in the forward. This gives the same cosine similarities and is
  easier to read. See [walkthrough §4](walkthrough.md#4-head) and, for their norm,
  [§8.2](walkthrough.md#82-optimizer).
- **Representation loss.** Rewritten as a single masked contrastive loss that mirrors the math,
  instead of two functions built on the generic SupCon code.
  See [walkthrough §6.2](walkthrough.md#62-representation-loss).
- **Datasets.** Two separate CIFAR100 instances wrapped in subsets, instead of the `deepcopy` the
  official code relies on. This avoids sharing a `transform` attribute.
  See [walkthrough §1](walkthrough.md#1-cifar-100-dataset-split).
- **Metrics.** Only one accuracy metric is reimplemented, `split_cluster_acc_v2` (renamed
  `acc_metrics`), since it is the one used for the paper's results.
- **Structure and tests.** A modular file structure instead of the flat layout, a test suite under
  `tests/` (the official repo has none), and asserts at construction time.

## Minor changes

- `torch.roll` instead of a nested loop to compute the unsupervised classification loss.
- The student temperature is read from the config instead of being hardcoded.
- `.to(device)` instead of `.cuda()`, to allow debugging on CPU.
- `__getitem__` doesn't return the original dataset index.
- An `eval_freq` config parameter, which saves two to three hours of training per run.
