# What is this?
This project contains scripts to reproduce my paper 
[On the Convergence and Calibration of Deep Learning with Differential Privacy](https://arxiv.org/abs/2106.07830)
by Zhiqi Bu, Hua Wang, Zongyu Dai and Qi Long. We only add **one line of code** into the [Pytorch Opacus](https://github.com/pytorch/opacus)  library v0.15.0.

# The Problem of Interest
Deep learning models are vulnerable to privacy attacks and raise severe privacy concerns. To protect the privacy, Abadi et. al. applied [deep learning with differential privacy](https://arxiv.org/abs/1607.00133) (DP) and trained DP neural networks. Notably, if you train a neural network with SGD, you get regular non-DP network; if you train with differentially private SGD (DP-SGD), you get DP network.

Any regular optimizers (SGD, HeavyBall, Adam, etc.) can be turned into DP optimizers, with per-sample clipping and noise addition, via the Gaussian Mechanism. However, the convergence of DP optimizers is usually much slower in terms of iterations and results in low accuracy (e.g. in [recent Google paper](https://arxiv.org/abs/2007.14191), state-of-the-art CIFAR10 accuracy without pretraining is 66\% when privacy risk $\epsilon=8$).

We give the first **general convergence analysis** on the training dynamics of DP optimizers in deep learning, taking a close look at neural tangent kernel (**NTK**) matrix **H(t)**.
<p align="center"><img src="https://github.com/woodyx218/opacus_global_clipping/blob/master/website/static/dp_not_GD.png" alt="Opacus" width="800"/></p>

We show that existing per-sample clipping, with small clipping norm, breaks the positive semi-definiteness of NTK and leads to undesirable convergence behavior. We thus propose to use larger clipping norm to preserve the positive semi-definiteness and significantly improve the convergence as well as the calibration. This is based on the following insight:

$$\text{clipping/normalization} \Longleftrightarrow R/\|\frac{\partial \ell_i}{\partial w}\|\overset{\text{small} R}{\longleftarrow}C_i=\min\{1,R/\|\frac{\partial \ell_i}{\partial w}\|\}\overset{\text{large} R}{\longrightarrow}C_i=1\Longleftrightarrow\text{no clipping}.$$


For experiments on CIFAR10 (image) and SNLI (text):
<p align="center"><img src="https://github.com/woodyx218/opacus_global_clipping/blob/master/website/static/cifar10.png" alt="Opacus" width="800"/></p>
<p align="center"><img src="https://github.com/woodyx218/opacus_global_clipping/blob/master/website/static/cifar10_calibration.png" alt="Opacus" width="800"/></p>

The SNLI is trained on BERT (108 million parameters) in [Opacus BERT tutorial](https://github.com/pytorch/opacus/blob/master/tutorials/building_text_classifier.ipynb).
<p align="center"><img src="https://github.com/woodyx218/opacus_global_clipping/blob/master/website/static/bert.png" alt="Opacus" width="800"/></p>
<p align="center"><img src="https://github.com/woodyx218/opacus_global_clipping/blob/master/website/static/bert_calibration.png" alt="Opacus" width="800"/></p>

# New clipping function
and additionally a new clipping function -- the **global** per-sample clipping --
<p align="center"><img src="https://github.com/woodyx218/opacus_global_clipping/blob/master/website/static/clippings.png" alt="Opacus" width="800"/></p>
<p align="center"><img src="https://github.com/woodyx218/opacus_global_clipping/blob/master/website/static/clippings_summary.png" alt="Opacus" width="800"/></p>

# Codes
We add
```python
clip_factor=(clip_factor>=1)
```
between line 178 and line 179 in (https://github.com/pytorch/opacus/blob/v0.15.0/opacus/per_sample_gradient_clip.py), to implement our global per-sample clipping. 

Alternatively, one can directly use this repository, which imports the **config** library and introduces one new variable: config.clipping_fn={'local','global'} to indicate whether using global clipping; note that setting config.clipping_fn='local' (by default) is exactly using the original Opacus with local clipping.

To be specific, the only difference between this repo and Opacus is in the per_sample_gradient_clip.py.

## Installation
```bash
git clone https://github.com/woodyx218/opacus_global_clipping.git
cd opacus_global_clipping
pip install -e .
```

When using the code, the user still refer to the Opacus, e.g.
```python
import opacus
```

## Citation
```
@article{bu2021convergence,
  title={On the Convergence and Calibration of Deep Learning with Differential Privacy},
  author={Bu, Zhiqi and Wang, Hua and Long, Qi},
  journal={arXiv preprint arXiv:2106.07830},
  year={2021}
}
```

# Introducing Opacus
The below contents are forked from [Opacus github](https://github.com/pytorch/opacus). We do not claim ownership of the codes in this open-sourced repository and we sincerely thank the Opacus community for maintaining this amazing library.

[Opacus](https://opacus.ai) is a library that enables training PyTorch models with differential privacy. It supports training with minimal code changes required on the client, has little impact on training performance and allows the client to online track the privacy budget expended at any given moment.

## Target audience
This code release is aimed at two target audiences:
1. ML practitioners will find this to be a gentle introduction to training a model with differential privacy as it requires minimal code changes.
2. Differential Privacy scientists will find this easy to experiment and tinker with, allowing them to focus on what matters.



## Getting started
To train your model with differential privacy, all you need to do is to declare the screening threshold Z, whether to use global clpping, a `PrivacyEngine` and attach it to your optimizer before running, eg:

```python
model = Net()
optimizer = SGD(model.parameters(), lr=0.05)

config.G=True # using global clipping; reduces to original Opacus if False
config.Z=100
privacy_engine = PrivacyEngine(
    model,
    sample_rate=0.01,
    alphas=[10, 100],
    noise_multiplier=1.3,
    max_grad_norm=1.0,
)
privacy_engine.attach(optimizer)
# Now it's business as usual
```

The [MNIST example](https://github.com/pytorch/opacus/tree/master/examples/mnist.py) shows an end-to-end run using opacus. The [examples](https://github.com/pytorch/opacus/tree/master/examples/) folder contains more such examples.


