# What is this?
This project contains scripts to reproduce my paper 
[On the Convergence of Deep Learning with Differential Privacy](https://arxiv.org/abs/2106.07830)
by Zhiqi Bu, Hua Wang, Qi Long, and Weijie J. Su. We only add **one line of code** into the [Pytorch Opacus library](https://github.com/pytorch/opacus).

# The Problem of Interest
Deep learning models are vulnerable to privacy attacks and raise severe privacy concerns. To protect the privacy, Abadi et. al. applied [deep learning with differential privacy](https://arxiv.org/abs/1607.00133) (DP) and obtain DP neural networks. Notably, if you train a neural network with SGD, you get regular non-DP network; if you train with differentially private SGD (DP-SGD), you get DP network.

Any regular optimizers (SGD, HeavyBall, Adam) can be turned into DP optimizers, with per-sample clipping and noise addition, via the Gaussian Mechanism. However, the convergence of DP optimizers is usually much slower and results in low accuracy (e.g. in [recent Google paper](https://arxiv.org/abs/2007.14191), state-of-the-art CIFAR10 accuracy without pretraining is 66\% when privacy risk $\epsilon=8$).

We give the first **general convergence analysis** on the training dynamics of DP optimizers in deep learning, taking a close look at neural tangent kernel (**NTK**) matrix **H(t)**.
<p align="center"><img src="https://github.com/woodyx218/opacus_global_clipping/blob/master/website/static/dp_not_GD.png" alt="Opacus" width="800"/></p>


We show that existing per-sample clipping (which we refer to as the **local** clipping) breaks the positive semi-definiteness of NTK, which leads to undesirable convergence behavior. We thus propose the **global** per-sample clipping to preserve the positive semi-definiteness and significantly improve the convergence as well as the calibration.
<p align="center"><img src="https://github.com/woodyx218/opacus_global_clipping/blob/master/website/static/clippings.png" alt="Opacus" width="800"/></p>
<p align="center"><img src="https://github.com/woodyx218/opacus_global_clipping/blob/master/website/static/clippings_summary.png" alt="Opacus" width="800"/></p>

For experiments on CIFAR10 (image) and SNLI (text):
<p align="center"><img src="https://github.com/woodyx218/opacus_global_clipping/blob/master/website/static/cifar10.png" alt="Opacus" width="800"/></p>
<p align="center"><img src="https://github.com/woodyx218/opacus_global_clipping/blob/master/website/static/cifar10_calibration.png" alt="Opacus" width="800"/></p>

The SNLI is trained on BERT (108 million parameters in [Opacus BERT tutorial](https://github.com/pytorch/opacus/blob/master/tutorials/building_text_classifier.ipynb).
<p align="center"><img src="https://github.com/woodyx218/opacus_global_clipping/blob/master/website/static/bert.png" alt="Opacus" width="800"/></p>
<p align="center"><img src="https://github.com/woodyx218/opacus_global_clipping/blob/master/website/static/bert_calibration.png" alt="Opacus" width="800"/></p>


# Codes
We add
```python
import config; clip_factor=torch.where(clip_factor > 1/config.Z, torch.ones_like(clip_factor)/config.Z, torch.zeros_like(clip_factor))
```
between line 197 and line 198 in (https://github.com/pytorch/opacus/blob/ee6867e6364781e67529664261243c16c3046b0b/opacus/per_sample_gradient_clip.py) as in Feburary 2021, to implement our global per-sample clipping. Here **config.Z** is a global variable that you can tune.

Alternatively, one can directly use this repository, which introduces two new variables: config.Z for the screening threshold Z and config.G={True,False} to indicate whether using global clipping; note that setting config.G=False is exactly using the original Opacus with local clipping, and that config.Z has no effect in this case.

To be specific, the only difference between this repo and Opacus is in the per_sample_gradient_clip.py:

In line 33, we insert
```python
import config
```

In line 197 (within for loop), we insert
```python
if config.G==True:
    Z=config.Z
    clip_factor=torch.where(clip_factor > 1/Z, torch.ones_like(clip_factor)*1/Z, torch.zeros_like(clip_factor))
```

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

<p align="center"><img src="https://github.com/pytorch/opacus/blob/master/website/static/img/opacus_logo.svg" alt="Opacus" width="500"/></p>

<hr/>

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


