# mergenet

<h2 align="center"><a href="https://arxiv.org/abs/2404.13322">MergeNet: Knowledge Migration across Heterogeneous Models, Tasks, and Modalities </a></h2>

> *<sup>1</sup>Zhejiang University, <sup>2</sup>Shanghai Jiao Tong University*

<h5 align=center> 

[![arXiv](https://img.shields.io/badge/Arxiv-2404.13322-red?logo=arxiv&label=Arxiv&color=green)](https://arxiv.org/abs/2404.13322)
![GitHub Repo stars](https://img.shields.io/github/stars/Fantasylii/mergenet)
</h5>

## Abstract
In this study, we focus on heterogeneous knowledge transfer across entirely different model architectures, tasks, and modalities. Existing knowledge transfer methods (e.g., backbone sharing, knowledge distillation) often hinge on shared elements within model structures or task-specific features/labels, limiting transfers to complex model types or tasks. To overcome these challenges, we present MergeNet, which learns to bridge the gap of parameter spaces of heterogeneous models, facilitating the direct interaction, extraction, and application of knowledge within these parameter spaces. The core mechanism of MergeNet lies in the parameter adapter, which operates by querying the source model's low-rank parameters and adeptly learning to identify and map parameters into the target model. MergeNet is learned alongside both models, allowing our framework to dynamically transfer and adapt knowledge relevant to the current stage, including the training trajectory knowledge of the source model. Extensive experiments on heterogeneous knowledge transfer demonstrate significant improvements in challenging settings, where representative approaches may falter or prove less applicable.

## Citation

#### If you find our work valuable, we would appreciate your citation: 🎈


```bibtex
@article{li2024mergenet,
  title={MergeNet: Knowledge Migration across Heterogeneous Models, Tasks, and Modalities},
  author={Li, Kunxi and Zhan, Tianyu and Fu, Kairui and Zhang, Shengyu and Kuang, Kun and Li, Jiwei and Zhao, Zhou and Wu, Fei},
  journal={arXiv preprint arXiv:2404.13322},
  year={2024}
}
```


#### The code is still being organized.🚧
