# DCMHT
[paper](https://dl.acm.org/doi/abs/10.1145/3744567)

## Motivation
This work introduces a novel generative hashing method inspired by the text generation process in language models. It first adopts a Mixture-of-Experts (MoE) framework to streamline the conventional Two-Tower cross-modal hashing architecture, implementing hash learning with a unified Transformer Decoder structure within different modality features. Furthermore, the method emulates the encoder-decoder language generation mechanism to produce discrete hash codes in a sequential manner, thereby enhancing the model’s understanding of the discrete coding space during end-to-end training. To the best of our knowledge, this is the first effort that adapts language model generation strategies to formulate an end-to-end generative hashing framework.

## Training
>cd ../../
>
> python main.py --config-file configs/UMoED/config.yaml--save-dir result/UMoED/coco/16
