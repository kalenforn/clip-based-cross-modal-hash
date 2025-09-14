# DCMHT
[paper](https://ieeexplore.ieee.org/document/10974482)

## Motivation
Traditional cross-modal hashing methods have primarily focused on one-to-one feature alignment, often overlooking the inconsistent multi-instance information inherent in multi-modal datasets. Since textual descriptions tend to emphasize the most salient instances within an image, they frequently convey less comprehensive multi-instance information compared to the visual content. To address this limitation, we introduce a multi-instance learning (MIL) framework to capture latent multi-instance features, enabling many-to-many feature alignment in cross-modal hashing. This approach allows the model to learn and align multiple features across modalities, facilitating latent instance-level matching and thereby mitigating the inherent inconsistencies in cross-modal datasets.

## Training
>cd ../../
>
> python main.py --config-file configs/DIMCH/config.yaml--save-dir result/DIMCH/coco/16
