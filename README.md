# CLIP-based Cross-Modal Hashing

[中文介绍](./README_chinese.md)

This is a library for cross-modal hashing method relied on CLIP model. We implement the follow methods:
- DCMHT, _MM22_, [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548187)
- MITH, _MM23_, [paper](https://dl.acm.org/doi/10.1145/3581783.3612411)
- DSPH, _TCSVT23_, [paper](https://ieeexplore.ieee.org/document/10149001)
- DNPH, _TOMM24_, [paper](https://dl.acm.org/doi/10.1145/3643639)
- TwDH, _TMM24_, [paper](https://ieeexplore.ieee.org/document/10487033) 

**News:** 
- Our TwDH paper (Two-Step Discrete Hashing for Cross-Modal Retrieval) has been accepted in 2024/03/21 by TMM. [code](./runners/TwDH/README.md) for details, [paper](https://ieeexplore.ieee.org/document/10487033).

Thanks for these authors. We re-construct their code with a common structure for the feature research. Compared with DCMHT, this project is more flexible. It consists of the following parts:

```
├── common
├── configs
│   ├── DCMHT
│   ├── DiHE
│   ├── DNPH
│   ├── DSPH
│   ├── MITH
│   └── TwDH
├── data
│   └── transformer
│       ├── coco
│       ├── nuswide
│       ├── mirflickr
├── dataset
├── models
│   ├── baseline
│   ├── CLIP
│   ├── common
│   ├── DCMHT
│   │   └── hash
│   ├── DNPH
│   │   ├── hash
│   │   └── loss
│   ├── DSPH
│   │   ├── hash
│   │   └── loss
│   ├── MITH
│   │   └── hash
├── runners
│   ├── baseline
│   ├── DCMHT
│   ├── DNPH
│   ├── DSPH
│   ├── MITH
└── utils
```

## Package Introduction：
### The common package：
It includes：
```
├── calc_utils.py
├── __init__.py
└── register.py
```
where "calc_utils" contains many cross-modal hashing metrics, including mAP，cosine、euclidean distance, label similarity.

"register" part is the most important part in this project. It oversees the invocation of the training, model, and optimizer modules, which enables our code to be concise. 
### data package：
It is used to save the training data. The training data is the same as DCMHT([link](https://github.com/kalenforn/DCHMT/tree/main)), or you can download at 

链接：https://pan.baidu.com/s/1d5MQNRPagem_3a-4LeQn8Q?pwd=0aju 
提取码：0aju

**Notice:** please run the change_index_path.py in the mat dir for each dataset, and replace the "path_replace" with the "images" dir.

### dataset package：
It consists of：
```
├── base.py
├── builder.py
├── __init__.py
└── transformer_dataset.py
```
This package is used to build dataset.

### models package：
This package is used to construct the model. The loss function should be added into the model structure.

### runners package：
This package is used to build the trainer.

### utils package：
It consists of:
```
├── get_args.py
├── __init__.py
├── logger.py
└── set_seed.py
```

## Details for Each Method：

- [DCMHT](./runners/DCMHT/README.md)
- [MITH](./runners/MITH/README.md)
- [DNPH](./runners/DNPH/README.md)
- [DSPH](./runners/DSPH/README.md)

## Training step


Before training:
- 1. Creating environment:
    > conda create -n clip-hash python=3.8
    >
    > source clip-hash
    >
    > pip install -r requirements.txt

- 2. Downloading CLIP pretrained model to this dir. we adopt the ViT-B-32 pretrained model, the download link is in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py).

Training on a single GPU.
> python main.py --config-file configs/DCMHT/config.yaml --save-dir result/DCMHT/coco/16

Training on multiple GPUs with a distribution method (**Testing**).
> python main.py --config-file configs/DCMHT/config.yaml --save-dir result/DCMHT/coco/16 --device "0,1,2,3" --distribute

## Acknowledegements

- [DCMHT](https://github.com/kalenforn/DCHMT/tree/main)
- [MITH](https://github.com/DarrenZZhang/MITH)
- [DSPH](https://github.com/QinLab-WFU/DSPH)
- [DNPH](https://github.com/QinLab-WFU/OUR-DNPH)

Thanks for the Key Laboratory of Knowledge Engineering with Big Dat in Hefei University of Technology!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=kalenforn/clip-based-cross-modal-hash&type=Date)](https://star-history.com/#kalenforn/clip-based-cross-modal-hash&Date)

## In the End
Thanks to the authors who have contributed code to github in the field of cross-modal hashing. I may study other content in the future. This project serves as a final summary and gives back to the open source community. If any author needs to merge projects, please contact ganlantee@gmail.com
