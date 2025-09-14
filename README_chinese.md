# CLIP-based Cross-Modal Hashing

本项目是一个依赖于CLIP模型的跨模态哈希检索的代码库，我们实现的方法有:
- DCMHT, _MM22_, [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548187)
- MITH, _MM23_, [paper](https://dl.acm.org/doi/10.1145/3581783.3612411)
- DSPH, _TCSVT23_, [paper](https://ieeexplore.ieee.org/document/10149001)
- DNPH, _ToMM24_, [paper](https://dl.acm.org/doi/10.1145/3643639)
- TwDH, _TMM24_, [paper](https://ieeexplore.ieee.org/document/10487033) 
- DIMCH, _TIP25_, [paper](https://ieeexplore.ieee.org/document/10974482)
- UMoED, _ToMM25_, [paper](https://dl.acm.org/doi/abs/10.1145/3744567)

**🔥最新消息** 
- TwDH于2024/03/21被TMM收录，详情请查看 [代码](./runners/TwDH/README.md)，[paper](https://ieeexplore.ieee.org/document/10487033)。
- DIMCH于2025/04/23被TIP收录，详情请查看 [代码](./runners/DIMCH/README.md)，[paper](https://ieeexplore.ieee.org/abstract/document/10974482)。
- UMoED于2025/06/08被ToMM收录，详情请查看[代码](./runners/UMoED/README.md)，[paper](https://dl.acm.org/doi/abs/10.1145/3744567)。

感谢这些工作的作者提供公开的代码，本项目基于DCMHT代码进行整体重构，并结合其所有衍生方法。本项目相较于DCMHT，具有更好的拓展性，更加便于研究者构建属于自己的跨模态哈希方法。
项目主要代码结构为
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

## 代码库简介：
### common包：
其包含代码：
```
├── calc_utils.py
├── __init__.py
└── register.py
```
其中，calc_utils包含多种cross-modal hashing需要的计算，包括mAP，cosine、euclidean距离，lable相关性等

register模块是整个代码的核心，所有的模型、训练器、优化器等都是由该模块进行调度。这种模式使整个代码框架更便于管理以及减少代码量。（**注意：** 添加任何新模型、训练器、优化器请先引入register并进行注册）
### data包：
该包用于存储原始数据，数据生成方式来自于DCMHT代码[link](https://github.com/kalenforn/DCHMT/tree/main)，或者你可以在我的百度云里下载

链接：https://pan.baidu.com/s/1d5MQNRPagem_3a-4LeQn8Q?pwd=0aju 
提取码：0aju

**注意：** 下载好以后请执行各自mat文件夹下的change_index_path.py文件，请修改其中的path_replace参数为各文件夹下的images文件夹

### dataset包：
其包含代码：
```
├── base.py
├── builder.py
├── __init__.py
└── transformer_dataset.py
```
该库用于数据集加载

### models包：
该库包含各种不同方法的模型，模型包含其独特的loss计算过程。

### runners包：
该库用于驱动训练过程

### utils包：
其包含代码：
```
├── get_args.py
├── __init__.py
├── logger.py
└── set_seed.py
```
其主要用于获取可变参数以及记录日志、设置随机种子

## 各方法介绍请参考：

- [DCMHT](./runners/DCMHT/README.md)
- [MITH](./runners/MITH/README.md)
- [DNPH](./runners/DNPH/README.md)
- [DSPH](./runners/DSPH/README.md)

## 训练

安装环境:
> conda create -n clip-hash python=3.8
>
> source clip-hash
>
> pip install -r requirements.txt

下载ViT-B-32预训练模型，下载链接在[CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py)的第30行中找

单卡训练
> python main.py --config-file configs/DCMHT/config.yaml --save-dir result/DCMHT/coco/16

多卡分布式训练(尚未测试)
> python main.py --config-file configs/DCMHT/config.yaml --save-dir result/DCMHT/coco/16 --device "0,1,2,3" --distribute

当前只测试了所有代码在coco数据集上的16比特结果，其于原始论文中结果偏差不大。

## 鸣谢

- [DCMHT](https://github.com/kalenforn/DCHMT/tree/main)
- [MITH](https://github.com/DarrenZZhang/MITH)
- [DSPH](https://github.com/QinLab-WFU/DSPH)
- [DNPH](https://github.com/QinLab-WFU/OUR-DNPH)

特别鸣谢合肥工业大学媒体计算实验室(数据知识工程重点实验室)对本人的支持！

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=kalenforn/clip-based-cross-modal-hash&type=Date)](https://star-history.com/#kalenforn/clip-based-cross-modal-hash&Date)

## 写最后
感谢cross-modal hashing领域在github贡献过代码的作者，本人后续可能会研究其他内容，此项目作为最终总结，并回馈与开源社区。如有作者需要合并项目请联系邮箱 ganlantee@gmail.com