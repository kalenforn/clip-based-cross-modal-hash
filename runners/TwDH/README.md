# TwDH
This is the official code for paper of "Two-Step Discrete Hashing for Cross-Modal Retrieval"

[paper] just has been accepted by TMM, paper will be published soon!

## Framework
![framework](./data/structure.png)
In this work, we propose a novel two-step discrete hashing, which first extracts a long hash code and second utilizes a transform matrix T to project the long hash code to the short one. The projection procedure is shown in (b). The quantization method we inherit from the [DCMHT](https://github.com/kalenforn/DCHMT/tree/main) method. 

## Results
![result](./data/result.png)

## Training
>cd ../../
>
> python main.py --config-file configs/TwDH/config.yaml--save-dir result/TwDH/coco/16

## Citation
It will be updated soon!