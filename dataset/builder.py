
import os
import numpy as np
import scipy.io as scio

from common.register import registry


def split_data(captions, indexs, labels, query_num=5000, train_num=10000, random_index=None):
    # np.random.seed(seed=seed)
    if random_index is None:
        random_index = np.random.permutation(range(len(indexs)))
    query_index = random_index[: query_num]
    train_index = random_index[query_num: query_num + train_num]
    retrieval_index = random_index[query_num:]

    query_indexs = indexs[query_index]
    query_captions = captions[query_index]
    query_labels = labels[query_index]
    
    train_indexs = indexs[train_index]
    train_captions = captions[train_index]
    train_labels = labels[train_index]

    retrieval_indexs = indexs[retrieval_index]
    retrieval_captions = captions[retrieval_index]
    retrieval_labels = labels[retrieval_index]
    
    split_indexs = (query_indexs, train_indexs, retrieval_indexs)
    split_captions = (query_captions, train_captions, retrieval_captions)
    split_labels = (query_labels, train_labels, retrieval_labels)
    return split_indexs, split_captions, split_labels

def build_dataloader(captionFile: str,
                indexFile: str,
                labelFile: str,
                imageResolution=224,
                query_num=5000, 
                train_num=10000, 
                dataset_cls=None,
                **kwags):
    assert dataset_cls is not None, "'dataset_cls' must be provided!"
    dataset = registry.get_dataset_class(dataset_cls)
    if captionFile.endswith("mat"):
        
        captions = scio.loadmat(captionFile)
        if "caption" in captions:
            captions = captions["caption"]
        elif "tags" in captions:
            captions = captions["tags"]
        elif "YAll" in captions:
            captions = captions["YAll"]
        else:
            raise RuntimeError("text file is not support, we only read the keys of [caption, tags, YAll].")
        captions = captions[0] if captions.shape[0] == 1 else captions
    elif captionFile.endswith("txt"):
        with open(captionFile, "r") as f:
            captions = f.readlines()
        captions = np.asarray([[item.strip()] for item in captions])
    else:
        raise ValueError("the format of 'captionFile' doesn't support, only support [txt, mat] format.")
    
    if indexFile.endswith("mat"):
        npy = False
        indexs = scio.loadmat(indexFile)
        if "index" in indexs:
            indexs = indexs["index"]
        elif "imgs" in indexs:
            indexs = indexs["imgs"]
        elif "FAll" in indexs:
            indexs = indexs["FAll"]
        else:
            raise RuntimeError("image file is not support, we only read the keys of [caption, tags, YAll].")
    elif indexFile.endswith("npy"):
        npy = True
        indexs = np.load(indexFile)
    else:
        npy = False
        raise RuntimeError("index file is not support, we only read the keys of [*.mat, *.npy].")
    labels = scio.loadmat(labelFile)
    if "category" in labels:
        labels = labels["category"]
    elif "LAll" in labels:
        labels = labels["LAll"]
    elif "labels" in labels:
        labels = labels["labels"]
    else:
        raise RuntimeError("label file is not support, we only read the keys of [caption, tags, YAll].")
    
    split_indexs, split_captions, split_labels = split_data(captions, indexs, labels, query_num=query_num, train_num=train_num)

    img_train_transform = kwags['img_train_transform'] if 'img_train_transform' in kwags.keys() else None
    txt_train_transform = kwags['txt_train_transform'] if 'txt_train_transform' in kwags.keys() else None
    img_valid_transform = kwags['img_valid_transform'] if 'img_valid_transform' in kwags.keys() else None
    txt_valid_transform = kwags['txt_valid_transform'] if 'txt_valid_transform' in kwags.keys() else None

    train_data = dataset(captions=split_captions[1], indexs=split_indexs[1], labels=split_labels[1], 
                         imageResolution=imageResolution, img_transformer=img_train_transform, txt_transformer=txt_train_transform, npy=npy, **kwags)
    query_data = dataset(captions=split_captions[0], indexs=split_indexs[0], labels=split_labels[0], 
                         imageResolution=imageResolution, is_train=False, img_transformer=img_valid_transform, txt_transformer=txt_valid_transform, npy=npy, **kwags)
    retrieval_data = dataset(captions=split_captions[2], indexs=split_indexs[2], labels=split_labels[2], 
                             imageResolution=imageResolution, is_train=False, img_transformer=img_valid_transform, txt_transformer=txt_valid_transform, npy=npy, **kwags)

    return train_data, query_data, retrieval_data
    

