from torch.utils.data import Dataset
import torch
import random
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop
from common.register import registry
from .base import BaseDataset



@registry.register_dataset("transformer_dataset")
class Transformer_Dataset(BaseDataset):

    def __init__(self, 

            captions: dict,
            indexs: dict,
            labels: dict,
            is_train=True,
            imageResolution=224,
            tokenizer=None,
            maxWords=32,
            npy=False, 
            **kwags):
        
        super().__init__()
        self.captions = captions
        self.indexs = indexs
        self.labels = labels
        self.is_train = is_train

        self.__length = len(self.indexs)

        self.transform = Compose([
                # Resize(imageResolution, interpolation=Image.BICUBIC),
                # CenterCrop(imageResolution),
                RandomHorizontalFlip(),
                RandomResizedCrop(imageResolution),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]) if is_train else Compose([
                Resize((imageResolution, imageResolution), interpolation=Image.BICUBIC),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        self.npy = npy
        
        self.maxWords = maxWords
        self.tokenizer = tokenizer
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        # self.__length = len(self.indexs)
        
    def __len__(self):
        return self.__length

    def _load_image(self, index: int) -> torch.Tensor:
        if not self.npy:
            image_path = self.indexs[index].strip()
            # print(image_path)
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        else:
            image = self.transform(Image.fromarray(self.indexs[index], mode="RGB"))

        return image

    def _load_text(self, index: int):
        captions = self.captions[index]
        
        if self.tokenizer is not None:
            use_cap = captions[random.randint(0, len(captions) - 1)]
            words = self.tokenizer.tokenize(use_cap)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.maxWords - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            caption = self.tokenizer.convert_tokens_to_ids(words)

            while len(caption) < self.maxWords:
                caption.append(0)

        caption = torch.tensor(caption)
        key_padding_mask = (caption == 0)
        return caption, key_padding_mask
    
    def _load_label(self, index: int) -> torch.Tensor:
        label = self.labels[index]
        label = torch.from_numpy(label)

        return label

    def get_all_label(self):
        labels = torch.zeros([self.__length, len(self.labels[0])], dtype=torch.int64)
        for i, item in enumerate(self.labels):

            labels[i] = torch.from_numpy(item)
        return labels

    def __getitem__(self, index):
        image = self._load_image(index)
        caption, key_padding_mask = self._load_text(index)
        label = self._load_label(index)

        return image, caption, key_padding_mask, label, index
