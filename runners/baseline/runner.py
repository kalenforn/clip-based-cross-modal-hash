import torch
import os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base import BaseTrainer
from common.register import registry
from dataset.builder import build_dataloader


@registry.register_runner("BaselineTrainer")
class BaselineTrainer(BaseTrainer):
    def __init__(self,
                cfg, 
                is_train=True, 
                logger=None, 
                device=None, 
                world_size=torch.cuda.device_count(), 
                output_dim=16, 
                train_num=10000,
                query_num=5000,
                epochs=100, 
                save_dir="./result", 
                batch_size=128, 
                num_workers=4, 
                pin_memory=True, 
                shuffle=True,
                display_step=20,
                top_k=5000,
                model_state="",
                loss_type="l1", 
                distributed=False, 
                **kwags) -> None:
        self.hash_func = cfg.model.get("hash_func", "tanh")
        self.loss_type = loss_type
        super().__init__(cfg=cfg, is_train=is_train, logger=logger, device=device, output_dim=output_dim, train_num=train_num, distributed=distributed, 
                         query_num=query_num, epochs=epochs, save_dir=save_dir, display_step=display_step, top_k=top_k, model_state=model_state, batch_size=batch_size, world_size=world_size)
        self.build_dataset(cfg.dataset, train_num=train_num, query_num=query_num, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
        self.build_model(cfg.model, output_dim=output_dim)
        self.optimizer, self.lr_schedu = self.build_optimizer(cfg_optimizer=cfg.optimizer)
        assert self.hash_func == "tanh", "Baseline must adopt the 'tahn' hash technique."
        self.hash_scale = 1
        self.run()
    
    @classmethod
    def from_config(cls, rank=0, world_size=torch.cuda.device_count(), distributed=False, cfg=None, logger=None):
        assert cfg is not None, "config is None!"
        is_train = cfg.run.get("is_train", False)

        if distributed:
            device = rank
        else:
            device = cfg.run.get("device", 0)
        
        output_dim = cfg.run.get("output_dim", 16)
        train_num = cfg.run.get("train_num", 10000)
        query_num = cfg.run.get("query_num", 5000)
        epochs = cfg.run.get("epochs", 10)
        batch_size = cfg.run.get("batch_size", 128)
        num_workers = cfg.run.get("num_workers", 4)
        pin_memory = cfg.run.get("pin_memory", True)
        shuffle = cfg.run.get("shuffle", True)
        save_dir = cfg.run.get("save_dir", True)
        display_step = cfg.run.get("display_step", 20)
        top_k = cfg.run.get("top_k", None)
        model_state = cfg.run.get("resume_model", "")

        return cls(cfg, 
                is_train=is_train,
                logger=logger, 
                device=device, 
                output_dim=output_dim, 
                train_num=train_num,
                query_num=query_num,
                epochs=epochs, 
                save_dir=save_dir, 
                batch_size=batch_size, 
                num_workers=num_workers, 
                pin_memory=pin_memory, 
                shuffle=shuffle, 
                model_state=model_state, 
                display_step=display_step, 
                top_k=top_k,
                world_size=world_size, 
                distributed=distributed)

    def build_model(self, cfg_model, output_dim=16):

        arch = cfg_model.get("arch", "DCMHT")
        # print(arch)
        self.model = registry.get_model_class(arch).from_config(cfg_model, output_dim=output_dim, train_num=self.train_num)
        if os.path.isfile(self.model_state):
            self.logger.info("loading model...")
            self.model.load_state_dict(torch.load(self.model_state, map_location=f"cuda:{self.device}"))
        self.model.float()
        self.model.to(self.device)

        if self.distributed:
            self.logger.info("use distribution mode.")
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model_ddp = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device], find_unused_parameters=True)
        else:
            self.model_ddp = None

        self.logger.info("Building model!")
        # print(self.model)
        self.logger.info(f"Output dim: {self.output_dim}")
    
    def build_optimizer(self, cfg_optimizer=None, parameters=None):

        lr_schedu = None
        arch = cfg_optimizer.get("arch", "BertAdam")
        clip_lr = cfg_optimizer.get("clip_lr", 0.00001)
        lr = cfg_optimizer.get("lr", 0.001)
        warmup_proportion = cfg_optimizer.get("warmup_proportion", 0.1)
        schedule = cfg_optimizer.get("schedule", "warmup_cosine")
        b1 = cfg_optimizer.get("b1", 0.9)
        b2 = cfg_optimizer.get("b2", 0.98) 
        e = cfg_optimizer.get("e", 0.000001) 
        max_grad_norm = cfg_optimizer.get("max_grad_norm", 1.0)  
        weight_decay = cfg_optimizer.get("weight_decay",  0.2)  

        if parameters is None:
            parameters = [{'params': self.model.clip.parameters(), 'lr': clip_lr},
                        {'params': self.model.hash.parameters(), 'lr': lr}]
            
        optimizer = registry.get_optimizer_class(arch)(parameters, lr=lr, warmup=warmup_proportion, 
                                                            schedule=schedule, b1=b1, b2=b2, e=e, t_total=len(self.train_loader) * self.epochs,
                                                            weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        self.logger.info("Building optimizer!")
        return optimizer, lr_schedu
        
    
    def build_dataset(self, cfg, train_num=10000, query_num=5000, batch_size=128, num_workers=4, pin_memory=True, shuffle=True):
        dataname = cfg.get("name", "mirflickr25k")
        path = cfg.get("path", "./data")
        self.logger.info(f"Using {dataname} dataset.")
        image_file = os.path.join(path, dataname, cfg.get("img_file", "index.mat"))
        text_file = os.path.join(path, dataname, cfg.get("txt_file", "caption.mat"))
        label_file = os.path.join(path, dataname, cfg.get("label_file", "caption.mat"))
        max_word = cfg.get("max_word", 32)
        image_resolution = cfg.get("image_resolution", 224)
        dataset_cls = cfg.get("arch", "transformer_dataset")

        train_data, query_data, retrieval_data = build_dataloader(
            captionFile=text_file, indexFile=image_file, labelFile=label_file, imageResolution=image_resolution, 
            maxWords=max_word, query_num=query_num, train_num=train_num, dataset_cls=dataset_cls, tokenizer=registry.get_tokenizer_class(cfg.get("tokenizer_arch", "clip_tokenizer"))()
        )
        self.build_loader(train_data=train_data, query_data=query_data, retrieval_data=retrieval_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
    
    def train(self):

        for epoch in range(self.epochs):
            self.train_epoch(epoch=epoch)
            self.valid(epoch, k=self.top_k)
        
        self.logger.info(f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")

    def compute_loss(self, img_hash=None, txt_hash=None, label=None, index=None, epoch=0, times=0, global_step=0, **kwags):
        
        all_loss, loss_dict = self.model.object_function(img_hash=img_hash, txt_hash=txt_hash, labels=label, indexs=index, **kwags)
        
        if global_step % self.display_step == 0:
            bits = img_hash.shape[-1] // self.hash_scale
            self.print_loss_dict(loss_dict, bits=bits, epoch=epoch, times=times)
                
        return all_loss
    
    def train_epoch(self, epoch: int):
        self.change_state(mode="train")

        self.logger.info(">>>>>> epochs: %d/%d"%(epoch, self.epochs))
        all_loss = 0
        times = 0
        for image, text, key_padding_mask, label, index in self.train_loader:
            self.global_step += 1
            times += 1
            image = image.to(self.device, non_blocking=True)
            text = text.to(self.device, non_blocking=True)
            index = index.numpy()

            hash_img, hash_text = self.model_ddp(image, text, return_loss=False) if self.model_ddp is not None else self.model(image, text, return_loss=False) 
            loss = self.compute_loss(img_hash=hash_img, txt_hash=hash_text, label=label, index=index, epoch=epoch, times=times, global_step=self.global_step)
                    
            all_loss += loss 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.logger.info(f">>>>>> [{epoch}/{self.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")

def hash_convert(hash_label):
    # 用作转换hash code，匹配我的模型
    if len(hash_label.shape) == 2:
        result = torch.zeros([hash_label.shape[0], hash_label.shape[1] ,2])
        hash_label = (hash_label > 0).long()
        # 构造坐标
        i = torch.arange(hash_label.shape[0]).view(hash_label.shape[0], -1).expand_as(hash_label)
        j = torch.arange(hash_label.shape[1]).expand_as(hash_label)
        result[i, j, hash_label] = 1
        result = result.view(hash_label.shape[0], -1)
    elif len(hash_label.shape) == 1:
        result = torch.zeros([hash_label.shape[0], 2])
        hash_label = (hash_label > 0).long()
        result[torch.arange(hash_label.shape[0]), hash_label] = 1
        result = result.view(hash_label.shape[0], -1)
    result = result.to(hash_label.device)
    return result


