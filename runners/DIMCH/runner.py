import torch
import os
import torch.nn as nn

from ..base import BaseTrainer
from common.register import registry


@registry.register_runner("DIMCHTrainer")
class DIMCHTrainer(BaseTrainer):
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
        self.hash_func = cfg.model.get("hash_func", "softmax")
        self.loss_type = loss_type
        super().__init__(cfg=cfg, is_train=is_train, logger=logger, device=device, output_dim=output_dim, train_num=train_num, distributed=distributed, 
                         query_num=query_num, epochs=epochs, save_dir=save_dir, display_step=display_step, top_k=top_k, model_state=model_state, batch_size=batch_size, world_size=world_size)
        self.build_dataset(cfg.dataset, train_num=train_num, query_num=query_num, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
        self.build_model(cfg.model, output_dim=output_dim, txt_token_size=cfg.dataset.get("max_word", 32))
        self.optimizer, self.lr_schedu = self.build_optimizer(cfg_optimizer=cfg.optimizer)
        if self.hash_func == "softmax":
            self.hash_scale = 2
        else:
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

    def build_model(self, cfg_model, output_dim=16, txt_token_size=32):

        arch = cfg_model.get("arch", "DCMHT")
        # print(arch)
        self.model = registry.get_model_class(arch).from_config(cfg_model, output_dim=output_dim, train_num=self.train_num, txt_token_size=txt_token_size)
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

    def compute_loss(self, img_embeds=None, img_hash=None, txt_embeds=None, txt_hash=None, label=None, index=None, epoch=0, times=0, global_step=0, **kwags):
        
        all_loss, loss_dict = self.model.object_function(img_embeds=img_embeds, img_hash=img_hash, txt_embeds=txt_embeds, txt_hash=txt_hash, labels=label, indexs=index, **kwags)
        
        if global_step % self.display_step == 0:
            bits = img_embeds.shape[-1] // self.hash_scale
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

            img_embeds, hash_img, txt_embeds, hash_text = self.model_ddp(image, text, return_loss=False) if self.model_ddp is not None else self.model(image, text, return_loss=False) 
            loss = self.compute_loss(img_hash=hash_img, txt_hash=hash_text, img_embeds=img_embeds, txt_embeds=txt_embeds, label=label, index=index, epoch=epoch, times=times, global_step=self.global_step)
                    
            all_loss += loss 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.logger.info(f">>>>>> [{epoch}/{self.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")

    def merge_hash(self, image_embed, text_embed):
        
        hash = torch.sign((image_embed + text_embed).detach() / 2)
        return hash

    def generate_hash(self, image, text, key_padding_mask=None):
        _, image_hash, _, text_hash = self.model_ddp(image, text, return_loss=False) if self.model_ddp is not None else self.model(image, text, return_loss=False) 
        return image_hash, text_hash
