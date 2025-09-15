import torch
import os
import torch.nn as nn
import scipy.io as scio
from tqdm import tqdm

from ..base import BaseTrainer
from common.register import registry
from torch import distributed as dist


@registry.register_runner("UMoEDTrainer")
class UMoEDTrainer(BaseTrainer):
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
        self.build_dataset(cfg.dataset, train_num=train_num, query_num=query_num, batch_size=batch_size, num_workers=num_workers, \
                           pin_memory=pin_memory, shuffle=shuffle)
        self.build_model(cfg.model, output_dim=output_dim, txt_token_size=cfg.dataset.get("max_word", 32))
        self.optimizer, self.lr_schedu = self.build_optimizer(cfg_optimizer=cfg.optimizer)
        if self.hash_func == "softmax":
            self.hash_scale = 2
        else:
            self.hash_scale = 1

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params / 1e6:.2f}M")

        self.max_mapi2t = 0
        self.max_mapt2i = 0
        self.max_mapi2i = 0
        self.max_mapt2t = 0
        self.best_epoch_i2t = 0
        self.best_epoch_t2i = 0
        self.best_epoch_i2i = 0
        self.best_epoch_t2t = 0
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
            self.logger.info("Using distribution mode.")
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model_ddp = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device], find_unused_parameters=True)
        else:
            self.model_ddp = None

        self.logger.info("Building model!")
        self.logger.info(f"Output dim: {self.output_dim}")

    def compute_loss(self, img_embeds=None, img_hash=None, txt_embeds=None, txt_hash=None, fusion_embeds=None, fusion_hash=None, label=None, index=None, epoch=0, times=0, global_step=0, **kwags):
        
        all_loss, loss_dict = self.model.object_function(img_embeds=img_embeds, img_hash=img_hash, txt_embeds=txt_embeds, txt_hash=txt_hash, fusion_embeds=fusion_embeds, fusion_hash=fusion_hash, labels=label, indexs=index, **kwags)
        
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

            img_embeds, hash_img, txt_embeds, hash_text, fusion_embeds, fusion_hash = self.model_ddp(image, text, return_loss=False) if self.model_ddp is not None else self.model(image, text, return_loss=False) 
            loss = self.compute_loss(img_hash=hash_img, txt_hash=hash_text, img_embeds=img_embeds, txt_embeds=txt_embeds, fusion_embeds=fusion_embeds, fusion_hash=fusion_hash, label=label, index=index, epoch=epoch, times=times, global_step=self.global_step)
                    
            all_loss += loss 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.logger.info(f">>>>>> [{epoch}/{self.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")


    def generate_hash(self, image, text, key_padding_mask=None):
        _, image_hash, _, text_hash, _, fusion_hash = self.model_ddp(image, text, return_loss=False) if self.model_ddp is not None else self.model(image, text, return_loss=False) 
        return image_hash, text_hash, fusion_hash

    def get_code(self, data_loader, length: int):
        self.change_state(mode="valid")
        img_buffer = torch.empty(length, self.output_dim, dtype=torch.float).to(self.device)
        text_buffer = torch.empty(length, self.output_dim, dtype=torch.float).to(self.device)
        fusion_buffer = torch.empty(length, self.output_dim, dtype=torch.float).to(self.device)

        for image, text, key_padding_mask, label, index in tqdm(data_loader):
            image = image.to(self.device, non_blocking=True)
            text = text.to(self.device, non_blocking=True)
            index = index.numpy()
            image_hash, text_hash, fusion_hash = self.generate_hash(image=image, text=text, key_padding_mask=key_padding_mask)
            

            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data
            # fusion_buffer[index, :] = fusion_hash.data
        
        if self.distributed:
            # ensure all ranks have finished writing their local shards
            dist.barrier()
            dist.all_reduce(img_buffer, op=dist.ReduceOp.SUM)
            dist.all_reduce(text_buffer, op=dist.ReduceOp.SUM)
            dist.all_reduce(fusion_buffer, op=dist.ReduceOp.SUM)
         
        return img_buffer, text_buffer, fusion_buffer
    
    def valid(self, epoch, k=None):
        assert self.query_loader is not None and self.retrieval_loader is not None
        save_dir = os.path.join(self.save_dir, "mat_files")
        os.makedirs(save_dir, exist_ok=True)
        self.logger.info("Valid.")
        # self.change_state(mode="valid")

        query_img, query_txt, query_fusion = self.get_code(self.query_loader, self.query_num)
        retrieval_img, retrieval_txt, retrieval_fusion = self.get_code(self.retrieval_loader, self.retrieval_num)

        mAPi2t = self.calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, k)
        mAPt2i = self.calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, k)
        mAPi2i = self.calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, k)
        mAPt2t = self.calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, k)

        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i2t = epoch
            if not self.distributed or (self.distributed and self.device == 0):
                self.save_mat(query_img, query_txt, query_fusion, self.query_labels, retrieval_img, retrieval_txt, retrieval_fusion, self.retrieval_labels, save_file=os.path.join(save_dir, "i2t-best.mat"))
                # self.save_model(save_dir=self.save_dir, epoch=epoch)
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)

        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t2i = epoch
            if not self.distributed or (self.distributed and self.device == 0):
                self.save_mat(query_img, query_txt, query_fusion, self.query_labels, retrieval_img, retrieval_txt, retrieval_fusion, self.retrieval_labels, save_file=os.path.join(save_dir, "t2i-best.mat"))
                # self.save_model(save_dir=self.save_dir, epoch=epoch)
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)

        if self.max_mapi2i < mAPi2i:
            self.best_epoch_i2i = epoch
            if not self.distributed or (self.distributed and self.device == 0):
                self.save_mat(query_img, query_txt, query_fusion, self.query_labels, retrieval_img, retrieval_txt, retrieval_fusion, self.retrieval_labels, save_file=os.path.join(save_dir, "i2i-best.mat"))
                # self.save_model(save_dir=self.save_dir, epoch=epoch)
        self.max_mapi2i = max(self.max_mapi2i, mAPi2i)

        if self.max_mapt2t < mAPt2t:
            self.best_epoch_t2t = epoch
            if not self.distributed or (self.distributed and self.device == 0):
                self.save_mat(query_img, query_txt, query_fusion, self.query_labels, retrieval_img, retrieval_txt, retrieval_fusion, self.retrieval_labels, save_file=os.path.join(save_dir, "t2t-best.mat"))
                # self.save_model(save_dir=self.save_dir, epoch=epoch)
        self.max_mapt2t = max(self.max_mapt2t, mAPt2t)

        self.logger.info(f">>>>>> [{epoch}/{self.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, "\
                        f"MAX MAP(i->t): {self.max_mapi2t}, epoch: {self.best_epoch_i2t}, MAX MAP(t->i): {self.max_mapt2i}, epoch: {self.best_epoch_t2i}, "\
                        f"MAX MAP(i->i): {self.max_mapi2i}, epoch: {self.best_epoch_i2i}, MAX MAP(t->t): {self.max_mapt2t}, epoch: {self.best_epoch_t2t}, "\
                        )
    
    def train(self):

        for epoch in range(self.epochs):
            self.train_epoch(epoch=epoch)
            self.valid(epoch, k=self.top_k)
        
        self.logger.info(f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i2t}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t2i}, mAP: {self.max_mapt2i}, I-I: {self.best_epoch_i2i}, MAP: {self.max_mapi2i}, T-T: {self.best_epoch_t2t}, MAP: {self.max_mapt2t}")
    
    @classmethod
    def save_mat(cls, query_img, query_txt, query_fusion, query_labels, retrieval_img, retrieval_txt, retrieval_fusion, retrieval_labels, save_file="i2t"):

        if isinstance(query_img, torch.Tensor):
            query_img = query_img.cpu().detach().numpy()
            query_txt = query_txt.cpu().detach().numpy()
            retrieval_img = retrieval_img.cpu().detach().numpy()
            retrieval_txt = retrieval_txt.cpu().detach().numpy()
            query_fusion = query_fusion.cpu().detach().numpy()
            retrieval_fusion = retrieval_fusion.cpu().detach().numpy()
            query_labels = query_labels.numpy()
            retrieval_labels = retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'q_fus': query_fusion,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'r_fus': retrieval_fusion,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_file), result_dict)
    
