import os
import torch
from tqdm import tqdm
from ..base import BaseTrainer
from common.register import registry
from torch import distributed as dist


@registry.register_runner("TwDHTrainer")
class TwDHTrainer(BaseTrainer):
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
        self.build_model(cfg.model, output_dim=output_dim)
        self.optimizer, self.lr_schedu = self.build_optimizer(cfg_optimizer=cfg.optimizer)
        assert self.hash_func == "softmax", "DCMHT must adopt the 'softmax' hash technique."
        self.hash_scale = 2
        self.long_dim = cfg.model.get("long_dim", 512)
        self.max_short = {}
        self.best_epoch_short = {}
        for item in self.model.get_short_dims():
            self.max_short.update({item: {"i2t": 0, "t2i": 0}})
            self.best_epoch_short.update({item: {"i2t": 0, "t2i": 0}})
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
    
    @classmethod
    def make_hash_code(cls, code):

        if isinstance(code, list):
            code = torch.stack(code).permute(1, 0, 2)
        elif len(code.shape) < 3:
            code = code.view(code.shape[0], -1, 2)
        else:
            code = code
        hash_code = torch.argmax(code, dim=-1)
        hash_code[torch.where(hash_code == 0)] = -1
        hash_code = hash_code.float()

        return hash_code

    def compute_loss(self, long_img_hash, long_txt_hash, short_img_hash, short_txt_hash, label=None, index=None, epoch=0, times=0, global_step=0, **kwags):
        
        all_loss, loss_dict = self.model.object_function(long_img_hash=long_img_hash, long_txt_hash=long_txt_hash, short_img_hash=short_img_hash, short_txt_hash=short_txt_hash, labels=label, indexs=index, **kwags)
        
        if global_step % self.display_step == 0:
            bits = long_img_hash.shape[-1] // self.hash_scale
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

            img_long_hash, img_short_hash, txt_long_hash, txt_short_hash = self.model_ddp(image, text, return_loss=False) if self.model_ddp is not None else self.model(image, text, return_loss=False) 
            loss = self.compute_loss(long_img_hash=img_long_hash, long_txt_hash=txt_long_hash, short_img_hash=img_short_hash, short_txt_hash=txt_short_hash, label=label, index=index, epoch=epoch, times=times, global_step=self.global_step)
                    
            all_loss += loss 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.logger.info(f">>>>>> [{epoch}/{self.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")
    
    def generate_hash(self, image, text, key_padding_mask=None):
        long_image_hash, short_image_hash = self.model.encode_image(image) if self.model_ddp is None else self.model_ddp.module.encode_image(image)
        long_text_hash, short_text_hash = self.model.encode_text(text) if self.model_ddp is None else self.model_ddp.module.encode_text(text)

        return long_image_hash, short_image_hash, long_text_hash, short_text_hash

    def get_code(self, data_loader, length: int):
        short_dims = self.model.get_short_dims()
        self.change_state(mode="valid")
        long_img_buffer = torch.empty(length, self.long_dim, dtype=torch.float).to(self.device)
        long_txt_buffer = torch.empty(length, self.long_dim, dtype=torch.float).to(self.device)
        short_img_buffers = {}
        short_txt_buffers = {}
        for dim in short_dims:
            short_img_buffers.update({str(dim): torch.empty(length, dim, dtype=torch.float).to(self.device)})
            short_txt_buffers.update({str(dim): torch.empty(length, dim, dtype=torch.float).to(self.device)})

        for image, text, key_padding_mask, label, index in tqdm(data_loader):
            image = image.to(self.device, non_blocking=True)
            text = text.to(self.device, non_blocking=True)
            index = index.numpy()
            long_image_hash, short_image_hash, long_text_hash, short_text_hash = self.generate_hash(image=image, text=text, key_padding_mask=key_padding_mask)
            

            long_img_buffer[index, :] = self.make_hash_code(long_image_hash.data)
            long_txt_buffer[index, :] = self.make_hash_code(long_text_hash.data)
            for k, v in short_image_hash.items():
                short_img_buffers[k][index, :] = self.make_hash_code(v.data)
            for k, v in short_text_hash.items():
                short_txt_buffers[k][index, :] = self.make_hash_code(v.data)
        
        # If distributed, aggregate shards across ranks by summing (each index is unique to one rank)
        if self.distributed:
            # ensure all ranks have finished writing their local shards
            dist.barrier()
            dist.all_reduce(long_img_buffer, op=dist.ReduceOp.SUM)
            dist.all_reduce(long_txt_buffer, op=dist.ReduceOp.SUM)
            dist.all_reduce(short_img_buffers, op=dist.ReduceOp.SUM)
            dist.all_reduce(short_txt_buffers, op=dist.ReduceOp.SUM)
         
        return long_img_buffer, long_txt_buffer, short_img_buffers, short_txt_buffers
    
    def valid(self, epoch, k=None):
        assert self.query_loader is not None and self.retrieval_loader is not None
        save_dir = os.path.join(self.save_dir, "mat_files")
        os.makedirs(save_dir, exist_ok=True)
        self.logger.info("Valid.")

        q_long_img_buffer, q_long_txt_buffer, q_short_img_buffers, q_short_txt_buffers = self.get_code(self.query_loader, self.query_num)
        r_long_img_buffer, r_long_txt_buffer, r_short_img_buffers, r_short_txt_buffers = self.get_code(self.retrieval_loader, self.retrieval_num)

        self.valid_each(epoch=epoch, query_img=q_long_img_buffer, query_txt=q_long_txt_buffer, retrieval_img=r_long_img_buffer, retrieval_txt=r_long_txt_buffer, k=k, save_dir=save_dir)
        for key, v in q_short_img_buffers.items():
            self.valid_each(epoch=epoch, query_img=q_short_img_buffers[key], query_txt=q_short_txt_buffers[key], retrieval_img=r_short_img_buffers[key], 
                            retrieval_txt=r_short_txt_buffers[key], k=k, save_dir=save_dir, short=key)

    def valid_each(self, epoch, query_img=None, query_txt=None, retrieval_img=None, retrieval_txt=None, k=None, save_dir=None, short=None):

        mAPi2t = self.calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, k)
        mAPt2i = self.calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, k)
        mAPi2i = self.calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, k)
        mAPt2t = self.calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, k)

        if short is None:
            if self.max_mapi2t < mAPi2t:
                self.best_epoch_i = epoch
                if not self.distributed or (self.distributed and self.device == 0):
                    self.save_mat(query_img, query_txt, self.query_labels, retrieval_img, retrieval_txt, self.retrieval_labels, save_file=os.path.join(save_dir, "i2t-long.mat"))
                    self.save_model(save_dir=self.save_dir, epoch=epoch)
            self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
            if self.max_mapt2i < mAPt2i:
                self.best_epoch_t = epoch
                if not self.distributed or (self.distributed and self.device == 0):
                    self.save_mat(query_img, query_txt, self.query_labels, retrieval_img, retrieval_txt, self.retrieval_labels, save_file=os.path.join(save_dir, "t2i-long.mat"))
                    self.save_model(save_dir=self.save_dir, epoch=epoch)
            self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
            self.logger.info(f">>>>>> [{epoch}/{self.epochs}], Long, {query_img.shape[-1]} Bit, MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, "\
                    f"MAX MAP(i->t): {self.max_mapi2t}, epoch: {self.best_epoch_i}, MAX MAP(t->i): {self.max_mapt2i}, epoch: {self.best_epoch_t}")
        else:
            if self.max_short[short]["i2t"] < mAPi2t:
                self.best_epoch_short[short]["i2t"] = epoch
                if not self.distributed or (self.distributed and self.device == 0):
                    self.save_mat(query_img, query_txt, self.query_labels, retrieval_img, retrieval_txt, self.retrieval_labels, save_file=os.path.join(save_dir, f"i2t-short-{short}.mat"))
            self.max_short[short]["i2t"] = max(self.max_short[short]["i2t"], mAPi2t)
            if self.max_short[short]["t2i"] < mAPt2i:
                self.best_epoch_short[short]["t2i"] = epoch
                if not self.distributed or (self.distributed and self.device == 0):
                    self.save_mat(query_img, query_txt, self.query_labels, retrieval_img, retrieval_txt, self.retrieval_labels, save_file=os.path.join(save_dir, f"t2i-short-{short}.mat"))
            self.max_short[short]["t2i"] = max(self.max_short[short]["t2i"], mAPt2i)
            self.logger.info(f">>>>>> [{epoch}/{self.epochs}], Short, {query_img.shape[-1]} Bit, MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, "\
                    f"MAX MAP(i->t): {self.max_short[short]['i2t']}, epoch: {self.best_epoch_short[short]['i2t']}, MAX MAP(t->i): {self.max_short[short]['t2i']}, epoch: {self.best_epoch_short[short]['t2i']}")