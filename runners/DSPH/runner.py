import torch

from ..base import BaseTrainer
from common.register import registry


@registry.register_runner("DSPHTrainer")
class DSPHTrainer(BaseTrainer):
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
        self.optimizer, self.optimizer_loss, self.lr_schedu = self.build_optimizer(cfg_optimizer=cfg.optimizer)
        assert self.hash_func == "tanh", "DCMHT must adopt the 'tanh' hash technique."
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
    
    def build_optimizer(self, cfg_optimizer=None, parameters=None):

        optimizer, lr_schedu = super().build_optimizer(cfg_optimizer=cfg_optimizer, parameters=parameters)
        lr_ = cfg_optimizer.hyp.get("lr", 0.02)
        momentum_ = cfg_optimizer.hyp.get("momentum", 0.9)
        weight_decay_ = cfg_optimizer.hyp.get("weight_decay", 0.0005)
        optimizer_loss = torch.optim.SGD(params=self.model.hyp.parameters(), lr=lr_, momentum=momentum_, weight_decay=weight_decay_)
        self.logger.info("Building optimizer!")
        return optimizer, optimizer_loss, lr_schedu

    def compute_loss(self, img_hash=None, txt_hash=None, label=None, index=None, epoch=0, times=0, global_step=0, **kwags):
        
        all_loss, loss_dict = self.model.object_function(img_hash=img_hash, txt_hash=txt_hash, labels=label, indexs=index, **kwags)
        
        if global_step % self.display_step == 0:
            bits = img_hash.shape[-1] // self.hash_scale
            self.print_loss_dict(loss_dict, bits=bits, epoch=epoch, times=times)
                
        return all_loss
    
    # notice it is different with others
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
            self.optimizer_loss.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_loss.step()

        self.logger.info(f">>>>>> [{epoch}/{self.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")

