import torch

from ..base import BaseTrainer
from common.register import registry
from common.calc_utils import calc_label_sim


@registry.register_runner("MITHTrainer")
class MITHTrainer(BaseTrainer):
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
        assert self.hash_func == "tanh"
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

    def compute_loss(self, res_img_cls, img_cls_hash, tokens_hash_i, trans_tokens_i, 
                     res_txt_cls, txt_cls_hash, tokens_hash_t, trans_tokens_t, 
                     label=None, index=None, epoch=0, times=0, global_step=0, **kwags):
        label_sim = calc_label_sim(self.train_labels, label)
        all_loss, loss_dict = self.model.object_function(res_img_cls=res_img_cls, img_cls_hash=img_cls_hash, tokens_hash_i=tokens_hash_i, trans_tokens_i=trans_tokens_i,
                                res_txt_cls=res_txt_cls, txt_cls_hash=txt_cls_hash, tokens_hash_t=tokens_hash_t, trans_tokens_t=trans_tokens_t, 
                                labels=label, indexs=index, label_sim=label_sim, **kwags)
        
        if global_step % self.display_step == 0:
            bits = img_cls_hash.shape[-1] // self.hash_scale
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

            res_img_cls, img_cls_hash, tokens_hash_i, trans_tokens_i, res_txt_cls, txt_cls_hash, tokens_hash_t, trans_tokens_t = \
                self.model_ddp(image, text, key_padding_mask=key_padding_mask, return_loss=False) if self.model_ddp is not None \
                    else self.model(image, text, key_padding_mask=key_padding_mask, return_loss=False) 
            loss = self.compute_loss(res_img_cls=res_img_cls, img_cls_hash=img_cls_hash, tokens_hash_i=tokens_hash_i, trans_tokens_i=trans_tokens_i,
                                     res_txt_cls=res_txt_cls, txt_cls_hash=txt_cls_hash, tokens_hash_t=tokens_hash_t, trans_tokens_t=trans_tokens_t, 
                                     label=label, index=index, epoch=epoch, times=times, global_step=self.global_step)
                    
            all_loss += loss 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.logger.info(f">>>>>> [{epoch}/{self.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")
    
    def generate_hash(self, image, text, key_padding_mask=None):
        _, img_cls_hash, tokens_hash_i, _, \
                _, txt_cls_hash, tokens_hash_t, _ = self.model_ddp(image, text, key_padding_mask=key_padding_mask, return_loss=False) if self.model_ddp is not None \
                    else self.model(image, text, key_padding_mask=key_padding_mask, return_loss=False)
        image_hash = img_cls_hash.detach() + tokens_hash_i.detach()
        text_hash = txt_cls_hash.detach() + tokens_hash_t.detach()
        return image_hash, text_hash

