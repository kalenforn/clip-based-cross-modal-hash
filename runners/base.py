import os
from tqdm import tqdm
import torch
import scipy.io as scio
from common.calc_utils import calc_map_k
from torch.utils.data import DataLoader

from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from dataset.builder import build_dataloader
from utils.logger import get_color_logger
from common.register import registry

# @registry.register_runner("BaseTrainer")
class BaseTrainer():

    def __init__(self, 
                cfg, 
                is_train=True, 
                device=None, 
                world_size=torch.cuda.device_count(), 
                output_dim=16, 
                train_num=10000,
                query_num=5000,
                epochs=100, 
                save_dir="./result", 
                display_step=20,
                top_k=5000,
                model_state="",
                batch_size=128,
                distributed=False, 
                # used to open the softmax hash method.
                **kwags) -> None:
        
        self.cfg = cfg

        self.is_train = is_train
        if distributed:
            # world_size=2
            # self.gpus = self.cfg.run.device
            self.logger = get_color_logger(cfg.run.log_dir, cfg.dataset.name + "-" + str(device), display=(device==0))
            self._init_distribution(rank=device, world_size=world_size)
            # self.logger.info(device==self.gpus[0])
            # self.logger.info("distribution inited!")
        else:
            self.logger = get_color_logger(cfg.run.log_dir, cfg.dataset.name + "-" + str(device))
        # self.logger.info(f"display: {device==self.gpus[0]} {device} {self.gpus}")
        self.logger.info(f"parameters: {cfg}")
        self.device = device
        self.output_dim = output_dim
        self.train_num = train_num
        self.query_num = query_num
        self.epochs = epochs
        self.display_step = display_step
        self.top_k=top_k
        self.model_state = model_state
        self.batch_size = batch_size
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # used to display the middle step.
        self.global_step = 0

        # used to record the best result.
        self.max_mapi2t = 0
        self.max_mapt2i = 0
        self.best_epoch_i = 0
        self.best_epoch_t = 0

        self.calc_map_k = calc_map_k
        self.distributed = distributed
        self.world_size = world_size
    
    def _init_distribution(self, rank=0, world_size=4):
        self.rank = rank
        self.world_size = world_size
        self.logger.info("Initializing distributed")
        os.environ['MASTER_ADDR'] = self.cfg.run.distributed_addr
        os.environ['MASTER_PORT'] = str(self.cfg.run.distributed_port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    def build_model(self, cfg_model, output_dim=16, **kwags):

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
        backbone_lr = cfg_optimizer.get("backbone_lr", 0.00001)
        lr = cfg_optimizer.get("lr", 0.001)
        warmup_proportion = cfg_optimizer.get("warmup_proportion", 0.1)
        schedule = cfg_optimizer.get("schedule", "warmup_cosine")
        b1 = cfg_optimizer.get("b1", 0.9)
        b2 = cfg_optimizer.get("b2", 0.98) 
        e = cfg_optimizer.get("e", 0.000001) 
        max_grad_norm = cfg_optimizer.get("max_grad_norm", 1.0)  
        weight_decay = cfg_optimizer.get("weight_decay",  0.2)  

        if parameters is None:
            parameters = [{'params': self.model.backbone.parameters(), 'lr': backbone_lr},
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
    
    def build_loader(self, train_data, query_data, retrieval_data, batch_size, num_workers, pin_memory, shuffle, drop_last=False):

        self.train_labels = train_data.get_all_label()
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"train shape: {self.train_labels.shape}")
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")

        if self.distributed:
            train_data_sampler = DistributedSampler(
                dataset=train_data,
                rank=self.rank,
                num_replicas=self.world_size,
                shuffle=True
            )
            batch_size = batch_size // self.world_size
        else:
            train_data_sampler = None

        self.train_loader = DataLoader(
                dataset=train_data,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=train_data_sampler, 
                shuffle=shuffle if train_data_sampler is None else False,
                drop_last=drop_last
            )
        self.query_loader = DataLoader(
                dataset=query_data,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=shuffle,
                drop_last=False
            )
        self.retrieval_loader = DataLoader(
                dataset=retrieval_data,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=shuffle,
                drop_last=False
            )
    
    def run(self):
        if self.is_train:
            self.train()
        else:
            self.test()

    def generate_hash(self, image, text, key_padding_mask=None):
        image_hash = self.model.encode_image(image) if self.model_ddp is None else self.model_ddp.module.encode_image(image)
        text_hash = self.model.encode_text(text) if self.model_ddp is None else self.model_ddp.module.encode_text(image)

        return image_hash, text_hash

    def get_code(self, data_loader, length: int):
        self.change_state(mode="valid")
        img_buffer = torch.empty(length, self.output_dim, dtype=torch.float).to(self.device)
        text_buffer = torch.empty(length, self.output_dim, dtype=torch.float).to(self.device)

        for image, text, key_padding_mask, label, index in tqdm(data_loader):
            image = image.to(self.device, non_blocking=True)
            text = text.to(self.device, non_blocking=True)
            index = index.numpy()
            image_hash, text_hash = self.generate_hash(image=image, text=text, key_padding_mask=key_padding_mask)
            

            img_buffer[index, :] = self.make_hash_code(image_hash.data)
            text_buffer[index, :] = self.make_hash_code(text_hash.data)
         
        return img_buffer, text_buffer
    
    def change_state(self, mode):
        """
        This method need to be rewrote if the self.model is not the single model, splited as the image_model and the text model.
        """
        if self.model_ddp is None:
            if mode == "train":
                self.model.train()
                self.model.unfreezen()
            else:
                self.model.eval()
                self.model.freezen()
        else:
            if mode == "train":
                self.model_ddp.train()
                self.model.train()
            else:
                self.model_ddp.eval()
                self.model.eval()
    
    def train(self):

        for epoch in range(self.epochs):
            self.train_epoch(epoch=epoch)
            self.valid(epoch, k=self.top_k)
        
        self.logger.info(f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")
    
    def train_epoch(self, epoch: int):
        raise NotImplementedError()

    def compute_loss(self, image_embed, text_embed, label, index, epoch=0, times=0, global_step=0, **kwags):
        """
        This method work with computing loss and displaying the loss result.

        To adapt the projection method, it is separated from the train_epoch code.
        """
        raise NotImplementedError()
    
    def valid(self, epoch, k=None):
        assert self.query_loader is not None and self.retrieval_loader is not None
        save_dir = os.path.join(self.save_dir, "mat_files")
        os.makedirs(save_dir, exist_ok=True)
        self.logger.info("Valid.")
        # self.change_state(mode="valid")

        query_img, query_txt = self.get_code(self.query_loader, self.query_num)
        retrieval_img, retrieval_txt = self.get_code(self.retrieval_loader, self.retrieval_num)

        mAPi2t = self.calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, k)
        # print("map map")
        mAPt2i = self.calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, k)
        mAPi2i = self.calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, k)
        mAPt2t = self.calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, k)
        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            if not self.distributed or (self.distributed and self.device == 0):
                self.save_mat(query_img, query_txt, self.query_labels, retrieval_img, retrieval_txt, self.retrieval_labels, save_file=os.path.join(save_dir, "i2t-best.mat"))
                self.save_model(save_dir=self.save_dir, epoch=epoch)
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            if not self.distributed or (self.distributed and self.device == 0):
                self.save_mat(query_img, query_txt, self.query_labels, retrieval_img, retrieval_txt, self.retrieval_labels, save_file=os.path.join(save_dir, "t2i-best.mat"))
                self.save_model(save_dir=self.save_dir, epoch=epoch)
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)

        if not self.distributed or (self.distributed and self.device == 0):
            self.save_mat(query_img, query_txt, self.query_labels, retrieval_img, retrieval_txt, self.retrieval_labels, save_file=os.path.join(save_dir, "last.mat"))

        self.logger.info(f">>>>>> [{epoch}/{self.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, "\
                    f"MAX MAP(i->t): {self.max_mapi2t}, epoch: {self.best_epoch_i}, MAX MAP(t->i): {self.max_mapt2i}, epoch: {self.best_epoch_t}")
    
    def test(self):
        assert not self.model_state == "", "test step must provide the model file!"
        self.logger.info("Test.")
        self.change_state(mode="valid")
        save_dir = os.path.join(self.save_dir, "mat_files")
        os.makedirs(save_dir, exist_ok=True)

        query_img, query_txt = self.get_code(self.query_loader, self.query_num)
        retrieval_img, retrieval_txt = self.get_code(self.retrieval_loader, self.retrieval_num)

        mAPi2t = self.calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, self.top_k)
        # print("map map")
        mAPt2i = self.calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, self.top_k)
        mAPi2i = self.calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, self.top_k)
        mAPt2t = self.calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, self.top_k)
        self.save_mat(query_img, query_txt, self.query_labels, retrieval_img, retrieval_txt, self.retrieval_labels, save_file=os.path.join(save_dir, "test.mat"))
        self.logger.info(f">>>>>> TEST, MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}")
    
    def print_loss_dict(self, loss_dict, bits=16, epoch=0, times=0):

        print_str = f">>>>>> Display ({self.loss_type} loss-{bits}) >>>>>> [{epoch}/{self.epochs}], [{times}/{len(self.train_loader)}]: "

        def leaf_str(dict_, key, print_str):
            
            print_str += f"{key}: "
            if isinstance(dict_[key], dict):
                for k in dict_[key]:
                    print_str = leaf_str(dict_[key], k, print_str)
            else:
                print_str += f"{dict_[key]}, "
            return print_str
        
        for key in loss_dict.keys():
            print_str += leaf_str(loss_dict, key=key, print_str="")
        
        print_str += f"lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}"
        self.logger.info(print_str)
    
    def save_model(self, save_dir, epoch, other=""):
        """
        I haven't write the load_state() code. It will be added in the furture. 2023/10/19
        """
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model-" + other + str(epoch) + ".pth"))
        self.logger.info("save mode to {}".format(os.path.join(save_dir, "model-" + other + str(epoch) + ".pth")))
    
    @classmethod
    def save_mat(cls, query_img, query_txt, query_labels, retrieval_img, retrieval_txt, retrieval_labels, save_file="i2t"):

        if isinstance(query_img, torch.Tensor):
            query_img = query_img.cpu().detach().numpy()
            query_txt = query_txt.cpu().detach().numpy()
            retrieval_img = retrieval_img.cpu().detach().numpy()
            retrieval_txt = retrieval_txt.cpu().detach().numpy()
            query_labels = query_labels.numpy()
            retrieval_labels = retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_file), result_dict)
    
    @classmethod
    def make_hash_code(cls, code):
        # print("aaaaaaaa")
        return code.sign_()
    
    @classmethod
    def from_config(cls, cfg, logger=None):
        raise NotImplementedError()

