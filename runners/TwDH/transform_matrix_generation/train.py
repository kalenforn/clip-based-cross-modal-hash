from .model import Matrix
from .optimizer import BertAdam
from .dataset import build_dataloader

import os
import torch
from torch.utils.data import DataLoader

import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--index-file", type=str, default="./dataset/index.mat", help="index.mat")
    parser.add_argument("--caption-file", type=str, default="./dataset/caption.mat", help="caption.mat")
    parser.add_argument("--label-file", type=str, default="./dataset/label.mat", help="label.mat")
    parser.add_argument("--max-words", type=int, default=32, help="word embedding size.")
    parser.add_argument("--resolution", type=int, default=224, help="image resolution.")
    parser.add_argument("--query-num", type=int, default=5000, help="query set size")
    parser.add_argument("--train-num", type=int, default=10000, help="training dataset size")
    parser.add_argument("--seed", type=int, default=1814, help="seed")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--post-epochs", type=int, default=100)
    parser.add_argument("--long-dim", type=int, default=512, help="long center dim, please checking it is same as 'long-center' file!")
    parser.add_argument("--output-dim", type=int, default=16, help="short center dim, please checking it is same as 'short-center' file!")

    parser.add_argument("--post-lr", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=0.9)
    parser.add_argument("--clip-lr", type=float, default=0.00001)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")

    parser.add_argument("--long-center-path", type=str, default="./data/transformer/TwDH/coco/long/512.pkl", help="long-center path")
    parser.add_argument("--short-center-path", type=str, default="./data/transformer/TwDH/coco/short/16.pkl", help="short-center path")
    parser.add_argument("--save-dir", type=str, default="./")

    args = parser.parse_args()

    return args

def hash_center_multilables(labels, Hash_center):
    is_start = True
    random_center = torch.randint_like(Hash_center[0], 2)
    for label in labels:
        one_labels = (label == 1).nonzero() 
        one_labels = one_labels.squeeze(1)
        Center_mean = torch.mean(Hash_center[one_labels], dim=0)
        Center_mean[Center_mean<0] = -1
        Center_mean[Center_mean>0] = 1
        random_center[random_center==0] = -1   
        Center_mean[Center_mean == 0] = random_center[Center_mean == 0]  
        Center_mean = Center_mean.view(1, -1) 

        if is_start: 
            hash_center = Center_mean
            is_start = False
        else:
            hash_center = torch.cat((hash_center, Center_mean), 0)
    return hash_center

def hash_convert(hash_label):
    if len(hash_label.shape) == 2:
        result = torch.zeros([hash_label.shape[0], hash_label.shape[1] ,2])
        hash_label = (hash_label > 0).long()
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

def soft_argmax_hash_loss(code):
    if len(code.shape) < 3:
        code = code.view(code.shape[0], -1, 2)
    
    hash_loss = 1 - (torch.pow(code[:, :, 0] - code[:, :, 1], 2)).mean()
    return hash_loss

def check(long_file, low_file, transfor_file):
    long_code = torch.load(long_file)
    low_code = torch.load(low_file)
    transfor = torch.load(transfor_file)

    low_code[torch.where(low_code < 0)] = 0
    low_ = hash_convert(long_code).mm(transfor).view(low_code.shape[0], low_code.shape[1], 2)
    low_ = torch.argmax(low_, dim=-1)

    return torch.equal(low_.int(), low_code.int())

def train(args=get_args(), alpha=0.001):

    model = Matrix(input_dim=args.long_dim, output_dim=args.output_dim).to(0)

    args.index_file = os.path.join("./dataset", args.dataset, args.index_file)
    args.caption_file = os.path.join("./dataset", args.dataset, args.caption_file)
    args.label_file = os.path.join("./dataset", args.dataset, args.label_file)

    train_data, query_data, retrieval_data = build_dataloader(captionFile=args.caption_file, 
                                        indexFile=args.index_file, 
                                        labelFile=args.label_file, 
                                        maxWords=args.max_words,
                                        imageResolution=args.resolution,
                                        query_num=args.query_num,
                                        train_num=args.train_num,
                                        seed=args.seed,
                                        random_index=None)
    
    long_center = torch.load(args.long_center_path).float()
    low_center = torch.load(args.short_center_path).float()

    train_loader = DataLoader(
                dataset=train_data,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                shuffle=True
            )
    query_loader = DataLoader(
                dataset=query_data,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                shuffle=True
            )
    retrieval_loader = DataLoader(
                dataset=retrieval_data,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                shuffle=True
            )
    
    optimizer = BertAdam(model.parameters(), lr=args.post_lr, warmup=args.warmup_proportion, schedule='warmup_cosine', 
                    b1=0.9, b2=0.98, e=1e-6, t_total=len(train_loader) * args.post_epochs,
                    weight_decay=args.weight_decay, max_grad_norm=1.0)
    
    criterion = torch.nn.BCELoss()

    for epoch in range(args.epochs):
        for _, _, _, label, index in train_loader:
            long_hash = hash_convert(hash_center_multilables(label, long_center)).to(0, non_blocking=True).float()
            low_hash = hash_convert(hash_center_multilables(label, low_center)).to(0, non_blocking=True).float()
            target = model(long_hash)
            hash_loss = soft_argmax_hash_loss(target)

            class_loss = criterion(target, low_hash)
            lasso = alpha * model.matrix.norm(p=1)
            loss = hash_loss + class_loss + lasso

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("loss:", hash_loss.data, class_loss.data, lasso.data)
    
        matrix = model.matrix.cpu().detach()
        print("save matrix: ", matrix.shape, os.path.join(args.save_dir, str(args.long_dim), f"{args.output_dim}.pkl"))
        torch.save(matrix, os.path.join(args.save_dir, str(args.long_dim), f"{args.output_dim}.pkl"))
        if check(args.long_center_path, 
                 args.short_center_path, 
                 os.path.join(args.save_dir, str(args.long_dim), f"{args.output_dim}.pkl")):
            torch.save(matrix, os.path.join(args.save_dir, str(args.long_dim), f"{args.output_dim}_zeros.pkl"))
            print("find a lossless transform matrix!")
            break

    matrix = model.matrix.cpu().detach()
    print("save matrix: ", matrix.shape, os.path.join(args.save_dir, str(args.long_dim), f"{args.output_dim}.pkl"))
    torch.save(matrix, os.path.join(args.save_dir, str(args.long_dim), f"{args.output_dim}.pkl"))


if __name__ == "__main__":
    train()
