import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file", type=str, default="configs/DCMHT/config.yaml", help="choices a hash model to run.")
    parser.add_argument("--save-dir", type=str, default="./result/DCMHT", help="save dir.")
    parser.add_argument("--device", type=str, default="0", help="only supports single GPU runing.")

    parser.add_argument("--seed", type=int, default=1814, help="The number of query dataset.")

    parser.add_argument("--distribute", action="store_true", default=False, help="Wethear utilize distributation training")

    args = parser.parse_args()

    return args

