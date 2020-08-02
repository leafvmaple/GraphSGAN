import pickle as pkl
from enum import Enum

from model.GraphSGAN import GraphSGAN
from data_loader import load_data

class args(Enum):
    momentum       = 0.5
    lr             = 0.003
    tempdir        = "temp"
    epochs         = 100
    batch_size     = 100
    unlabel_weight = 1

if __name__ == "__main__":
    adj, features, labels, idx_train, idx_val, idx_test = load_data('cora')
    gan = GraphSGAN(adj, features, labels, idx_train, idx_val, idx_test, args, None)
    gan.train()
