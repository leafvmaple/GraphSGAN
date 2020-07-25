from torch
import pickle as pkl

from model.GraphSGAN import GraphSGAN

if __name__ == "__main__":
    torch.cuda.manual_seed(1)
    with open('cora.dataset', 'r') as fdata:
        dataset = pkl.load(fdata)

    args = {
        momentum = 0.5,
        lr = 0.003,
        tempdir = "temp",
        epochs = 100,
        batch_size = 100,
        unlabel_weight = 1,
    }

    gan = GraphSGAN(Generator(dataset, args)
    gan.train() 

