import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from utils.generator import Generator
from utils.discriminator import Discriminator
from utils.functional import log_sum_exp, pull_away_term

class GraphSGAN(object):
    def __init__(self, dataset, args):
        self.generator = Generator(200, dataset.k + dataset.d)
        self.discriminator = Discriminator(dataset.k + dataset.d, dataset.m)
        self.args = args
        self.dataset = dataset

        os.makedirs(args.tempdir)
        self.embedding_layer = nn.Embedding(dataset.n, dataset.d)
        self.embedding_layer.weight = Parameter(torch.Tensor(dataset.embbedings))
        torch.save(self.generator, os.path.join(args.tempdir, 'generator.pkl'))
        torch.save(self.discriminator, os.path.join(args.tempdir, 'discriminator.pkl'))
        torch.save(self.embedding_layer, os.path.join(args.tempdir, 'embedding.pkl'))

        self.generator.cuda()
        self.discriminator.cuda()
        self.discrim_optim = optim.Adam(self.discriminator.parameters(), lr=args.lr, betas = (args.momentum, 0.999))
        self.gen_optim = optim.Adam(self.generator.parameters(), lr=args.lr, betas = (args.momentum, 0.999))

    def train_discrim(self, idf_label, y, idf_unlabel):
        x_label   = self.make_input(*idf_label).cuda()
        x_unlabel = self.make_input(*idf_unlabel).cuda()
        y = Variable(y, requires_grad = False).cuda()

        output_label = self.discriminator(x_label)
        output_unlabel, mom_un = self.discriminator(x_unlabel, feature=True)
        output_fake = self.discriminator(self.generator(x_unlabel.size()[0]).view(x_unlabel.size()).detach())

        logz_label = log_sum_exp(output_label)
        logz_unlabel = log_sum_exp(output_unlabel)
        logz_fake = log_sum_exp(output_fake)

        prob_label = torch.gather(output_label, 1, y.unsqueeze(1))
        loss_supervised = -torch.mean(prob_label) + torch.mean(logz_label)
        loss_unsupervised = 0.5 * (-torch.mean(logz_unlabel) + torch.mean(F.softplus(logz_unlabel)) + torch.mean(F.softplus(logz_fake)) ) 
        entropy = -torch.mean(F.softmax(output_unlabel, dim = 1) * F.log_softmax(output_unlabel, dim = 1))
        pt = pull_away_term(mom_un)
        loss = loss_supervised + self.args.unlabel_weight * loss_unsupervised + entropy + pt
        acc = torch.mean((output_label.max(1)[1] == y).float())
        self.discrim_optim.zero_grad()
        loss.backward()
        self.discrim_optim.step()
        return loss_supervised.data.cpu().numpy(), loss_unsupervised.data.cpu().numpy(), acc
    
    def train_gen(self, idf_unlabel):
        x_unlabel = self.make_input(*idf_unlabel).cuda()
        fake = self.generator(x_unlabel.size()[0]).view(x_unlabel.size())
        output_fake, mom_gen = self.discriminator(fake, feature=True)
        output_unlabel, mom_unlabel = self.discriminator(x_unlabel, feature=True)
        loss_pt = pull_away_term(mom_gen)
        mom_gen = torch.mean(mom_gen, dim = 0)
        mom_unlabel = torch.mean(mom_unlabel, dim = 0) 
        loss_fm = torch.mean(torch.abs(mom_gen - mom_unlabel))
        loss = loss_fm + loss_pt 
        self.gen_optim.zero_grad()
        self.discrim_optim.zero_grad()
        loss.backward()
        self.gen_optim.step()
        return loss.data.cpu().numpy()

    def make_input(self, ids, feature, volatile = False):
        embedding = self.embedding_layer(Variable(ids, volatile = volatile)).detach()
        return torch.cat((Variable(feature), embedding), dim = 1)

    def train(self):
        NUM_BATCH = 100
        for epoch in range(self.args.epochs):
            self.generator.train()
            self.discriminator.train()
            self.discriminator.turn = epoch
            loss_supervised = loss_unsupervised = loss_gen = accuracy = 0.
            for batch_num in range(NUM_BATCH):
                idf_unlabel1 = self.dataset.unlabel_batch(self.args.batch_size)
                idf_unlabel2 = self.dataset.unlabel_batch(self.args.batch_size)
                id0, xf, y = self.dataset.label_batch(self.args.batch_size)

                ll, lu, acc = self.train_discrim((id0, xf), y, idf_unlabel1)
                loss_supervised += ll
                loss_unsupervised += lu
                accuracy += acc

                lg = self.train_gen(idf_unlabel2)
                loss_gen += lg

                if (batch_num + 1) % self.args.log_interval == 0:
                    print('Training: %d / %d' % (batch_num + 1, NUM_BATCH))
                    print('loss', {'loss_supervised':ll, 'loss_unsupervised':lu, 'loss_gen':lg})

            batch_num += 1
            loss_supervised /= batch_num
            loss_unsupervised /= batch_num
            loss_gen /= batch_num
            accuracy /= batch_num
            print("Iteration %d, loss_supervised = %.4f, loss_unsupervised = %.4f, loss_gen = %.4f train acc = %.4f" % (epoch, loss_supervised, loss_unsupervised, loss_gen, accuracy))

            tmp = self.eval()
            print("Eval: correct %d / %d, Acc: %.2f"  % (tmp, self.dataset.test_num, tmp * 100. / self.dataset.test_num))
            torch.save(self.generator, os.path.join(self.args.tempdir, 'generator.pkl'))
            torch.save(self.discriminator, os.path.join(self.args.tempdir, 'discriminator.pkl'))


    def predict(self, x):
        return torch.max(self.discriminator(x), 1)[1].data

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()
        ids, f, y = self.dataset.test_batch()
        x = self.make_input(ids, f, volatile = True)
        if self.args.cuda:
            x, y = x.cuda(), y.cuda()
        pred1 = self.predict(x)

        return torch.sum(pred1 == y)