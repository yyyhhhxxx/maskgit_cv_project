import torch
import random
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms

from configs import FLAGS
#from datasets.dataset import TinyImageNet
from torch.utils.data.dataloader import DataLoader
from models.vqvae import VQVAE
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class VQVAESolver:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS          # config
        self.start_epoch = 1
        self.model = None           # torch.nn.Module
        self.optimizer = None       # torch.optim.Optimizer
        self.scheduler = None       # torch.optim.lr_scheduler._LRScheduler
        self.summary_writer = None  # torch.utils.tensorboard.SummaryWriter
        self.train_loader = None    # dataloader for training dataset
        self.test_loader = None     # dataloader for testing dataset
        self.transform = None       # transform for dataset

        # choose device for train or test 
        if FLAGS.device < 0:       
           self.device = torch.device('cpu')
        else:
           self.device = torch.device(f'cuda:{FLAGS.device}')
        # ......

    def config_model(self):
        self.model = VQVAE(in_channels=self.FLAGS.model.in_channels, 
                           out_channels=self.FLAGS.model.out_channels,
                           embedding_dim=self.FLAGS.model.d_embedding, 
                           num_embeddings=self.FLAGS.model.n_embedding, 
                           hidden_dims=self.FLAGS.model.channels_list, 
                           beta=self.FLAGS.model.beta, 
                           img_size=32)

    def get_dataset(self, flag):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        if flag == 'train':
            dataset = datasets.CIFAR100('cifar100',
                                     train=True,
                                     download=False,
                                     transform=transform)
                                              
        elif flag == 'test':
            dataset = datasets.CIFAR100('cifar100',
                                     train=False,
                                     download=False,
                                     transform=transform)
        else:
            raise ValueError(f'Invalid dataset flag {flag}')
        
        return dataset
    
    def config_dataloader(self, disable_train=False):
        
        if not disable_train:
            self.train_loader = DataLoader(self.get_dataset('train'), 
                                           batch_size=FLAGS.loader.train_batch_size,
                                           shuffle=True,
                                           num_workers=FLAGS.loader.num_workers,
                                           pin_memory=True)
        self.test_loader = DataLoader(self.get_dataset('test'), 
                                      batch_size=FLAGS.loader.test_batch_size,
                                      shuffle=False,
                                      num_workers=FLAGS.loader.num_workers,
                                      pin_memory=True)

    def config_optimizer(self):
        OptimizerClass = getattr(torch.optim, self.FLAGS.optimizer.type)
        self.optimizer = OptimizerClass(self.model.parameters(), lr = self.FLAGS.optimizer.base_lr)


    def config_scheduler(self):
        SchedulerClass = getattr(torch.optim.lr_scheduler, self.FLAGS.scheduler.type)
        self.scheduler = SchedulerClass(self.optimizer, gamma=self.FLAGS.scheduler.gamma, step_size=self.FLAGS.scheduler.step_size)

    def train(self):
        self.manual_seed()
        self.config_model()
        self.config_dataloader()
        self.config_optimizer()
        self.config_scheduler()
        # set model as train mode
        self.model.train()
        self.model.to(self.device)

        for epoch in range(0, 5):

            train_losses = []
            for i, (image_data, _) in enumerate(self.train_loader):
                # read data
                image_data = image_data.to(self.device)
                # model forward process、
                x_hat, vq_loss, _ = self.model(image_data)
                # compute loss

                recon_loss = F.mse_loss(x_hat, image_data)
                loss = recon_loss + 10 * vq_loss
                train_losses.append(loss.item())
                if i % 400 == 399:    # print every 100 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {sum(train_losses) / len(train_losses):.3f}')
                    train_losses = []

                # compute gradient
                self.optimizer.zero_grad()
                loss.backward()
                # optimize parameters
                self.optimizer.step()
                # update learning rate
            self.scheduler.step()
        # save model
        torch.save(self.model.state_dict(), 'vqvae.pth')




    def test(self):
        self.config_model()
        self.config_dataloader(True)

        image_data, _ = next(iter(self.test_loader))
        image_data = image_data.to(self.device)

        self.model.eval()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('vqvae.pth'))

        # model forward process
        x_hat, vq_loss, _ = self.model(image_data)
        # compute loss
        recon_loss = F.mse_loss(x_hat, image_data)
        loss = recon_loss + vq_loss
        print(loss.item())
        # compare original image and reconstructed image
        new_x = torch.cat([image_data, x_hat.detach()], dim=0)
        new_x = new_x.to('cpu')
        grid = make_grid(new_x, nrow=10)

        grid = (grid  * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        plt.imshow(grid.permute(1, 2, 0).squeeze())
        plt.show()

    def manual_seed(self):
        rand_seed = self.FLAGS.rand_seed
        if rand_seed > 0:
            random.seed(rand_seed)
            np.random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def run(self):
        eval('self.%s()' % self.FLAGS.run)

    @classmethod
    def main(cls):
        completion = cls(FLAGS)
        completion.run()

if __name__ == '__main__':
    VQVAESolver.main()
    
