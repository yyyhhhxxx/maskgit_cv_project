import torch
import random
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms

from configs import FLAGS
#from datasets.dataset import TinyImageNet
from torch.utils.data.dataloader import DataLoader
from models.vqgan import VQGAN
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class VQGANSolver:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS          # config
        self.start_epoch = 1
        self.model = None           # torch.nn.Module
        self.optimizer_G = None     # torch.optim.Optimizer
        self.optimizer_D = None     # torch.optim.Optimizer
        self.scheduler_G = None     # torch.optim.lr_scheduler._LRScheduler
        self.scheduler_D = None     # torch.optim.lr_scheduler._LRScheduler
        self.summary_writer = None  # torch.utils.tensorboard.SummaryWriter
        self.train_loader = None    # dataloader for training dataset
        self.test_loader = None     # dataloader for testing dataset
        self.transform = None       # transform for dataset

        # choose device for train or test 
        if FLAGS.device < 0:       
           self.device = torch.device('cpu')
        else:
           self.device = torch.device(f'cuda:{FLAGS.device}')
        from torch.utils.tensorboard import SummaryWriter
        self.logger = SummaryWriter(FLAGS.logdir)
        # ......

    def config_model(self):
        self.model = VQGAN(vae_in_channels=self.FLAGS.model.vqvae.in_channels,
                           vae_out_channels=self.FLAGS.model.vqvae.out_channels,
                           vae_embedding_dim=self.FLAGS.model.vqvae.d_embedding,
                           vae_num_embeddings=self.FLAGS.model.vqvae.n_embedding,
                           vae_hidden_dims=self.FLAGS.model.vqvae.channels_list,
                           vae_beta=self.FLAGS.model.vqvae.beta,
                           img_size=32,
                           input_channel=self.FLAGS.model.D.input_channel,
                           ndf=self.FLAGS.model.D.ndf,
                           n_layers=self.FLAGS.model.D.n_layers)

    def get_dataset(self, flag):
        transform = transforms.Compose([
            transforms.ToTensor(),
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
        self.optimizer_G = OptimizerClass(self.model.vqvae.parameters(), lr = self.FLAGS.optimizer.vqvae.lr)
        self.optimizer_D = OptimizerClass(self.model.discriminator.parameters(), lr = self.FLAGS.optimizer.D.lr)

    def config_scheduler(self):
        SchedulerClass = getattr(torch.optim.lr_scheduler, self.FLAGS.scheduler.type)
        self.scheduler_G = SchedulerClass(self.optimizer_G,
                                          gamma=self.FLAGS.scheduler.vqvae.gamma, 
                                          step_size=self.FLAGS.scheduler.vqvae.step_size)
        self.scheduler_D = SchedulerClass(self.optimizer_D,
                                          gamma=self.FLAGS.scheduler.D.gamma, 
                                          step_size=self.FLAGS.scheduler.D.step_size)

    def train(self):
        self.manual_seed()
        self.config_model()
        self.config_dataloader()
        steps_one_epoch = len(self.train_loader)
        self.config_optimizer()
        self.config_scheduler()
        # set model as train mode
        self.model.train()
        self.model.to(self.device)

        for epoch in range(0, 10):

            train_losses_G = []
            Discloss = []
            VQVAEloss = []
            for i, (image_data, _) in enumerate(self.train_loader):
                # read data
                image_data = image_data.to(self.device)
                # model forward process
                x_hat, vq_loss, _ = self.model.vqvae(image_data)
                
                Real = self.model.discriminator(image_data)
                Fake = self.model.discriminator(x_hat.detach())
                # compute D-loss
                D_loss = -torch.mean(torch.log(Real + 1e-12) + torch.log(1 - Fake + 1e-12))
                # Optimize Discriminator
                for _ in range(3):
                    self.optimizer_D.zero_grad()
                    D_loss.backward(retain_graph=True)
                    self.optimizer_D.step()
                # compute G-loss
                rec_loss = F.mse_loss(x_hat, image_data)
                vq_vae_loss = vq_loss + rec_loss
                disloss = -torch.mean(torch.log(Fake + 1e-12))
                G_loss = disloss + vq_vae_loss
                Discloss.append(disloss.item())
                VQVAEloss.append(vq_vae_loss.item())
                train_losses_G.append(G_loss.item())
                # print loss
                if i % 200 == 199:
                    print(f'[{epoch + 1}, {i + 1:5d}] vqvae_loss: {np.mean(VQVAEloss):.3f} discriminator_loss: {np.mean(Discloss):.3f} generator_loss: {np.mean(train_losses_G):.3f}')
                    train_losses_G = []
                    Discloss = []
                    VQVAEloss = []
                # Optimize Generator
                self.optimizer_G.zero_grad()
                G_loss.backward()
                self.optimizer_G.step()
                
                self.logger.add_scalar("Loss/VQ loss", np.round(G_loss.cpu().detach().numpy().item(), 5), (epoch * steps_one_epoch) + i)
                self.logger.add_scalar("Loss/Q loss", np.round(vq_loss.cpu().detach().numpy().item(), 5), (epoch * steps_one_epoch) + i)
                # self.logger.add_scalar("Loss/perceptual loss", np.round(perceptual_loss.cpu().detach().numpy().mean(), 5), (epoch * steps_one_epoch) + i)
                self.logger.add_scalar("Loss/rec loss", np.round(rec_loss.cpu().detach().numpy().item(), 5), (epoch * steps_one_epoch) + i)
                self.logger.add_scalar("Loss/dis loss", np.round(disloss.cpu().detach().numpy().item(), 5), (epoch * steps_one_epoch) + i)

                self.logger.add_scalar("Loss/GAN loss", np.round(D_loss.cpu().detach().numpy().item(), 5), (epoch * steps_one_epoch) + i)
                from torchvision import utils as vutils
                import os
                os.makedirs(self.FLAGS.logdir, exist_ok=True)
                if i % 500 == 499:
                    with torch.no_grad():
                        both = torch.cat((image_data[:4], x_hat.add(1).mul(0.5)[:4]))
                        vutils.save_image(both, os.path.join(self.FLAGS.logdir, f"{epoch}_{i}.jpg"), nrow=4)
                # compute adaptive coefficient lambda before vae_loss
            self.scheduler_G.step()
            self.scheduler_D.step()
        # save model
        # torch.save(self.model.state_dict(), 'vqgan.pth')




    def test(self):
        self.config_model()
        self.config_dataloader(True)

        image_data, _ = next(iter(self.test_loader))
        image_data = image_data.to(self.device)

        self.model.eval()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('vqgan.pth'))

        # model forward process
        x_hat, vq_loss, _ = self.model.vqvae(image_data)
        # compute loss
        recon_loss = F.mse_loss(x_hat, image_data)
        loss = recon_loss + vq_loss
        print(loss.item())
        # compare original image and reconstructed image
        new_x = torch.cat([image_data, x_hat.detach()], dim=0)
        new_x = new_x.to('cpu')
        grid = make_grid(new_x, nrow=10)
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
    VQGANSolver.main()
    
