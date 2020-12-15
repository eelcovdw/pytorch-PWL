import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.distributions import kl_divergence, Normal, Bernoulli, RelaxedBernoulli
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

from vae import build_model, train_step_arm, train_step_grep, train_step_arm_2, train_step_disarm

def main(args):
    latent_size = args.latent_size
    latent_dist = args.latent_type
    estimator = args.estimator
    output_dir = f'./runs/fashion/{latent_dist}-{estimator}-{latent_size}'
    writer = SummaryWriter(output_dir)
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'batch_size': args.batch_size}
    transform=transforms.ToTensor()
    dataset1 = datasets.FashionMNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.FashionMNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    
    
    model, train_step = build_model(latent_size, latent_dist, estimator)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters())
    print(model)

    step = 0
    for epoch in range(args.num_epochs):
        model.train()
        pbar = tqdm(train_loader)
        for bx, _ in pbar:
            step += 1
            bx = Bernoulli(probs=bx).sample()

            opt.zero_grad()
            bx = bx.to(device).view(-1, 28*28)
            out_dict = train_step(model, bx)

            out_dict['loss'].backward()
            opt.step()
            pbar.set_postfix(ll=out_dict['ll'].detach().cpu().numpy().mean(), kl=out_dict['kl'].detach().cpu().numpy().mean(), refresh=False)
            writer.add_scalar('train/ll', out_dict['ll'].detach().cpu().numpy().mean(), step)
            writer.add_scalar('train/kl', out_dict['ll'].detach().cpu().numpy().mean(), step)
        
        model.eval()
        total_size = 0
        total_ll = 0
        total_kl = 0
        for bx, by in test_loader:
            with torch.no_grad():
                bx = Bernoulli(probs=bx).sample().to(device)
                bx = bx.to(device).view(-1, 28*28)
                out_dict = train_step(model, bx)
                total_ll += out_dict['ll'].cpu().sum()
                total_kl += out_dict['kl'].cpu().sum()
                total_size += bx.size(0)
        print("{} VAL ll {:.3f}\tkl {:.3f}".format(epoch, total_ll / total_size, total_kl / total_size))
        writer.add_scalar('val/ll', total_ll / total_size, step)
        writer.add_scalar('val/kl',total_kl / total_size, step)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", help="Number of epochs", type=int, default=200)
    parser.add_argument("--batch_size", help="Minibatch size", type=int, default=128)
    parser.add_argument("--latent_size", help="size of z", type=int, default=64)
    parser.add_argument("--latent_type", help="Type of prior and posterior.", default="bernoulli")
    parser.add_argument("--estimator", help="Estimator used, sfe|grep|arm|disarm", default="grep")
    args = parser.parse_args()
    for arg in args.__dict__:
        print("{}: {}".format(arg, getattr(args, arg)))
    main(args)
