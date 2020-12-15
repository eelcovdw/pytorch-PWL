import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.distributions import kl_divergence, Normal, Bernoulli, RelaxedBernoulli
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from vae import *

def main():
    latent_size = 200
    latent_dist = 'bernoulli'
    output_dir = f'./runs/mnist/{latent_dist}Z-{latent_size}'
    writer = SummaryWriter(output_dir)
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'batch_size': 128}
    transform=transforms.ToTensor()
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    
    
    model = build_model(latent_size, latent_dist).to(device)
    if latent_dist == "bernoulli":
        train_step = train_step_disarm
    else:
        train_step = train_step_grep
    opt = optim.Adam(model.parameters(), lr=0.001)
    print(model)

    step = 0
    for epoch in range(100):
        model.train()
        pbar = tqdm(train_loader)
        for bx, by in pbar:
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
        print("VAL ll {:.3f}\tkl {:.3f}".format(total_ll / total_size, total_kl / total_size))
        writer.add_scalar('val/ll', total_ll / total_size, step)
        writer.add_scalar('val/kl',total_kl / total_size, step)
                

if __name__ == "__main__":
    main()
