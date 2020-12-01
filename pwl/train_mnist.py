import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from logistic import DLogistic
from QLinear import QLinear, Quantizer
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical
import itertools

class MnistMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            QLinear(28 * 28, 256, Quantizer),
            nn.ReLU(),
            QLinear(256, 256, Quantizer),
            nn.ReLU(),
            QLinear(256, 10, Quantizer)
        ])

    def init_quantizers(self):
        for layer in self.layers:
            if isinstance(layer, QLinear):
                layer.quantizer.init_params()

    def forward(self, input, quantize=False, hard=False):
        h = input
        for layer in self.layers:
            if isinstance(layer, QLinear):
                h = layer(h, quantize=quantize, train_quantize=True)
            else:
                h = layer(h)
        return F.log_softmax(h, dim=-1)

    def qparams(self):
        qlayers = [l for l in self.layers if isinstance(l, QLinear)]
        return itertools.chain(*[l.quantizer.parameters() for l in qlayers])

    def get_qvals(self):
        with torch.no_grad():
            alphas = []
            betas = []
            scales = []
            for layer in self.layers:
                if isinstance(layer, QLinear):
                    alphas.append(layer.quantizer.alpha.view(1))
                    betas.append(layer.quantizer.beta.view(1))
                    scales.append(layer.quantizer.scale.view(1))
        return torch.cat(alphas), torch.cat(betas), torch.cat(scales)


def train(model, device, train_loader, optimizer, epoch, quantize):
    model.train()
    num_steps = 0
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        num_steps += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.view(data.size(0), -1), quantize)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach().cpu().item()
        if batch_idx % 50 == 0:
            loss = loss_sum / num_steps
            loss_sum = 0
            num_steps = 0
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))
            if quantize:
                for v in model.get_qvals():
                    print(v)


def test(model, device, test_loader, quantize):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(data.size(0), -1), quantize, hard=True)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'batch_size': 128}
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    
    model = MnistMLP().to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)

    quantize = False

    for epoch in range(2):
        train(model, device, train_loader, opt, epoch, quantize)
        test(model, device, test_loader, quantize)

    quantize = True
    model.init_quantizers()
    opt = optim.Adam(model.parameters(), lr=1e-4)
    for v in model.get_qvals():
        print(v)   
    for epoch in range(2):
        train(model, device, train_loader, opt, epoch, quantize)
        test(model, device, test_loader, quantize)
        for v in model.get_qvals():
            print(v)

if __name__ == "__main__":
    main()