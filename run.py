# Set up data loaders
from dataset import SiameseFaceDataset
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.optim import lr_scheduler
import torch.optim as optim
from trainer import fit
import numpy as np
import collections
# Set up the network and training parameters
from networks import EmbeddingNet, SiameseNet
from losses import ContrastiveLoss
from utils import Scale




transform = transforms.Compose([
                Scale(96),
                transforms.ToTensor(),
                transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                          std = [ 0.5, 0.5, 0.5 ])
                ])
cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_path = './data/train'
test_path = './data/val'
batch_size = 8
embedding_size = 16
margin = 1.

siamese_train_dataset = SiameseFaceDataset(dir=train_path, transform=transform)
siamese_test_dataset = SiameseFaceDataset(dir=test_path, transform=transform, train=False)



siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)




embedding_net = EmbeddingNet(embedding_size=embedding_size)
model = SiameseNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 10
log_interval = 100

fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)