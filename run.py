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
from logger import Logger
import torchvision
from torchvision import datasets
from dataset import SiameseFaceDataset, TripletFaceDataset, BalancedBatchSampler
from networks import EmbeddingNet
from losses import OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from metrics import AverageNonzeroTripletsMetric

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
embedding_net = EmbeddingNet(embedding_size=embedding_size)
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 5
log_interval = 25
log_dir = './log'
optimizer_name = 'Adam'
LOG_DIR =log_dir + '/run-optim{}-lr{}-embbeding_size{}'.format(optimizer_name, lr, embedding_size)
logger = Logger(LOG_DIR)




train_dataset = datasets.ImageFolder(train_path, transform = transform)
test_dataset = datasets.ImageFolder(test_path, transform=transform)


# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(train_dataset.targets, n_classes=4, n_samples=2)
test_batch_sampler = BalancedBatchSampler(test_dataset.targets, n_classes=4, n_samples=2)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

print(len(train_loader))
print(len(test_loader))



fit(train_loader, test_loader, model, logger, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)