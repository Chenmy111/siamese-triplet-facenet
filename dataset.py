from __future__ import print_function

import torch
from PIL import Image
import torchvision.datasets as datasets
import os
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data.sampler import BatchSampler



class SiameseFaceDataset(datasets.ImageFolder):
    def __init__(self, dir, transform=None, train=True, *args, **kw):
        super(SiameseFaceDataset, self).__init__(dir, transform)
        self.train = train
        self.img_path = []
        for path, _ in self.imgs:
            self.img_path.append(path)
        self.labels_set = set(self.targets)
        self.label_to_indices = {label: np.where(np.array(self.targets) == label)[0] for label in self.labels_set}

        if not train:
            random_state = np.random.RandomState(29)
            positive_pairs = [[i, random_state.choice(self.label_to_indices[self.targets[i]]), 1] for i in
                              range(0, len(self.targets), 2)]
            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[np.random.choice(
                                   list(self.labels_set - set([self.targets[i]])))]),
                               0]
                              for i in range(1, len(self.targets), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        def img_path_to_img(img_path):
            img = self.loader(img_path)
            return img

        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = img_path_to_img(self.img_path[index]), self.targets[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = img_path_to_img(self.img_path[siamese_index])
        else:
            img1 = img_path_to_img(self.img_path[self.test_pairs[index][0]])
            img2 = img_path_to_img(self.img_path[self.test_pairs[index][1]])
            target = self.test_pairs[index][2]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.targets)


class TripletFaceDataset(datasets.ImageFolder):
    def __init__(self, dir, transform=None, train=True, *arg, **kw):
        super(TripletFaceDataset, self).__init__(dir, transform)
        # labels即self.targets
        self.train = train
        self.img_path = []
        for path, _ in self.imgs:
            self.img_path.append(path)
        self.labels_set = set(self.targets)
        self.label_to_indices = {label: np.where(np.array(self.targets) == label)[0] for label in self.labels_set}

        if not train:
            random_state = np.random.RandomState(29)
            triplets = [[i, random_state.choice(self.label_to_indices[self.targets[i]]), random_state.choice(
                self.label_to_indices[np.random.choice(list(self.labels_set - set([self.targets[i]])))])] for i in
                        range(len(self.targets))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        def img_path_to_img(img_path):
            img = self.loader(img_path)
            return img

        if self.train:
            img1, label1 = img_path_to_img(self.img_path[index]), self.targets[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = img_path_to_img(self.img_path[positive_index])
            img3 = img_path_to_img(self.img_path[negative_index])
        else:
            img1 = img_path_to_img(self.img_path[self.test_triplets[index][0]])
            img2 = img_path_to_img(self.img_path[self.test_triplets[index][1]])
            img3 = img_path_to_img(self.img_path[self.test_triplets[index][2]])

        # img1 = Image.fromarray(img1.numpy(), mode='L')
        # img2 = Image.fromarray(img2.numpy(), mode='L')
        # img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.targets)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(np.array(self.labels)))
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size