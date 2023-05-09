# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import json
import os
import warnings

from continuum import ClassIncremental
# from continuum import Permutations
from continual.mycontinual import Rotations, IncrementalRotation, Permutations
from continuum.datasets import CIFAR100, MNIST, ImageNet100, ImageFolderDataset, CIFAR10, TinyImageNet200, STL10
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.transforms import functional as Fv

from typing import Tuple, Union

import numpy as np

from continuum.datasets import ImageFolderDataset
from continuum.download import download, unzip

try:
    interpolation = Fv.InterpolationMode.BICUBIC
except:
    interpolation = 3


from torch.utils.data import DataLoader
import torch.nn.functional as F
from argparse import Namespace
from copy import deepcopy
import torch
from PIL import Image
# from datasets.utils.validation import get_train_val
from typing import Tuple


class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set.lower() == 'cifar10':
        dataset = CIFAR10(args.data_path, train=is_train, download=True)
    elif args.data_set.lower() == 'cifar':
        dataset = CIFAR100(args.data_path, train=is_train, download=True)
    elif args.data_set.lower() == 'tinyimg':
        dataset = TinyImageNet200(args.data_path, train=is_train, download=True)
    elif args.data_set.lower() == 'imagenet100':
        dataset = ImageNet100_local(
            args.data_path, train=is_train,
            data_subset=os.path.join('./imagenet100_splits', "train_100.txt" if is_train else "val_100.txt")
        )
    elif args.data_set.lower() == 'imagenet1000':
        dataset = ImageNet1000(args.data_path, train=is_train)
    else:
        raise ValueError(f'Unknown dataset {args.data_set}.')

    scenario = ClassIncremental(
        dataset,
        initial_increment=args.initial_increment,
        increment=args.increment,
        transformations=transform.transforms,
        class_order=args.class_order
    )
    nb_classes = scenario.nb_classes #100

    return scenario, nb_classes


def build_transform(is_train, args):
    if args.aa == 'none':
        args.aa = None

    with warnings.catch_warnings():
        resize_im = args.input_size > 32
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
            if not resize_im:
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size, padding=4)

            if args.input_size == 32 and (args.data_set == 'CIFAR' or args.data_set == 'CIFAR10'):
                transform.transforms[-1] = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            elif args.data_set == 'STL10':
                transform.transforms[-1] = transforms.Normalize((0.4192, 0.4124, 0.3804), (0.2714, 0.2679, 0.2771))
            return transform

        t = []
        if resize_im and args.data_set != 'TINYIMG':
            size = int((256 / 224) * args.input_size)
            t.append(
                transforms.Resize(size, interpolation=interpolation),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        if args.input_size == 32 and (args.data_set == 'CIFAR' or args.data_set == 'CIFAR10'):
            # Normalization values for CIFAR100
            t.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))
        else:
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

        composed_transforms = transforms.Compose(t)
        return composed_transforms


class ImageNet1000_local(ImageFolderDataset):
    """ImageNet1000 dataset.

    Simple wrapper around ImageFolderDataset to provide a link to the download
    page.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if self.train:
        #     self.data_path = os.path.join(self.data_path, "train")
        # else:
        #     self.data_path = os.path.join(self.data_path, "val")

    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

    def _download(self):
        if not os.path.exists(self.data_path):
            raise IOError(
                "You must download yourself the ImageNet dataset."
                " Please go to http://www.image-net.org/challenges/LSVRC/2012/downloads and"
                " download 'Training images (Task 1 & 2)' and 'Validation images (all tasks)'."
            )
        print("ImageNet already downloaded.")


class ImageNet100_local(ImageNet1000_local):
    """Subset of ImageNet1000 made of only 100 classes.

    You must download the ImageNet1000 dataset then provide the images subset.
    If in doubt, use the option at initialization `download=True` and it will
    auto-download for you the subset ids used in:
        * Small Task Incremental Learning
          Douillard et al. 2020
    """

    train_subset_url = "https://github.com/Continvvm/continuum/releases/download/v0.1/train_100.txt"
    test_subset_url = "https://github.com/Continvvm/continuum/releases/download/v0.1/val_100.txt"

    def __init__(
            self, *args, data_subset: Union[Tuple[np.array, np.array], str, None] = None, **kwargs
    ):
        self.data_subset = data_subset
        super().__init__(*args, **kwargs)

    def _download(self):
        super()._download()

        filename = "val_100.txt"
        self.subset_url = self.test_subset_url
        if self.train:
            filename = "train_100.txt"
            self.subset_url = self.train_subset_url

        if self.data_subset is None:
            self.data_subset = os.path.join(self.data_path, filename)
            download(self.subset_url, self.data_path)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        data = self._parse_subset(self.data_subset, train=self.train)  # type: ignore
        return (*data, None)

    def _parse_subset(
            self,
            subset: Union[Tuple[np.array, np.array], str, None],
            train: bool = True
    ) -> Tuple[np.array, np.array]:
        if isinstance(subset, str):
            x, y = [], []

            with open(subset, "r") as f:
                for line in f:
                    split_line = line.split(" ")
                    path = split_line[0].strip()
                    x.append(os.path.join(self.data_path, path))
                    y.append(int(split_line[1].strip()))
            x = np.array(x)
            y = np.array(y)
            return x, y
        return subset  # type: ignore
