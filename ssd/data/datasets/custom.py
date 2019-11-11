import os
import torch.utils.data
import numpy as np
from PIL import Image

from ssd.structures.container import Container


class CustomDataset(torch.utils.data.Dataset):
    class_names = ('__background__', 'plane')

    def __init__(self, data_dir, transform=None, target_transform=None, is_train=True):
        self.transform = transform
        self.target_transform = target_transform
        self.data_dir = data_dir
        self.image_files = np.array([x.path for x in os.scandir(data_dir) if
                                     x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")])

    def __getitem__(self, index):
        image = Image.open(self.image_files[index]).convert("RGB")
        image = np.array(image)
        boxes, labels = self._get_annotation(index)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        return image, targets, index

    def __len__(self):
        return len(self.image_files)

    def _get_annotation(self, index):
        boxes = []
        labels = []
        fname = os.path.splitext(self.image_files[index])[0] + ".txt"
        objs = np.loadtxt(fname, dtype=np.float32)
        if len(objs.shape)<2:
            objs = np.array([objs])
        for obj in objs:
            x1 = obj[0] - 1
            y1 = obj[1] - 1
            x2 = obj[2] - 1
            y2 = obj[5] - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(1)
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))
