import os
import numpy as np
import torch
from PIL import ImageFilter
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataloader import EstimationDataset

class TrainEstimation():
    def __init__(self, model):
        self._model = model