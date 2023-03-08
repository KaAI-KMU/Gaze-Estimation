import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GazeEstimationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()