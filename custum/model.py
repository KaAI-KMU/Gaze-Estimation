import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GazeEstimationModel_fc(nn.Module):
    """Some Information about GazeEstimationModel_face"""
    def __init__(self):
        super(GazeEstimationModel_fc, self).__init__()

    @staticmethod
    def _fc_layers(in_features, out_features):
        x_l = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        x_r = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        concat = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        fc = nn.Sequential(
            nn.Linear(514, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_features)
        )

        return x_l, x_r, concat, fc

    def forward(self, left_eye, right_eye, head_pose):
        left_x = self.left_feature()
        return left_x
    
class GazeEstimationModel_vgg16(GazeEstimationModel_fc):
    """Some Information about GazeEstimationModel_vgg16"""
    def __init__(self, num_out=2):
        super (GazeEstimationModel_vgg16, self).__init__()
        _left_eye_model = models.vgg16(pretrained=True)
        _right_eye_model = models.vgg16(pretrained=True)

        _left_modules = [module for module in _left_eye_model.features]
        _left_modules.append(_left_eye_model.avgpool)
        self.left_features = nn.Sequential(*_left_modules)

        _right_modules = [module for module in _right_eye_model.features]
        _right_modules.append(_right_eye_model.avgpool)
        self.right_feature = nn.Sequential(*_right_modules)

    def forward(self, x):

        return x
    
class GazeEstiamationModel_resent18(GazeEstimationModel_fc):
    """Some Information about GazeEstiamationModel_resent18"""
    def __init__(self, num_out=2):
        super(GazeEstiamationModel_resent18, self).__init__()
        _left_eye_model = models.resnet18(pretrained=True)
        _right_eye_model = models.resnet18(pretrained=True)

        self.left_feature = nn.Sequential(
            _left_eye_model.conv1,
            _left_eye_model.bn1,
            _left_eye_model.relu,
            _left_eye_model.maxpool,
            _left_eye_model.layer1,
            _left_eye_model.layer2,
            _left_eye_model.layer3,
            _left_eye_model.layer4,
            _left_eye_model.avgpool
        )

        self.right_feature = nn.Sequential(
            _right_eye_model.conv1,
            _right_eye_model.bn1,
            _right_eye_model.relu,
            _right_eye_model.maxpool,
            _right_eye_model.layer1,
            _right_eye_model.layer2,
            _right_eye_model.layer3,
            _right_eye_model.layer4,
            _right_eye_model.avgpool
        )

    def forward(self, x):

        return x