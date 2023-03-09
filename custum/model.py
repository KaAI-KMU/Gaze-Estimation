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
        left_x = self.left_feature(left_eye)
        left_x = nn.Flatten(left_x, 1)
        left_x = self.xl(left_x)

        right_x = self.right_feature(right_eye)
        right_x = nn.Flatten(right_x, 1)
        right_x = self.xr(right_x)

        eyes_x = torch.cat((left_x, right_x), dim=1)
        eyes_x = self.concat(eyes_x)

        eyes_headpose = torch.cat((eyes_x, head_pose), dim=1)

        fc_output = self.fc(eyes_headpose)
        return fc_output
    
    @staticmethod
    def _init_weights(modules):
        for md in modules:
            if isinstance(md, nn.Linear):
                nn.init.kaiming_uniform_(md.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(md.bias)

    
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

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self.xl, self.xr, self.concat, self.fc = GazeEstimationModel_fc._fc_layers(in_features=_left_eye_model.classifier[0].in_features,
                                                                                   out_features=num_out)
        GazeEstimationModel_fc._init_weights(self.modules())
    
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

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self.xl, self.xr, self.concat, self.fc = GazeEstimationModel_fc._fc_layers(in_features=_left_eye_model.fc.in_features,
                                                                                   out_features=num_out)
        GazeEstimationModel_fc._init_weights(self.modules())