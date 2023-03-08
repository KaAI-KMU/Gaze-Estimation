import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

class EstimationFileDataset(Dataset):
    def __init__(self, img_dir, depth_dir, landmark_dir, left_eye_dir, right_eye_dir, label_dir, face_transform=None, eye_transform=None):
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.landmark_dir = landmark_dir
        self.left_eye_dir = left_eye_dir
        self.right_eye_dir = right_eye_dir
        self.label_dir = label_dir

        self._face_transform = face_transform
        self._eye_transform = eye_transform
        if self._face_transform is None:
            self._face_transform = transforms.Compose([transforms.Resize((256,256), Image.BICUBIC),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if self._eye_transform is None:
            self._eye_transform = transforms.Compose([transforms.Resize((64,64), Image.BICUBIC),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        # load image
        img_path = os.path.join(self.img_dir, f"00000{idx:06d}_face.jpg")
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        # load depth image
        depth_path = os.path.join(self.depth_dir, f"00000{idx:06d}_depth.jpg")
        depth = Image.open(depth_path).convert('L')
        depth = np.array(depth)

        # load landmark data
        landmark_path = os.path.join(self.landmark_dir, f"00000{idx:06d}.txt")
        with open(landmark_path, "r") as f:
            landmark = f.read().split(',')
        landmark = landmark[:-1]

        # load left eye image
        left_eye_path = os.path.join(self.left_eye_dir, f"00000{idx:06d}_leye.jpg")
        left_eye = Image.open(left_eye_path).convert('RGB')
        left_eye = np.array(left_eye)

        # load right eye image
        right_eye_path = os.path.join(self.right_eye_dir, f"00000{idx:06d}_reye.jpg")
        right_eye = Image.open(right_eye_path).convert('RGB')
        right_eye = np.array(right_eye)

        # load label, no fix
        label_path = os.path.join(self.label_dir, f"00000{idx:06d}_label.txt")
        with open(label_path, "r") as f:
            label = f.read().strip()

        return {
            'image': img,
            'depth': depth,
            'landmark': landmark,
            'left_eye': left_eye,
            'right_eye': right_eye,
            'label': label
        }

# create dataset and dataloader objects
dataset = EstimationFileDataset(
    img_dir='path/to/image/directory',
    depth_dir='path/to/depth/directory',
    landmark_dir='path/to/landmark/directory',
    left_eye_dir='path/to/left/eye/directory',
    right_eye_dir='path/to/right/eye/directory',
    label_dir='path/to/label/directory'
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)