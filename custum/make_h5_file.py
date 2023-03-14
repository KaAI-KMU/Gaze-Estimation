import h5py
import numpy as np
import os
from PIL import Image

orginal_dir = "custum/img_data/rgb_img/color_img"
img_dir="custum/img_data/rgb_img/faces"
depth_dir='custum/img_data/rgb_img/faces_depth'
landmark_dir='custum/img_data/rgb_img/facial_landmark'
left_eye_dir='custum/img_data/rgb_img/left_eyes'
right_eye_dir='custum/img_data/rgb_img/right_eyes'
label_dir='custum/img_data/rgb_img/csv/'

h5_file = h5py.File('custum/KaAI_dataset.hdf5', 'w') #'a'

h5_file.create_group("KaAI")

file_list = os.listdir(img_dir)
csv_max = os.listdir(orginal_dir)

# file_list_jpg = [os.path.splitext(file)[0] for file in file_list if os.path.splitext(file)[1] == ".jpg"]#.png

for i in file_list:
    i = i[0:6]
    h5_file.create_group("KaAI/{}/left_eye".format(i))
    h5_file.create_group("KaAI/{}/right_eye".format(i))
    h5_file.create_group("KaAI/{}/face_color".format(i))
    h5_file.create_group("KaAI/{}/face_depth".format(i))
    h5_file.create_group("KaAI/{}/label".format(i))
    h5_file.create_group("KaAI/{}/facial_landmark".format(i))

    rgb_img = Image.open('{}/{}_face.jpg'.format(img_dir,i))
    h5_file.create_dataset('KaAI/{}/face_color/{}.jpg'.format(i,i), data=rgb_img)
    
    depth_img = Image.open('{}/{}_face_depth.jpg'.format(depth_dir,i))
    depth_img = np.asarray(depth_img)
    h5_file.create_dataset('KaAI/{}/face_depth/{}.jpg'.format(i,i), data=depth_img)

    left_eye_img = Image.open('{}/{}_leye.jpg'.format(left_eye_dir,i))
    h5_file.create_dataset('KaAI/{}/left_eye/{}.jpg'.format(i,i), data=left_eye_img)

    right_eye_img = Image.open('{}/{}_reye.jpg'.format(right_eye_dir,i))
    h5_file.create_dataset('KaAI/{}/right_eye/{}.jpg'.format(i,i), data=right_eye_img)

    facial_landmark = open("{}/{}.txt".format(landmark_dir,i), 'r')
    facial_landmark_data = facial_landmark.read()
    facial_landmark_data = facial_landmark_data.split('\n')[:-1]
    h5_file.create_dataset("KaAI/{}/facial_landmark/{}_facial_landmark".format(i,i), data=facial_landmark_data)

    label = open("{}/driver.csv".format(label_dir), 'r')
    label_data = label.read()
    label_data = label_data.split('\n')
    label_data = label_data[:len(csv_max)+1]
    label_slice_data = [[index.split(',')]for index in label_data]
    h5_file.create_dataset("KaAI/{}/label/{}_label".format(i,i), data=label_slice_data)