# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
from PIL import Image

import json

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


from utils import smpl_to_openpose

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(dataset='openpose', data_folder='data', **kwargs):
    if dataset.lower() == 'openpose':
        return OpenPose(data_folder, **kwargs)
    elif dataset.lower() == 'animal':
        return AnimalData(data_folder)
    elif dataset.lower() == 'video_animal':
        return VideoAnimalData(data_folder)
    elif dataset.lower() == 'image_animal':
        return ImageAnimalData(data_folder)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt)




class AnimalData(Dataset):
    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 dtype=torch.float32,
                 model_type='smplx'):
        super(AnimalData, self).__init__()
        self.model_type = model_type
        self.dtype = dtype
        self.img_folder = osp.join(data_folder, img_folder)
        self.keyp_folder = osp.join(data_folder, keyp_folder)
        self.img_dir_paths = [osp.join(self.img_folder, img_fn)
                          for img_fn in os.listdir(self.img_folder)
                          if not img_fn.startswith('.')]
        self.img_dir_paths = sorted(self.img_dir_paths)
        self.cnt = 0

    def __len__(self):
        return len(self.img_dir_paths)

    def __getitem__(self, idx):
        img_dir_path = self.img_dir_paths[idx]
        return self.read_item(img_dir_path)

    def read_item(self, img_dir_path):
        img_paths = [osp.join(img_dir_path, img_fn)
                     for img_fn in os.listdir(img_dir_path)
                     if img_fn.endswith('.png') or
                     img_fn.endswith('.jpg') and not img_fn.startswith('.')]

        cam_names = []
        keypoints = []
        imgs = []

        for img_path in img_paths:

            img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
            cam_name, _ = osp.splitext(osp.split(img_path)[1])
            snapshot_name = osp.split(img_dir_path)[-1]
            keypoint_fn = osp.join(self.keyp_folder,snapshot_name,cam_name + '_keypoints.json')
            keyp_tuple = read_keypoints(keypoint_fn)
            if len(keyp_tuple.keypoints) < 1:
                return {}
            individual_keypoints = np.stack(keyp_tuple.keypoints)
            cam_names.append(cam_name)
            keypoints.append(individual_keypoints)
            imgs.append(img)

        output_dict = {'cam_names': [cam_names],
                        'snapshot_name': osp.split(osp.split(img_paths[0])[0])[-1],
                        'keypoints': [keypoints],
                        'imgs': [imgs]}
        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt >= len(self.img_dir_paths):
            raise StopIteration
        img_dir_path = self.img_dir_paths[self.cnt]
        self.cnt += 1
        return self.read_item(img_dir_path)
        

class VideoAnimalData(Dataset):
    # returns data for one timestamp (images, keypoints)
    def __init__(self, data_folder,
                 fps=30,
                 dtype=torch.float32):
        super(VideoAnimalData, self).__init__()

        self.data_folder = data_folder
        self.vid_paths = [osp.join(self.data_folder, vid_fn) for vid_fn in os.listdir(self.data_folder) if (vid_fn.endswith('.MP4') or vid_fn.endswith('.mp4'))]
        self.vid_paths = sorted(self.vid_paths)
        self.vidcaps = [cv2.VideoCapture(vid_path) for vid_path in self.vid_paths]
        self.cam_names = [vid_path.split('.')[-2].split('/')[-1] for vid_path in self.vid_paths]

        self.kp_paths = [vid_path.split('.')[-2]+'.json' for vid_path in self.vid_paths]
        self.kps = []
        for kp_path in self.kp_paths:
            with open(kp_path, 'r') as f:
                self.kps.append(json.load(f))

        self.pose_paths = [vid_path.split('.')[-2]+'_pose.json' for vid_path in self.vid_paths]
        self.poses = []
        for pose_path in self.pose_paths:
            with open(pose_path, 'r') as f:
                self.poses.append(json.load(f))

        self.cnt = 0

    def __len__(self):
        return len(self.kps[0])

    def __getitem__(self, idx):
        timestamp = list(self.kps[0].keys())[idx]
        return self.read_item(timestamp)

    def read_item(self, timestamp):
        keypoints = []
        poses = []
        imgs = []
        for vidcap, kp, pose in zip(self.vidcaps, self.kps, self.poses):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp)*1000/30)
            success,image = vidcap.read()
            if not success:
                break
            #im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            im = cv2.resize(im, (1280,720)) / 255.0
            imgs.append(im)
            keypoints.append(np.array([kp[timestamp]['zebra_0']]))
            poses.append(np.array([[float(el) for el in pose[timestamp]]]))
        output_dict = {'cam_names': [self.cam_names],
                        'snapshot_name': timestamp,
                        'keypoints': [keypoints],
                        'cam_poses': [poses],
                        'imgs': [imgs]}
        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt >= len(self.kps[0]):
            raise StopIteration
        timestamp =list(self.kps[0].keys())[self.cnt]
        self.cnt += 1
        return self.read_item(timestamp)



class ImageAnimalData(Dataset):
    def __init__(self, data_folder,
                 fps=30,
                 dtype=torch.float32):
        super(ImageAnimalData, self).__init__()

        self.data_folder = data_folder
        self.cam_names = os.listdir(osp.join(self.data_folder, "images/"))
        self.cam_names = sorted(self.cam_names)
        #self.vid_paths = sorted(self.vid_paths)
        #self.cam_names = [vid_path.split('/')[-1] for vid_path in self.vid_paths]

        self.kp_paths = [osp.join(self.data_folder, cam_name+".json") for cam_name in self.cam_names]
        self.kps = []
        for kp_path in self.kp_paths:
            with open(kp_path, 'r') as f:
                self.kps.append(json.load(f))

        self.pose_paths = [osp.join(self.data_folder, cam_name+"_pose.json") for cam_name in self.cam_names]
        self.poses = []
        for pose_path in self.pose_paths:
            with open(pose_path, 'r') as f:
                self.poses.append(json.load(f))

        self.cnt = 0

    def __len__(self):
        return len(self.kps[0])

    def __getitem__(self, idx):
        timestamp = list(self.kps[0].keys())[idx]
        return self.read_item(timestamp)

    def read_item(self, timestamp):
        keypoints = []
        poses = []
        imgs = []
        for cam_name, kp, pose in zip(self.cam_names, self.kps, self.poses):
            image = cv2.imread(osp.join(self.data_folder, "images/", cam_name, str(timestamp)+".jpg"))
            im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            im = cv2.resize(im, (1280,720)) / 255.0
            imgs.append(im)
            keypoints.append(np.array([kp[timestamp]['zebra_0']]))
            poses.append(np.array([[float(el) for el in pose[timestamp]]]))
        output_dict = {'cam_names': [self.cam_names],
                        'snapshot_name': timestamp,
                        'keypoints': [keypoints],
                        'cam_poses': [poses],
                        'imgs': [imgs]}
        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt >= len(self.kps[0]):
            raise StopIteration
        timestamp =list(self.kps[0].keys())[self.cnt]
        self.cnt += 1
        return self.read_item(timestamp)
