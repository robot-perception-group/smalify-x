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
from collections import namedtuple
import torch
import torch.nn as nn
from smplx.lbs import transform_mat
from smplx.lbs import batch_rodrigues

PerspParams = namedtuple('ModelOutput',
                         ['rotation', 'translation', 'center',
                          'focal_length'])


def create_camera(camera_type='persp', **kwargs):
    if camera_type.lower() == 'persp':
        return PerspectiveCamera(**kwargs)
    else:
        raise ValueError('Uknown camera type: {}'.format(camera_type))


class PerspectiveCamera(nn.Module):
    FOCAL_LENGTH = 5000

    def __init__(self, rotation=None, translation=None, global_translation=None,
                 focal_length_x=None, focal_length_y=None,
                 batch_size=1,
                 center=None, dtype=torch.float32, **kwargs):
        super(PerspectiveCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype

        if focal_length_x is None or type(focal_length_x) == float:
            focal_length_x = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_x is None else
                focal_length_x,
                dtype=dtype)

        if focal_length_y is None or type(focal_length_y) == float:
            focal_length_y = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_y is None else
                focal_length_y,
                dtype=dtype)

        self.register_buffer('focal_length_x', focal_length_x)
        self.register_buffer('focal_length_y', focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)

        rotation_aa = torch.Tensor([0, 0, 0])
        rotation_aa = nn.Parameter(rotation_aa, requires_grad=True)
        self.register_parameter('rotation_aa', rotation_aa)

        rotation = batch_rodrigues(torch.unsqueeze(rotation_aa, 0))
        self.register_buffer('rotation', rotation)


        if global_translation is None:
            global_translation = torch.zeros([batch_size, 3], dtype=dtype)
        global_translation = nn.Parameter(global_translation, requires_grad=True)
        self.register_parameter('global_translation', global_translation)

        translation = -torch.matmul(self.rotation,self.global_translation.T).T # check!!
        self.register_buffer('translation', translation)

        '''if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)
        translation = nn.Parameter(translation, requires_grad=True)
        self.register_parameter('translation', translation)'''


        intrinsic = torch.Tensor([[focal_length_x, 0, center[0, 0]], [0, focal_length_y, center[0, 1]], [0, 0, 1]])
        self.register_buffer('intrinsic', intrinsic)


    def forward(self, points):

        self.intrinsic = torch.Tensor([[self.focal_length_x, 0, self.center[0, 0]], [0, self.focal_length_y, self.center[0, 1]], [0, 0, 1]])
        self.rotation = batch_rodrigues(torch.unsqueeze(self.rotation_aa, 0))
        self.translation = -torch.matmul(self.rotation,self.global_translation.T).T
        extr_intr_mul = torch.matmul(self.intrinsic, torch.cat((self.rotation[0], self.translation.view(3, -1)), dim=1))
        hom_points = torch.cat((points[0], torch.Tensor([1.0] * len(points[0])).unsqueeze(-1)), dim=1)
        points2d = torch.matmul(extr_intr_mul, hom_points.T).T


        return (points2d[:, :2] / points2d[:, 2:3]).unsqueeze(dim=0)  # + self.center.unsqueeze(dim=1)