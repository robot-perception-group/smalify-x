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

import time

import numpy as np

import torch
import torch.nn as nn

from mesh_viewer import MeshViewer
import utils

import itertools

@torch.no_grad()
def guess_init(model,
               joints_2d,
               edge_idxs,
               focal_length=5000,
               pose_embedding=None,
               vposer=None,
               use_vposer=True,
               dtype=torch.float32,
               model_type='smpl',
               **kwargs):
    ''' Initializes the camera translation vector

        Parameters
        ----------
        model: nn.Module
            The PyTorch module of the body
        joints_2d: torch.tensor 1xJx2
            The 2D tensor of the joints
        edge_idxs: list of lists
            A list of pairs, each of which represents a limb used to estimate
            the camera translation
        focal_length: float, optional (default = 5000)
            The focal length of the camera
        pose_embedding: torch.tensor 1x32
            The tensor that contains the embedding of V-Poser that is used to
            generate the pose of the model
        dtype: torch.dtype, optional (torch.float32)
            The floating point type used
        vposer: nn.Module, optional (None)
            The PyTorch module that implements the V-Poser decoder
        Returns
        -------
        init_t: torch.tensor 1x3, dtype = torch.float32
            The vector with the estimated camera location

    '''

    #body_pose = vposer.decode(
    #    pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
    body_pose = (vposer.decode(pose_embedding).get( 'pose_body')).reshape(1, -1) if use_vposer else None

    if use_vposer and model_type == 'smpl':
        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
        body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    #output = model(body_pose=body_pose, return_verts=False, return_full_pose=False)
    output = model(body_pose=body_pose, return_verts=True, return_full_pose=False)


    # SMALR method
    joints_2d = joints_2d / 3.75 # °°°°°°°°° fix this!

    use_ids0 = [18, 12, 13, 10, 11, 7, 8, 9]

    use_ids = [id for id in use_ids0 if (joints_2d[0][id,0]) and (joints_2d[0][id,1])]
    pairs = [p for p in itertools.combinations(use_ids, 2)]

    key_vids = [np.array([np.array([1068, 1080, 1029, 1226], dtype=np.uint16),
                          np.array([2660, 3030, 2675, 3038], dtype=np.uint16), np.array([910]),
                          np.array([360, 1203, 1235, 1230], dtype=np.uint16),
                          np.array([3188, 3156, 2327, 3183], dtype=np.uint16),
                          np.array([1976, 1974, 1980, 856], dtype=np.uint16),
                          np.array([3854, 2820, 3852, 3858], dtype=np.uint16),
                          np.array([452, 1811], dtype=np.uint16),
                          np.array([416, 235, 182], dtype=np.uint16),
                          np.array([2156, 2382, 2203], dtype=np.uint16), np.array([829]),
                          np.array([2793]), np.array([60, 114, 186, 59], dtype=np.uint8),
                          np.array([2091, 2037, 2036, 2160], dtype=np.uint16),
                          np.array([384, 799, 1169, 431], dtype=np.uint16),
                          np.array([2351, 2763, 2397, 3127], dtype=np.uint16),
                          np.array([221, 104], dtype=np.uint8), np.array([2754, 2192], dtype=np.uint16),
                          np.array([191, 1158, 3116, 2165], dtype=np.uint16),
                          np.array([28, 1109, 1110, 1111, 1835, 1836, 3067, 3068, 3069], dtype=np.uint16),
                          np.array([149, 150, 368, 542, 543, 544], dtype=np.uint16),
                          np.array([2124, 2125, 2335, 2507, 2508, 2509], dtype=np.uint16),
                          np.array([1873, 1876, 1877, 1885, 1902, 1905, 1906, 1909, 1920, 1924],
                                   dtype=np.uint16),
                          np.array([2963, 2964, 3754, 3756, 3766, 3788, 3791, 3792, 3802, 3805],
                                   dtype=np.uint16),
                          np.array([764, 915, 916, 917, 934, 935, 956], dtype=np.uint16),
                          np.array([2878, 2879, 2880, 2897, 2898, 2919, 3751], dtype=np.uint16),
                          np.array([795, 796, 1054, 1058, 1060], dtype=np.uint16),
                          np.array([2759, 2760, 3012, 3015, 3016, 3018], dtype=np.uint16)],
                         dtype=object)]

    def mean_model_point(row_id):
        return torch.mean(torch.index_select(output.vertices, 1, torch.tensor(key_vids[0][row_id].astype(np.int32)))[0], axis=0)


    dist3d = torch.Tensor([np.linalg.norm(mean_model_point(p[0]) - mean_model_point(p[1])) for p in pairs])
    dist2d = torch.Tensor([np.linalg.norm(joints_2d[0][p[0], :] - joints_2d[0][p[1], :]) for p in pairs])
    est_ds = focal_length * dist3d / dist2d


    '''joints_3d = output.joints
    joints_2d = joints_2d.to(device=joints_3d.device)

    diff3d = []
    diff2d = []

    #°°°°°°°°
    single_vert_indices = np.array([1068, 2660,  910,  360, 3188, 1976, 3854,  452,  416, 2156,  829, 2793,   60, 2091,  384, 2351,  221, 2754,  191,   28,  542, 2507, 1039,  0])
    joints_3d = torch.index_select(output.vertices, 1, torch.tensor(single_vert_indices))

    for edge in edge_idxs:
        diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
        diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])
        #print('°°°°°° joints_2d, _3d \n', edge, '\n', joints_3d[:, edge[0]], '\n', joints_3d[:, edge[1]], '\n')



    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)
    #print('°°°°°° diff3d \n', diff3d, '\n')

    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()
    #print('°°°°°° length_3d \n', length_3d, '\n')

    height2d = length_2d.mean(dim=1)
    height3d = length_3d.mean(dim=1)
    #print('°°°°°° height3d \n', height3d, '\n')

    est_d = focal_length * (height3d / height2d)

    # just set the z value'''
    batch_size = 1
    x_coord = torch.zeros([batch_size], device=output.joints.device, dtype=dtype)
    y_coord = x_coord.clone()
    init_t = torch.stack([x_coord, y_coord, torch.unsqueeze(torch.median(est_ds),0)], dim=1)
    #print('°°°°° init_t \n', init_t, '\n\n')

    return init_t


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        for n in range(self.maxiters):

            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            if self.visualize and n % self.summary_steps == 0:
                #body_pose = vposer.decode(
                #    pose_embedding, output_type='aa').view(
                #        1, -1) if use_vposer else None
                body_pose = (vposer.decode(pose_embedding).get( 'pose_body')).reshape(1, -1) if use_vposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                model_output = body_model(
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()

        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_joints=None, loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               create_graph=False,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            #body_pose = vposer.decode(
            #    pose_embedding, output_type='aa').view(
            #        1, -1) if use_vposer else None

            body_pose = (vposer.decode(pose_embedding).get( 'pose_body')).reshape(1, -1) if use_vposer else None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)
            total_loss = loss(body_model_output, camera=camera,
                              gt_joints=gt_joints,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              **kwargs)

            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),body_model.faces)

            return total_loss

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


class SMPLifyLoss(nn.Module):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 expr_prior=None,
                 angle_prior=None,
                 jaw_prior=None,
                 use_joints_conf=True,
                 use_face=True, use_hands=True,
                 left_hand_prior=None, right_hand_prior=None,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0, jaw_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 reduction='sum',
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior

        rho = 150.0 # from SMALR, is rho same as sig??
        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance

        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                                 torch.tensor(hand_prior_weight, dtype=dtype))
        if self.use_face:
            self.register_buffer('expr_prior_weight',
                                 torch.tensor(expr_prior_weight, dtype=dtype))
            self.register_buffer('jaw_prior_weight',
                                 torch.tensor(jaw_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, joints_conf,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embedding=None,
                **kwargs):

        # °°°°° Actual joints vs vertex-based pseudo-joints:

        # Actual joints
        #projected_joints = camera(body_model_output.joints)

        """key_vids = [np.array([np.array([1068, 1080, 1029, 1226]),
                             np.array([2660, 3030, 2675, 3038]), np.array([910]),
                             np.array([360, 1203, 1235, 1230]),
                             np.array([3188, 3156, 2327, 3183]),
                             np.array([1976, 1974, 1980, 856]),
                             np.array([3854, 2820, 3852, 3858]),
                             np.array([452, 1811]),
                             np.array([416, 235, 182]),
                             np.array([2156, 2382, 2203]), np.array([829]),
                             np.array([2793]), np.array([60, 114, 186, 59]),
                             np.array([2091, 2037, 2036, 2160]),
                             np.array([384, 799, 1169, 431]),
                             np.array([2351, 2763, 2397, 3127]),
                             np.array([221, 104]), np.array([2754, 2192]),
                             np.array([191, 1158, 3116, 2165]), np.array([28]),
                             np.array([542]), np.array([2507]),
                             np.array([1039, 1845, 1846, 1870, 1879, 1919, 2997, 3761, 3762]),
                             np.array([0, 464, 465, 726, 1824, 2429, 2430, 2690])], dtype=object)]"""

        key_vids = [np.array([np.array([1068, 1080, 1029, 1226], dtype=np.uint16),
                np.array([2660, 3030, 2675, 3038], dtype=np.uint16), np.array([910]),
                np.array([360, 1203, 1235, 1230], dtype=np.uint16),
                np.array([3188, 3156, 2327, 3183], dtype=np.uint16),
                np.array([1976, 1974, 1980, 856], dtype=np.uint16),
                np.array([3854, 2820, 3852, 3858], dtype=np.uint16),
                np.array([452, 1811], dtype=np.uint16),
                np.array([416, 235, 182], dtype=np.uint16),
                np.array([2156, 2382, 2203], dtype=np.uint16), np.array([829]),
                np.array([2793]), np.array([60, 114, 186, 59], dtype=np.uint8),
                np.array([2091, 2037, 2036, 2160], dtype=np.uint16),
                np.array([384, 799, 1169, 431], dtype=np.uint16),
                np.array([2351, 2763, 2397, 3127], dtype=np.uint16),
                np.array([221, 104], dtype=np.uint8), np.array([2754, 2192], dtype=np.uint16),
                np.array([191, 1158, 3116, 2165], dtype=np.uint16),
                np.array([28, 1109, 1110, 1111, 1835, 1836, 3067, 3068, 3069], dtype=np.uint16),
                np.array([149, 150, 368, 542, 543, 544], dtype=np.uint16),
                np.array([2124, 2125, 2335, 2507, 2508, 2509], dtype=np.uint16),
                np.array([1873, 1876, 1877, 1885, 1902, 1905, 1906, 1909, 1920, 1924],
                      dtype=np.uint16),
                np.array([2963, 2964, 3754, 3756, 3766, 3788, 3791, 3792, 3802, 3805],
                      dtype=np.uint16),
                np.array([764, 915, 916, 917, 934, 935, 956], dtype=np.uint16),
                np.array([2878, 2879, 2880, 2897, 2898, 2919, 3751], dtype=np.uint16),
                np.array([795, 796, 1054, 1058, 1060], dtype=np.uint16),
                np.array([2759, 2760, 3012, 3015, 3016, 3018], dtype=np.uint16)],
               dtype=object)]

        nCameras = 1

        j2d = [None] * nCameras
        kp_weights = [None] * nCameras
        assignments = [None] * nCameras
        num_points = [None] * nCameras
        use_ids = [None] * nCameras
        visible_vids = [None] * nCameras
        all_vids = [None] * nCameras

        i = 0

        landmarks_names = [['leftEye', 'rightEye', 'chin', 'frontLeftFoot', 'frontRightFoot',
               'backLeftFoot', 'backRightFoot', 'tailStart', 'frontLeftKnee',
               'frontRightKnee', 'backLeftKnee', 'backRightKnee', 'leftShoulder',
               'rightShoulder', 'frontLeftAnkle', 'frontRightAnkle',
               'backLeftAnkle', 'backRightAnkle', 'neck', 'TailTip', 'leftEar',
               'rightEar', 'noseTip', 'halfTail']]

        landmarks = [np.array([[  0.        ,   0.        ,   0.        ],
           [362.4       , 137.06666667,   1.        ],
           [356.8       , 187.73333333,   1.        ],
           [270.13333333, 257.86666667,   1.        ],
           [298.93333333, 257.33333333,   1.        ],
           [191.73333333, 258.4       ,   1.        ],
           [181.86666667, 261.06666667,   1.        ],
           [163.73333333, 124.26666667,   1.        ],
           [  0.        ,   0.        ,   0.        ],
           [275.2       , 176.26666667,   1.        ],
           [193.6       , 187.73333333,   1.        ],
           [169.86666667, 182.66666667,   1.        ],
           [  0.        ,   0.        ,   0.        ],
           [278.93333333, 147.46666667,   1.        ],
           [268.53333333, 222.93333333,   1.        ],
           [303.2       , 209.86666667,   1.        ],
           [181.86666667, 208.26666667,   1.        ],
           [172.53333333, 212.        ,   1.        ],
           [314.93333333, 130.93333333,   1.        ],
           [131.2       , 203.73333333,   1.        ],
           [  0.        ,   0.        ,   0.        ],
           [360.26666667,  96.8       ,   1.        ],
           [  0.        ,   0.        ,   0.        ],
           [364.53333333, 182.13333333,   1.        ],
           [  0.        ,   0.        ,   0.        ],
           [354.66666667, 172.26666667,   1.        ],
           [  0.        ,   0.        ,   0.        ],
           [360.8       , 181.6       ,   1.        ]])]

        visible = landmarks[i][:, 2].astype(bool)

        use_ids[i] = [id for id in np.arange(landmarks[i].shape[0]) if visible[id]]
        visible_vids[i] = np.hstack([key_vids[i][id] for id in use_ids[i]])

        group = np.hstack([index * np.ones(len(key_vids[i][row_id])) for index, row_id in enumerate(use_ids[i])])
        assignments[i] = np.vstack([group == j for j in np.arange(group[-1] + 1)])
        #assignments[i] = torch.Tensor(assignments[i])
        num_points[i] = len(use_ids[i])

        all_vids[i] = visible_vids[i]
        #cam[i].v = sv[i][all_vids[i], :]
        j2d[i] = torch.Tensor(landmarks[i][use_ids[i], :2])

        kp_weights[i] = np.ones((landmarks[i].shape[0], 1))
        kp_weights[i] = np.ones((landmarks[i].shape[0], 1))

        kp_weights[i][landmarks_names[i].index('leftEye'), :] *= 2.
        kp_weights[i][landmarks_names[i].index('rightEye'), :] *= 2.
        kp_weights[i][landmarks_names[i].index('leftEar'), :] *= 2.
        kp_weights[i][landmarks_names[i].index('rightEar'), :] *= 2.
        if 'noseTip' in landmarks_names[i]:
            kp_weights[i][landmarks_names[i].index('noseTip'), :] *= 2.

        kp_weights[i] = torch.Tensor(kp_weights[i])

        dummy_verts_to_project = torch.index_select(body_model_output.vertices, 1, torch.tensor(all_vids[0]))
        projected_joints = camera(torch.index_select(body_model_output.vertices, 1, torch.tensor(all_vids[0])))# / 3.75
        projected_joints = projected_joints[0]

        choice = assignments[0][0]
        dum_var0 = projected_joints[choice]

        dummy_var0 = torch.vstack([projected_joints[choice] if np.sum(choice) == 1 else projected_joints[choice].mean(axis=0) for choice in assignments[0]]) - j2d[0]
        dummy_var = self.robustifier(torch.vstack([projected_joints[choice] if np.sum(choice) == 1 else projected_joints[choice].mean(axis=0) for choice in assignments[0]]) - j2d[0])

        kp_proj = 1500.0 * kp_weights[i][use_ids[0]] * torch.sqrt(self.robustifier(torch.vstack([projected_joints[choice] if np.sum(choice) == 1 else projected_joints[choice].mean(axis=0) for choice in assignments[0]]) - j2d[0])) / np.sqrt(num_points[i])
        joint_loss = torch.sum(torch.square(kp_proj))





        '''# Single vertex-based pseudo-joints
        single_vert_indices = np.array([1068, 2660,  910,  360, 3188, 1976, 3854,  452,  416, 2156,  829, 2793,   60, 2091,  384, 2351,  221, 2754,  191,   28,  542, 2507, 1039,  0])
        projected_joints = camera(torch.index_select(body_model_output.vertices, 1, torch.tensor(single_vert_indices)))
        # Calculate the weights for each joints
        # °°°°°°°°°
        joint_weights = torch.index_select(joint_weights, 1, torch.from_numpy(np.array(range(24))))

        weights = (joint_weights * joints_conf if self.use_joints_conf else joint_weights).unsqueeze(dim=-1)
        #weights = joints_conf.unsqueeze(dim=-1)
        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        #°°°°°°
        projected_joints = weights ** 2 * projected_joints
        joint_diff = self.robustifier(gt_joints - projected_joints)
        joint_loss = (torch.sum(weights ** 2 * joint_diff) *
                      self.data_weight ** 2)'''









        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = self.body_pose_prior(body_model_output.body_pose) * self.body_pose_weight ** 2

        shape_loss = self.shape_prior(body_model_output.betas) * self.shape_weight ** 2
        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight

        # Apply the prior on the pose space of the hand
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        '''if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = torch.sum(
                self.left_hand_prior(
                    body_model_output.left_hand_pose)) * \
                self.hand_prior_weight ** 2

        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = torch.sum(
                self.right_hand_prior(
                    body_model_output.right_hand_pose)) * \
                self.hand_prior_weight ** 2'''

        expression_loss = 0.0
        jaw_prior_loss = 0.0
        if self.use_face:
            expression_loss = torch.sum(self.expr_prior(
                body_model_output.expression)) * \
                self.expr_prior_weight ** 2

            if hasattr(self, 'jaw_prior'):
                jaw_prior_loss = torch.sum(
                    self.jaw_prior(
                        body_model_output.jaw_pose.mul(
                            self.jaw_prior_weight)))

        pen_loss = 0.0
        # Calculate the loss due to interpenetration
        '''if (self.interpenetration and self.coll_loss_weight.item() > 0):
            batch_size = projected_joints.shape[0]
            triangles = torch.index_select(
                body_model_output.vertices, 1,
                body_model_faces).view(batch_size, -1, 3, 3)

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)

            # Remove unwanted collisions
            if self.tri_filtering_module is not None:
                collision_idxs = self.tri_filtering_module(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(
                    self.coll_loss_weight *
                    self.pen_distance(triangles, collision_idxs))'''

        #°°°°°°
        total_loss = joint_loss + shape_loss + pprior_loss #2500*joint_loss + 20*shape_loss + pprior_loss #+ angle_prior_loss
        #print('°°°°°° losses: \n\n', joint_loss, '\n', shape_loss, '\n', pprior_loss, '\n\n')
        #total_loss = (joint_loss + pprior_loss + shape_loss +
        #              angle_prior_loss + pen_loss +
        #              jaw_prior_loss + expression_loss +
        #              left_hand_prior_loss + right_hand_prior_loss)
        return total_loss


class SMPLifyCameraInitLoss(nn.Module):

    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2, dtype=torch.float32,
                 **kwargs):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints,
                **kwargs):

        #projected_joints = camera(body_model_output.joints)
        single_vert_indices = np.array([1068, 2660, 910, 360, 3188, 1976, 3854, 452, 416, 2156, 829, 2793, 60, 2091, 384, 2351, 221, 2754, 191, 28,542, 2507, 1039, 0])
        projected_joints = camera(torch.index_select(body_model_output.vertices, 1, torch.tensor(single_vert_indices)))


        joint_error = torch.pow(
            torch.index_select(gt_joints, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2

        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
                None):
            depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))

        return joint_loss + depth_loss
