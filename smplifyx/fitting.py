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
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
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
               key_vids=None,
               **kwargs):
    #body_pose = vposer.decode(
    #    pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
    body_pose = (vposer.decode(pose_embedding).get( 'pose_body')).reshape(1, -1) if use_vposer else None
    if use_vposer and model_type == 'smpl':
        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
        body_pose = torch.cat([body_pose, wrist_pose], dim=1)
    output = model(body_pose=body_pose, return_verts=True, return_full_pose=False)
    use_ids0 = edge_idxs
    use_ids = [id for id in use_ids0 if (joints_2d[0][id,0]) and (joints_2d[0][id,1])]
    pairs = [p for p in itertools.combinations(use_ids, 2)]
    def mean_model_point(row_id):
        return torch.mean(torch.index_select(output.vertices, 1, torch.tensor(key_vids[0][row_id].astype(np.int32)))[0], axis=0)
    dist3d = torch.Tensor([np.linalg.norm(mean_model_point(p[0]) - mean_model_point(p[1])) for p in pairs])
    dist2d = torch.Tensor([np.linalg.norm(joints_2d[0][p[0], :] - joints_2d[0][p[1], :]) for p in pairs])
    est_ds = focal_length * dist3d / dist2d
    batch_size = 1
    x_coord = torch.zeros([batch_size], device=output.joints.device, dtype=dtype)
    y_coord = x_coord.clone()
    init_t = torch.stack([x_coord, y_coord, torch.unsqueeze(torch.median(est_ds),0)], dim=1)

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
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        for n in range(self.maxiters):

            '''if n==4:
                with autograd.detect_anomaly():
                    loss = optimizer.step(closure)
            else:
                pass'''
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

                self.mv.update_mesh(vertices.squeeze(),body_model.faces)
            prev_loss = loss.item()
        return prev_loss

    def create_fitting_closure(self,optimizer, body_model, camera=None,
                               gt_joints=None, loss=None,
                               joints_conf=None,
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
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              **kwargs)
            if backward:
                total_loss.backward(create_graph=create_graph)
            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(return_verts=True,body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()
                try:
                    self.mv.update_mesh(vertices.squeeze(),body_model.faces)
                except:
                    print('mesh update error')
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
    def __init__(self,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 angle_prior=None,
                 use_joints_conf=True,
                 dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 reduction='sum',
                 key_vids=None,
                 **kwargs):
        super(SMPLifyLoss, self).__init__()
        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior
        self.key_vids = key_vids
        rho = 150.0 # from SMALR
        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho
        self.body_pose_prior = body_pose_prior
        self.shape_prior = shape_prior

        self.register_buffer('data_weight',torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',torch.tensor(bending_prior_weight, dtype=dtype))

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
                body_model_faces,
                use_vposer=False, pose_embedding=None,
                **kwargs):
        key_vids = self.key_vids
        nCameras = 1
        j2d = [None] * nCameras
        kp_weights = [None] * nCameras
        assignments = [None] * nCameras
        num_points = [None] * nCameras
        use_ids = [None] * nCameras
        visible_vids = [None] * nCameras
        all_vids = [None] * nCameras
        i = 0
        landmarks_names = \
            [['leftEye','rightEye','chin','frontLeftFoot','frontRightFoot','backLeftFoot','backRightFoot',
             'tailStart','frontLeftKnee','backLeftKnee','backRightKnee','leftShoulder','rightShoulder',
             'frontLeftAnkle','frontRightAnkle','backLeftAnkle','backRightAnkle','neck','TailTip','leftEar',
             'rightEar','nostrilLeft','nostrilRight','mouthLeft','mouthRight','cheekLeft','cheekRight',
             'mane','back','croup']]

        landmarks = np.array(torch.unsqueeze(torch.hstack([gt_joints[0], joints_conf.t()]),0)) # fix the 3.75

        visible = landmarks[i][:, 2].astype(bool)

        use_ids[i] = [id for id in np.arange(landmarks[i].shape[0]) if visible[id]]
        visible_vids[i] = np.hstack([key_vids[i][id] for id in use_ids[i]])

        group = np.hstack([index * np.ones(len(key_vids[i][row_id])) for index, row_id in enumerate(use_ids[i])])
        assignments[i] = np.vstack([group == j for j in np.arange(group[-1] + 1)])
        num_points[i] = len(use_ids[i])

        all_vids[i] = visible_vids[i]
        #cam[i].v = sv[i][all_vids[i], :]
        j2d[i] = torch.Tensor(landmarks[i][use_ids[i], :2])
        kp_weights[i] = np.ones((landmarks[i].shape[0], 1))
        kp_weights[i] = np.ones((landmarks[i].shape[0], 1))
        kp_weights[i][landmarks_names[i].index('mane'), :] *= 2.
        kp_weights[i][landmarks_names[i].index('leftEye'), :] *= .5
        kp_weights[i][landmarks_names[i].index('rightEye'), :] *= .5
        kp_weights[i][landmarks_names[i].index('leftEar'), :] *= .5
        kp_weights[i][landmarks_names[i].index('leftEar'), :] *= .5
        kp_weights[i][landmarks_names[i].index('nostrilLeft'), :] *= .5
        kp_weights[i][landmarks_names[i].index('nostrilRight'), :] *= .5
        kp_weights[i][landmarks_names[i].index('mouthLeft'), :] *= .5
        kp_weights[i][landmarks_names[i].index('mouthRight'), :] *= .5

        kp_weights[i] = torch.Tensor(kp_weights[i])
        projected_joints = camera(torch.index_select(body_model_output.vertices, 1, torch.tensor(all_vids[0])))
        projected_joints = projected_joints[0]
        kp_proj = 1500.0 * kp_weights[i][use_ids[0]] * torch.sqrt(self.robustifier(torch.vstack([projected_joints[choice] if np.sum(choice) == 1 else projected_joints[choice].mean(axis=0) for choice in assignments[0]]) - j2d[0])+1e-8) / np.sqrt(num_points[i])
        joint_loss = torch.sum(torch.square(kp_proj))

        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = self.body_pose_prior(body_model_output.body_pose) * self.body_pose_weight ** 2

        shape_loss = self.shape_prior(body_model_output.betas) * self.shape_weight ** 2
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight
        total_loss = joint_loss + shape_loss + pprior_loss
        return total_loss

class SMPLifyCameraInitLoss(nn.Module):
    def __init__(self, init_joints_idxs, trans_estimation=None,
                 data_weight=1.0,
                 depth_loss_weight=1e2,
                 dtype=torch.float32,
                 key_vids=None,
                 **kwargs):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype
        self.robustifier = utils.GMoF(rho=150.0)

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))
        self.key_vids = key_vids

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, body_model_faces, joints_conf, **kwargs):
        key_vids = self.key_vids

        nCameras = 1

        j2d = [None] * nCameras
        kp_weights = [None] * nCameras
        assignments = [None] * nCameras
        num_points = [None] * nCameras
        use_ids = [None] * nCameras
        visible_vids = [None] * nCameras
        all_vids = [None] * nCameras
        i = 0
        landmarks = np.array(torch.unsqueeze(torch.hstack([gt_joints[0], joints_conf.t()]), 0))

        visible = landmarks[i][:, 2].astype(bool)

        init_joints_idxs = self.init_joints_idxs #torch.tensor([12, 10, 11, 7, 8, 9]) # removed 18 (neck), 13
        use_ids[i] = [id for id in np.arange(landmarks[i].shape[0]) if (visible[id] and id in init_joints_idxs)]
        visible_vids[i] = np.hstack([key_vids[i][id].astype(int) for id in use_ids[i]])

        group = np.hstack([index * np.ones(len(key_vids[i][row_id])) for index, row_id in enumerate(use_ids[i])])
        assignments[i] = np.vstack([group == j for j in np.arange(group[-1] + 1)])
        # assignments[i] = torch.Tensor(assignments[i])
        num_points[i] = len(use_ids[i])

        all_vids[i] = visible_vids[i]
        # cam[i].v = sv[i][all_vids[i], :]
        j2d[i] = torch.Tensor(landmarks[i][use_ids[i], :2])

        kp_weights[i] = np.ones((landmarks[i].shape[0], 1))
        kp_weights[i] = torch.Tensor(kp_weights[i])

        projected_joints = camera(torch.index_select(body_model_output.vertices, 1, torch.tensor(all_vids[0])))
        projected_joints = projected_joints[0]

        kp_proj = torch.sqrt(self.robustifier(torch.vstack(
            [projected_joints[choice] if np.sum(choice) == 1 else projected_joints[choice].mean(axis=0) for choice in
             assignments[0]]) - j2d[0])) / np.sqrt(num_points[i])
        joint_loss = torch.square(self.data_weight) * torch.sum(torch.square(kp_proj))

        depth_loss = self.depth_loss_weight ** 2 * torch.sum((camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))

        return joint_loss + depth_loss
