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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img

from optimizers import optim_factory

import fitting
#from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.tools.model_loader import load_model 
from human_body_prior.models.vposer_model import VPoser


def fit_single_frame(img,
                     keypoints,
                     body_model,
                     camera,
                     body_pose_prior,
                     shape_prior,
                     angle_prior,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     out_img_fn='overlay.png',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     data_weights=None,
                     body_pose_prior_weights=None,
                     shape_weights=None,
                     depth_loss_weight=1e2,
                     focal_length=5000.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     batch_size=1,
                     dtype=torch.float32,
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        #vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer, _ = load_model(vposer_ckpt, model_code=VPoser, remove_words_in_model_weights='vp_model.', disable_grad=True)
        vposer = vposer.to(device=device)
        vposer.eval()

    if use_vposer:
        body_mean_pose = torch.zeros([batch_size, vposer_latent_dim],dtype=dtype)
    else:
        body_mean_pose = np.load('smplifyx/walking_toy_symmetric_35parts_mean_pose.npz')['mean_pose'][3:]

    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :2]
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}

    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],device=device,dtype=dtype)

    # The indices of the joints used for the initialization of the camera

    init_joints_idxs = torch.tensor([12,13,10,11,7,8,9], device=device) # excluded 18 (neck) #torch.tensor([7, 18,  13,  18], device=device) # torch.tensor([23, 25,  13,  15], device=device)#torch.tensor(init_joints_idxs, device=device)

    key_vids = [np.array([np.array([1068, 1080, 1029, 1226], dtype=np.uint16),
                          np.array([2660, 3030, 2675, 3038], dtype=np.uint16),
                          np.array([910]),
                          np.array([360, 1203, 1235, 1230], dtype=np.uint16),
                          np.array([3188, 3156, 2327, 3183], dtype=np.uint16),
                          np.array([1976, 1974, 1980, 856], dtype=np.uint16),
                          np.array([3854, 2820, 3852, 3858], dtype=np.uint16),
                          np.array([452, 1811], dtype=np.uint16),
                          np.array([416, 235, 182], dtype=np.uint16),
                          np.array([2156, 2382, 2203], dtype=np.uint16),
                          np.array([829]),
                          np.array([2793]),
                          np.array([60, 114, 186, 59], dtype=np.uint8),
                          np.array([2091, 2037, 2036, 2160], dtype=np.uint16),
                          np.array([384, 799, 1169, 431], dtype=np.uint16),
                          np.array([2351, 2763, 2397, 3127], dtype=np.uint16),
                          np.array([221, 104], dtype=np.uint8),
                          np.array([2754, 2192], dtype=np.uint16),
                          np.array([191, 1158, 3116, 2165], dtype=np.uint16),
                          np.array([28, 1109, 1110, 1111, 1835, 1836, 3067, 3068, 3069], dtype=np.uint16),
                          np.array([149, 150, 368, 542, 543, 544], dtype=np.uint16),
                          np.array([2124, 2125, 2335, 2507, 2508, 2509], dtype=np.uint16),
                          np.array([1873, 1876, 1877, 1885, 1902, 1905, 1906, 1909, 1920, 1924],dtype=np.uint16),
                          np.array([2963, 2964, 3754, 3756, 3766, 3788, 3791, 3792, 3802, 3805],dtype=np.uint16),
                          np.array([764, 915, 916, 917, 934, 935, 956], dtype=np.uint16),
                          np.array([2878, 2879, 2880, 2897, 2898, 2919, 3751], dtype=np.uint16),
                          np.array([795, 796, 1054, 1058, 1060], dtype=np.uint16),
                          np.array([2759, 2760, 3012, 3015, 3016, 3018], dtype=np.uint16),
                          np.array([1810]), #°°°°°°°
                          np.array([19]),
                          np.array([55])],dtype=object)]

    init_t = fitting.guess_init(body_model, gt_joints, init_joints_idxs,#[(23, 25), (13, 15)],#edge_indices,
                                use_vposer=use_vposer, vposer=vposer,
                                pose_embedding=pose_embedding,
                                model_type=kwargs.get('model_type', 'smpl'),
                                focal_length=focal_length, dtype=dtype, key_vids=key_vids)

    camera_loss = fitting.create_loss('camera_init',
                                      trans_estimation=init_t,
                                      init_joints_idxs=init_joints_idxs,
                                      depth_loss_weight=depth_loss_weight,
                                      dtype=dtype,
                                      key_vids=key_vids).to(device=device)

    loss = fitting.create_loss(loss_type=loss_type,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               dtype=dtype,
                               key_vids=key_vids,
                               **kwargs)
    loss = loss.to(device=device)

    with fitting.FittingMonitor(batch_size=batch_size, visualize=visualize, **kwargs) as monitor:
        img = torch.tensor(img, dtype=dtype)

        H, W, _ = img.shape

        data_weight = 650 / W
        camera_loss.reset_loss_weights({'data_weight': data_weight})

        zebra_betas = [4.01370676e-01, 1.23658677e+00, -8.94257279e-01,
                       3.19973349e-01, 7.19024035e-01, -1.05410595e-01,
                       3.99230129e-01, 1.58862240e-01, 3.85614217e-01,
                       -8.16620447e-02, 1.46995142e-01, -2.31515581e-01,
                       -3.10253925e-01, -3.42558453e-01, -2.16503877e-01,
                       4.97941459e-02, 8.76565450e-03, 1.12414110e-01,
                       9.20290504e-02, 5.10690930e-02]

        # Reset the parameters to estimate the initial translation of the
        # body model
        body_model.reset_params(body_pose=body_mean_pose, betas=zebra_betas)

        with torch.no_grad():
            body_model.betas[:] = torch.Tensor([zebra_betas])

        with torch.no_grad():
            camera.translation[:] = init_t.view_as(camera.translation)
            camera.center[:] = torch.Tensor([W/2, H/2])

        camera.translation.requires_grad = True

        camera_opt_params = [camera.translation, body_model.global_orient]

        camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
            camera_opt_params,
            **kwargs)
        camera_optimizer.zero_grad()

        # The closure passed to the optimizer
        fit_camera = monitor.create_fitting_closure(
            camera_optimizer, body_model, camera, gt_joints,
            camera_loss, joints_conf=joints_conf, create_graph=camera_create_graph,
            use_vposer=use_vposer, vposer=vposer,
            pose_embedding=pose_embedding,
            return_full_pose=False, return_verts=True)

        # Step 1: Optimize over the torso joints the camera translation
        # Initialize the computational graph by feeding the initial translation
        # of the camera and the initial pose of the body model.
        camera_init_start = time.time()
        cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                fit_camera,
                                                camera_opt_params, body_model,
                                                use_vposer=use_vposer,
                                                pose_embedding=pose_embedding,
                                                vposer=vposer)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            tqdm.write('Camera initialization done after {:.4f}'.format(
                time.time() - camera_init_start))
            tqdm.write('Camera initialization final loss {:.4f}'.format(
                cam_init_loss_val))

        orientations = [body_model.global_orient.detach().cpu().numpy()]

        results = []

        # Step 2: Optimize the full model
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()

            new_params = defaultdict(global_orient=orient,
                                     body_pose=body_mean_pose,
                                     betas=zebra_betas)

            body_model.reset_params(**new_params)
            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):

                body_params = list(body_model.parameters())

                final_params = list(
                    filter(lambda x: x.requires_grad, body_params))

                if use_vposer:
                    final_params.append(pose_embedding)

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()

                curr_weights['data_weight'] = data_weight
                curr_weights['bending_prior_weight'] = (
                    3.17 * curr_weights['body_pose_weight'])
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()

                print(opt_idx, curr_weights)
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                            opt_idx, elapsed))

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                tqdm.write('Body final loss val = {:.5f}'.format(
                    final_loss_val))

            result = {'camera_' + str(key): val.detach().cpu().numpy()
                      for key, val in camera.named_parameters()}
            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})
            if use_vposer:
                result['body_pose'] = pose_embedding.detach().cpu().numpy()

            results.append({'loss': final_loss_val,
                            'result': result})

        with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)

    if save_meshes or visualize:
        #body_pose = vposer.decode(
        #    pose_embedding,
        #    output_type='aa').view(1, -1) if use_vposer else None
        body_pose = (vposer.decode(pose_embedding).get( 'pose_body')).reshape(1, -1) if use_vposer else None

        model_output = body_model(return_verts=True, body_pose=body_pose)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        import trimesh

        out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        out_mesh.apply_transform(rot)
        out_mesh.export(mesh_fn)

    persp_camera = camera

    if visualize:
        import pyrender

        vis_start = time.time()
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_center = camera.center.detach().cpu().numpy().squeeze()
        camera_transl = camera.translation.detach().cpu().numpy().squeeze()
        # Equivalent to 180 degrees around the y-axis. Transforms the fit to
        # OpenGL compatible coordinate system.
        camera_transl[0] *= -1.0 # ??? find out why it changes camera.translation itself, causes issues later

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_transl

        camera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=camera_center[0], cy=camera_center[1])
        scene.add(camera, pose=camera_pose)

        # Get the lights from the viewer
        light_nodes = monitor.mv.viewer._create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        r = pyrender.OffscreenRenderer(viewport_width=W,
                                       viewport_height=H,
                                       point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        input_img = img.detach().cpu().numpy()

        output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img)

        with torch.no_grad():
            persp_camera.translation[0][0] *= -1 # fix this
        #model_output = body_model(return_verts=True)
        projected_keypoints = persp_camera(torch.Tensor([[torch.mean(torch.index_select(model_output.vertices, 1, torch.tensor(keypoint_ids.astype(np.int32)))[0],axis=0).tolist() for keypoint_ids in key_vids[0]]]))

        img = pil_img.fromarray((output_img * 255).astype(np.uint8))

        plt.clf()

        plt.imshow(img)
        plt.scatter(x=projected_keypoints[0,:,0].detach().numpy(), y=projected_keypoints[0,:,1].detach().numpy(), c='r', s=input_img.shape[1]*0.001)
        plt.scatter(x=keypoints[0,:,0], y=keypoints[0,:,1], c='g', s=input_img.shape[1]*0.001)
        plt.scatter(x=keypoints[0, init_joints_idxs, 0], y=keypoints[0, init_joints_idxs, 1], c='b', s=input_img.shape[1] * 0.001)

        for gt, proj in zip(keypoints[0,:,:2], projected_keypoints[0,:].detach().numpy()):
            if gt[0] or gt[1]:
                plt.plot([gt[0], proj[0]], [gt[1], proj[1]], c='r', lw=input_img.shape[1] * 0.0002)
        plt.axis('off')
        plt.show()
        plt.savefig(out_img_fn+'_keypoints.png',bbox_inches='tight', dpi=387.1, pad_inches=0)
        #img.save(out_img_fn)

        print('Took ', time.time()-vis_start, 'for the visualisation stage')
