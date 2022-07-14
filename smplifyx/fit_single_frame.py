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
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     out_img_fn='overlay.png',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     coll_loss_weights=None,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    '''if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg'''

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    '''if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg'''

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg

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

    # °°°°°°°°
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

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None
    '''if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=point2plane,
                vectorized=True, penalize_outside=penalize_outside)

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)'''

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    '''if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights'''

    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],device=device,dtype=dtype)

    # The indices of the joints used for the initialization of the camera

    '''['leftEye'] 0
        ['rightEye'] 1
        ['chin'] 2
        ['frontLeftFoot'] 3
        ['frontRightFoot'] 4
        ['backLeftFoot'] 5
        ['backRightFoot'] 6
        ['tailStart'] 7
        ['frontLeftKnee'] 8
        ['frontRightKnee'] 9
        ['backLeftKnee'] 10
        ['backRightKnee'] 11
        ['leftShoulder'] 12
        ['rightShoulder'] 13
        ['frontLeftAnkle'] 14
        ['frontRightAnkle'] 15
        ['backLeftAnkle'] 16
        ['backRightAnkle'] 17
        ['neck'] 18
        ['TailTip'] 19
        ['leftEar'] 20
        ['rightEar'] 21
        ['nostrilLeft'] 22
        ['nostrilRight'] 23
        ['mouthLeft'] 24
        ['mouthRight'] 25
        ['cheekLeft'] 26
        ['cheekRight'] 27'''
    #init_joints_idxs = torch.tensor([12, 13, 10, 11, 7, 8, 9], device=device) # excluded 18 (neck) #torch.tensor([7, 18,  13,  18], device=device) # torch.tensor([23, 25,  13,  15], device=device)#torch.tensor(init_joints_idxs, device=device)
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
    #camera_loss.trans_estimation[:] = init_t

    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face, use_hands=use_hands,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
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

        # If the distance between the 2D shoulders is smaller than a
        # predefined threshold then try 2 fits, the initial one and a 180
        # degree rotation
        #shoulder_dist = torch.dist(gt_joints[:, left_shoulder_idx],
        #                           gt_joints[:, right_shoulder_idx])
        
        # °°°°°°
        try_both_orient = False
        #try_both_orient = shoulder_dist.item() < side_view_thsh

        with torch.no_grad():
            camera.translation[:] = init_t.view_as(camera.translation)
            camera.center[:] = torch.Tensor([W/2, H/2])

        # Re-enable gradient calculation for the camera translation
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


        # °°°°°°°°° fix
        #forced_camera_translation = torch.Tensor([0.05380275, -0.23544808,  7.48446028])
        #forced_model_orient = torch.Tensor([1.52352594, 0.29095589, -0.06503418]) # forced_model_orient and forced_camera_translation need to be verified
        #with torch.no_grad():
        #    camera.translation[:] = forced_camera_translation.view_as(camera.translation)
        #    body_model.global_orient[:] = forced_model_orient.view_as(body_model.global_orient)





        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            tqdm.write('Camera initialization done after {:.4f}'.format(
                time.time() - camera_init_start))
            tqdm.write('Camera initialization final loss {:.4f}'.format(
                cam_init_loss_val))

        # If the 2D detections/positions of the shoulder joints are too
        # close the rotate the body by 180 degrees and also fit to that
        # orientation

        '''if False: # °°°°°°°°try_both_orient:
            body_orient = body_model.global_orient.detach().cpu().numpy()
            flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
                cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
            flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

            flipped_orient = torch.tensor(flipped_orient,
                                          dtype=dtype,
                                          device=device).unsqueeze(dim=0)
            orientations = [body_orient, flipped_orient]
        else:'''
        orientations = [body_model.global_orient.detach().cpu().numpy()]

        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []

        # Step 2: Optimize the full model
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()

            new_params = defaultdict(global_orient=orient,
                                     body_pose=body_mean_pose,
                                     betas=zebra_betas)

            #for name, param in body_model.named_parameters():
            #    print('\n\n', name,param,'\n\n')

            body_model.reset_params(**new_params)
            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):

                body_params = list(body_model.parameters())

                #body_model.global_orient.requires_grad = False #°°°°°°°°°°
                #body_model.global_orient = torch.nn.Parameter(torch.Tensor([[1.52352594, 0.0, 0.0]]))

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
                '''if use_hands:
                    joint_weights[:, 25:67] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 67:] = curr_weights['face_weight']'''
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
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

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
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

        model_type = kwargs.get('model_type', 'smpl')
        append_wrists = model_type == 'smpl' and use_vposer
        if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

        model_output = body_model(return_verts=True, body_pose=body_pose)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        #print(camera(model_output.joints))

        import trimesh

        out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        out_mesh.apply_transform(rot)
        out_mesh.export(mesh_fn)

    persp_camera = camera

    if visualize: #visualize:
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



        #projected_joints = persp_camera(model_output.joints)
        #plt.scatter(x=projected_joints[0, :, 0].detach().numpy(), y=projected_joints[0, :, 1].detach().numpy(), c='y', s=input_img.shape[1] * 0.04)


        for gt, proj in zip(keypoints[0,:,:2], projected_keypoints[0,:].detach().numpy()):
            if gt[0] or gt[1]:
                plt.plot([gt[0], proj[0]], [gt[1], proj[1]], c='r', lw=input_img.shape[1] * 0.0002)
        plt.axis('off')
        plt.show()
        plt.savefig(out_img_fn+'_keypoints.png',bbox_inches='tight', dpi=387.1, pad_inches=0)

        #img.save(out_img_fn)

        print('Took ', time.time()-vis_start, 'for the visualisation stage')
