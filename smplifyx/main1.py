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

import time
import yaml
import torch

import smplx

import cv2

from utils import JointMapper
from cmd_parser import parse_config
from data_parser import create_dataset
from fit_single_frame import fit_single_frame

from camera import create_camera
from prior import create_prior

torch.backends.cudnn.enabled = False

# !
def main(**args):
    output_folder = args.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)

    result_folder = args.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = args.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder)

    float_dtype = args['float_dtype']
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float64
    else:
        print('Unknown float type {}, exiting!'.format(float_dtype))
        sys.exit(-1)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    img_folder = args.pop('img_folder', 'images')
    dataset_obj = create_dataset(img_folder=img_folder, **args)

    start = time.time()

    max_persons = args.pop('max_persons', -1)

    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    zebra_betas = [4.01370676e-01, 1.23658677e+00, -8.94257279e-01,
             3.19973349e-01, 7.19024035e-01, -1.05410595e-01,
             3.99230129e-01, 1.58862240e-01, 3.85614217e-01,
             -8.16620447e-02, 1.46995142e-01, -2.31515581e-01,
             -3.10253925e-01, -3.42558453e-01, -2.16503877e-01,
             4.97941459e-02, 8.76565450e-03, 1.12414110e-01,
             9.20290504e-02, 5.10690930e-02]

    model_params = dict(model_path=args.get('model_folder'),
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_transl=False,
                        dtype=dtype,
                        betas=torch.Tensor([zebra_betas]),
                        num_betas=20,
                        **args)
    body_model = smplx.create(**model_params)

    focal_length = args.get('focal_length')
    camera = create_camera(focal_length_x=focal_length,
                           focal_length_y=focal_length,
                           dtype=dtype,
                           **args)

    if hasattr(camera, 'rotation'):
        camera.rotation.requires_grad = False

    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        dtype=dtype,
        **args)

    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'mahalanobis_shape'),
        dtype=dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        camera = camera.to(device=device)
        body_model = body_model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)

    for idx, data in enumerate(dataset_obj):
        img = data['img'][0]
        fn = data['fn'][0]
        keypoints = data['keypoints'][0]
        img_scaled_w = 1280
        img_scaling_factor = img.shape[1] / img_scaled_w
        img_scaled_h = round(img.shape[0] / img_scaling_factor)
        img = cv2.resize(img, dsize=(img_scaled_w, img_scaled_h), interpolation=cv2.INTER_CUBIC)
        keypoints[0][:,:2] = keypoints[0][:,:2] / img_scaling_factor

        print('Processing: {}'.format(data['img_path'][0]))

        curr_result_folder = osp.join(result_folder, fn)
        if not osp.exists(curr_result_folder):
            os.makedirs(curr_result_folder)
        curr_mesh_folder = osp.join(mesh_folder, fn)
        if not osp.exists(curr_mesh_folder):
            os.makedirs(curr_mesh_folder)
        for person_id in range(keypoints.shape[0]):
            if person_id >= max_persons and max_persons > 0:
                continue
            curr_result_fn = osp.join(curr_result_folder,'{:03d}.pkl'.format(person_id))
            curr_mesh_fn = osp.join(curr_mesh_folder,'{:03d}.obj'.format(person_id))
            curr_img_folder = osp.join(output_folder, 'images', fn,'{:03d}'.format(person_id))
            if not osp.exists(curr_img_folder):
                os.makedirs(curr_img_folder)
            out_img_fn = osp.join(curr_img_folder, 'output.png')

            fit_single_frame(img, keypoints[[person_id]],
                             body_model=body_model,
                             camera=camera,
                             dtype=dtype,
                             output_folder=output_folder,
                             result_folder=curr_result_folder,
                             out_img_fn=out_img_fn,
                             result_fn=curr_result_fn,
                             mesh_fn=curr_mesh_fn,
                             shape_prior=shape_prior,
                             body_pose_prior=body_pose_prior,
                             angle_prior=angle_prior,
                             **args)
    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))

if __name__ == "__main__":
    args = parse_config()
    main(**args)
