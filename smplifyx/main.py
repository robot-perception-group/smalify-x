from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import os.path as osp
import time
import torch
import smplx
import cv2
from cmd_parser import parse_config
from data_parser import create_dataset
from fit_single_frame import fit_single_frame
from camera import create_camera
from prior import create_prior

def main(**args):
    output_folder = 'output'
    result_folder = output_folder+'/results'
    mesh_folder = output_folder+'/meshes'
    dataset_obj = create_dataset(**args)
    start = time.time()
    betas = torch.Tensor([[float(el) for el in args['zebra_betas']]])

    model_params = dict(model_path=args.get('model_folder'),
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_transl=False,
                        betas=betas,
                        **args)
    body_model = smplx.create(**model_params)
    camera = create_camera(focal_length_x=args['focal_length'],
                           focal_length_y=args['focal_length'],
                           **args)
    camera.rotation_aa.requires_grad = False
    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        **args)
    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'mahalanobis_shape'),
        **args)
    angle_prior = create_prior(prior_type='angle', dtype=args['dtype'])
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
            curr_result_fn = osp.join(curr_result_folder,'{:03d}.pkl'.format(person_id))
            curr_mesh_fn = osp.join(curr_mesh_folder,'{:03d}.obj'.format(person_id))
            curr_img_folder = osp.join(output_folder, 'images', fn,'{:03d}'.format(person_id))
            if not osp.exists(curr_img_folder):
                os.makedirs(curr_img_folder)
            out_img_fn = osp.join(curr_img_folder, 'output.png')

            fit_single_frame(img, keypoints[[person_id]],
                             body_model=body_model,
                             camera=camera,
                             result_folder=curr_result_folder,
                             out_img_fn=out_img_fn,
                             result_fn=curr_result_fn,
                             mesh_fn=curr_mesh_fn,
                             shape_prior=shape_prior,
                             body_pose_prior=body_pose_prior,
                             angle_prior=angle_prior,
                             betas=betas,
                             **args)
    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))

if __name__ == "__main__":
    args = parse_config()
    main(**args)




