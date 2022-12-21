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
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import PIL.Image as pil_img
from optimizers import optim_factory
import fitting
import json
import sys, os
sys.path.append(os.path.abspath(os.getcwd())+"/pytorch3d/")
from pytorch3d import transforms
import os.path as osp

def fit_single_frame(imgs,
                     keypoints,
                     cam_poses,
                     body_model,
                     cameras,
                     body_pose_prior,
                     shape_prior,
                     angle_prior,
                     cam_pose_prior,
                     betas,
                     output_dir,
                     snapshot_name,
                     loss_type='smplify',
                     depth_loss_weight=1e2,
                     save_meshes=True,
                     batch_size=1,
                     init_joints_idxs=(9, 12, 2, 5), # the next ones are in kwargs
                     use_cuda=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     shape_weights=None,
                     rho=100,
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     dtype=torch.float32,
                     **kwargs):
    #assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'
    use_yaw = kwargs.get('use_yaw', False)
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    body_mean_pose = np.load('smplifyx/walking_toy_symmetric_35parts_mean_pose.npz')['mean_pose'][3:]

    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:,:, :, :2]
    joints_conf = keypoint_data[:,:, :, 2]#.reshape(1, -1)
    gt_joints = gt_joints.to(device=device, dtype=dtype)
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
    key_vids = [np.load('smplifyx/key_vids.npy', allow_pickle=True)]

    camera_loss = fitting.create_loss('camera_init',
                                init_joints_idxs=init_joints_idxs,
                                depth_loss_weight=depth_loss_weight,
                                cam_pose_prior=cam_pose_prior,
                                #cam_prior_poses=cam_prior_poses,
                                dtype=dtype,
                                key_vids=key_vids).to(device=device)

    with fitting.FittingMonitor(batch_size=batch_size, visualize=visualize, **kwargs) as monitor:
        imgs = torch.tensor(imgs, dtype=dtype)
        H, W, _ = imgs[0].shape
        data_weight = 650 / W
        camera_loss.reset_loss_weights({'data_weight': data_weight})
        betas = betas.tolist()
        body_model.reset_params(body_pose=body_mean_pose, betas=betas, yaw=torch.Tensor([2]))#, yaw=yaw_init)
        
        with torch.no_grad():
            body_model.betas[:] = torch.Tensor([betas])
            body_model.global_orient = torch.nn.Parameter(torch.Tensor([[1.2,1.2, -1.2]]))#torch.Tensor([[0, 2.2, -2.2]])) #[[0, 2.2, -2.2]]))
            #body_model.yaw = torch.nn.Parameter(torch.Tensor([np.pi/2]))#torch.Tensor([[0, 2.2, -2.2]])) #[[0, 2.2, -2.2]]))


            for camera_i, camera in enumerate(cameras):
                log_translation = torch.Tensor(cam_poses[camera_i][0][14:17])
                log_rotation_euler = torch.Tensor(cam_poses[camera_i][0][17:20])
                log_rotation_euler = torch.Tensor([0,-np.pi/3,np.pi/2])

                camera.global_translation[:] = log_to_smalify(log_translation,log_rotation_euler)[0]
                #cR_euler = log_to_smalify(log_translation,log_rotation_euler)[1]
                #cR = transforms.euler_angles_to_matrix(cR_euler, convention="XYZ")
                cR = transforms.euler_angles_to_matrix(log_to_smalify(log_translation,log_rotation_euler)[1], convention="YXZ")
                R = torch.inverse(cR)
                #rotation_matrix = transforms.euler_angles_to_matrix(torch.Tensor([-float(cam_poses[camera_i][0][17]), np.pi/2+float(cam_poses[camera_i][0][19]),-np.pi/2+float(cam_poses[camera_i][0][18])]), convention="ZXY")
                rotation_aa = transforms.matrix_to_axis_angle(R)
                camera.rotation_aa[:] = rotation_aa 
                camera.global_translation.requires_grad = False
                camera.rotation_aa.requires_grad = False
                camera.center[:] = torch.Tensor([W/2, H/2])
                camera.forward(torch.Tensor([[[0.,0.,0.]]]))

            intrinsic = cameras[0].intrinsic
            extrinsic = torch.cat((cameras[0].rotation[0], cameras[0].translation.view(3, -1)), dim=1) # check rotation and translation coordinate system
            M = torch.matmul(intrinsic, extrinsic)
            init_2d_keypoint = np.mean(np.array([kp for kp in keypoints[0][0] if kp[0] and kp[1]]), axis=0)[:2]
            init_2d_keypoint = np.append(init_2d_keypoint,np.array([1.0,0.0]))
            init_2d_keypoint = torch.Tensor(init_2d_keypoint)
            M_ext = torch.cat((M,torch.Tensor([[0,1,0,0]])))
            init_3d_keypoint = torch.inverse(M_ext)@init_2d_keypoint.T
            body_model.transl[:] = torch.nn.Parameter((init_3d_keypoint / init_3d_keypoint[3])[:3])
            


        camera_opt_params = []
        #for camera in cameras:
        #    camera.rotation_aa.requires_grad = True
        #    camera_opt_params.append(camera.rotation_aa)

        body_model.transl.requires_grad = True
        camera_opt_params.append(body_model.transl)
        
        if use_yaw:
            body_model.yaw.requires_grad = True
            camera_opt_params.append(body_model.yaw)
        else:
            body_model.global_orient.requires_grad = True
            camera_opt_params.append(body_model.global_orient)

        


        camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
            camera_opt_params,
            **kwargs)
        camera_optimizer.zero_grad()
        fit_camera = monitor.create_fitting_closure(
            camera_optimizer, body_model, cameras, gt_joints,
            camera_loss, joints_conf=joints_conf, create_graph=camera_create_graph,
            return_full_pose=False, return_verts=True)
        # Step 1: Optimize over the torso joints the camera translation
        camera_init_start = time.time()

        cam_init_loss_val = 0
        cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                fit_camera,
                                                camera_opt_params, body_model)
        
        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            tqdm.write('Camera initialization done after {:.4f}'.format(
                time.time() - camera_init_start))
            tqdm.write('Camera initialization final loss {:.4f}'.format(
                cam_init_loss_val))

        results = []







        # Step 2: Optimize the full model
        final_loss_val = 0

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

        opt_start = time.time()

        with torch.no_grad():
            body_model.global_orient = torch.nn.Parameter(torch.Tensor([[1.2,1.2, -1.2]]))

        new_params = defaultdict(body_pose=body_mean_pose,
                                 betas=betas,
                                 global_orient=body_model.global_orient,
                                 transl = body_model.transl)
        body_model.reset_params(**new_params)

        for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):

            body_model.transl.requires_grad = True
            body_model.betas.requires_grad = True
            body_model.body_pose.requires_grad = True
            body_model.global_orient.requires_grad = True

            body_params = list(body_model.parameters())

            final_params = list(
                filter(lambda x: x.requires_grad, body_params))
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
                cameras=cameras, gt_joints=gt_joints,
                joints_conf=joints_conf,
                loss=loss, create_graph=body_create_graph,
                return_verts=True, return_full_pose=True)

            if interactive:
                stage_start = time.time()

            print(opt_idx, curr_weights)
            final_loss_val = monitor.run_fitting(
                body_optimizer,
                closure, final_params,
                body_model)

            if interactive:
                elapsed = time.time() - stage_start
                tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                        opt_idx, elapsed))

        if interactive:
            elapsed = time.time() - opt_start
            tqdm.write(
                'Body fitting done after {:.4f} seconds'.format(elapsed))
            tqdm.write('Body final loss val = {:.5f}'.format(
                final_loss_val))

        result = {'camera_' + str(key): val.detach().cpu().numpy()
                  for key, val in camera.named_parameters()}
        result.update({key: val.detach().cpu().numpy()
                       for key, val in body_model.named_parameters()})

        results.append({'loss': final_loss_val,
                        'result': result})

        '''with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)'''

    if save_meshes or visualize:
        body_pose = (vposer.decode(pose_embedding).get( 'pose_body')).reshape(1, -1) if use_vposer else None
        model_output = body_model(return_verts=True, body_pose=body_pose)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()
        import trimesh
        out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        out_mesh.apply_transform(rot)
        mesh_fn = osp.join(output_dir,'meshes/'+'out_mesh_'+snapshot_name+'.obj')
        out_mesh.export(mesh_fn)

        mesh_dict = {}
        mesh_dict['vertices'] = vertices.tolist()
        mesh_dict['faces'] = body_model.faces.tolist()
        with open(mesh_fn.split(".")[-2]+".json", 'w') as fp:
            json.dump(mesh_dict, fp)


    persp_camera = camera

    if visualize:
        import pyrender
        vis_start = time.time()
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)
        for camera_index, camera in enumerate(cameras):
            persp_camera = camera
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],ambient_light=(0.3, 0.3, 0.3))
            scene.add(mesh, 'mesh')
            camera_center = camera.center.detach().cpu().numpy().squeeze()
            camera_transl = camera.translation.detach().cpu().numpy().squeeze().copy()
            camera_transl[0] *= -1.0 # ??? find out why it changes camera.translation itself, causes issues later
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_transl

            input_img = imgs[camera_index].detach().cpu().numpy()
            output_img = input_img

            cam_param_dict = {}

            R = camera.rotation[0].detach()
            t = camera.translation[0].detach()
            cam_param_dict['translation'] = (-R.T@t).tolist()
            cam_param_dict['rotation'] = (R.T).tolist()

            with open(osp.join(output_dir,"results/","cam_pose_"+snapshot_name+"_"+str(camera_index)+".json"), 'w') as fp:
                json.dump(cam_param_dict, fp)

            #model_output = body_model(return_verts=True)
            
            projected_keypoints = persp_camera(torch.Tensor([[torch.mean(torch.index_select(model_output.vertices, 1, torch.tensor(keypoint_ids.astype(np.int32)))[0],axis=0).tolist() for keypoint_ids in key_vids[0]]]))
            all_vertices_projected = persp_camera(model_output.vertices)
            img = pil_img.fromarray((output_img * 255).astype(np.uint8))
            plt.clf()
            plt.imshow(img)
            keypoint_circle_size = input_img.shape[1]*0.00005
            plt.scatter(x=all_vertices_projected[0,:,0].detach().numpy(), y=all_vertices_projected[0,:,1].detach().numpy(), c='w', s=keypoint_circle_size*1e-2)
            plt.scatter(x=projected_keypoints[0,:,0].detach().numpy(), y=projected_keypoints[0,:,1].detach().numpy(), c='r', s=keypoint_circle_size)
            plt.scatter(x=keypoints[camera_index][0,:,0], y=keypoints[camera_index][0,:,1], c='g', s=keypoint_circle_size)
            plt.scatter(x=keypoints[camera_index][0, init_joints_idxs, 0], y=keypoints[camera_index][0, init_joints_idxs, 1], c='b', s=keypoint_circle_size)
            #plt.scatter(x=init_2d_keypoint[0], y=init_2d_keypoint[1], s=30*keypoint_circle_size,c='orange', marker='o')
            plt.scatter(x=init_2d_keypoint[0], y=init_2d_keypoint[1], color='orange', s=keypoint_circle_size*2)

            # a grid around origin as a reference
            '''unit_z_point = persp_camera(torch.Tensor([[[0,0,1]]]))
            x = np.linspace(-1,1, 3)
            X,Z = np.meshgrid(x,x)
            grid_points_n = len(x)**2
            grid = torch.cat((torch.Tensor(X).view((grid_points_n,1)), torch.zeros(grid_points_n,1), torch.Tensor(Z).view((grid_points_n,1))), dim=1).unsqueeze(dim=0)
            projected_grid = persp_camera(grid)
            plt.scatter(x=projected_grid[0,:,0].detach().numpy(), y=projected_grid[0,:,1].detach().numpy(), color='pink', s=keypoint_circle_size*3)
            plt.scatter(x=unit_z_point[0,:,0].detach().numpy(), y=unit_z_point[0,:,1].detach().numpy(), color='red', s=keypoint_circle_size*3)'''

            for gt, proj in zip(keypoints[camera_index][0,:,:2], projected_keypoints[0,:].detach().numpy()):
                if gt[0] or gt[1]:
                    plt.plot([gt[0], proj[0]], [gt[1], proj[1]], c='r', lw=input_img.shape[1] * 0.0002)
            plt.axis('off')
            plt.show()
            #plt.savefig(out_img_fn+'_cam_'+str(camera_index)+'_keypoints.png',bbox_inches='tight', dpi=387.1, pad_inches=0)
            image_dir = osp.join(output_dir, "images/",snapshot_name+"/")
            plt.savefig(osp.join(image_dir,str(camera_index)+'_keypoints.png'),bbox_inches='tight', dpi=387.1, pad_inches=0)
            #img.save(out_img_fn)
        print('Took ', time.time()-vis_start, 'for the visualisation stage')

def log_to_smalify(translation,rotation_euler):
    return (torch.Tensor([translation[1],translation[2],translation[0]]), torch.Tensor([rotation_euler[2],rotation_euler[1], rotation_euler[0]]))
