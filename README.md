## Fitting Articulated Animal Model (SMAL) to Multiview Keypoints Using pytorch

[[Project Page](https://smpl-x.is.tue.mpg.de/)] 
[[Paper](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf)]
[[Supp. Mat.](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/498/SMPL-X-supp.pdf)]

![SMPL-X Examples](./images/teaser_fig.png)

## Table of Contents
  * [License](#license)
  * [Description](#description)
  * [Dependencies](#dependencies)
  * [Citation](#citation)
  * [Contact](#contact)


## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and conditions](https://github.com/vchoutas/smplx/blob/master/LICENSE) and any accompanying documentation before you download and/or use the SMPL-X/SMPLify-X model, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](./LICENSE).

## Description

This repository contains the fitting code for the SMALR animal model, optimized for data collected from drones.

### Fitting 
Run the following command to execute the code:
```Shell
python smplifyx/main.py 
  --config cfg_files/fit_smal.yaml 
  --data_folder DATA_FOLDER
  --visualize True 
  --model_folder MODEL_FILE 
```
where the `DATA_FOLDER` should contain video files, e.g. cam1.mp4, cam2.mp4 etc, animal keypoint files, e.g. cam1.json, and camera pose files, e.g. cam1_pose.json. The model file is the SMAL model, e.g. models/SMAL.pkl. You can download the SMAL model [here] (https://smal.is.tue.mpg.de/download.php)

## Dependencies

Follow the installation instructions for each of the following before using the
fitting code.

1. [PyTorch](https://pytorch.org/)
2. [SMPL-X](https://github.com/vchoutas/smplx) dependencies

### Optional Dependencies

1. [PyTorch Mesh self-intersection](https://github.com/vchoutas/torch-mesh-isect) for interpenetration penalty 
   * Download the per-triangle part segmentation: [smplx_parts_segm.pkl](https://owncloud.tuebingen.mpg.de/index.php/s/MWnr8Kso4K8T8at)
2. [Trimesh](https://trimsh.org/) for loading triangular meshes
3. [Pyrender](https://pyrender.readthedocs.io/) for visualization

### LBFGS with Strong Wolfe Line Search

The LBFGS optimizer with Strong Wolfe Line search is taken from this [Pytorch pull request](https://github.com/pytorch/pytorch/pull/8824). Special thanks to 
[Du Phan](https://github.com/fehiepsi) for implementing this. 
We will update the repository once the pull request is merged.

## Contact
This project was built upon [SMALR](https://github.com/silviazuffi/smalr_online) and [SMPLify-X](https://github.com/vchoutas/smplify-x) by [Egor Iuganov](mailto:egor.iuganov@ifr.uni-stuttgart.de) from the [Flight Robotics and Perception Group](https://www.aamirahmad.de/).

For commercial licensing (and all related questions for business applications), please contact [ps-licensing@tue.mpg.de](mailto:ps-licensing@tue.mpg.de).
