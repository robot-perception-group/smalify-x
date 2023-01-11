import pickle
import numpy as np
import os
with open('../smalr_online/smpl_models/smal_00781_4_all.pkl', 'rb') as smpl_file:
    data3 = pickle.load(smpl_file,encoding='latin1')
data3["shapedirs"] = np.array(data3["shapedirs"])
with open('SMAL.pkl', 'wb') as handle:
    pickle.dump(data3, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../smalr_online/pose_priors/walking_toy_symmetric_pose_prior_with_cov_35parts.pkl', 'rb') as smpl_file:
    data2 = pickle.load(smpl_file,encoding='latin1')
data2["pic"] = np.array(data2["pic"])
with open('walking_toy_symmetric_pose_prior_with_cov_35parts.pkl', 'wb') as handle:
    pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)

os.system('cp ../smalr_online/smpl_models/smal_data_00781_4_all.pkl smal_data_00781_4_all.pkl') 
os.system('cp ../smalr_online/pose_priors/walking_toy_symmetric_35parts_mean_pose.npz walking_toy_symmetric_35parts_mean_pose.npz')