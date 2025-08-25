import argparse
import glob
import os
import numpy as np
import joblib 
from smplx import SMPL
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch3d.transforms as transforms
import pickle as pkl
from PIL import Image
import tqdm
import logging
def run_smpl_model(root_trans, aa_rot_rep, gender, bm_dict):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 24 X 3
    # gender: BS 
    assert aa_rot_rep.shape[2] == 24
    assert aa_rot_rep.shape[3] == 3
    assert root_trans.shape[2] == 3
    assert root_trans.shape[1] == aa_rot_rep.shape[1]

    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    # num_joints = num_joints - 1
    aa_rot_rep = aa_rot_rep.reshape(bs*num_steps, -1, 3) # (BS*T) X 24 X 3 
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist() # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3) # (BS*T) X 3 
    smpl_root_orient = aa_rot_rep[:, 0, :] # (BS*T) X 3 
    smpl_pose_body = aa_rot_rep[:, 1:, :].reshape(-1, 69) # (BS*T) X 69

    B = smpl_trans.shape[0] # (BS*T) 

    smpl_vals = [smpl_trans, smpl_root_orient, smpl_pose_body]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ['male', 'female']
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=np.int64)*-1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=np.int64)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue
        
        # reconstruct SMPL
        cur_pred_trans, cur_pred_orient, cur_pred_pose = gender_smpl_vals
        bm = bm_dict[gender_name]
        pred_body = bm(body_pose=cur_pred_pose, global_orient=cur_pred_orient, transl=cur_pred_trans)
        
        pred_joints.append(pred_body.joints)
        pred_verts.append(pred_body.vertices)

    # cat all genders and reorder to original batch ordering
    x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]
        
    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map] # (BS*T) X 23 X 3 

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map] # (BS*T) X 6890 X 3 
    
    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3) # BS X T X 23 X 3  
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3) # BS X T X 6890 X 3 

    mesh_faces = bm.faces_tensor
    
    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces

def local2global_pose(local_pose, parents):
    # local_pose: T X J X 3 X 3 
    # kintree = get_smpl_parents() 

    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(parents)):
        parent_id = parents[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose # T X J X 3 X 3 

def quat_ik_torch(grot_mat, parents):
    # grot: T X J X 3 X 3 
    # parents = get_smpl_parents() 

    grot = transforms.matrix_to_quaternion(grot_mat) # T X J X 4 

    res = torch.cat(
            [
                grot[..., :1, :],
                transforms.quaternion_multiply(transforms.quaternion_invert(grot[..., parents[1:], :]), \
                grot[..., 1:, :]),
            ],
            dim=-2) # T X J X 4 

    res_mat = transforms.quaternion_to_matrix(res) # T X J X 3 X 3 

    return res_mat 

def quat_fk_torch(lrot_mat, lpos, parents):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    # parents = get_smpl_parents() 

    lrot = transforms.matrix_to_quaternion(lrot_mat)

    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
        )
        gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

    return res

class MultiModalDataset(Dataset):
    def __init__(
        self,
        opt,
        mode='train',
        window=120,
        device=torch.device('cuda:0'),
    ):
        self.opt = opt 

        self.train = True if mode == 'train' else False
        
        self.window = window
        self.device = device
        # Prepare SMPLH model 
        surface_model_male_fname = '/LOCAL2/anguyen/faic/quang/zigma/SMPL_python_v.1.0.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
        surface_model_female_fname = '/LOCAL2/anguyen/faic/quang/zigma/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

        self.male_bm = SMPL(surface_model_male_fname, gender='male')
        self.female_bm = SMPL(surface_model_female_fname, gender='female')

        for p in self.male_bm.parameters():
            p.requires_grad = False
        for p in self.female_bm.parameters():
            p.requires_grad = False 

        self.male_bm = self.male_bm
        self.female_bm = self.female_bm
        self.parents = self.male_bm.parents
        self.bm_dict = {'male' : self.male_bm, 'female' : self.female_bm}

        self.rest_human_offsets = self.get_rest_pose_joints() # 1 X J X 3 

        self.egocentric_folder = '/LOCAL2/anguyen/faic/quang/zigma/data/egocentric_aist'
        logging.info("Loaing egocentric data from {}".format(self.egocentric_folder))
        if mode == 'train':
            self.egocentric_paths = np.loadtxt(os.path.join(self.egocentric_folder, 'train_data.txt'), dtype=str).tolist()
        elif mode == 'val':
            self.egocentric_paths = np.loadtxt(os.path.join(self.egocentric_folder, 'val_data.txt'), dtype=str).tolist()
        else:
            self.egocentric_paths = np.loadtxt(os.path.join(self.egocentric_folder, 'test_data.txt'), dtype=str).tolist()
        self.audio_feats_path = '/LOCAL2/anguyen/faic/quang/zigma/data/aist++/audio_feats'
        logging.info(f"Total number of windows for {mode}:{len(self.egocentric_paths)}")
        mean_std_path = '/LOCAL2/anguyen/faic/quang/zigma/data/mean_std.npz'
        mean_std = np.load(mean_std_path)
        self.mean = torch.from_numpy(mean_std['mean']).float()
        self.std = torch.from_numpy(mean_std['std']).float()

    def get_rest_pose_joints(self):
        zero_root_trans = torch.zeros(1, 1, 3).float()
        zero_rot_aa_rep = torch.zeros(1, 1, 24, 3).float()
        bs = 1
        gender = ["male"] * bs 

        rest_human_jnts, _, _ = \
        run_smpl_model(zero_root_trans, zero_rot_aa_rep, gender, self.bm_dict)
        # 1 X 1 X 24 X 3 

        parents = self.bm_dict['male'].parents
        rest_human_offsets = rest_human_jnts.squeeze(0) - rest_human_jnts.squeeze(0)[:, parents, :]

        return rest_human_offsets # 1 X 24 X 3 

    def fk_smpl(self, root_trans, lrot_aa):
        # root_trans: N X 3 
        # lrot_aa: N X J X 3 

        # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
        # lpos: N X J X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
        
        # parents = get_smpl_parents() 

        lrot_mat = transforms.axis_angle_to_matrix(lrot_aa) # N X J X 3 X 3 

        lrot = transforms.matrix_to_quaternion(lrot_mat)

        # Generate global joint position 
        lpos = self.rest_human_offsets.repeat(lrot_mat.shape[0], 1, 1) # T' X 24 X 3 

        gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
        for i in range(1, len(self.parents)):
            gp.append(
                transforms.quaternion_apply(gr[self.parents[i]], lpos[..., i : i + 1, :]) + gp[self.parents[i]]
            )
            gr.append(transforms.quaternion_multiply(gr[self.parents[i]], lrot[..., i : i + 1, :]))

        global_rot = torch.cat(gr, dim=-2) # T X 23 X 4 
        global_jpos = torch.cat(gp, dim=-2) # T X 23 X 3 

        global_jpos += root_trans[:, None, :] # T X 23 X 3

        return global_rot, global_jpos 

    def filter_data(self, ori_data_dict):
        new_cnt = 0
        new_data_dict = {}
        max_len = 0 
        for k in ori_data_dict:
            curr_data = ori_data_dict[k]
            seq_len = curr_data['head_qpos'].shape[0]
         
            if seq_len >= self.window:
                new_data_dict[new_cnt] = curr_data 
                new_cnt += 1 

            if seq_len > max_len:
                max_len = seq_len 

        print("The numer of sequences in original data:{0}".format(len(ori_data_dict)))
        print("After filtering, remaining sequences:{0}".format(len(new_data_dict)))
        print("Max length:{0}".format(max_len))

        return new_data_dict 
    
    def de_normalize_jpos(self, normalized_jpos):
        de_jpos = normalized_jpos * self.std.to(normalized_jpos.device) + self.mean.to(normalized_jpos.device)  

        return de_jpos # T X 23 X 3 

    # def process_window_data(self, seq_root_trans, seq_root_orient, seq_pose_body, random_t_idx, end_t_idx):
    #     window_root_trans = torch.from_numpy(seq_root_trans[random_t_idx:end_t_idx]).float()
    #     window_root_orient = torch.from_numpy(seq_root_orient[random_t_idx:end_t_idx]).float()
    #     window_pose_body  = torch.from_numpy(seq_pose_body[random_t_idx:end_t_idx]).float()

    #     # window_root_rot_mat = transforms.axis_angle_to_matrix(window_root_orient) # T' X 3 X 3 
    #     # window_root_quat = transforms.matrix_to_quaternion(window_root_rot_mat)

    #     # window_pose_rot_mat = transforms.axis_angle_to_matrix(window_pose_body) # T' X 23 X 3 X 3 

    #     # # Generate global joint rotation 
    #     # local_joint_rot_mat = torch.cat((window_root_rot_mat[:, None, :, :], window_pose_rot_mat), dim=1) # T' X 24 X 3 X 3 
        

    #     if self.opt.canonicalize_init_head:
    #         # print("Canonicalize init head")
    #         # Canonicalize first frame's facing direction based on global head joint rotation. 
    #         global_joint_rot_mat = local2global_pose(local_joint_rot_mat, self.parents) # T' X 24 X 3 X 3 
    #         global_joint_rot_quat = transforms.matrix_to_quaternion(global_joint_rot_mat) # T' X 24 X 4 
    #         head_idx = 15 
    #         global_head_joint_rot_quat = global_joint_rot_quat[:, head_idx, :].detach().cpu().numpy() # T' X 4 

    #         aligned_root_trans, aligned_head_quat, recover_rot_quat = \
    #         rotate_at_frame_smplh(window_root_trans.detach().cpu().numpy()[np.newaxis], \
    #         global_head_joint_rot_quat[np.newaxis], cano_t_idx=0)
    #         # BS(1) X T' X 3, BS(1) X T' X 4, BS(1) X 1 X 1 X 4  
    #         # recover_rot_quat: from [1, 0, 0] to the actual forward direction 

    #         # Apply the rotation to the root orientation 
    #         cano_window_root_quat = transforms.quaternion_multiply( \
    #         transforms.quaternion_invert(torch.from_numpy(recover_rot_quat[0, 0]).float().to(\
    #         window_root_quat.device)).repeat(window_root_quat.shape[0], 1), window_root_quat) # T' X 4 
    #         cano_window_root_rot_mat = transforms.quaternion_to_matrix(cano_window_root_quat) # T' X 3 X 3 

    #         cano_local_joint_rot_mat = torch.cat((cano_window_root_rot_mat[:, None, :, :], window_pose_rot_mat), dim=1) # T' X 23 X 3 X 3 
    #         cano_global_joint_rot_mat = local2global_pose(cano_local_joint_rot_mat, self.parents) # T' X 23 X 3 X 3 
            
    #         cano_local_rot_aa_rep = transforms.matrix_to_axis_angle(cano_local_joint_rot_mat) # T' X 23 X 3 

    #         cano_local_rot_6d = transforms.matrix_to_rotation_6d(cano_local_joint_rot_mat)
    #         cano_global_rot_6d = transforms.matrix_to_rotation_6d(cano_global_joint_rot_mat)

    #         # Generate global joint position 
    #         local_jpos = self.rest_human_offsets.repeat(cano_local_rot_aa_rep.shape[0], 1, 1) # T' X 23 X 3 
    #         _, human_jnts = quat_fk_torch(cano_local_joint_rot_mat, local_jpos, self.parents) # T' X 23 X 3 
    #         human_jnts = human_jnts + torch.from_numpy(aligned_root_trans[0][:, None, :]).float().to(human_jnts.device)# T' X 23 X 3 

    #         # Move the trajectory based on global head position. Make the head joint to x = 0, y = 0. 
    #         global_head_jpos = human_jnts[:, head_idx, :] # T' X 3 
    #         move_to_zero_trans = global_head_jpos[0:1].clone() # 1 X 3
    #         move_to_zero_trans[:, 2] = 0  
        
    #         global_jpos = human_jnts - move_to_zero_trans[None] # T' X 23 X 3  

    #         global_jvel = global_jpos[1:] - global_jpos[:-1] # (T'-1) X 23 X 3 

    #         query = {}

    #         query['local_rot_mat'] = cano_local_joint_rot_mat # T' X 23 X 3 X 3 
    #         query['local_rot_6d'] = cano_local_rot_6d # T' X 23 X 6

    #         query['global_jpos'] = global_jpos # T' X 23 X 3 
    #         query['global_jvel'] = torch.cat((global_jvel, \
    #             torch.zeros(1, 24, 3).to(global_jvel.device)), dim=0) # T' X 23 X 3 
            
    #         query['global_rot_mat'] = cano_global_joint_rot_mat # T' X 23 X 3 X 3 
    #         query['global_rot_6d'] = cano_global_rot_6d # T' X 23 X 6
    #     else:
    #         curr_seq_pose_aa = torch.cat((window_root_orient[:, None, :], window_pose_body), dim=1) # T' X 24 X 3 
    #         rot_matrix = transforms.axis_angle_to_matrix(curr_seq_pose_aa) # T' X 24 X 3 X 3
    #         rot_6d = transforms.matrix_to_rotation_6d(rot_matrix) # T' X 24 x 6
    #         # curr_seq_local_jpos = self.rest_human_offsets.repeat(curr_seq_pose_aa.shape[0], 1, 1) # T' X 24 X 3 

    #         # curr_seq_pose_rot_mat = transforms.axis_angle_to_matrix(curr_seq_pose_aa) # T x 24 x 3 x 3
    #         # _, human_jnts = quat_fk_torch(curr_seq_pose_rot_mat, curr_seq_local_jpos, self.parents)
    #         # human_jnts = human_jnts + window_root_trans[:, None, :] # T' X 24 X 3  

    #         # head_idx = 15 
    #         # # Move the trajectory based on global head position. Make the head joint to x = 0, y = 0. 
    #         # global_head_jpos = human_jnts[:, head_idx, :] # T' X 3 
    #         # move_to_zero_trans = global_head_jpos[0:1].clone() # 1 X 3
    #         # move_to_zero_trans[:, 2] = 0  
        
    #         # global_jpos = human_jnts - move_to_zero_trans[None] # T' X 24 X 3  

    #         # # global_jvel = global_jpos[1:] - global_jpos[:-1] # (T'-1) X 24 X 3 

    #         # local_joint_rot_mat = transforms.axis_angle_to_matrix(curr_seq_pose_aa) # T' X 24 X 3 X 3 
    #         # global_joint_rot_mat = local2global_pose(local_joint_rot_mat, self.parents) # T' X 24 X 3 X 3 

    #         # local_rot_6d = transforms.matrix_to_rotation_6d(local_joint_rot_mat)
    #         # global_rot_6d = transforms.matrix_to_rotation_6d(global_joint_rot_mat)

    #         # query = {}

    #         # query['local_rot_mat'] = local_joint_rot_mat # T' X 24 X 3 X 3 
    #         # query['local_rot_6d'] = local_rot_6d # T' X 24 X 6

    #         # query['global_jpos'] = global_jpos # T' X 24 X 3 
    #         # # query['global_jvel'] = torch.cat((global_jvel, \
    #         # #     torch.zeros(1, 24, 3).to(global_jvel.device)), dim=0) # T' X 23 X 3 
            
    #         # query['global_rot_mat'] = global_joint_rot_mat # T' X 24 X 3 X 3 
    #         # query['global_rot_6d'] = global_rot_6d # T' X 24 X 6
    #         return rot_6d
    #     return query 

    def __len__(self):
        return len(self.egocentric_paths)

    def __getitem__(self, index):
        data_id = self.egocentric_paths[index]
        motion_file = os.path.join(self.egocentric_folder, data_id, 'motion.npz')
        data = np.load(motion_file)
        body_pose = torch.from_numpy(data['pose_body'])
        global_orient = torch.from_numpy(data['root_orient'])
        local_aa = torch.cat((global_orient, body_pose), dim=-1).reshape(-1, 24, 3) # T X 24 X 3
        rot_mat = transforms.axis_angle_to_matrix(local_aa) # T X 24 X 3 X 3
        rot_6d = transforms.matrix_to_rotation_6d(rot_mat).reshape(-1, 24*6) # T X 24 X 6
        trans = torch.from_numpy(data['trans'])
        seq_length = body_pose.shape[0]
        start_idx = np.random.randint(0, seq_length - self.window)

        # rot_6d = self.process_window_data(trans, global_orient, body_pose, start_idx, start_idx + self.window).reshape(-1, 24*6)
        # global_jpos = query['global_jpos'].detach().cpu().reshape(-1, 72)
        # global_rot_6d = query['global_rot_6d'].detach().cpu().reshape(-1, 24*6)
        motion_input = torch.cat((rot_6d, trans), dim=-1) # T X (24x6 + 3)
        motion_input = motion_input[start_idx:start_idx + self.window]
        motion_input = (motion_input - self.mean) / self.std
        # Load image
        image_path = os.path.join(self.egocentric_folder, data_id + '/images.npy')
        images = torch.from_numpy(np.load(image_path)).float().permute(0, 3, 1, 2)
        image_input = images[start_idx:start_idx + self.window]

        # Load music
        audio_name = data_id.split('_')[-4]
        audio = np.load(os.path.join(self.audio_feats_path, audio_name + '.npy'))
        audio = torch.from_numpy(audio).float()
        audio_input = audio[start_idx:start_idx + self.window]

        actual_seq_len = motion_input.shape[0]
        if actual_seq_len < self.window:
            # Add padding
            padded_motion_input = torch.zeros(self.window-actual_seq_len, motion_input.shape[1]) 
            motion_input = torch.cat((motion_input, padded_motion_input), dim=0)

            padded_image_input = torch.zeros(self.window-actual_seq_len, image_input.shape[1], image_input.shape[2], image_input.shape[3]) 
            image_input = torch.cat((image_input, padded_image_input), dim=0)

            padded_audio_input = torch.zeros(self.window-actual_seq_len, audio_input.shape[1]) 
            audio_input = torch.cat((audio_input, padded_audio_input), dim=0)

        data_input_dict = {}
        data_input_dict['motion'] = motion_input
        data_input_dict['image'] = image_input
        data_input_dict['audio'] = audio_input
        data_input_dict['seq_len'] = actual_seq_len

        return data_input_dict 

def main(opt):
    opt.canonicalize_init_head = False
    dataset = MultiModalDataset(opt, mode=opt.mode)
    length = len(dataset)
    # data = dataset[0]
    # for k, v in data.items():
    #     print(k, v.shape)
    loader = DataLoader(dataset, batch_size=opt.batch_size,  shuffle=False, num_workers=0)
    data_dict = next(iter(loader))
    for k, v in data_dict.items():
        print(k, v.shape)
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--window', type=int, default=120, help='horizon')

    parser.add_argument('--batch_size', type=int, default=2, help='batch size')

    parser.add_argument("--canonicalize_init_head",type=bool, default=True)

    parser.add_argument("--mode", type=str, default='train')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)