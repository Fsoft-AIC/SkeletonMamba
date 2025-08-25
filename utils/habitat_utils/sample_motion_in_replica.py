import os
import json
import random
import torch
from smplx import SMPL, SMPLH
from smplx.utils import Struct
from smplx.vertex_ids import vertex_ids
import pickle as pkl
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import argparse
import os
import argparse
import math
import magnum as mn
# import quaternion
from smplx.lbs import batch_rigid_transform
import imageio.v2 as imageio
import logging
from utils.data_utils.transformation import batch_rodrigues, rotation_matrix_to_angle_axis
# from habitat_utils.utils import batch_rigid_transform, REGION_NAME_DIC

OUT_FPS = 30
DISCARD_TERRAIN_SEQUENCES = True # throw away sequences where the person steps onto objects (determined by a heuristic)
DISCARD_SHORTER_THAN = 1.0 # seconds
# optional viz during processing
VIZ_PLOTS = False
VIZ_SEQ = False
# if sequence is longer than this, splits into sequences of this size to avoid running out of memory
# ~ 4000 for 12 GB GPU, ~2000 for 8 GB
SPLIT_FRAME_LIMIT = 2000
NUM_BETAS = 16 # size of SMPL shape parameter to use
# for determining floor height
FLOOR_VEL_THRESH = 0.005
FLOOR_HEIGHT_OFFSET = 0.01
# for determining contacts
CONTACT_VEL_THRESH = 0.005 #0.015
CONTACT_TOE_HEIGHT_THRESH = 0.04
CONTACT_ANKLE_HEIGHT_THRESH = 0.08
# for determining terrain interaction
TERRAIN_HEIGHT_THRESH = 0.04 # if static toe is above this height
ROOT_HEIGHT_THRESH = 0.04 # if maximum "static" root height is more than this + root_floor_height
CLUSTER_SIZE_THRESH = 0.25 # if cluster has more than this faction of fps (30 for 120 fps)
SMPL_JOINTS = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 2, 'spine' : 3, 'leftLeg' : 4, 'rightLeg' : 5,
                'spine1' : 6, 'leftFoot' : 7, 'rightFoot' : 8, 'spine2' : 9, 'leftToeBase' : 10, 'rightToeBase' : 11,
                'neck' : 12, 'leftShoulder' : 13, 'rightShoulder' : 14, 'head' : 15, 'leftArm' : 16, 'rightArm' : 17,
                'leftForeArm' : 18, 'rightForeArm' : 19, 'leftHand' : 20, 'rightHand' : 21}

def images_to_video_w_imageio(img_folder, output_vid_file):
    img_files = os.listdir(img_folder)
    img_files.sort()
    im_arr = []
    for img_name in img_files:
        img_path = os.path.join(img_folder, img_name)
        im = imageio.imread(img_path)
        im_arr.append(im)

    im_arr = np.asarray(im_arr)
    imageio.mimwrite(output_vid_file, im_arr, fps=30, quality=8)

def determine_floor_height_and_contacts(body_joint_seq, fps):
    '''
    Input: body_joint_seq N x 21 x 3 numpy array
    Contacts are N x 4 where N is number of frames and each row is left heel/toe, right heel/toe
    '''
    num_frames = body_joint_seq.shape[0]

    # compute toe velocities
    root_seq = body_joint_seq[:, SMPL_JOINTS['hips'], :]
    left_toe_seq = body_joint_seq[:, SMPL_JOINTS['leftToeBase'], :]
    right_toe_seq = body_joint_seq[:, SMPL_JOINTS['rightToeBase'], :]
    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    # if VIZ_PLOTS:
    #     fig = plt.figure()
    #     steps = np.arange(num_frames)
    #     plt.plot(steps, left_toe_vel, '-r', label='left vel')
    #     plt.plot(steps, right_toe_vel, '-b', label='right vel')
    #     plt.legend()
    #     plt.show()
    #     plt.close()

    # now foot heights (z is up)
    left_toe_heights = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights = root_seq[:, 2]

    # if VIZ_PLOTS:
    #     fig = plt.figure()
    #     steps = np.arange(num_frames)
    #     plt.plot(steps, left_toe_heights, '-r', label='left toe height')
    #     plt.plot(steps, right_toe_heights, '-b', label='right toe height')
    #     plt.plot(steps, root_heights, '-g', label='root height')
    #     plt.legend()
    #     plt.show()
    #     plt.close()

    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_toe_vel < FLOOR_VEL_THRESH]
    right_static_foot_heights = right_toe_heights[right_toe_vel < FLOOR_VEL_THRESH]
    right_static_inds = all_inds[right_toe_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(left_static_foot_heights, right_static_foot_heights)
    all_static_inds = np.append(left_static_inds, right_static_inds)

    # if VIZ_PLOTS:
    #     fig = plt.figure()
    #     steps = np.arange(left_static_foot_heights.shape[0])
    #     plt.plot(steps, left_static_foot_heights, '-r', label='left static height')
    #     plt.legend()
    #     plt.show()
    #     plt.close()

    # fig = plt.figure()
    # plt.hist(all_static_foot_heights)
    # plt.show()
    # plt.close()

    discard_seq = False
    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(all_static_foot_heights.reshape(-1, 1))
        all_labels = np.unique(clustering.labels_)
        # print(all_labels)
        if VIZ_PLOTS:
            plt.figure()
        min_median = min_root_median = float('inf')
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(all_static_inds[clustering.labels_ == cur_label]) # inds in the original sequence that correspond to this cluster
            if VIZ_PLOTS:
                plt.scatter(cur_clust, np.zeros_like(cur_clust), label='foot %d' % (cur_label))
            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)
            if VIZ_PLOTS:
                plt.scatter(cur_root_clust, np.zeros_like(cur_root_clust), label='root %d' % (cur_label))

            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        # print(cluster_heights)
        # print(cluster_root_heights)
        # print(cluster_sizes)
        # if VIZ_PLOTS:
        #     plt.show()
        #     plt.close()

        floor_height = min_median
        offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET # toe joint is actually inside foot mesh a bit

        if DISCARD_TERRAIN_SEQUENCES:
            # print(min_median + TERRAIN_HEIGHT_THRESH)
            # print(min_root_median + ROOT_HEIGHT_THRESH)
            for cluster_root_height, cluster_height, cluster_size in zip (cluster_root_heights, cluster_heights, cluster_sizes):
                root_above_thresh = cluster_root_height > (min_root_median + ROOT_HEIGHT_THRESH)
                toe_above_thresh = cluster_height > (min_median + TERRAIN_HEIGHT_THRESH)
                cluster_size_above_thresh = cluster_size > int(CLUSTER_SIZE_THRESH*fps)
                if root_above_thresh and toe_above_thresh and cluster_size_above_thresh:
                    discard_seq = True
                    print('DISCARDING sequence based on terrain interaction!')
                    break
    else:
        floor_height = offset_floor_height = 0.0

    # now find contacts (feet are below certain velocity and within certain range of floor)
    # compute heel velocities
    left_heel_seq = body_joint_seq[:, SMPL_JOINTS['leftFoot'], :]
    right_heel_seq = body_joint_seq[:, SMPL_JOINTS['rightFoot'], :]
    left_heel_vel = np.linalg.norm(left_heel_seq[1:] - left_heel_seq[:-1], axis=1)
    left_heel_vel = np.append(left_heel_vel, left_heel_vel[-1])
    right_heel_vel = np.linalg.norm(right_heel_seq[1:] - right_heel_seq[:-1], axis=1)
    right_heel_vel = np.append(right_heel_vel, right_heel_vel[-1])

    left_heel_contact = left_heel_vel < CONTACT_VEL_THRESH
    right_heel_contact = right_heel_vel < CONTACT_VEL_THRESH
    left_toe_contact = left_toe_vel < CONTACT_VEL_THRESH
    right_toe_contact = right_toe_vel < CONTACT_VEL_THRESH

    # compute heel heights
    left_heel_heights = left_heel_seq[:, 2] - floor_height
    right_heel_heights = right_heel_seq[:, 2] - floor_height
    left_toe_heights =  left_toe_heights - floor_height
    right_toe_heights =  right_toe_heights - floor_height

    left_heel_contact = np.logical_and(left_heel_contact, left_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    right_heel_contact = np.logical_and(right_heel_contact, right_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    left_toe_contact = np.logical_and(left_toe_contact, left_toe_heights < CONTACT_TOE_HEIGHT_THRESH)
    right_toe_contact = np.logical_and(right_toe_contact, right_toe_heights < CONTACT_TOE_HEIGHT_THRESH)

    contacts = np.zeros((num_frames, len(SMPL_JOINTS)))
    contacts[:,SMPL_JOINTS['leftFoot']] = left_heel_contact
    contacts[:,SMPL_JOINTS['leftToeBase']] = left_toe_contact
    contacts[:,SMPL_JOINTS['rightFoot']] = right_heel_contact
    contacts[:,SMPL_JOINTS['rightToeBase']] = right_toe_contact

    # hand contacts
    left_hand_contact = detect_joint_contact(body_joint_seq, 'leftHand', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    right_hand_contact = detect_joint_contact(body_joint_seq, 'rightHand', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    contacts[:,SMPL_JOINTS['leftHand']] = left_hand_contact
    contacts[:,SMPL_JOINTS['rightHand']] = right_hand_contact

    # knee contacts
    left_knee_contact = detect_joint_contact(body_joint_seq, 'leftLeg', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    right_knee_contact = detect_joint_contact(body_joint_seq, 'rightLeg', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    contacts[:,SMPL_JOINTS['leftLeg']] = left_knee_contact
    contacts[:,SMPL_JOINTS['rightLeg']] = right_knee_contact

    return offset_floor_height, contacts, discard_seq

def detect_joint_contact(body_joint_seq, joint_name, floor_height, vel_thresh, height_thresh):
    # calc velocity
    joint_seq = body_joint_seq[:, SMPL_JOINTS[joint_name], :]
    joint_vel = np.linalg.norm(joint_seq[1:] - joint_seq[:-1], axis=1)
    joint_vel = np.append(joint_vel, joint_vel[-1])
    # determine contact by velocity
    joint_contact = joint_vel < vel_thresh
    # compute heights
    joint_heights = joint_seq[:, 2] - floor_height
    # compute contact by vel + height
    joint_contact = np.logical_and(joint_contact, joint_heights < height_thresh)

    return joint_contact

def translate_to_scene(body_joint_seq, trans, root_orient_matrix, house_dir):
    """Translate the human location to a random location on the floor."""

    region_ply_path = os.path.join(house_dir, 'habitat/mesh_semantic.ply')
    region_ply = PlyData.read(region_ply_path)

    # Read semantic json
    sem_json = os.path.join(house_dir, 'habitat/info_semantic.json')
    sem_data = json.load(open(sem_json, 'r'))
    obj_data = sem_data['objects']
    num_objs = len(obj_data)
    floor_obj_id_list = []
    for o_idx in range(num_objs):
        if obj_data[o_idx]['class_name'] == "floor":
            floor_obj_id_list.append(obj_data[o_idx]['id'])

    # Random pick one idx fron floor obj idx list
    floor_idx = random.sample(floor_obj_id_list, 1)[0]

    floor_face_ids = np.where(region_ply['face']['object_id'] == floor_idx)
    floor_vertex_ids = np.stack(region_ply['face']['vertex_indices'][floor_face_ids], axis=0).reshape(-1,)

    rand_vertex_id = np.random.choice(floor_vertex_ids)

    rand_vertex_x = region_ply['vertex']['x'][rand_vertex_id]
    rand_vertex_y = region_ply['vertex']['y'][rand_vertex_id]
    rand_vertex_z = region_ply['vertex']['z'][rand_vertex_id]

    # Randomize the initial frame's root orientation
    rot_angle = random.sample(list(range(0, 360, 20)), 1)[0]
    rot_angle = torch.tensor(rot_angle)
    rot_z = torch.deg2rad(rot_angle)

    curr_rot_mat = torch.zeros(3, 3).float()
    curr_rot_mat[0, 0] = torch.cos(rot_z)
    curr_rot_mat[1, 0] = torch.sin(rot_z)
    curr_rot_mat[0, 1] = -torch.sin(rot_z)
    curr_rot_mat[1, 1] = torch.cos(rot_z)
    curr_rot_mat[2, 2] = 1
    curr_rot_mat = curr_rot_mat[None] # 1 X 3 X 3

    # Rotate along with the z axis
    ori_root_rot_mat = root_orient_matrix.reshape(-1, 9).unsqueeze(0) # 1 X T X 9
    rotated_root_mat = torch.matmul(curr_rot_mat.to(ori_root_rot_mat.device).repeat(ori_root_rot_mat.shape[1], 1, 1), \
        ori_root_rot_mat.reshape(-1, 3, 3)).unsqueeze(1) # T X 1 X 3 X 3

    # Rotate trans
    ori_root_trans = trans.unsqueeze(0) # 1 X T X 3
    aligned_root_trans = torch.matmul(curr_rot_mat[0].to(ori_root_trans.device), ori_root_trans[0].T).T # T X 3

    # Get the offsets between joints and trans
    T = body_joint_seq.shape[0]
    joins = body_joint_seq[:, :23, :].unsqueeze(0).reshape(-1, T, 69)
    trans2joint = torch.zeros(1, 1, 3).to(ori_root_trans.device) # 1 X 1 X 3
    trans2joint[0, 0, :2] = ori_root_trans[0, 0, :2] - joins[0, 0, :2] # Translation from origin to root joint

    # Rotate joints
    ori_joints = joins.reshape(1, -1, 23, 3) # 1 X T X 23 X 3
    aligned_joints = ori_joints.clone()
    aligned_joints += trans2joint[None]
    aligned_joints = torch.matmul(curr_rot_mat[0].to(aligned_joints.device), aligned_joints.reshape((-1, 3)).T).T.reshape((-1, 23, 3)) # T X 23 X 3
    aligned_joints -= trans2joint # T X 23 X 3

    # Translate SMPL mesh
    init_delta_joints_x = rand_vertex_x - aligned_joints[0, 0, 0].data.cpu().item()
    init_delta_joints_y = rand_vertex_y - aligned_joints[0, 0, 1].data.cpu().item()
    init_delta_joints_z = rand_vertex_z
    init_delta_trans = torch.from_numpy(np.asarray([init_delta_joints_x, init_delta_joints_y, init_delta_joints_z])).float()

    aligned_root_trans += init_delta_trans.to(aligned_root_trans.device)[None]
    aligned_joints += init_delta_trans.to(aligned_joints.device)[None, None]

    return aligned_root_trans.squeeze(0), rotated_root_mat, aligned_joints

def process_single_sequence(file_name, args):
    # Load smpl model
    smpl_model = SMPL(args.smpl_model_path, gender="MALE")

    sequence_path = os.path.join(args.data_motion_path, file_name)
    bdata = pkl.load(open(sequence_path, 'rb'))
    fps = 60
    num_frames = bdata['smpl_poses'].shape[0]
    trans = torch.from_numpy(bdata['smpl_trans']/bdata['smpl_scaling']).float()
    smpl_poses = bdata['smpl_poses']
    root_orient = torch.from_numpy(smpl_poses[:, :3]).float()       # global root orientation (1 joint)
    pose_body = torch.from_numpy(smpl_poses[:, 3:]).float()         # body joint rotations (23 joints)

    # Rotate the pose by 90 degrees around x axis
    # logging.info('Rotating pose by 90 degrees around x axis')
    angle_90 = torch.tensor([np.pi / 2])
    rotation_matrix_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_90), -torch.sin(angle_90)],
        [0, torch.sin(angle_90), torch.cos(angle_90)]
    ]).reshape(3, 3).unsqueeze(0)
    root_orient_matrix = batch_rodrigues(root_orient.reshape(-1, 3))
    root_orient_rotated_matrix = torch.matmul(rotation_matrix_x.repeat(pose_body.shape[0], 1, 1), root_orient_matrix.reshape(-1, 3, 3))
    root_orient = rotation_matrix_to_angle_axis(root_orient_rotated_matrix).reshape(-1,3)
    trans = torch.matmul(rotation_matrix_x[0], trans.T).T

    parents = smpl_model.parents

    body = smpl_model.forward(
        global_orient=root_orient,
        body_pose=pose_body,
        transl=trans
    )

    body_joint_seq = body.joints.detach().numpy()
    floor_height, contacts, discard_seq = determine_floor_height_and_contacts(body_joint_seq, fps)
    trans[:,2] -= floor_height
    body_joint_seq[:,:,2] -= floor_height

    # logging.info('Downsampling data to {} fps'.format(OUT_FPS))
    if OUT_FPS != fps:
        if OUT_FPS > fps:
            print('Cannot supersample data, saving at data rate!')
        else:
            fps_ratio = float(OUT_FPS) / fps
            new_num_frames = int(fps_ratio*num_frames)

            downsamp_inds = np.linspace(0, num_frames-1, num=new_num_frames, dtype=int)

            fps = OUT_FPS
            num_frames = new_num_frames
            contacts = contacts[downsamp_inds]
            trans = trans[downsamp_inds]
            root_orient = root_orient[downsamp_inds]
            pose_body = pose_body[downsamp_inds]
            joint_seq = body_joint_seq[downsamp_inds]

    joint_seq = torch.from_numpy(joint_seq).float()

    # human_face = smpl_model.faces_tensor

    J = smpl_model.NUM_JOINTS #23

    pose_body_matrix = batch_rodrigues(pose_body.reshape(-1, 3)).reshape(-1, J, 3, 3)
    root_orient_matrix = batch_rodrigues(root_orient.reshape(-1, 3)).reshape(-1, 1, 3, 3)
    trans, root_orient_matrix, joint_seq = translate_to_scene(joint_seq, trans, root_orient_matrix, args.house_dir)
    root_orient = rotation_matrix_to_angle_axis(root_orient_matrix.reshape(-1, 3, 3)).reshape(-1, 3)
    # forward again to get vertices and joints
    body = smpl_model.forward(
        global_orient=root_orient,
        body_pose=pose_body,
        transl=trans
    )

    body_vtx_seq = body.vertices.detach().numpy()
    head_v_idx =  232
    head_cam_verts = body_vtx_seq[:, head_v_idx, :] # T X 3

    rot_mats = torch.cat([root_orient_matrix, pose_body_matrix], dim=1) # T x (J + 1) x 3 x 3
    body_joint_seq = body.joints[:, :24, :]
    dtype = body_joint_seq.dtype
    J_transformed, A = batch_rigid_transform(rot_mats, body_joint_seq, parents, dtype=dtype)
    ori_head = A[:, 15, :3, :3].detach().numpy()

    house_name = args.house_dir.split('/')[-1]
    output_dir = f"{args.egocentric_path}/{file_name[:-4]}_{house_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, "motion.npz")
    np.savez(save_path,
             fps=fps,
             root_orient=root_orient.detach().cpu().numpy(), # T X 3
             pose_body=pose_body.detach().cpu().numpy(), # T X 69
             trans=trans.detach().cpu().numpy(), # T X 3
             body_joint_seq=body_joint_seq.detach().cpu().numpy(), # T X 23 X 3
             head_cam_verts=head_cam_verts, # T X 3
             ori_head=ori_head, # T X 3 X 3
             )

def sample_motion_in_replica(args):
    file_names = os.listdir(args.data_motion_path)
    for idx, file_name in enumerate(file_names):
        logging.info(f"Process index: {idx} - {file_name}")
        process_single_sequence(file_name, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-motion-path',
                        default='/home/quangnv89re/Project1/egoego_release/data/aist++/motions',
                        type=str,
                        help='path to motion data'
                        )

    parser.add_argument('--egocentric-path',
                        default='/home/quangnv89re/Project1/egoego_release/data/egocentric_aist',
                        type=str,
                        help='path to output egocentric-aist++ dataset')

    parser.add_argument('--smpl-model-path',
                        type=str,
                        default='/home/quangnv89re/Project1/egoego_release/SMPL_python_v.1.0.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl',
                        help='path to smpl model')

    parser.add_argument('--house-dir',
                        type=str,
                        default='/home/quangnv89re/Project1/Replica_Dataset/data/room_0',
                        help='path to replica dataset')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info("EgoEgo: Generating egocentric video")
    logging.info(args)

    sample_motion_in_replica(args)

if __name__ == "__main__":
    main()