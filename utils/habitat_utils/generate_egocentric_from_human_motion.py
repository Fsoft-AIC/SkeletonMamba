import os, sys
import argparse
import math
import numpy as np
import torch
import subprocess
import habitat_sim
from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector
from habitat_sim import geo
import magnum as mn
import quaternion
import imageio
import logging

# from habitat_utils.utils import batch_rigid_transform, REGION_NAME_DICT

if "sim" not in globals():
    global sim
    sim = None

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

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []
    if settings["color_sensor_1st_person"]:
        color_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        color_sensor_1st_person_spec.uuid = "color_sensor_1st_person"
        color_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        color_sensor_1st_person_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        color_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_1st_person_spec)
    if settings["depth_sensor_1st_person"]:
        depth_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_1st_person_spec.uuid = "depth_sensor_1st_person"
        depth_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        depth_sensor_1st_person_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        depth_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_1st_person_spec)
    if settings["semantic_sensor_1st_person"]:
        semantic_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_1st_person_spec.uuid = "semantic_sensor_1st_person"
        semantic_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        semantic_sensor_1st_person_spec.position = [
            0.0,
            settings["sensor_height"],
            0.0,
        ]
        semantic_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        semantic_sensor_1st_person_spec.sensor_subtype = (
            habitat_sim.SensorSubType.PINHOLE
        )
        sensor_specs.append(semantic_sensor_1st_person_spec)
    if settings["color_sensor_3rd_person"]:
        color_sensor_3rd_person_spec = habitat_sim.CameraSensorSpec()
        color_sensor_3rd_person_spec.uuid = "color_sensor_3rd_person"
        color_sensor_3rd_person_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_3rd_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        color_sensor_3rd_person_spec.position = [
            0.0,
            settings["sensor_height"] + 0.2,
            0.2,
        ]
        color_sensor_3rd_person_spec.orientation = [-math.pi / 4, 0., 0.]
        color_sensor_3rd_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_3rd_person_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def make_default_settings():
    settings = {
        "width": 64,  # Spatial resolution of the observations
        "height": 64,
        "scene": "/home/quangnv89re/Project1/Replica_Dataset/data/room_0/habitat/mesh_semantic.ply",  # Scene path
        "default_agent": 0,
        "sensor_height": 1.5,  # Height of sensors in meters
        "sensor_pitch": -math.pi / 8.0,  # sensor pitch (x rotation in rads)
        "color_sensor_1st_person": True,  # RGB sensor
        "color_sensor_3rd_person": False,  # RGB sensor 3rd person
        "depth_sensor_1st_person": False,  # Depth sensor
        "semantic_sensor_1st_person": False,  # Semantic sensor
        "seed": 1,
        "enable_physics": False,  # enable dynamics simulation
    }
    return settings

def make_simulator_from_settings(sim_settings):
    cfg = make_cfg(sim_settings)
    # clean-up the current simulator instance if it exists
    global sim

    if sim != None:
        sim.close()
    # initialize the simulator
    sim = habitat_sim.Simulator(cfg)

sim_settings = make_default_settings()
sim_settings["sensor_pitch"] = 0.
sim_settings["sensor_height"] = 1. 

make_simulator_from_settings(sim_settings)
    
visual_sensor = sim._sensors["color_sensor_1st_person"]

# set sensor initial position
visual_sensor._spec.position = np.array([0., 0., 0.]) 
visual_sensor._spec.orientation = np.array([0., 0., 0.]) 

visual_sensor._sensor_object.set_transformation_from_spec()

save_video = False

quat_mp3d_to_habitat = quat_from_two_vectors(np.array([0, 0, -1]), geo.GRAVITY)
rot_mat_mp3d_to_habitat = quaternion.as_rotation_matrix(quat_mp3d_to_habitat)

data_path = '/home/quangnv89re/Project1/egoego_release/data/egocentric_aist'
list_paths = os.listdir(data_path)

for path in list_paths:
    print('Process {0}'.format(path))
    path_name = os.path.join(data_path, path)
    data = np.load(os.path.join(path_name, 'motion.npz'))

    trans = data['head_cam_verts']  
    ori = data['ori_head']

    image_path = os.path.join(path_name, "images")
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    print('Save images to {0}'.format(image_path))

    for t in range(trans.shape[0]):
        camera_pos = trans[t]
        #camera_rot = root_ori[t]
        camera_rot = ori[t]
        
        root_ori_t_x = -camera_rot[:, 0]
        root_ori_t_y = camera_rot[:, 1]
        root_ori_t_z = -camera_rot[:, 2]
        camera_rot = np.stack((root_ori_t_x, root_ori_t_y, root_ori_t_z), axis=1)

        #camera_pos_habitat = quat_rotate_vector(quat_mp3d_to_habitat, camera_pos)
        camera_pos_habitat = rot_mat_mp3d_to_habitat @ camera_pos
        camera_rot_habitat = rot_mat_mp3d_to_habitat @ camera_rot

        sim.get_agent(0).scene_node.translation = camera_pos_habitat
        sim.get_agent(0).scene_node.rotation = mn.Quaternion.from_matrix(camera_rot_habitat)
        observation = habitat_sim.utils.viz_utils.observation_to_image(sim.get_sensor_observations()['color_sensor_1st_person'], \
                                    observation_type='color')
        
        output_path = os.path.join(image_path, "%05d.png"%t)
        observation.save(output_path)

    if save_video:
        output_path = os.path.join(path_name, "demo.mp4")
        images_to_video_w_imageio(image_path, output_path)