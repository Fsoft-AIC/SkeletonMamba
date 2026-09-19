"""Canonical per-joint motion and versioned, train-only normalization."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from dataset.manifest import sha256_file
from utils.kinematics import (SMPL_PARENTS, axis_angle_to_matrix, forward_kinematics,
                             matrix_to_rotation_6d, rotation_6d_to_matrix)

REPRESENTATION = 'smpl24_global_xyz_rotation6d_v1'


def read_metadata(archive):
    if 'metadata' not in archive:
        raise ValueError('Prepared arrays require JSON metadata')
    metadata = json.loads(str(archive['metadata'].item()))
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported prepared-data metadata schema')
    return metadata


def atomic_save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def pack_motion(positions, global_rotations):
    if positions.shape[-2:] != (24, 3) or global_rotations.shape != positions.shape[:-1]+(3,3):
        raise ValueError('Expected matching positions[...,24,3] and rotations[...,24,3,3]')
    return torch.cat((positions, matrix_to_rotation_6d(global_rotations)), dim=-1)


def unpack_motion(motion):
    if motion.shape[-2:] != (24, 9):
        raise ValueError('Motion must end in [24,9]')
    return motion[..., :3], rotation_6d_to_matrix(motion[..., 3:])


def load_raw_motion(path):
    with np.load(path, allow_pickle=False) as data:
        keys = [k for k in ('pose_body','body_pose') if k in data]
        if not keys:
            raise ValueError('Raw motion has neither pose_body nor body_pose')
        if len(keys) == 2 and not np.array_equal(data[keys[0]], data[keys[1]]):
            raise ValueError('Conflicting body-pose aliases')
        body = np.asarray(data[keys[0]], dtype=np.float32)
        if body.ndim not in (2,3) or body.shape[1:] not in ((69,), (23,3)):
            raise ValueError('Body pose must be [T,69] or [T,23,3]')
        body = body.reshape(-1,23,3)
        trans = np.asarray(data['trans'], dtype=np.float32)
        root = np.asarray(data['root_orient'], dtype=np.float32)
    if len(body) == 0 or trans.shape != (len(body),3) or root.shape != trans.shape:
        raise ValueError('Invalid or inconsistent raw motion length')
    if not all(np.isfinite(a).all() for a in (body,trans,root)):
        raise ValueError('Raw motion contains nonfinite values')
    return tuple(torch.from_numpy(a.copy()) for a in (trans,root,body))


def load_geometry(path):
    """Read SMPL rest geometry from a supplied asset.

    NPZ: rest_joints[24,3], optional parents[24], metadata JSON with
    schema_version, model_id, gender, betas, units='m', up_axis='x'/'y'/'z'.
    Rest joints stay in the SMPL model's native frame; global root rotations
    already encode any source-to-scene basis rotation.
    """
    with np.load(path, allow_pickle=False) as data:
        joints = torch.as_tensor(np.array(data['rest_joints'], dtype=np.float32))
        parents = tuple(int(p) for p in data['parents']) if 'parents' in data else SMPL_PARENTS
        metadata = read_metadata(data)
    for key in ('model_id','gender','betas','units','up_axis'):
        if key not in metadata:
            raise ValueError(f'Geometry metadata missing {key}')
    if metadata['units'] != 'm' or metadata['up_axis'] not in ('x','y','z'):
        raise ValueError('Geometry requires meter units and an explicit world up axis')
    if joints.shape != (24,3) or not torch.isfinite(joints).all() or parents != SMPL_PARENTS:
        raise ValueError('Expected finite SMPL24 rest joints and standard parent order')
    if not metadata['model_id'] or not isinstance(metadata['betas'], list):
        raise ValueError('Declare the supplied model identity and shape coefficients')
    return joints, parents, metadata


def convert_raw_motion(trans, root_orient, body_pose, rest_joints, *,
                       canonicalization='none', up_axis='z'):
    local = axis_angle_to_matrix(torch.cat((root_orient[:,None,:], body_pose), dim=1))
    global_rotations, positions = forward_kinematics(local, trans, rest_joints)
    offset = torch.zeros(3, dtype=positions.dtype, device=positions.device)
    if canonicalization == 'head_translation':
        offset = positions[0,15].clone()
        offset[('x','y','z').index(up_axis)] = 0
        positions = positions - offset
    elif canonicalization != 'none':
        raise ValueError('Supported canonicalization: none or head_translation; heading needs a verified convention')
    transform = {'world_translation_offset':offset.tolist(),
                 'world_rotation':torch.eye(3).tolist(),
                 'inverse_rule':'world_position = model_position + world_translation_offset'}
    return pack_motion(positions,global_rotations), transform


def load_stats(path):
    with np.load(path, allow_pickle=False) as data:
        mean = torch.from_numpy(np.array(data['mean'], dtype=np.float32))
        std = torch.from_numpy(np.array(data['std'], dtype=np.float32))
        metadata = read_metadata(data)
    if mean.shape != (24,9) or std.shape != mean.shape:
        raise ValueError('Statistics must be per-joint/per-channel [24,9]')
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Statistics must be finite with strictly positive std')
    if metadata.get('representation') != REPRESENTATION or metadata.get('split') != 'train':
        raise ValueError('Statistics must be train-only and match the motion representation')
    return mean,std,metadata


class MotionNormalizer(nn.Module):
    def __init__(self, mean, std, metadata=None):
        super().__init__()
        mean, std = torch.as_tensor(mean,dtype=torch.float32), torch.as_tensor(std,dtype=torch.float32)
        if mean.shape != (24,9) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError('Normalizer requires finite [24,9] mean and positive std')
        self.register_buffer('mean',mean.clone())
        self.register_buffer('std',std.clone())
        self.metadata = metadata or {}

    @classmethod
    def from_npz(cls,path):
        mean,std,metadata = load_stats(path)
        return cls(mean,std,metadata)

    def normalize(self,motion):
        if motion.shape[-2:] != (24,9):
            raise ValueError('Expected motion ending [24,9]')
        return (motion-self.mean)/self.std

    def denormalize(self,motion):
        return motion*self.std+self.mean
