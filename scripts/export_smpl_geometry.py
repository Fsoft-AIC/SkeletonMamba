"""Export rest joints from a licensed SMPL model using smplx.

Joint offsets stay in the native SMPL frame; --up-axis describes the released
motion/world frame. Supply the model file and its shape coefficients locally.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset.manifest import sha256_file
from dataset.motion_representation import atomic_save_npz,load_geometry
from utils.kinematics import SMPL_PARENTS


def export_geometry(model_path,gender,betas,up_axis,output):
    path=Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Supply the licensed SMPL model file: {path}')
    if gender not in ('male','female','neutral') or up_axis not in ('x','y','z'):
        raise ValueError('Gender and world up axis must be explicitly selected')
    betas=np.asarray(betas,dtype=np.float32)
    if betas.ndim!=1 or len(betas)==0 or not np.isfinite(betas).all():
        raise ValueError('Supply a finite list of shape coefficients')
    try:
        from smplx import SMPL
    except ImportError as exc:
        raise ImportError('SMPL export requires the optional smplx package and your licensed model file') from exc
    model=SMPL(str(path),gender=gender,num_betas=len(betas),batch_size=1)
    if model.num_betas!=len(betas):
        raise ValueError('Supplied shape-coefficient count differs from the model capacity')
    with torch.no_grad():
        body=model(betas=torch.from_numpy(betas)[None],body_pose=torch.zeros(1,69),
                   global_orient=torch.zeros(1,3),transl=torch.zeros(1,3))
        joints=body.joints[0,:24].detach().cpu().numpy()
    parents=model.parents.detach().cpu().numpy()
    if tuple(parents.tolist())!=SMPL_PARENTS:
        raise ValueError('Model is not the expected SMPL24 skeleton')
    metadata=dict(schema_version=1,model_id=sha256_file(path),model_filename=path.name,
                  gender=gender,betas=betas.tolist(),units='m',up_axis=up_axis,
                  rest_joint_frame='native_smpl',world_up_axis_status='user_declared',
                  exporter='smplx.SMPL',smplx_version=__import__('importlib.metadata',fromlist=['version']).version('smplx'))
    atomic_save_npz(output,rest_joints=joints,parents=parents,
                    metadata=np.array(json.dumps(metadata,sort_keys=True)))
    load_geometry(output)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',required=True)
    parser.add_argument('--gender',required=True,choices=['male','female','neutral'])
    parser.add_argument('--betas',required=True,type=float,nargs='+')
    parser.add_argument('--up-axis',required=True,choices=['x','y','z'])
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    export_geometry(args.model_path,args.gender,args.betas,args.up_axis,args.output)
    print(f'Exported supplied SMPL geometry: {args.output}')


if __name__=='__main__':
    main()
