"""Prepare per-joint motion using supplied SMPL rest geometry.

The geometry NPZ contains rest_joints[24,3], optional parents[24], and JSON
metadata with schema_version=1, model_id, gender, betas, units='m', up_axis.
"""
import argparse
import json

import numpy as np
import torch

from dataset.manifest import load_manifest,resolve_path,sha256_file
from dataset.cache_integrity import arrays_hash,validate_content_checksum
from dataset.motion_representation import (REPRESENTATION,atomic_save_npz,load_geometry,
                                          load_raw_motion,convert_raw_motion,read_metadata)


def prepare_motion(manifest_path,geometry_path,canonicalization='none',overwrite=False):
    records=load_manifest(manifest_path)
    joints,parents,geometry=load_geometry(geometry_path)
    geometry_hash=sha256_file(geometry_path)
    completed=0
    for record in records:
        source=resolve_path(record['raw']['motion'],manifest_path)
        output=resolve_path(record['prepared']['motion'],manifest_path)
        source_hash=sha256_file(source)
        if record.get('raw_motion_sha256') and record['raw_motion_sha256'] != source_hash:
            raise ValueError(f'Raw file changed after audit: {source}')
        fingerprint=dict(schema_version=1,representation=REPRESENTATION,
                         sequence_id=record['sequence_id'],geometry_hash=geometry_hash,
                         canonicalization=canonicalization,raw_motion_sha256=source_hash,
                         fps=record['fps'])
        if output.exists() and not overwrite:
            with np.load(output,allow_pickle=False) as data:
                metadata=read_metadata(data)
                arrays={key:np.array(data[key]) for key in data.files if key!='metadata'}
            if any(metadata.get(k)!=v for k,v in fingerprint.items()):
                raise ValueError(f'Stale prepared motion at {output}; use --overwrite deliberately')
            validate_content_checksum(arrays,metadata,output)
            completed+=1
            continue
        trans,root,body=load_raw_motion(source)
        if len(trans)!=record['num_frames']:
            raise ValueError(f'Length changed after audit: {source}')
        with torch.no_grad():
            motion,transform=convert_raw_motion(trans,root,body,joints,
                canonicalization=canonicalization,up_axis=geometry['up_axis'])
        metadata=dict(fingerprint,geometry=geometry,parents=list(parents),transform=transform,
                      root_convention='pelvis_position = smpl_translation + native_rest_pelvis',
                      rotation6d_convention='first_two_matrix_rows',
                      timestamp_origin='clip_start',timing_status=record.get('timing_status','declared'))
        arrays=dict(motion=motion.numpy(),timestamps=np.arange(len(motion),dtype=np.float64)/record['fps'])
        metadata['content_sha256']=arrays_hash(arrays)
        atomic_save_npz(output,**arrays,metadata=np.array(json.dumps(metadata,sort_keys=True)))
        completed+=1
    return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--geometry',required=True)
    parser.add_argument('--canonicalization',choices=['none','head_translation'],default='none')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    print(f'Prepared/verified {prepare_motion(args.manifest,args.geometry,args.canonicalization,args.overwrite)} motions')


if __name__=='__main__':
    main()
