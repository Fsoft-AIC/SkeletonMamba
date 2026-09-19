"""Streaming train-only per-joint/channel statistics for prepared motion."""
import argparse
import json

import numpy as np

from dataset.manifest import load_manifest,resolve_path,sha256_file
from dataset.egoaistpp_dataset import load_prepared,validate_prepared_file_checksum
from dataset.motion_representation import REPRESENTATION,atomic_save_npz


def compute_statistics(manifest_path,output,epsilon=1e-6):
    if epsilon<=0:
        raise ValueError('epsilon must be positive')
    records=load_manifest(manifest_path)
    if not records or any(r['split']!='train' for r in records):
        raise ValueError('Statistics require a nonempty training-only manifest')
    count=0
    mean=np.zeros((24,9),dtype=np.float64)
    m2=np.zeros_like(mean)
    contract=None
    prepared_hashes=[]
    for record in records:
        path=resolve_path(record['prepared']['motion'],manifest_path)
        validate_prepared_file_checksum(path,record,'motion')
        motion,_,metadata=load_prepared(path,'motion',record['sequence_id'])
        values=motion.numpy().astype(np.float64)
        if len(values)!=record['num_frames']:
            raise ValueError('Prepared motion length differs from manifest')
        current={k:metadata.get(k) for k in ('geometry_hash','canonicalization')}
        if any(v is None for v in current.values()) or (contract is not None and current!=contract):
            raise ValueError('Statistics cannot mix geometry or canonicalization contracts')
        contract=current
        n=len(values)
        batch_mean=values.mean(0)
        batch_m2=((values-batch_mean)**2).sum(0)
        delta=batch_mean-mean
        total=count+n
        m2+=batch_m2+delta**2*count*n/total
        mean+=delta*n/total
        count=total
        prepared_hashes.append({'sequence_id':record['sequence_id'],'sha256':sha256_file(path)})
    std=np.maximum(np.sqrt(m2/count),epsilon)
    metadata=dict(schema_version=1,representation=REPRESENTATION,split='train',
                  axes='valid_frames_per_joint_per_channel',count=count,epsilon=epsilon,
                  variance='population',manifest_sha256=sha256_file(manifest_path),
                  prepared_motion_hashes=prepared_hashes,**contract)
    atomic_save_npz(output,mean=mean.astype(np.float32),std=std.astype(np.float32),
                    metadata=np.array(json.dumps(metadata,sort_keys=True)))
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--epsilon',type=float,default=1e-6)
    args=parser.parse_args()
    metadata=compute_statistics(args.manifest,args.output,args.epsilon)
    print(f'Statistics computed from {metadata["count"]} valid training frames: {args.output}')


if __name__=='__main__':
    main()
