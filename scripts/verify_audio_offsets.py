"""Match released motion to source frames by joint rotations.

Supports original AIST `smpl_poses` and EDGE `q` axis angles, or NPZ body/root
aliases. Supply the source FPS. Body rotations are compared as
matrices; root rotations must agree under one constant scene-basis rotation.
Motion matching alone does not establish the source motion's music origin.
Only an explicit --music-origin-map can mark absolute music offsets verified.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import importlib
import json
import math
from pathlib import Path
import pickle

import numpy as np

from dataset.manifest import load_manifest,write_manifest,resolve_path,sha256_file,atomic_json


class NumpyDataUnpickler(pickle.Unpickler):
    """Limit legacy dataset deserialization to NumPy numeric array primitives."""
    ALLOWED={('numpy','ndarray'),('numpy','dtype'),
             ('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),
             ('numpy.core.multiarray','scalar'),('numpy._core.multiarray','scalar'),
             ('numpy.core.numeric','_frombuffer'),('numpy._core.numeric','_frombuffer')}

    def find_class(self,module,name):
        if (module,name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f'Unsupported object in numeric motion pickle: {module}.{name}')
        return getattr(importlib.import_module(module),name)


def numeric_pickle_load(file):
    return NumpyDataUnpickler(file,encoding='latin1').load()


def _numeric(array,name):
    array=np.asarray(array)
    if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite numeric data')
    return array.astype(np.float64)


def motion_arrays(data):
    if not isinstance(data,dict):
        raise ValueError('Source motion must be a dictionary of numeric arrays')
    pose_key=next((key for key in ('smpl_poses','q') if key in data),None)
    if pose_key is not None:
        poses=_numeric(data[pose_key],pose_key)
        if poses.ndim not in (2,3) or poses.shape[1:] not in ((72,),(24,3)):
            raise ValueError(f'{pose_key} must contain SMPL24 axis angles [T,72] or [T,24,3]')
        poses=poses.reshape(-1,24,3)
        return poses[:,1:],poses[:,0]
    keys=[key for key in ('pose_body','body_pose') if key in data]
    if not keys:
        raise ValueError('No supported AIST/EDGE SMPL axis-angle representation found')
    if len(keys)==2 and not np.array_equal(data[keys[0]],data[keys[1]]):
        raise ValueError('Conflicting body-pose aliases')
    body=_numeric(data[keys[0]],keys[0])
    if body.ndim not in (2,3) or body.shape[1:] not in ((69,),(23,3)):
        raise ValueError('Body axis angles must be [T,69] or [T,23,3]')
    root=_numeric(data['root_orient'],'root_orient')
    if root.shape!=(len(body),3):
        raise ValueError('Invalid root axis-angle dimensions')
    return body.reshape(-1,23,3),root


def read_motion(path):
    path=Path(path).resolve()
    stat=path.stat()
    return _read_motion_cached(str(path),stat.st_size,stat.st_mtime_ns)


@lru_cache(maxsize=8)
def _read_motion_cached(path,size,mtime_ns):
    # Size and mtime invalidate cached arrays when a source file changes.
    path=Path(path)
    if path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as archive:
            data={key:np.array(archive[key]) for key in archive.files if key!='metadata'}
    elif path.suffix in ('.pkl','.pickle'):
        with path.open('rb') as f:
            data=numeric_pickle_load(f)
    else:
        raise ValueError('Expected numeric motion NPZ or restricted NumPy pickle')
    body,root=motion_arrays(data)
    if not len(body):
        raise ValueError('Source motion is empty')
    fps=float(data['fps']) if 'fps' in data else None
    return body,root,fps,sha256_file(path)


def rotation_matrices(aa):
    aa=np.asarray(aa,dtype=np.float64)
    x,y,z=np.moveaxis(aa,-1,0)
    zero=np.zeros_like(x)
    skew=np.stack((zero,-z,y,z,zero,-x,-y,x,zero),axis=-1).reshape(aa.shape[:-1]+(3,3))
    theta=np.linalg.norm(aa,axis=-1)
    a=np.sinc(theta/np.pi)[...,None,None]
    b=(0.5*np.sinc(theta/(2*np.pi))**2)[...,None,None]
    return np.eye(3)+a*skew+b*(skew@skew)


def match_motion_arrays(released_body,source_body,*,source_fps,target_fps,
                        released_root=None,source_root=None,tolerance=1e-4):
    """Return all matches for the caller to check for ambiguity.

    Error is the maximum joint rotation-matrix Frobenius distance. Equivalent
    axis-angle wraps therefore match. Sampling policies are merged only when
    their source frame indices are identical.
    """
    if any(not math.isfinite(v) or v<=0 for v in (source_fps,target_fps,tolerance)):
        raise ValueError('Frame rates and tolerance must be finite and positive')
    target=rotation_matrices(np.asarray(released_body).reshape(-1,23,3))
    source=rotation_matrices(np.asarray(source_body).reshape(-1,23,3))
    if not len(target) or not len(source):
        raise ValueError('Cannot match empty motion')
    if (released_root is None)!=(source_root is None):
        raise ValueError('Supply both source and released root rotations for the scene-basis check')
    root_target=rotation_matrices(released_root) if released_root is not None else None
    root_source=rotation_matrices(source_root) if source_root is not None else None
    ratio=source_fps/target_fps
    policies={}
    if ratio>=1 and math.isclose(ratio,round(ratio),abs_tol=1e-9):
        # Include all source phases instead of assuming frame-zero decimation.
        for phase in range(int(round(ratio))):
            policies[f'integer_stride_{int(round(ratio))}_phase_{phase}']=np.arange(phase,len(source),int(round(ratio)))
    count=int(len(source)*target_fps/source_fps)
    if 0<count<=len(source):
        policies['legacy_global_linspace']=np.linspace(0,len(source)-1,num=count,dtype=np.int64)
    if not policies:
        raise ValueError('Upsampling source motion is unsupported; supply the original or correctly sampled source')
    matches={}
    for policy,indices in policies.items():
        possible=len(indices)-len(target)+1
        if possible<=0:
            continue
        first_error=np.linalg.norm(source[indices[:possible]]-target[0],axis=(-2,-1)).max(-1)
        for start in np.flatnonzero(first_error<=tolerance):
            source_indices=indices[start:start+len(target)]
            max_error=float(np.linalg.norm(source[source_indices]-target,axis=(-2,-1)).max())
            if max_error>tolerance:
                continue
            root_error=None
            if root_target is not None:
                change=root_target[0]@root_source[source_indices[0]].T
                root_error=float(np.linalg.norm(change@root_source[source_indices]-root_target,axis=(-2,-1)).max())
                if root_error>tolerance:
                    continue
            key=tuple(source_indices.tolist())
            if key in matches:
                matches[key]['sampling_policies'].append(policy)
            else:
                matches[key]=dict(source_frame_indices=list(key),
                    source_frame_times_seconds=(source_indices/source_fps).tolist(),
                    sampling_policies=[policy],body_max_matrix_error=max_error,
                    root_max_matrix_error=root_error)
    return list(matches.values())


def verify_offsets(manifest_path,source_root,output_manifest,report_path,*,source_fps,
                   music_origin_map=None,tolerance=1e-4,recursive=False):
    if any(not math.isfinite(v) or v<=0 for v in (source_fps,tolerance)):
        raise ValueError('Source FPS and tolerance must be finite and positive')
    input_manifest_hash=sha256_file(manifest_path)
    records=load_manifest(manifest_path)
    source_root=Path(source_root).resolve()
    source_files=sorted(source_root.rglob('*') if recursive else source_root.glob('*'))
    source_files=[p for p in source_files if p.is_file() and p.suffix in ('.pkl','.pickle','.npz')]
    if not source_files:
        raise FileNotFoundError(f'No source motion arrays in {source_root}')
    origins=json.loads(Path(music_origin_map).read_text()) if music_origin_map else {}
    if not isinstance(origins,dict):
        raise ValueError('Music origin map must be an object keyed by source motion stem')
    origin_hash=sha256_file(music_origin_map) if music_origin_map else None
    results=[]
    for record in records:
        source_id=record['source_motion_id']
        candidates=[p for p in source_files if p.stem==source_id or p.stem.startswith(source_id+'_slice')]
        released_path=resolve_path(record['raw']['motion'],manifest_path)
        released_body,released_root,_,released_hash=read_motion(str(released_path))
        matches=[]
        errors=[]
        for path in candidates:
            try:
                body,root,declared_fps,source_hash=read_motion(str(path))
                if declared_fps is not None and not math.isclose(declared_fps,source_fps):
                    raise ValueError('Source FPS metadata disagrees with --source-fps')
                found=match_motion_arrays(released_body,body,source_fps=source_fps,target_fps=record['fps'],
                    released_root=released_root,source_root=root,tolerance=tolerance)
                matches.extend(dict(match,source_path=str(path),source_sha256=source_hash,
                                    source_stem=path.stem,source_fps=source_fps) for match in found)
            except (ValueError,KeyError,pickle.UnpicklingError,OSError) as exc:
                errors.append({'source_path':str(path),'error':str(exc)})
        record.pop('motion_match',None)
        # Clear stale verification instead of retaining an earlier guessed mapping.
        for key in ('source_offset_evidence','source_audio_timestamps_seconds','audio_reference_kind','audio_reference_id'):
            record.pop(key,None)
        record['source_start_time_seconds']=None
        record['source_offset_status']='unverified'
        status='ambiguous' if len(matches)>1 else 'unmatched'
        if len(matches)==1 and not errors:
            match=matches[0]
            match.update(unique_match=True,algorithm='body_rotation_matrix_and_constant_root_basis_v1',
                         tolerance=tolerance,released_motion_sha256=released_hash)
            record['motion_match']=match
            status='motion_verified_music_origin_unknown'
            origin=origins.get(match['source_stem'])
            if origin is not None:
                if not isinstance(origin,dict):
                    raise ValueError(f'Invalid music origin entry for {match["source_stem"]}')
                offset=origin.get('music_start_time_seconds')
                evidence=origin.get('evidence')
                if isinstance(offset,bool) or not isinstance(offset,(float,int)) or not math.isfinite(offset) or offset<0 or not isinstance(evidence,(str,dict)) or not evidence:
                    raise ValueError(f'Invalid music origin/evidence for {match["source_stem"]}')
                times=np.asarray(match['source_frame_times_seconds'])+offset
                record.update(source_start_time_seconds=float(times[0]),source_audio_timestamps_seconds=times.tolist(),
                              source_offset_status='verified',audio_reference_kind='full_song',audio_reference_id=record['music_id'],
                              source_offset_evidence={'motion_match':match,'music_origin':origin,'music_origin_map_sha256':origin_hash})
                status='audio_verified'
        # Output manifests can live elsewhere; preserve raw/prepared path meaning.
        for key in ('raw','prepared'):
            record[key]={k:str(resolve_path(v,manifest_path)) for k,v in record.get(key,{}).items()}
        results.append({'sequence_id':record['sequence_id'],'status':status,
                        'compatible_matches':len(matches),'matches':matches,'source_errors':errors})
    write_manifest(output_manifest,records)
    report=dict(schema_version=1,source_fps=source_fps,tolerance=tolerance,
                input_manifest_sha256=input_manifest_hash,results=results,
                note='Source FPS and music origin are explicit inputs; segment/slice suffixes are never converted to time')
    atomic_json(report_path,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--source-fps',required=True,type=float)
    parser.add_argument('--output-manifest',required=True)
    parser.add_argument('--report',required=True)
    parser.add_argument('--music-origin-map')
    parser.add_argument('--tolerance',type=float,default=1e-4)
    parser.add_argument('--recursive',action='store_true')
    args=parser.parse_args()
    report=verify_offsets(args.manifest,args.source_root,args.output_manifest,args.report,
                          source_fps=args.source_fps,music_origin_map=args.music_origin_map,
                          tolerance=args.tolerance,recursive=args.recursive)
    counts={status:sum(r['status']==status for r in report['results']) for status in sorted({r['status'] for r in report['results']})}
    print(json.dumps(counts,indent=2))
    if any(r['status'] in ('ambiguous','unmatched') for r in report['results']):
        raise SystemExit(1)


if __name__=='__main__':
    main()
