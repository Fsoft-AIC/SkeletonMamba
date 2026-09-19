"""Audit raw archive headers and split membership without NumPy or Torch.

FPS is supplied by the caller; media timestamps are not verified here.
"""
import argparse
from collections import Counter

from dataset.manifest import atomic_json, audit_records


def summarize(records, errors):
    partitions = {}
    for split in ('train','test','auxiliary'):
        subset = [r for r in records if r['split'] == split]
        partitions[split] = dict(clips=len(subset),
            source_ids=len({r['source_motion_id'] for r in subset}),
            scenes=sorted({r['scene_id'] for r in subset}),
            frames=sum(r['num_frames'] for r in subset),
            length_counts=dict(sorted(Counter(r['num_frames'] for r in subset).items())),
            schema_counts=dict(Counter(r['body_pose_key'] for r in subset)),
            missing_videos=[r['sequence_id'] for r in subset if 'missing_video' in r['quality_flags']])
    train = [r for r in records if r['split']=='train']
    test = [r for r in records if r['split']=='test']
    overlap = {key: sorted({r[key] for r in train}&{r[key] for r in test})
               for key in ('source_motion_id','music_id','scene_id','candidate_choreography_id')}
    names = Counter(r['sequence_id'] for r in records)
    return dict(schema_version=1,partitions=partitions,overlap=overlap,
                duplicate_folder_names=sorted(k for k,v in names.items() if v>1),
                errors=errors,scope='NPZ headers, file presence and listed source membership; media not fully decoded',
                choreography_key_status='candidate only; source dataset semantics need confirmation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--fps',required=True,type=float)
    parser.add_argument('--output',required=True)
    parser.add_argument('--hash-files',action='store_true')
    args = parser.parse_args()
    records,errors = audit_records(args.root,args.fps,args.hash_files)
    report = summarize(records,errors)
    atomic_json(args.output,report)
    print(f'Audited {len(records)} readable motions; {len(errors)} errors. Report: {args.output}')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
