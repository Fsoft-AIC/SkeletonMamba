"""Build deterministic manifests preserving the official train/test membership."""
import argparse
import hashlib
from pathlib import Path

from dataset.manifest import audit_records, atomic_json, write_manifest, sha256_file
from scripts.audit_dataset import summarize


def build_manifests(root, output, fps, validation_fraction=0.1, seed=0,
                    grouping='source', hash_files=False):
    if not 0 <= validation_fraction < 1:
        raise ValueError('validation_fraction must lie in [0,1)')
    if grouping not in ('source','candidate_choreography'):
        raise ValueError('Grouping must be source or explicitly provisional candidate_choreography')
    records,errors = audit_records(root,fps,hash_files)
    report = summarize(records,errors)
    if report['overlap']['source_motion_id'] or report['overlap']['scene_id']:
        raise ValueError('Official train/test source or scene overlap; inspect release protocol')
    selected = [r for r in records if r['split'] in ('train','test') and 'missing_video' not in r['quality_flags']]
    selected_keys = {(r['raw_tree'],r['sequence_id']) for r in selected}
    excluded = [dict(r,disposition='auxiliary' if r['split']=='auxiliary' else 'excluded')
                for r in records if (r['raw_tree'],r['sequence_id']) not in selected_keys]
    group_key = 'source_motion_id' if grouping=='source' else 'candidate_choreography_id'
    groups = sorted({r[group_key] for r in selected if r['split']=='train'},
                    key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest())
    count = min(int(round(len(groups)*validation_fraction)),max(0,len(groups)-1))
    if validation_fraction > 0 and len(groups)>1:
        count = max(1,min(count,len(groups)-1))
    val_groups = set(groups[:count])
    output = Path(output).resolve()
    partitions = {split:[] for split in ('train','val','test')}
    for record in selected:
        if record['split']=='train' and record[group_key] in val_groups:
            record['split']='val'
        record['validation_grouping'] = grouping
        record['validation_seed'] = seed
        record['prepared'] = {kind:str(output/'features'/record['split']/record['sequence_id']/f'{kind}.npz')
                              for kind in ('motion','audio','video')}
        partitions[record['split']].append(record)
    for split,subset in partitions.items():
        write_manifest(output/f'{split}.jsonl',subset)
    # Excluded records can share sequence names across source trees; retain both.
    write_manifest(output/'excluded.jsonl',excluded)
    report.update(validation=dict(fraction=validation_fraction,seed=seed,grouping=grouping,
                                  train_groups=len(groups)-count,validation_groups=count),
                  selected_counts={k:len(v) for k,v in partitions.items()},
                  split_file_hashes={s:sha256_file(Path(root)/'split'/f'{s}.txt') for s in ('train','test')},
                  timing_note='FPS explicitly declared; source music offsets remain unverified')
    atomic_json(output/'audit.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--fps',required=True,type=float)
    parser.add_argument('--validation-fraction',type=float,default=0.1)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--grouping',choices=['source','candidate_choreography'],default='source')
    parser.add_argument('--hash-files',action='store_true')
    args=parser.parse_args()
    report=build_manifests(args.root,args.output,args.fps,args.validation_fraction,args.seed,args.grouping,args.hash_files)
    print(report['selected_counts'])
    print(f"Excluded/auxiliary sequences and raw errors recorded in {args.output}/audit.json")
    if report['errors']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
