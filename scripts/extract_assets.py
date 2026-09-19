"""Extract split tar/gzip archives with checksums and resumable writes.

Full extraction requires all parts; --preview can inspect a downloaded prefix.
Use repeated --include patterns to select members. The JSONL journal records
completed files; the full inventory is published after archive validation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fnmatch
import glob
import gzip
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import tarfile
import tempfile

from dataset.manifest import atomic_json,sha256_file,_atomic_text


def part_info(path):
    path=Path(path).resolve()
    with path.open('rb') as f:
        prefix=f.read(256)
    result={'path':str(path),'bytes_on_disk':path.stat().st_size}
    if prefix.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
        fields=dict(line.split(' ',1) for line in prefix.decode().strip().splitlines())
        result.update(lfs_pointer=True,expected_bytes=int(fields['size']),
                      expected_sha256=fields['oid'].split(':',1)[1])
    else:
        result['lfs_pointer']=False
    return result


class SplitReader(io.RawIOBase):
    def __init__(self,parts,expected_sha256=None):
        self.parts=iter(parts)
        self.current=None
        self.expected_sha256=({str(Path(path).resolve()):digest for path,digest in expected_sha256.items()}
                              if expected_sha256 is not None else None)
        self.current_path=None
        self.current_digest=None
        self.current_bytes=0
        self.part_hashes=[]
        self.finished=False

    def _finish_part(self):
        self.current.close()
        self.current=None
        if self.current_digest is not None:
            actual=self.current_digest.hexdigest()
            expected=self.expected_sha256.get(str(self.current_path))
            if expected is not None and actual!=expected:
                raise ValueError(f'Archive part SHA256 mismatch: {self.current_path}; expected {expected}, got {actual}')
            self.part_hashes.append({'path':str(self.current_path),'sha256':actual,
                                    'sha256_verified':expected is not None,'bytes_read':self.current_bytes})
        self.current_digest=None

    def readable(self):
        return True

    def readinto(self,buffer):
        view=memoryview(buffer)
        total=0
        while total<len(view):
            if self.current is None:
                try:
                    path=next(self.parts)
                except StopIteration:
                    self.finished=True
                    break
                if part_info(path)['lfs_pointer']:
                    raise ValueError(f'Archive part still an LFS pointer: {path}')
                self.current=open(path,'rb')
                self.current_path=Path(path).resolve()
                self.current_digest=hashlib.sha256() if self.expected_sha256 is not None else None
                self.current_bytes=0
            count=self.current.readinto(view[total:])
            if count:
                if self.current_digest is not None:
                    self.current_digest.update(view[total:total+count])
                self.current_bytes+=count
                total+=count
            else:
                self._finish_part()
        return total

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current=None
        super().close()


def safe_name(name):
    if '\\' in name or '\x00' in name:
        raise ValueError(f'Unsafe archive path: {name!r}')
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or (path.parts and ':' in path.parts[0]):
        raise ValueError(f'Unsafe archive path: {name!r}')
    return path


def safe_target(root,name):
    relative=safe_name(name)
    target=root.joinpath(*relative.parts)
    current=root
    for part in relative.parts:
        current=current/part
        if current.is_symlink():
            raise ValueError(f'Extraction refuses existing symlink: {current}')
    if not target.resolve().is_relative_to(root):
        raise ValueError(f'Archive target escapes destination: {name}')
    return target


def inspect_archive(parts,limit=30):
    if limit<1:
        raise ValueError('Preview limit must be positive')
    members=[]
    with SplitReader(parts) as raw,io.BufferedReader(raw,buffer_size=65536) as stream:
        with tarfile.open(fileobj=stream,mode='r|*') as archive:
            for member in archive:
                safe_name(member.name)
                members.append({'name':member.name,'bytes':member.size,
                                'kind':'file' if member.isfile() else 'directory' if member.isdir() else 'unsupported'})
                if len(members)>=limit:
                    break
    return members


@contextmanager
def inventory_writer(path):
    """Publish the inventory atomically after successful archive traversal."""
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=path.name+'.partial.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def extract_archive(parts,output,*,includes=(),journal_path=None,expected_parts=None,inventory_path=None,
                    min_free_bytes=1024**3):
    if isinstance(min_free_bytes,bool) or not isinstance(min_free_bytes,int) or min_free_bytes<0:
        raise ValueError('Minimum free-space reserve must be a nonnegative integer')
    parts=[Path(p).resolve() for p in parts]
    if not parts or len(set(parts))!=len(parts):
        raise ValueError('Supply a nonempty ordered list of distinct archive parts')
    inventory=[part_info(p) for p in parts]
    if any(p['lfs_pointer'] for p in inventory):
        raise ValueError('Full extraction requires all archive parts; some remain LFS pointers')
    expected={Path(p.get('path',p.get('part',''))).name:p for p in (expected_parts or [])}
    if expected and {p.name for p in parts}!=set(expected):
        raise ValueError('Archive part list differs from the recorded complete inventory')
    for item in inventory:
        wanted=expected.get(Path(item['path']).name)
        if wanted:
            expected_size=wanted.get('expected_bytes',wanted.get('bytes'))
            if expected_size is not None and item['bytes_on_disk']!=expected_size:
                raise ValueError(f'Incomplete archive part: {item["path"]}')
            item['expected_sha256']=wanted.get('expected_sha256',wanted.get('sha256'))
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    journal_path=Path(journal_path) if journal_path else output/'.extraction.jsonl'
    journal_path=journal_path.resolve()
    inventory_path=Path(inventory_path).resolve() if inventory_path else output/'.archive_inventory.jsonl'
    report_path=output/'.extraction_report.json'
    reserved={journal_path,inventory_path,report_path}
    if len(reserved)!=3 or reserved.intersection(parts):
        raise ValueError('Extraction journal, inventory, report and archive parts must have distinct paths')
    journal_path.parent.mkdir(parents=True,exist_ok=True)
    context={'kind':'archive','schema_version':1,'parts':inventory,'destination':str(output)}
    completed={}
    lines=[]
    if journal_path.exists():
        raw_lines=journal_path.read_text().splitlines()
        for index,line in enumerate(raw_lines):
            try:
                item=json.loads(line)
            except json.JSONDecodeError:
                if index!=len(raw_lines)-1:
                    raise ValueError('Corrupt extraction journal')
                break
            lines.append(item)
        if not lines or lines[0]!=context:
            raise ValueError('Extraction journal belongs to different archive parts/destination')
        completed={entry['name']:entry for entry in lines[1:] if entry.get('kind')=='file'}
    if not lines:
        lines=[context]
    # Invalidate stale completion metadata only when it belongs to this extraction.
    stale_metadata=[]
    for path in (report_path,inventory_path):
        if not path.exists():
            continue
        try:
            if path==inventory_path:
                with path.open() as existing:
                    previous=json.loads(existing.readline())
            else:
                previous=json.loads(path.read_text())
        except (ValueError,OSError) as exc:
            raise FileExistsError(f'Refusing untracked extraction metadata: {path}') from exc
        if not isinstance(previous,dict) or previous.get('parts')!=inventory or previous.get('destination')!=str(output):
            raise FileExistsError(f'Refusing untracked extraction metadata: {path}')
        stale_metadata.append(path)
    for path in stale_metadata:
        path.unlink()
    _atomic_text(journal_path,''.join(json.dumps(item,sort_keys=True)+'\n' for item in lines))
    counts={'seen':0,'selected':0,'written':0,'verified_existing':0,'skipped':0,'bytes_written':0,
            'selected_bytes':0,'observed_archive_regular_bytes':0}
    seen=set()
    expected_hashes={item['path']:item.get('expected_sha256') for item in inventory}
    with inventory_writer(inventory_path) as member_inventory,journal_path.open('a') as journal,SplitReader(parts,expected_hashes) as raw,io.BufferedReader(raw) as stream:
        member_inventory.write(json.dumps(dict(context,kind='archive_inventory'),sort_keys=True)+'\n')
        decoded=gzip.GzipFile(fileobj=stream,mode='rb') if stream.peek(2)[:2]==b'\x1f\x8b' else stream
        with tarfile.open(fileobj=decoded,mode='r|') as archive:
            for member in archive:
                counts['seen']+=1
                if member.isfile():
                    counts['observed_archive_regular_bytes']+=member.size
                normalized=safe_name(member.name).as_posix()
                selected=not includes or any(fnmatch.fnmatchcase(normalized,pattern) for pattern in includes)
                member_inventory.write(json.dumps({'kind':'member','name':normalized,'bytes':member.size,
                    'type':'file' if member.isfile() else 'directory' if member.isdir() else 'unsupported',
                    'selected':bool(selected and member.isfile())},sort_keys=True)+'\n')
                if normalized in ('.','') or member.isdir():
                    continue
                if not selected:
                    counts['skipped']+=1
                    continue
                counts['selected']+=1
                if not member.isfile():
                    raise ValueError(f'Only regular files may be extracted; refused {member.name}')
                counts['selected_bytes']+=member.size
                if normalized in seen:
                    raise ValueError(f'Duplicate selected archive member: {normalized}')
                seen.add(normalized)
                target=safe_target(output,normalized)
                if target in reserved:
                    raise ValueError(f'Archive member conflicts with extraction metadata: {normalized}')
                prior=completed.get(normalized)
                if prior and target.is_file() and target.stat().st_size==member.size and sha256_file(target)==prior['sha256']:
                    # Verify resumed journal hashes against the current archive payload.
                    source=archive.extractfile(member)
                    if source is None:
                        raise ValueError(f'Unable to read resumed archive member: {normalized}')
                    digest=hashlib.sha256()
                    with source:
                        for block in iter(lambda:source.read(1024*1024),b''):
                            digest.update(block)
                    if digest.hexdigest()!=prior['sha256']:
                        raise ValueError(f'Resumed member differs from current archive: {normalized}; use a fresh destination')
                    counts['verified_existing']+=1
                    continue
                # Only damaged files tracked by the journal may be overwritten.
                untracked=target.exists() and prior is None
                if untracked and not target.is_file():
                    raise FileExistsError(f'Refusing untracked destination path: {target}')
                free=shutil.disk_usage(output).free
                if free<member.size+min_free_bytes:
                    raise OSError(f'Insufficient disk space for {normalized}: need {member.size} bytes plus '
                                  f'{min_free_bytes} reserve, available {free}; completed files remain resumable')
                target.parent.mkdir(parents=True,exist_ok=True)
                fd,temporary=tempfile.mkstemp(prefix=target.name+'.extracting.',dir=target.parent)
                digest=hashlib.sha256()
                written=0
                try:
                    source=archive.extractfile(member)
                    if source is None:
                        raise ValueError(f'Unable to read archive member: {normalized}')
                    with source,os.fdopen(fd,'wb') as destination:
                        for block in iter(lambda:source.read(1024*1024),b''):
                            destination.write(block)
                            digest.update(block)
                            written+=len(block)
                        destination.flush()
                        os.fsync(destination.fileno())
                    if written!=member.size:
                        raise ValueError(f'Truncated archive member: {normalized}')
                    # Recover a rename-before-journal crash only if contents match.
                    if untracked:
                        if target.stat().st_size!=written or sha256_file(target)!=digest.hexdigest():
                            raise FileExistsError(f'Refusing untracked destination file: {target}')
                    else:
                        os.replace(temporary,target)
                    record={'kind':'file','name':normalized,'bytes':written,'sha256':digest.hexdigest()}
                    journal.write(json.dumps(record,sort_keys=True)+'\n')
                    journal.flush()
                    os.fsync(journal.fileno())
                    counts['written']+=1
                    counts['bytes_written']+=written
                    if counts['written']%100==0:
                        print(json.dumps(counts),flush=True)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        # Read beyond tar's end marker to validate the gzip trailer and CRC.
        for _ in iter(lambda:decoded.read(1024*1024),b''):
            pass
        if not raw.finished:
            if stream.read(1) or not raw.finished:
                raise ValueError('Archive traversal did not consume all archive parts')
        if len(raw.part_hashes)!=len(parts):
            raise ValueError('Archive traversal did not finish every part checksum')
        part_hashes=list(raw.part_hashes)
        if decoded is not stream:
            decoded.close()
    report=dict(schema_version=1,complete=True,counts=counts,parts=inventory,
                selected_patterns=list(includes),journal=str(journal_path),destination=str(output),
                inventory=str(inventory_path),min_free_bytes=min_free_bytes,
                part_checksums=part_hashes,all_expected_part_checksums_verified=all(item['sha256_verified'] for item in part_hashes))
    atomic_json(report_path,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-glob',required=True,help='Quoted glob; parts read in lexicographic order')
    parser.add_argument('--output',required=True)
    parser.add_argument('--include',action='append',default=[])
    parser.add_argument('--preview',type=int,help='List first N headers; permits downloaded prefix only')
    parser.add_argument('--parts-manifest',help='JSON inventory containing parts with expected_bytes')
    parser.add_argument('--inventory-path',help='Complete JSONL member inventory; defaults to output/.archive_inventory.jsonl')
    parser.add_argument('--min-free-bytes',type=int,default=1024**3,help='Reserve this free space before writing each file (default 1 GiB)')
    args=parser.parse_args()
    parts=sorted(glob.glob(args.parts_glob))
    if not parts:
        parser.error('No archive parts match')
    if args.preview:
        members=inspect_archive(parts,args.preview)
        atomic_json(args.output,{'parts':[part_info(p) for p in parts],'preview':members,'complete_inventory':False})
        print(json.dumps(members,indent=2))
    else:
        expected=json.loads(Path(args.parts_manifest).read_text())['parts'] if args.parts_manifest else None
        print(json.dumps(extract_archive(parts,args.output,includes=args.include,expected_parts=expected,
                                        inventory_path=args.inventory_path,min_free_bytes=args.min_free_bytes),indent=2))


if __name__=='__main__':
    main()
