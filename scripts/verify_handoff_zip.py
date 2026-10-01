"""Verify a private research or local-checkpoint zip using only Python stdlib."""
import hashlib
import json
import sys
import zipfile


def main():
    with zipfile.ZipFile(sys.argv[1]) as archive:
        name = next(name for name in ['REPLICATION_MANIFEST.json', 'CHECKPOINT_MANIFEST.json'] if name in archive.namelist())
        manifest = json.loads(archive.read(name))
        for entry in manifest['files']:
            digest = hashlib.sha256()
            size = 0
            with archive.open(entry['path']) as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
                    size += len(chunk)
            if size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
                raise ValueError('File integrity mismatch: ' + entry['path'])
        print(json.dumps({'files_verified': len(manifest['files']), 'all_hashes_verified': True}))


if __name__ == '__main__':
    main()
