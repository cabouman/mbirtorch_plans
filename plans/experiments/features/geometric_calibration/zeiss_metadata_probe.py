"""What geometry metadata the two Zeiss .txrm files hold, beyond what the reader parses.

The reader `mbirtorch/preprocess/zeiss.py` reads a known set of OLE streams: the per-view angles,
the per-view source and detector distances, the per-view stage positions, the per-view alignment
x and y shifts (which `correct_sino_shifts` applies), a scalar reconstruction center shift, and
the axis name and unit lists.  The question this probe answers is what else the files hold: a
detector tilt, a per-view coordinate frame, or per-view correction data the reader does not use.

For each file the probe prints every stream with its size, and then the content of each stream
whose name suggests geometry.  A stream whose size is a multiple of four is tried as a float32
array and printed as a count with summary statistics.  A stream that decodes as text is printed
as text.  Run parameters are at the top and the output goes to the job log.
"""
import struct

import numpy as np
import olefile

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
FILES = (
    ('bga', '/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm'),
    ('z62', '/depot/bouman/data/ORNL/versa/ParAM-Round-1_Z62.txrm'),
)
# A stream is dumped when its lowercase path contains one of these.
KEYWORDS = ('tilt', 'align', 'shift', 'axis', 'center', 'rotat', 'correct', 'coord', 'position',
            'matrix', 'transform', 'frame', 'offset', 'angle', 'distance', 'recon')
# Streams the reader already parses, for the summary at the end.
PARSED = ('ImageInfo/Angles', 'ImageInfo/XPosition', 'ImageInfo/YPosition', 'ImageInfo/ZPosition',
          'ImageInfo/StoRADistance', 'ImageInfo/DtoRADistance', 'alignment/x-shifts',
          'alignment/y-shifts', 'AutoRecon/CenterShift', 'ReconSettings/CenterShift',
          'PositionInfo/AxisNames', 'PositionInfo/AxisUnits')
MAX_PRINT_FLOATS = 6
MAX_ARRAY_BYTES = 1 << 22       # streams beyond 4 MB are projection data, listed but not dumped


def as_text(raw):
    """The stream decoded as text, or None when it does not look like text."""
    for encoding in ('utf-8', 'utf-16-le', 'latin-1'):
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, AttributeError):
            continue
        cleaned = text.replace('\x00', ';').strip('; \r\n')
        printable = sum(c.isprintable() or c in '\r\n\t;' for c in cleaned)
        if cleaned and printable / max(len(cleaned), 1) > 0.95 and any(c.isalpha() for c in cleaned):
            return cleaned[:400]
    return None


def as_floats(raw):
    """The stream as a float32 array, or None when the values do not look like measurements."""
    if len(raw) % 4 != 0 or len(raw) == 0:
        return None
    values = np.frombuffer(raw, dtype='<f4')
    finite = np.isfinite(values)
    if finite.mean() < 0.9 or not np.all(np.abs(values[finite]) < 1e9):
        return None
    return values


def dump_stream(ole, path, num_views):
    raw = ole.openstream(path).read()
    label = f'  {path}  ({len(raw)} bytes)'
    if len(raw) > MAX_ARRAY_BYTES:
        print(label + '  [large, not dumped]')
        return
    text = as_text(raw)
    if text is not None:
        print(label + f'  text: {text}')
        return
    values = as_floats(raw)
    if values is not None and values.size >= 1:
        head = ', '.join(f'{v:.6g}' for v in values[:MAX_PRINT_FLOATS])
        note = ''
        if num_views and values.size % num_views == 0 and values.size >= num_views:
            note = f'  [{values.size // num_views} per view]'
        print(label + f'  float32 x{values.size}{note}: min {np.nanmin(values):.6g}, '
              f'max {np.nanmax(values):.6g}, mean {np.nanmean(values):.6g}; first: {head}')
        return
    if len(raw) in (4, 8):
        as_int = struct.unpack('<I' if len(raw) == 4 else '<Q', raw)[0]
        print(label + f'  int: {as_int}')
        return
    print(label + f'  bytes, head: {raw[:24].hex()}')


def probe(name, path):
    print(f'\n================ {name}: {path} ================', flush=True)
    ole = olefile.OleFileIO(path)
    paths = ['/'.join(entry) for entry in ole.listdir(streams=True, storages=False)]
    try:
        num_views = struct.unpack('<I', ole.openstream('ImageInfo/NoOfImages').read()[:4])[0]
    except Exception:
        num_views = None
    print(f'{len(paths)} streams, NoOfImages = {num_views}')
    print('\nALL STREAMS (name, bytes):')
    for p in sorted(paths):
        print(f'  {p}  {ole.get_size(p)}')
    print('\nGEOMETRY-RELATED STREAMS:')
    matched = [p for p in sorted(paths) if any(k in p.lower() for k in KEYWORDS)]
    for p in matched:
        try:
            dump_stream(ole, p, num_views)
        except Exception as error:
            print(f'  {p}  [failed: {error!r}]')
    unread = [p for p in matched if p not in PARSED]
    print(f'\nSUMMARY for {name}: {len(matched)} geometry-related streams, '
          f'{len(unread)} of them not parsed by the reader:')
    for p in unread:
        print(f'  {p}')
    ole.close()


def main():
    for name, path in FILES:
        probe(name, path)
    print('\nZEISS_METADATA_PROBE DONE', flush=True)


if __name__ == '__main__':
    main()
