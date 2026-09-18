"""Write ``web/lite_pins.txt``: the versions the Gradio-Lite page holds gradio's
dependencies at.

Why the file exists.  Gradio-Lite 5.45.0 is the last release of the Lite
runtime, and it resolves gradio's dependencies from PyPI at whatever versions
are current when a page loads.  Three of them had moved on by 2026-09-10 and
each broke the page in a different way (``gv5_web_findings.md``).  The page
therefore holds every dependency that comes from PyPI at the newest release
published before the runtime itself, so that the runtime runs against the
packages it was released with.  Packages that Pyodide ships are not listed,
because the runtime takes those from Pyodide's own lock file at fixed versions.

Where the names come from.  :data:`PACKAGES` is the list of packages that
``micropip.list()`` reported with source ``pypi`` after the runtime installed
gradio 5.45.0 and gradio_client 1.13.0 in a browser on 2026-09-10.  A package
missing from the list is installed at its newest version, as before.

There are no command-line arguments.  Run:

    cd plans/geometry_viewer/experiments
    python gv5_lite_pins.py

The script reads PyPI over the network, prints one line per package, and
rewrites ``web/lite_pins.txt``.  ``gv5_build_web.py`` reads that file.
"""

import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PINS_FILE = os.path.join(HERE, 'web', 'lite_pins.txt')

#: When Gradio-Lite 5.45.0 was published.  The gradio 5.45.0 wheel reached
#: PyPI at this time, and the npm package @gradio/lite 5.45.0 the same day.
#: Every pin is the newest release before this moment.
CUTOFF = '2025-09-10T17:06:22Z'

#: The packages the runtime installs from PyPI, in the order micropip listed
#: them.  gradio and gradio_client themselves come from the runtime's own
#: wheels and are not listed.
PACKAGES = (
    'aiofiles', 'annotated-doc', 'anyio', 'fastapi', 'filelock', 'groovy',
    'huggingface-hub', 'pydub', 'python-multipart', 'safehttpx',
    'semantic-version', 'starlette', 'tomlkit', 'typing-inspection',
    'websockets',
)


def is_final(version):
    """True for a release version: no alpha, beta, release candidate, or dev
    marker.  PEP 440 spells those with a letter, and a release has none."""
    return version.replace('.', '').isdigit()


def newest_before_cutoff(name):
    """The newest release of ``name`` on PyPI that has a pure Python wheel,
    is not yanked, and was uploaded before :data:`CUTOFF`.

    Returns:
        (str, str): the version and the upload time of its wheel.
    """
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/json',
                                timeout=60) as response:
        releases = json.load(response)['releases']
    candidates = []
    for version, files in releases.items():
        if not is_final(version):
            continue
        for entry in files:
            pure = entry['filename'].endswith(('py3-none-any.whl',
                                               'py2.py3-none-any.whl'))
            uploaded = entry['upload_time_iso_8601']
            if pure and not entry.get('yanked') and uploaded < CUTOFF:
                candidates.append((uploaded, version))
    if not candidates:
        raise RuntimeError(f'{name} has no pure wheel before {CUTOFF}')
    uploaded, version = max(candidates)
    return version, uploaded


def main():
    lines = [
        '# The PyPI packages the Gradio-Lite page installs, each at its newest',
        f'# release before {CUTOFF}, when Gradio-Lite 5.45.0 was published.',
        '# Written by gv5_lite_pins.py; change that script, not this file.',
    ]
    for name in PACKAGES:
        version, uploaded = newest_before_cutoff(name)
        print(f'{name:20s} {version:10s} uploaded {uploaded[:10]}')
        lines.append(f'{name}=={version}')
    with open(PINS_FILE, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(f'wrote {os.path.relpath(PINS_FILE, HERE)}')


if __name__ == '__main__':
    main()
