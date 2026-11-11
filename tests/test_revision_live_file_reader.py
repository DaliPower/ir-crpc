"""Real OS handle-sharing checks; no experiment file is changed."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.revision_live_file_reader import open_shared_read

REPLACE = '''
import json, os, sys
try:
    os.replace(sys.argv[1], sys.argv[2])
    print(json.dumps({'replaced': True}))
except OSError as exc:
    print(json.dumps({'replaced': False, 'winerror': exc.winerror}))
'''


def replace_in_process(source, target):
    value = subprocess.run([sys.executable, '-c', REPLACE, str(source), str(target)],
        capture_output=True, text=True, check=True, timeout=10,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return json.loads(value.stdout)


@pytest.mark.skipif(os.name != 'nt', reason='Windows sharing violation control')
def test_ordinary_python_reader_reproduces_windows_replace_failure(tmp_path):
    target, source = tmp_path/'progress.json', tmp_path/'progress.json.tmp'
    target.write_bytes(b'{"version":0}')
    source.write_bytes(b'{"version":1}')
    with target.open('rb'):
        result = replace_in_process(source, target)
    assert result['replaced'] is False
    assert result['winerror'] in (5, 32)
    assert replace_in_process(source, target)['replaced'] is True


def test_shared_reader_is_read_only_and_documents_destination_replace_limit(tmp_path):
    target, source = tmp_path/'progress.json', tmp_path/'progress.json.tmp'
    target.write_bytes(b'{"version":0}')
    source.write_bytes(b'{"version":1}')
    with open_shared_read(target) as handle:
        assert not handle.writable()
        assert not os.get_inheritable(handle.fileno())
        result = replace_in_process(source, target)
        if os.name == 'nt':
            assert result['replaced'] is False
            assert result['winerror'] == 5
        else:
            assert result['replaced'] is True
        assert json.loads(handle.read()) == {'version': 0}
    if os.name == 'nt':
        assert replace_in_process(source, target)['replaced'] is True
    with open_shared_read(target) as handle:
        assert json.loads(handle.read()) == {'version': 1}
