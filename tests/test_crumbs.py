import importlib
import importlib
import os
from pathlib import Path


def test_crumb_builder(tmp_path):
    mod = importlib.import_module('think.crumbs')
    builder = mod.CrumbBuilder(generator='test')
    file1 = tmp_path/'f1.txt'
    file1.write_text('data')
    builder.add_file(file1)
    builder.add_files([file1])
    builder.add_glob(str(file1))
    builder.add_model('m')
    crumb_path = builder.commit(str(tmp_path/'out.txt'))
    assert Path(crumb_path).is_file()
    data = Path(crumb_path).read_text()
    assert 'generator' in data
