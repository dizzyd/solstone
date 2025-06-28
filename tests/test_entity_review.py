import importlib
from pathlib import Path


def test_format_date():
    review = importlib.import_module('dream.entity_review')
    assert '2024' not in review.format_date('20240102')
    assert review.format_date('bad') == 'bad'


def test_modify_and_update(tmp_path):
    review = importlib.import_module('dream.entity_review')
    md = tmp_path/'entities.md'
    md.write_text('* Person: Jane - desc\n')
    review.modify_entity_in_file(str(md), 'Person', 'Jane', operation='remove')
    assert md.read_text() == ''
    md.write_text('* Person: Jane - desc\n')
    review.modify_entity_in_file(str(md), 'Person', 'Jane', new_name='J', operation='rename')
    assert 'J' in md.read_text()
    review.update_top_entry(str(tmp_path), 'Person', 'J', 'info')
    top_path = tmp_path/'entities.md'
    assert top_path.read_text()
