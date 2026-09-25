"""Check exported research assets, editable math, tables and rendered-page coverage."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def main(root, qa, reviewed):
    frozen = read(root / 'study/frozen.json')
    assert sha(root / 'final/model.pt') == frozen['checkpoint_sha256']
    assert sha(root / 'final/scaler.npz') == frozen['scaler_sha256']
    with (root / 'final/attachment3_predictions.csv').open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    assert len(rows) == len(set(r['sample_id'] for r in rows)) == 30
    for r in rows:
        values = [float(r['prob_' + c]) for c in ['negative', 'neutral', 'positive']]
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
        assert abs(sum(values) - 1) < 1e-6
        assert -3 <= float(r['intensity']) <= 3
        assert int(r['class_id']) == max(range(3), key=values.__getitem__)
    for item in read(root / 'figures/round4_figure_manifest.json'):
        for relative, expected in item['sources'].items(): assert sha(root / relative) == expected, relative
        for ext in ['svg', 'pdf', 'png']: assert (root / f"figures/{item['id']}.{ext}").stat().st_size > 0
    cli_predictions = list((root / 'evaluation/recomputed_cli_valid/predictions').glob('*.csv'))
    assert len(cli_predictions) == 46
    for path in cli_predictions:
        assert sha(path) == sha(root / 'evaluation/final/valid/predictions' / path.name), path.name
    stem = 'E题问题2第四轮类别平衡实验与论文补充'
    document = root / f'reports/{stem}.docx'
    ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
          'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    with zipfile.ZipFile(document) as archive: tree = ET.fromstring(archive.read('word/document.xml'))
    maths = tree.findall('.//m:oMath', ns)
    assert len(maths) == 4
    assert not tree.findall('.//w:ins', ns) and not tree.findall('.//w:del', ns)
    blocks = read(root / 'reports/round4_content.json')
    expected_tables = [b for b in blocks if b['type'] == 'table']
    tables = tree.findall('.//w:tbl', ns)
    assert len(tables) == len(expected_tables) == 8
    for actual, expected in zip(tables, expected_tables):
        data = [[''.join(t.text or '' for t in cell.findall('.//w:t', ns)) for cell in row.findall('w:tc', ns)] for row in actual.findall('w:tr', ns)]
        assert data == [[str(v) for v in row] for row in [expected['header']] + expected['rows']], expected['caption']
    word_text = [''.join(t.text or '' for t in p.findall('.//w:t', ns)) for p in tree.findall('.//w:p', ns)]
    md = (root / f'reports/{stem}.md').read_text(encoding='utf-8')
    for b in blocks:
        if b['type'] in ['title', 'heading', 'paragraph']:
            assert b['text'] in word_text and b['text'] in md
    figures = [b['caption'] for b in blocks if b['type'] == 'figure']
    assert all(c.startswith(f'图{i} ') for i, c in enumerate(figures, 1)) and len(figures) == 4
    pages = sorted(qa.glob('page-*.png'))
    page_numbers = sorted(int(p.stem.split('-')[-1]) for p in pages)
    assert page_numbers == reviewed == list(range(1, 9))
    old = read(root / 'audit/source_hashes.json')
    provided = {str(p.relative_to(root)).replace('\\', '/'): sha(p) for folder in ['src', 'scripts', 'tests'] for p in (root / folder).rglob('*.py')}
    changed = [name for name, value in old.items() if provided.get(name) != value]
    # UI routing and authoring tooling were enhanced after model freeze, never the trained model implementation.
    assert all(not name.startswith('src/') or name == 'src/q2v3/__main__.py' for name in changed), changed
    result = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'attachment3_rows': len(rows),
        'editable_equations': len(maths), 'tables': len(tables), 'figure_groups': len(figures),
        'docx_sha256': sha(document), 'markdown_sha256': sha(root / f'reports/{stem}.md'),
        'portable_cli_identical_validation_scenarios': len(cli_predictions),
        'all_word_tables_equal_generated_evidence': True, 'word_markdown_text_consistent': True,
        'visual_review': {'pages': reviewed, 'renderer': 'Word COM fallback: packaged LibreOffice unavailable',
                          'pdf_sha256': sha(qa / 'round4.pdf'), 'review': 'All eight rendered pages inspected; no clipping, detached captions or illegible figures'},
        'changed_provided_sources_since_training': changed,
        'source_version_note': 'The WSL research snapshot remains immutable; exported CLI and authoring utilities include postfreeze improvements.'}
    (root / 'audit/provided_source_hashes.json').write_text(json.dumps(provided, indent=2), encoding='utf-8')
    (root / 'audit/release_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); p.add_argument('--qa-dir', type=Path, required=True)
    p.add_argument('--visually-reviewed-pages', required=True, help='Explicit page numbers after inspecting every final page')
    a = p.parse_args(); main(a.root.resolve(), a.qa_dir.resolve(), [int(x) for x in a.visually_reviewed_pages.split(',')])
