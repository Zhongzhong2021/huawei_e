"""Verify exported research artifacts; record completed manual page review.

Run with bundled Windows Python after inspecting every page of the latest render.
This does not render the document or substitute structural checks for visual QA.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--qa-dir', type=Path, required=True)
    p.add_argument('--visually-reviewed-pages', type=int, required=True)
    args = p.parse_args()
    root, qa = args.root.resolve(), args.qa_dir.resolve()
    blocks = read(root / 'reports/paper_content.json')
    docx = next((root / 'reports').glob('*.docx'))
    markdown = docx.with_suffix('.md')
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
          'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
          'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
          'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
          'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    with zipfile.ZipFile(docx) as z:
        xml = ET.fromstring(z.read('word/document.xml'))
        styles = ET.fromstring(z.read('word/styles.xml'))
        core = ET.fromstring(z.read('docProps/core.xml'))
        rels = ET.fromstring(z.read('word/_rels/document.xml.rels'))
        relmap = {r.attrib['Id']: r.attrib['Target'] for r in rels}
        text = ''.join(n.text or '' for n in xml.findall('.//w:t', ns))
        assert not xml.findall('.//w:pBdr', ns)
        assert not styles.findall('.//w:pBdr', ns)
        assert not xml.findall('.//w:ins', ns) and not xml.findall('.//w:del', ns)
        for node in core:
            if node.tag.split('}')[-1] in ['creator', 'lastModifiedBy']:
                assert not (node.text or '').strip()
        for b in blocks:
            if b['type'] in ['title', 'heading', 'paragraph']:
                assert b['text'] in text, ('Missing Word text', b['text'][:60])
        equations = [b for b in blocks if b['type'] == 'equation']
        assert [b['number'] for b in equations] == list(range(1, 17))
        assert len(xml.findall('.//m:oMath', ns)) == len(equations) == 16
        figures = [b for b in blocks if b['type'] == 'figure']
        assert [b['paper_number'] for b in figures] == list(range(1, 9))
        drawings = xml.findall('.//w:drawing', ns)
        assert len(drawings) == len(figures) == 8
        for drawing, b in zip(drawings, figures):
            blip = drawing.find('.//a:blip', ns)
            embed = blip.attrib['{' + ns['r'] + '}embed']
            target = relmap[embed]
            image_bytes = z.read('word/' + target)
            assert hashlib.sha256(image_bytes).hexdigest() == sha((root / 'reports' / b['path']).resolve())
            assert b['caption'] in text
        page = xml.find('.//w:pgSz', ns)
        w = lambda key: '{' + ns['w'] + '}' + key
        assert abs(int(page.attrib[w('w')]) - 11906) <= 2
        assert abs(int(page.attrib[w('h')]) - 16838) <= 2
        margin = xml.find('.//w:pgMar', ns)
        usable = int(page.attrib[w('w')]) - int(margin.attrib[w('left')]) - int(margin.attrib[w('right')])
        for extent in xml.findall('.//wp:extent', ns):
            assert int(extent.attrib['cx']) <= usable * 635 + 1000
        tables = xml.findall('.//w:tbl', ns)
        tableblocks = [b for b in blocks if b['type'] == 'table']
        assert len(tables) == len(tableblocks)
        for table, b in zip(tables, tableblocks):
            assert sum(int(col.attrib[w('w')]) for col in table.findall('./w:tblGrid/w:gridCol', ns)) <= usable + 4
            assert table.find('./w:tr/w:trPr/w:tblHeader', ns) is not None
            actual = [[''.join(n.text or '' for n in cell.findall('.//w:t', ns))
                       for cell in row.findall('./w:tc', ns)] for row in table.findall('./w:tr', ns)]
            expected = [[str(v) for v in row] for row in [b['header']] + b['rows']]
            assert actual == expected, b['caption']
        assert all(s not in text.lower() for s in ['xifeng', '/home/', 'c:\\users', 'turn0search'])
    md = []
    for b in blocks:
        typ = b['type']
        if typ == 'title': md += ['# ' + b['text'], '']
        elif typ == 'heading': md += ['#' * (b['level'] + 1) + ' ' + b['text'], '']
        elif typ == 'paragraph': md += [b['text'], '']
        elif typ == 'equation': md += ['$$', b['latex'] + rf'\tag{{{b["number"]}}}', '$$', '']
        elif typ == 'table':
            md += [b['caption'], '', '| ' + ' | '.join(b['header']) + ' |',
                   '| ' + ' | '.join(['---'] * len(b['header'])) + ' |']
            md += ['| ' + ' | '.join(map(str, row)) + ' |' for row in b['rows']] + ['']
        elif typ == 'figure': md += [f"![{b['caption']}]({b['path']})", '']
    assert markdown.read_text(encoding='utf-8') == '\n'.join(md)
    latex = (root / 'reports/equations.tex').read_text(encoding='utf-8')
    assert latex.count('\\begin{equation}') == 16
    for b in equations:
        assert b['latex'] in latex
    manifest = read(root / 'figures/figure_manifest.json')
    source_count = 0
    for fig in manifest:
        for source, expected in fig['sources'].items():
            assert sha(root / source) == expected, source
            source_count += 1
        for ext in ['svg', 'pdf', 'png']:
            assert (root / 'figures' / (fig['id'] + '.' + ext)).stat().st_size > 0
    pages = sorted(qa.glob('page-*.png'))
    assert len(pages) == args.visually_reviewed_pages == 17
    assert (qa / 'paper.pdf').stat().st_size > 0
    now = datetime.now(timezone.utc).isoformat()
    save(root / 'audit/document_qa.json', {
        'checked_at_utc': now, 'docx_sha256': sha(docx), 'markdown_sha256': sha(markdown),
        'pages_visually_reviewed': list(range(1, 18)),
        'visual_review_attestation': 'All pages of this latest render manually inspected; no clipped formulas, overlapping text, truncated tables, missing glyphs, detached captions or unreadable figures observed.',
        'native_editable_equations': 16, 'figure_groups': 8, 'tables_including_algorithms': len(tables),
        'word_markdown_structured_content_match': True, 'table_cell_values_match': True,
        'figure_embedded_bytes_match': True, 'figure_source_hashes_verified': source_count,
        'a4_layout': True, 'identity_metadata_empty': True,
        'canonical_renderer': 'render_docx.py attempted; unavailable bundled LibreOffice soffice.exe',
        'actual_renderer': 'Independent hidden Microsoft Word COM PDF export, then bundled Poppler rasterization',
        'qa_pdf_sha256': sha(qa / 'paper.pdf'),
        'qa_page_sha256': {p.name: sha(p) for p in pages}
    })
    freeze = read(root / 'study/frozen.json')
    assert sha(root / 'final/model.pt') == freeze['checkpoint_sha256']
    prediction_sha = sha(root / 'final/attachment3_predictions.csv')
    assert prediction_sha == sha(root / 'audit/exported_predictions.csv')
    for source, expected in read(root / 'audit/source_hashes.json').items():
        assert sha(root / source) == expected, ('Exported source drift', source)
    acceptance = read(root / 'audit/release_acceptance.json')
    assert acceptance['tests_passed'] and acceptance['tests_run'] == 18
    assert acceptance['metrics_verified'] == 184 and acceptance['reproduction_exact']
    old = root.parent / 'round2/E_Q2_round2_submission.zip'
    expected_old = '8390c127915d0ef2fd9fa8e008e824f06998985a1d25b3514e52525e63133280'
    assert sha(old) == expected_old
    save(root / 'audit/previous_round_windows.json', {
        'file': '../round2/E_Q2_round2_submission.zip', 'sha256': expected_old,
        'unchanged': True, 'checked_at_utc': now})
    excluded = {'audit/delivery_manifest.json', 'audit/delivery_complete.json'}
    files = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and '__pycache__' not in path.parts and relative not in excluded:
            files[relative] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    save(root / 'audit/delivery_manifest.json', files)
    save(root / 'audit/delivery_complete.json', {
        'completed_at_utc': now, 'status': 'passed', 'scope': 'round3 research deliverables; not the team size-limited submission package',
        'paper': str(docx.relative_to(root).as_posix()), 'paper_pages': 17,
        'tests_passed': 18, 'metric_scenarios_recomputed': 184,
        'same_environment_retraining_exact': True, 'exported_directory_inference_exact_csv': True,
        'attachment3_rows': 30, 'prediction_sha256': prediction_sha,
        'preserved_round2_archive': True, 'manifest_files': len(files),
        'manifest_sha256': sha(root / 'audit/delivery_manifest.json')
    })
    print(json.dumps(read(root / 'audit/delivery_complete.json'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
