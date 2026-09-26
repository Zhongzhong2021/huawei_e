"""Assemble a self-contained Q1 draft with all 100 rows included in the body."""
from pathlib import Path
from common import digest, write

ROOT = Path(__file__).resolve().parents[1]


def main():
    body = ROOT / 'scripts/paper_body.txt'
    table = ROOT / 'data/delivery_summary/all_100.md'
    if (ROOT/'data/multigranular_alignment/all_100.md').is_file():
        table=ROOT/'data/multigranular_alignment/all_100.md'
    text = body.read_text() + '\n\n## 9 全100条结果表\n\n' + '\n'.join(table.read_text().splitlines()[2:]) + '\n'
    destination = ROOT / '问题1正文稿.md'
    destination.write_text(text)
    write(ROOT / 'data/delivery_summary/paper_manifest.json', {
        'status': 'q1_body_with_measured_results_and_explicit_uncertainty',
        'sources': {str(p.relative_to(ROOT)): digest(p) for p in [body, table]},
        'paper_sha256': digest(destination),
        'sample_rows': sum(line.startswith('| -') for line in table.read_text().splitlines())})


if __name__ == '__main__':
    main()
