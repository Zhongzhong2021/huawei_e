"""Build measured enhancement tables without manual example selection."""
import csv
from pathlib import Path
from common import read,write,digest

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'data/multigranular_alignment'
    summary=read(out/'summary.json');evaluation=read(out/'evaluation.json')
    selection=read(ROOT/'data/phrase_selection/evaluation.json')
    assert selection['current_manifest_sha256']==digest(out/'manifest.json')
    assert evaluation['enhanced_manifest_sha256']==digest(out/'manifest.json')
    total=summary['totals'];rows={s['sample_id']:s for s in summary['samples']}
    original=(ROOT/'data/delivery_summary/all_100.md').read_text()
    lines=[]
    for line in original.splitlines():
        if line.startswith('| 样本'):
            line=line.replace('候选/未定','基础词级/未定').rstrip()+' 增强词级 | 仅词组覆盖词数 | 完全未定词数 |'
        elif line.startswith('|---'):
            line+='---:|---:|---:|'
        elif line.startswith('| -'):
            sid=line.split('|')[1].strip();r=rows[sid]
            unresolved=r['words']-r['word_candidates']-r['phrase_only_words']
            line+=f" {r['word_candidates']} | {r['phrase_only_words']} | {unresolved} |"
        lines.append(line)
    lines[0]='# 全100条特征与多粒度关系汇总'
    lines[4]='基础词级列保留对照结果；增强词级、仅词组覆盖、完全未定三类词数互斥并覆盖全部原词。词组级关系保存组范围，内部词时间保持未确定。'
    (out/'all_100.md').write_text('\n'.join(lines)+'\n')
    with (out/'all_100.csv').open('w') as stream:
        keys=['sample_id','words','baseline_candidates','word_candidates','added_words','conflicting_baseline_words','phrase_candidates','phrase_only_words','unique_exact_asr_words','supported_runs']
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in summary['samples'])
    write(out/'report_manifest.json',{'selection_evaluation_sha256':digest(ROOT/'data/phrase_selection/evaluation.json'),'summary_sha256':digest(out/'summary.json'),'evaluation_sha256':digest(out/'evaluation.json'),
          'all_100_table_sha256':digest(out/'all_100.md')})

if __name__=='__main__':main()
