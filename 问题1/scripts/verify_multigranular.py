"""Verify full enhanced provenance, deterministic decisions and physical reads."""
import json
import hashlib
from pathlib import Path
from common import read,write,digest
from multigranular_reader import Dataset
from whisper_support import lexical_support
from phrase_selection import compatible
from evaluate_phrase_selection import evaluate as evaluate_selection

ROOT=Path(__file__).resolve().parents[1]

def main():
    ds=Dataset(ROOT);out=ROOT/'data/multigranular_alignment'
    manifest=read(ROOT/'data/whisper_recognition/manifest.json')
    protocol=read(ROOT/'data/whisper_recognition/protocol.json')
    assert manifest['complete'] and manifest['protocol_sha256']==digest(ROOT/'data/whisper_recognition/protocol.json')
    assert protocol['script_sha256']==digest(ROOT/'scripts/run_whisper_recognition.py')
    assert protocol['config_sha256']==digest(ROOT/'config/whisper_alignment.json')
    assert protocol['reference_prompt_used'] is False and protocol['human_data_used'] is False
    cfg=read(ROOT/'config/automatic_alignment.json')
    hashes={s['sample_id']:s['sha256'] for s in manifest['samples']}
    assert set(hashes)==set(ds.records)
    counts={'samples':0,'words':0,'word_candidates':0,'phrase_candidates':0,'phrase_only_words':0,'conflicting_baseline_words':0}
    evidence=read(out/'acoustic_evidence.json')
    prep=ROOT/'data/whisper_local/preparation.json';execution=ROOT/'data/whisper_local/execution.json'
    assert digest(prep)==evidence['preparation_sha256'] and digest(execution)==evidence['execution_sha256']
    assert read(execution)['returncode']==0
    for path,sha in evidence['raw_sha256'].items():
        assert digest(ROOT/'data/whisper_local'/path)==sha
    for sid,item in ds.records.items():
        path=ROOT/'data/whisper_recognition'/(sid+'.json');assert digest(path)==hashes[sid]
        recognition=read(path);assert recognition['source_sha256']==item['sha256']
        sample=ds.sample(sid);result=sample.automatic
        support=lexical_support(sample.metadata['words'],recognition['words'],cfg,item['audio']['start_s'],item['audio']['end_s'])
        assert support==ds.evidence[sid]['support']
        assert all(compatible(a,b) for a,b in zip(result['phrases'],result['phrases'][1:]))
        word_set={i for i,w in enumerate(result['words']) if w['automatic_eligible']}
        phrase_set={i for p in result['phrases'] for i in p['word_indices']}
        assert sorted(phrase_set-word_set)==result['phrase_only_words']
        for i in range(len(result['words'])):
            view=sample.word(i)
            if i in word_set:assert len(view['relation']['audio']['rows'])>0
            else:assert view['relation'] is None
        for i in range(len(result['phrases'])):
            view=sample.phrase(i)
            assert view['relation']['granularity']=='phrase' and not view['relation']['internal_word_times_assigned']
        if item['zero_audio']:assert not word_set and not phrase_set
        counts['samples']+=1;counts['words']+=len(result['words'])
        counts['word_candidates']+=len(word_set);counts['phrase_candidates']+=len(result['phrases'])
        counts['phrase_only_words']+=len(phrase_set-word_set)
        counts['conflicting_baseline_words']+=result['conflicting_baseline_words']
    expected=read(out/'summary.json')['totals']
    assert all(counts[k]==expected[k] for k in counts if k!='samples') and counts['samples']==100
    reproduction=read(out/'reproduction.json')
    assert reproduction['status']=='passed' and reproduction['relation_manifest_sha256']==digest(out/'manifest.json')
    for step in reproduction['rebuild_record']['steps']:
        assert step['exit_code']==0 and digest(out/'reproduction_logs'/step['log'])==step['log_sha256']
    for name in ['run_whisper_recognition.py','prepare_whisper_local.py','run_whisper_local_mfa.py','whisper_support.py','multigranular_alignment.py','phrase_selection.py','multigranular_reader.py','build_multigranular_alignment.py']:
        assert digest(ROOT/'scripts'/name)==reproduction['rebuild_record']['scripts'][name]
    for sid,sha in reproduction['reproduced_relation_sha256'].items():assert digest(out/(sid+'.json'))==sha
    assert set(reproduction['reproduced_relation_sha256'])==set(ds.records)
    evaluation=read(out/'evaluation.json')
    assert evaluation['enhanced_manifest_sha256']==digest(out/'manifest.json')
    assert evaluation['recognition_manifest_sha256']==digest(ROOT/'data/whisper_recognition/manifest.json')
    assert read(ROOT/'data/phrase_selection/evaluation.json') == evaluate_selection()
    write(out/'verification.json',{'status':'passed',**counts,'native_features_modified':False,
          'manifest_sha256':digest(out/'manifest.json'),'recognition_and_acoustic_sources_verified':True,
          'all_decisions_recomputed':True,'all_physical_reads_checked':True,
          'reproduction_evidence_checked':True,'phrase_selection_ablation_verified':True,'phrase_sequence_nonoverlapping':True,'independent_accuracy_measured':False})
    print(counts)

if __name__=='__main__':main()
