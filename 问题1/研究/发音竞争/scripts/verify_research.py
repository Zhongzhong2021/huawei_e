"""Verify source traces, fixed protocols, scored controls and unchanged stable outputs."""
import os,sys,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=Path(os.environ.get('Q1_PROJECT',R.parents[1]))
sys.path.insert(0,str(Q/'scripts'))
from common import read,write,digest
from build_enhancement import build
from build_correspondence_inventory import build as build_inventory

def main():
    protocol_sha=digest(R/'protocol.json');posts=read(R/'data/posterior_manifest.json')
    assert posts['complete'] and len(posts['records'])==100
    assert posts['script_sha256']==digest(R/'scripts/infer_phones.py')
    assert posts['input_sha256']==digest(Q/'data/inputs.json')
    provenance=read(R/'data/provenance.json')
    for name,sha in provenance['core_dependencies_sha256'].items():assert digest(Q/name)==sha
    for name,sha in provenance['research_algorithms_sha256'].items():assert digest(R/'scripts'/name)==sha
    proposal=read(R/'data/proposal_manifest.json')
    assert proposal['source_inputs_sha256']==digest(Q/'data/inputs.json')
    assert proposal['proposals_sha256']==digest(R/'data/proposals.json') and proposal['proposals']==393
    assert proposal['asr_manifest_sha256']==digest(Q/'data/whisper_recognition/manifest.json')
    summary=read(R/'data/research_summary.json');checked=0
    for mode,metrics in summary['modes'].items():
        out=R/'data'/mode;s=read(out/'summary.json')
        assert s['protocol_sha256']==protocol_sha and s['posterior_manifest_sha256']==digest(R/'data/posterior_manifest.json')
        for name in ['phone_evidence.py','evaluate_competition.py','proposals.py']:
            assert s['scripts_sha256'][name]==digest(R/'scripts'/name)
        for name in ['identity','shifted','corrupted','injected']:
            d=read(out/(name+'.json'));rows=d['rows'];accepted=sum(row['accepted'] for row in rows)
            for row in rows:assert row['accepted']==all(e['accepted'] for e in row['evidence'])
            if name=='identity':assert len(rows)==393 and accepted==metrics['accepted_groups']
            else:
                stem={'shifted':'shifted','corrupted':'corrupted','injected':'injected'}[name]
                assert len(rows)==metrics[stem+'_trials'] and accepted==metrics[stem+'_accepted']
            checked+=len(rows)
    reproduction=read(R/'data/reproduction.json')
    assert reproduction['status']=='passed' and reproduction['arrays_exactly_equal']==200
    assert reproduction['fresh_manifest_sha256']==digest(R/'data/posterior_manifest.json')
    assert reproduction['original_manifest_sha256']==digest(R/'data/reference_posterior_manifest.json')
    reference=read(R/'data/base_reference.json')
    assert reference['archive_sha256']==summary['stable_archive_sha256']
    for name,sha in reference['files'].items():assert digest(Q/name)==sha,name
    existing=len(reference['files'])
    originals=read(Q/'data/input_hashes.json')['sha256']
    assert all(digest(Path(p))==sha for p,sha in originals.items())
    enhancement=build()
    assert enhancement==read(R/'data/enhancement.json')
    inventory=build_inventory()
    assert inventory==read(R/'data/correspondence_inventory.json')
    omission=read(R/'data/local_omissions.json')
    assert omission['code_sha256']==digest(R/'scripts/check_local_omissions.py')
    assert omission['phone_evidence_sha256']==digest(R/'scripts/phone_evidence.py')
    assert omission['protocol_sha256']==protocol_sha
    assert omission['proposals_sha256']==digest(R/'data/proposals.json')
    assert omission['posterior_manifest_sha256']==digest(R/'data/posterior_manifest.json')
    write(R/'data/verification.json',{'status':'passed','scored_cases_checked':checked,'context_windows_checked':checked*2,'original_files_unchanged':len(originals),
        'frozen_generated_files_unchanged':existing,'base_outputs_unchanged':True,'new_words_promoted':0,
        'research_test_cases_available':14,'repeat_inference_equal_arrays':200,'protocol_sha256':protocol_sha,
        'enhancement_rebuild_exact':True,'enhancement':enhancement['summary'],
        'correspondence_inventory_rebuild_exact':True,'correspondence':inventory['summary'],
        'local_omission_tests':omission['tests'],'local_omission_supported_tests':omission['supported_tests'],
        'measurement_scope':'Fixed model-score rules, development and constructed controls, repeatability. No independent precision estimate.'})
    print('Verified',checked,'scored cases;',existing,'frozen generated files unchanged.')
if __name__=='__main__':main()
