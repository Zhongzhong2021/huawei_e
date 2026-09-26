"""Read word and phrase candidates with immutable native features and provenance."""
from common import read, digest
from delivery_reader import Dataset as BaselineDataset, DeliverySample
from multigranular_alignment import combine
from temporal_support import interval_relation


class MultigranularSample(DeliverySample):
    def phrase(self, phrase_index):
        if not isinstance(phrase_index, int) or not 0 <= phrase_index < len(self.automatic['phrases']):
            raise IndexError('Phrase index outside candidate list')
        phrase = self.automatic['phrases'][phrase_index]
        relation=interval_relation(self.arrays,self.metadata,self.source,
                    phrase['interval_s'],'automatic_candidate',phrase['word_indices'])
        relation['granularity']='phrase'
        relation['internal_word_times_assigned']=False
        return {'sample_id':self.source['sample_id'],'status':phrase['status'],
                'internal_word_times_assigned':False,'semantic_verified_by_this_function':False,
                'relation':relation}


class Dataset(BaselineDataset):
    def __init__(self,root):
        super().__init__(root)
        self.extension_root=self.root/'data/multigranular_alignment'
        self.extension_manifest=read(self.extension_root/'manifest.json')
        assert self.extension_manifest['complete']
        self.extension_config=read(self.root/'config/whisper_alignment.json')
        if digest(self.root/'config/whisper_alignment.json')!=self.extension_manifest['config_sha256']:
            raise ValueError('Enhanced configuration digest mismatch')
        for name,sha in self.extension_manifest['scripts'].items():
            if digest(self.root/'scripts'/name)!=sha:raise ValueError('Enhanced implementation changed: '+name)
        if digest(self.extension_root/'acoustic_evidence.json')!=self.extension_manifest['evidence_sha256']:
            raise ValueError('Enhanced acoustic evidence digest mismatch')
        self.evidence={r['sample_id']:r for r in read(self.extension_root/'acoustic_evidence.json')['records']}
        self.result_hashes={r['sample_id']:r['sha256'] for r in self.extension_manifest['samples']}
        if set(self.evidence)!=set(self.records) or set(self.result_hashes)!=set(self.records):
            raise ValueError('Enhanced sample coverage mismatch')

    def sample(self,sid):
        baseline=super().sample(sid)
        path=self.extension_root/(sid+'.json')
        if digest(path)!=self.result_hashes[sid]:raise ValueError('Enhanced output digest mismatch')
        result=read(path);evidence=self.evidence[sid]
        if result['sample_id']!=sid or result['source_sha256']!=baseline.source['sha256'] or evidence['source_sha256']!=baseline.source['sha256']:
            raise ValueError('Enhanced source mismatch')
        if result['baseline_result_sha256']!=digest(self.root/'data/local_evidence_corroborated'/(sid+'.json')):
            raise ValueError('Enhanced baseline mismatch')
        expected=combine(baseline.metadata['words'],baseline.automatic,evidence['support'],evidence['runs'],
                         self.extension_config,[baseline.source['audio']['start_s'],baseline.source['audio']['end_s']],
                         baseline.source['zero_audio'])
        for key,value in expected.items():
            if result[key]!=value:raise ValueError('Enhanced decision differs from rule: '+key)
        sample=MultigranularSample(baseline.arrays,baseline.metadata,baseline.source,result,baseline.temporal_index)
        sample.baseline_automatic=baseline.automatic
        return sample
