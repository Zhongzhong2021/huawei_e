"""Full accounting, fail-closed omission evidence, and preserved visual observations."""
import unittest
from build_correspondence_inventory import build, word_state, R
from check_local_omissions import confirmed_windows
from common import read


class CorrespondenceTests(unittest.TestCase):
    def test_omission_needs_both_windows(self):
        self.assertFalse(confirmed_windows([]))
        self.assertFalse(confirmed_windows([{'context_s':0,'accepted':True}]))
        evidence=[{'context_s':0.,'accepted':True},{'context_s':.1,'accepted':True}]
        self.assertTrue(confirmed_windows(evidence))
        self.assertFalse(confirmed_windows(list(reversed(evidence))))
        evidence[1]['accepted']=False
        self.assertFalse(confirmed_windows(evidence))

    def test_signal_absence_is_not_matching_failure(self):
        word={'automatic_eligible':False,'status':'sample_support_insufficient'}
        self.assertEqual(word_state(word,[],{'zero_audio':True},'no_signal'),'no_audio_signal')
        self.assertEqual(word_state(word,[],{'zero_audio':False},'no_informative_support'),'no_informative_anchor')
        word['status']='new_unknown_status'
        with self.assertRaises(ValueError):word_state(word,[],{'zero_audio':False},'supported')

    def test_full_accounting_and_visual_preservation(self):
        result=build()
        self.assertEqual(result,read(R/'data/correspondence_inventory.json'))
        self.assertEqual(result['summary']['words'],1932)
        self.assertEqual(result['summary']['unresolved_in_base'],464)
        for sample in result['samples']:
            self.assertEqual([w['word_index'] for w in sample['words']],list(range(len(sample['words']))))
            for w in sample['words']:
                self.assertEqual(w['global_absence']=='no_audio_signal',w['state']=='no_audio_signal')
            v=sample['visual']
            self.assertEqual(v['sampled_frames'],v['frames_with_face_detections']+v['frames_without_face_detections'])
            self.assertFalse(v['speaker_identity_assigned'])
            self.assertFalse(v['no_detection_means_face_absent'])

    def test_omission_records_are_local_and_traceable(self):
        evidence=read(R/'data/local_omissions.json');proposals=read(R/'data/proposals.json')
        self.assertEqual(evidence['tests'],sum(len(p['interior_indices']) for p in proposals))
        for row in evidence['rows']:
            proposal=proposals[row['proposal_row']]
            self.assertIn(row['word_index'],proposal['interior_indices'])
            self.assertEqual(row['full_words'],proposal['reference_words'])
            self.assertEqual(row['sample_id'],proposal['sample_id'])
            index=proposal['word_indices'].index(row['word_index'])
            self.assertEqual(row['deletion_words'],row['full_words'][:index]+row['full_words'][index+1:])
            self.assertFalse(row['whole_audio_absence_established'])
            self.assertEqual(row['local_omission_supported'],confirmed_windows(row['evidence']))


if __name__=='__main__':unittest.main()
