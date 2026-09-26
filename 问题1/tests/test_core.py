import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from math_features import words,cells,iou,acoustic
from feature_io import pad,candidate_rows
class CoreTests(unittest.TestCase):
    def test_spans_preserve_names_numbers_and_annotations(self):
        text="[Speaker:] River Stone didn't sell 21 units."
        ws=words(text)
        for w in ws:self.assertEqual(text[slice(*w['char_span'])],w['text'])
        self.assertEqual(ws[0]['text'],'Speaker');self.assertEqual(next(w for w in ws if w['text']=='21')['parts'],['TWENTY','ONE'])
    def test_cells_not_duration_scaled(self):
        np.testing.assert_allclose(cells([.1,.3,.6],0,.8),[[0,.2],[.2,.45],[.45,.8]])
        with self.assertRaises(ValueError):cells([.2,.1],0,1)
    def test_box_geometry(self):
        self.assertEqual(iou(np.array([0,0,1,1]),np.array([0,0,1,1])),1)
        self.assertEqual(iou(np.array([0,0,.2,.2]),np.array([.3,.3,1,1])),0)
    def test_silence_is_not_pitch(self):
        cfg={'audio_rate':16000,'audio_window':400,'audio_hop':320,'pitch_window':1024,'pitch_range_hz':[50,500],'pitch_periodicity_min':.6}
        a,m,c=acoustic(np.zeros(16000),cfg);self.assertTrue(np.isfinite(a).all());self.assertFalse(m[:,-2:].any());self.assertTrue(m[:,:15].all())
    def test_padding_and_dim_mask_differ(self):
        a={'audio':np.array([[1,0],[2,3]],np.float32),'audio_dim_mask':np.array([[1,0],[1,1]],bool)}
        b={'audio':np.array([[0,0]],np.float32),'audio_dim_mask':np.ones((1,2),bool)}
        batch=pad([(a,{'sample_id':'a'}),(b,{'sample_id':'b'})],'audio')
        self.assertTrue(batch['sequence_mask'][1,0]);self.assertFalse(batch['sequence_mask'][1,1]);self.assertFalse(batch['dimension_mask'][0,0,1])
    def test_visual_gaps_do_not_get_synthetic_support(self):
        a={'word_candidate_mask':np.array([True]),'word_intervals':np.array([[.1,.9]]),'audio_cells':np.array([[0,1]]),'video_cells':np.array([[0,1]])}
        native={'time_base':'1/10','pts_ticks':[0,8],'lengths':[2,2]}
        out=candidate_rows(a,0,native);self.assertAlmostEqual(out['audio'][0][1],.8);self.assertAlmostEqual(out['scene'][0][1],.2)
    def test_zero_track_batch_preserves_length_but_cannot_supply_features(self):
        a={'audio':np.array([[-20.,3.],[2.,4.]],np.float32),'audio_dim_mask':np.ones((2,2),bool)}
        original=a['audio'].copy()
        result=pad([(a,{'sample_id':'silent','alignment':{'status':'no_signal'}})],'audio')
        self.assertEqual(result['lengths'].tolist(),[2])
        self.assertTrue(result['sequence_mask'].all())
        self.assertFalse(result['sample_available'].any())
        self.assertFalse(result['dimension_mask'].any())
        self.assertFalse(result['values'].any())
        np.testing.assert_array_equal(a['audio'],original)
    def test_legitimate_zero_values_are_not_missing(self):
        a={'audio':np.zeros((2,2),np.float32),'audio_dim_mask':np.ones((2,2),bool)}
        result=pad([(a,{'sample_id':'valid','alignment':{'status':'candidate'}})],'audio')
        self.assertTrue(result['sample_available'].all())
        self.assertTrue(result['dimension_mask'].all())
if __name__=='__main__':unittest.main()
