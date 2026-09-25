import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from q2v3.__main__ import main, is_round4


class RoundRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'configs').mkdir()
        (self.root / 'configs/frozen_protocol.json').write_text(json.dumps({'version': 'round4-class-balance-predeclared-v1'}))

    def test_study_record_overrides_default_version(self):
        self.assertTrue(is_round4(self.root))
        (self.root / 'study').mkdir()
        (self.root / 'study/protocol.json').write_text(json.dumps({'version': 'round3'}))
        self.assertFalse(is_round4(self.root))

    def test_round4_figures_and_paper_route_to_matching_generators(self):
        for command, expected in [('figures', 'plot_round4'), ('paper', 'paper_round4')]:
            with self.subTest(command=command), mock.patch('sys.argv', ['q2v3', '--root', str(self.root), command]), mock.patch('q2v3.__main__.script') as factory:
                main(); factory.assert_called_once_with(self.root, expected)
                factory.return_value.main.assert_called_once_with(self.root)

    def test_round5_uses_saved_predictions_and_own_figures_and_paper(self):
        (self.root / 'configs/frozen_protocol.json').write_text(json.dumps({'version':'round5-deferred-weighting-predeclared-v1'}))
        for command, expected, method in [('analyze','analyze_round5','recompute_saved'),('figures','plot_round5','main'),('paper','paper_round5','main')]:
            with self.subTest(command=command), mock.patch('sys.argv',['q2v3','--root',str(self.root),command]), mock.patch('q2v3.__main__.script') as factory:
                main();factory.assert_called_once_with(self.root,expected)
                getattr(factory.return_value,method).assert_called_once_with(self.root)

    def test_round4_evaluate_passes_explicit_paths(self):
        with mock.patch('sys.argv', ['q2v3', '--root', str(self.root), 'evaluate', '--data-root', str(self.root), '--split', 'test', '--device', 'cuda']), mock.patch('q2v3.evaluation.evaluate_frozen') as run:
            main(); run.assert_called_once_with(self.root, self.root, None, 'test', 'cuda')

    def test_evaluate_refuses_existing_output_before_loading_model(self):
        from q2v3.evaluation import evaluate_frozen
        with mock.patch('q2v3.evaluation.load_deployment') as load:
            with self.assertRaises(FileExistsError): evaluate_frozen(self.root, output_dir=self.root)
            load.assert_not_called()

    def test_evaluate_requires_freeze_before_creating_output(self):
        from q2v3.evaluation import evaluate_frozen
        output = self.root / 'new_evaluation'
        with self.assertRaises(FileNotFoundError): evaluate_frozen(self.root, output_dir=output)
        self.assertFalse(output.exists())
