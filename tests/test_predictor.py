import tempfile
from pathlib import Path
from unittest import TestCase

import joblib
import numpy as np

from apa_polya_predictor.cli import apply_best_ensemble
from apa_polya_predictor.fasta_utils import encode_sequence_to_uint8, normalize_sequence, read_fasta
from apa_polya_predictor.predictor import Predictor


class TestFastaUtils(TestCase):
    def test_normalize_long(self):
        seq = "A" * 100 + "C" * 100 + "G" * 100 + "T" * 100 + "N" * 100
        out = normalize_sequence(seq, 240)
        self.assertEqual(len(out), 240)
        start = (len(seq) - 240) // 2  # 130
        self.assertEqual(out, seq[start:start + 240])

    def test_normalize_short(self):
        seq = "ACGT" * 10
        out = normalize_sequence(seq, 240)
        self.assertEqual(len(out), 240)
        self.assertEqual(out.count("ACGT"), 10)

    def test_encode(self):
        arr = encode_sequence_to_uint8("ACGTNACGTN", 10)
        self.assertListEqual(arr.tolist(), [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])


class TestPredictor(TestCase):
    def setUp(self):
        self.weights_dir = str(Path(__file__).resolve().parent.parent / "weights")

    def _make_fasta(self, n: int = 3):
        lines = []
        for i in range(n):
            lines.append(f">seq{i+1}")
            lines.append("ACGT" * 60)  # 240 nt
        return "\n".join(lines)

    def test_single_model_prediction(self):
        with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f:
            f.write(self._make_fasta())
            f.flush()
            fasta_path = f.name

        pred = Predictor(
            weights_dir=self.weights_dir,
            species="Human",
            models=["deeppolya"],
            device="cpu",
        )
        df = pred.predict_fasta(fasta_path)
        self.assertEqual(len(df), 3)
        self.assertIn("deeppolya_prob", df.columns)
        self.assertNotIn("ensemble_prob", df.columns)

    def test_single_model_cli_outputs_label(self):
        import tempfile
        from apa_polya_predictor.cli import add_per_model_predictions
        import pandas as pd

        df = pd.DataFrame({
            "header": ["s1", "s2"],
            "deeppolya_prob": [0.3, 0.7],
            "pasnet_prob": [0.6, 0.4],
        })
        out = add_per_model_predictions(df, threshold=0.5)
        self.assertIn("deeppolya_pred", out.columns)
        self.assertIn("pasnet_pred", out.columns)
        self.assertListEqual(out["deeppolya_pred"].tolist(), [0, 1])
        self.assertListEqual(out["pasnet_pred"].tolist(), [1, 0])

    def test_best_ensemble_keyword(self):
        with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f:
            f.write(self._make_fasta())
            f.flush()
            fasta_path = f.name

        pred = Predictor(
            weights_dir=self.weights_dir,
            species="Human",
            models=["ensemble"],
            device="cpu",
        )
        df = pred.predict_fasta(fasta_path)
        self.assertTrue(
            set(pred.models).issubset(
                {"deeppasta", "DNABERT2", "pasnet", "GENA-LM-base", "polyaid", "UTRLM", "DNABERT1"}
            )
        )
        for col in [f"{m}_prob" for m in pred.models]:
            self.assertIn(col, df.columns)


class TestApplyBestEnsemble(TestCase):
    def test_apply_best_ensemble(self):
        from sklearn.ensemble import GradientBoostingClassifier

        # Fake data: 4 samples, 2 models.
        df = {
            "header": ["a", "b", "c", "d"],
            "m1_prob": [0.1, 0.2, 0.8, 0.9],
            "m2_prob": [0.15, 0.25, 0.75, 0.85],
        }
        X = np.column_stack([df["m1_prob"], df["m2_prob"]])
        y = np.array([0, 0, 1, 1])
        gb = GradientBoostingClassifier(n_estimators=5, random_state=0).fit(X, y)
        bundle = {"model": gb, "model_names": ["m1_prob", "m2_prob"], "threshold": 0.5}

        import pandas as pd

        out, threshold = apply_best_ensemble(pd.DataFrame(df), bundle, threshold=0.5)
        self.assertIn("ensemble_prob", out.columns)
        self.assertIn("ensemble_pred", out.columns)
        self.assertEqual(threshold, 0.5)
        self.assertTrue(((out["ensemble_pred"] == 0) | (out["ensemble_pred"] == 1)).all())
