"""Train-only, whole-episode guard tests for full versus parity cell mode."""

import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "v7c-mainFiles"))
import slot_forecasting
slot_forecasting.__path__.append(str(pathlib.Path(__file__).resolve().parents[2] / "v7c-mainFiles" / "slot_forecasting"))
from slot_forecasting.schema import Play, Query
from slot_forecasting import cell_mode_gate


_FITS = []


class _FakeSizeRule:
    atoms = {"type": ((0.7, 0.1, 0.1, 0.1), (0.1, 0.7, 0.1, 0.1))}
    lags = (1, 2)
    static_prior = 0.5

    @classmethod
    def learn(cls, queries, history):
        _FITS.append({"kind": "size", "episodes": {query.episode_id for query in queries},
                      "history_episodes": {play.episode_id for play in history}})
        return cls()


class _FakePairMixer:
    def __init__(self, *, cell_mode, **_kwargs):
        self.cell_mode = cell_mode[0] if isinstance(cell_mode, (tuple, list)) else cell_mode

    def fit(self, queries, history):
        _FITS.append({"kind": "mixer", "mode": self.cell_mode,
                      "episodes": {query.episode_id for query in queries},
                      "history_episodes": {play.episode_id for play in history}})
        self.type_size_priors = {"type": [0.25, 0.25, 0.25, 0.25]}
        return self

    def predict(self, queries, _history):
        if self.cell_mode == "full":
            row = (0.05, 0.90, 0.02, 0.02, 0.01)
        else:
            row = (0.20, 0.20, 0.20, 0.20, 0.20)
        return {query.query_id: row for query in queries}


def _fixture(n_episodes):
    queries, history, labels = [], [], {}
    for index in range(n_episodes):
        episode = f"e{index}"
        queries.append(Query(f"q{index}", episode, "m", "type", 0, 1))
        history.append(Play(episode, "m", 0, 0))
        labels[f"q{index}"] = 1
    return tuple(queries), tuple(history), labels


class CellModeGateTest(unittest.TestCase):
    def setUp(self):
        _FITS.clear()

    def _run(self, count):
        queries, history, labels = _fixture(count)
        with patch.object(cell_mode_gate, "SizeRule", _FakeSizeRule), \
             patch.object(cell_mode_gate, "PairMixer", _FakePairMixer), \
             patch.object(cell_mode_gate, "choose_pairs_robust", return_value=((1, 2),)), \
             patch.object(cell_mode_gate, "discover_size_keys", return_value=((),)):
            result = cell_mode_gate.guard_full_cell_mode(queries, history, labels)
        return result, queries

    def test_insufficient_episode_support_keeps_parity(self):
        result, _ = self._run(6)
        self.assertFalse(result["allow_full"])
        self.assertEqual(result["diagnostics"]["chosen_mode"], "parity")
        self.assertEqual(result["diagnostics"]["n_episodes"], 6)

    def test_clear_full_gain_uses_disjoint_episode_fits(self):
        result, queries = self._run(10)
        self.assertTrue(result["allow_full"])
        self.assertEqual(result["diagnostics"]["chosen_mode"], "full")
        self.assertLess(result["diagnostics"]["upper_ci"], 0.0)
        all_episodes = {query.episode_id for query in queries}
        size_fits = [record["episodes"] for record in _FITS if record["kind"] == "size"]
        self.assertEqual(len(size_fits), 2)
        self.assertTrue(size_fits[0].isdisjoint(size_fits[1]))
        self.assertEqual(size_fits[0] | size_fits[1], all_episodes)
        mixer_fits = [record for record in _FITS if record["kind"] == "mixer"]
        self.assertEqual(len(mixer_fits), 4)
        fold_fit_sets = []
        for offset in (0, 2):
            self.assertEqual(mixer_fits[offset]["episodes"], mixer_fits[offset + 1]["episodes"])
            self.assertIn(mixer_fits[offset]["mode"], ("parity", "full"))
            self.assertIn(mixer_fits[offset + 1]["mode"], ("parity", "full"))
            self.assertEqual(mixer_fits[offset]["episodes"], size_fits[offset // 2])
            fold_fit_sets.append(mixer_fits[offset]["episodes"])
        self.assertTrue(fold_fit_sets[0].isdisjoint(fold_fit_sets[1]))
        self.assertEqual(fold_fit_sets[0] | fold_fit_sets[1], all_episodes)


if __name__ == "__main__":
    unittest.main()
