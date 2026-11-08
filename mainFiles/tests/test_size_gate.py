"""Causal support and episode-fold tests for the train-only size gate."""

import pathlib
import sys
import unittest
from unittest.mock import patch

# The working package includes the new SizeRule while the c2-combined tree
# supplies PairMixer until integration assembles them into one directory.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "v7c-mainFiles"))

import slot_forecasting
slot_forecasting.__path__.append(str(pathlib.Path(__file__).resolve().parents[2] / "v7c-mainFiles" / "slot_forecasting"))
from slot_forecasting.schema import Play, Query
from slot_forecasting import size_gate


class _FakeRule:
    fitted = []

    def __init__(self):
        self.atoms = {"type": ((0.7, 0.1, 0.1, 0.1), (0.1, 0.7, 0.1, 0.1))}
        self.lags = (1, 2)
        self.static_prior = 0.5

    @classmethod
    def learn(cls, queries, history):
        cls.fitted.append({"queries": {q.episode_id for q in queries},
                           "history": {p.episode_id for p in history}})
        return cls()

    def prequential(self, pay, machine_type):
        # Forecast is fixed before any raw payout is absorbed.
        for tick, _payout in enumerate(pay):
            yield tick, [0.97, 0.01, 0.01, 0.01]


class _FakeBaseline:
    type_size_priors = {"type": [0.25, 0.25, 0.25, 0.25]}

    def walk(self, own, machine_type, stop, emit):
        for tick in range(stop + 1):
            emit(tick, [0.2, 0.2, 0.2, 0.2, 0.2])


def _fixture(n_episodes):
    queries, history, labels = [], [], {}
    for index in range(n_episodes):
        episode = f"e{index}"
        query = Query(f"q{index}", episode, "m", "type", 0, 1)
        queries.append(query)
        history.append(Play(episode, "m", 0, 1))
        labels[query.query_id] = 1
    return tuple(queries), tuple(history), labels


class SizeGateTest(unittest.TestCase):
    def setUp(self):
        _FakeRule.fitted = []

    def _run(self, n_episodes):
        queries, history, labels = _fixture(n_episodes)
        with patch.object(size_gate, "SizeRule", _FakeRule), \
             patch.object(size_gate, "adapt", return_value=_FakeBaseline()):
            result = size_gate.fit_size_gate(queries, history, labels)
        return result, queries, history

    def test_insufficient_episode_support_defaults_to_v6(self):
        result, _, _ = self._run(6)
        self.assertFalse(result["use_candidate"])
        self.assertEqual(result["diagnostics"]["chosen_head"], "v6_size_head")
        self.assertLess(result["diagnostics"]["n_episodes"], 8)

    def test_clear_train_only_gain_selects_candidate_and_folds_are_disjoint(self):
        result, queries, history = self._run(10)
        self.assertTrue(result["use_candidate"])
        self.assertEqual(result["diagnostics"]["chosen_head"], "size_rule")
        self.assertEqual(result["diagnostics"]["n_wins"], 20)
        query_episodes = {q.episode_id for q in queries}
        history_episodes = {p.episode_id for p in history}
        self.assertEqual(len(_FakeRule.fitted), 2)
        for fit in _FakeRule.fitted:
            self.assertEqual(fit["queries"], fit["history"])
            self.assertTrue(fit["queries"] <= query_episodes)
            self.assertTrue(fit["history"] <= history_episodes)
        self.assertTrue(_FakeRule.fitted[0]["queries"].isdisjoint(_FakeRule.fitted[1]["queries"]))
        self.assertEqual(_FakeRule.fitted[0]["queries"] | _FakeRule.fitted[1]["queries"],
                         query_episodes)

    def test_missing_stream_emission_fails_closed_instead_of_scoring_uniform(self):
        class MissingEmission(_FakeRule):
            def prequential(self, pay, machine_type):
                return iter(())

        queries, history, labels = _fixture(8)
        with patch.object(size_gate, "SizeRule", MissingEmission), \
             patch.object(size_gate, "adapt", return_value=_FakeBaseline()):
            result = size_gate.fit_size_gate(queries, history, labels)
        self.assertFalse(result["use_candidate"])
        self.assertIn("omitted scored tick", result["diagnostics"]["reason"])


if __name__ == "__main__":
    unittest.main()
