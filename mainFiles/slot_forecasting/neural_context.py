"""E7: tiny shared hypernetwork for machine-specific contextual win logits.

The model has no machine, episode, signal, or condition embeddings. It consumes
per-machine causal prefix summaries and predicts coefficients for signed binary
main and pair lag features. It is an optional CPU experiment, not a required
runtime dependency of the baseline forecasters.
"""
from copy import deepcopy
import hashlib
import json
from itertools import combinations
from math import log
from pathlib import Path
from time import perf_counter

import numpy as np

from .pair_logistic import PairLogisticForecaster, _lag_vector
from .schema import index_history, unique_queries


class NeuralContextForecaster(PairLogisticForecaster):
    """E1 payout-shape head with a train-only learned contextual win head.

    ``recent_half_life`` adds a second, recency-weighted set of prefix
    sufficient statistics. ``None`` uses only all-history statistics. Fitted
    artifacts can be written with ``save_artifact`` and restored with
    ``load_artifact`` without refitting.
    """

    def __init__(self, *, recent_half_life=32.0, min_context=16,
                 examples_per_machine=24, max_epochs=50, patience=6,
                 batch_size=256, learning_rate=1e-3, seed=2026,
                 lags=8, C=3.0, half_life=32.0, min_rows=12,
                 base_half_life=24.0, clip=1e-4):
        super().__init__(lags=lags, pairs=True, C=C, half_life=half_life,
                         min_rows=min_rows, base_half_life=base_half_life,
                         clip=clip)
        self.recent_half_life = (None if recent_half_life is None
                                 else float(recent_half_life))
        self.min_context = int(min_context)
        self.examples_per_machine = int(examples_per_machine)
        self.max_epochs, self.patience = int(max_epochs), int(patience)
        self.batch_size = int(batch_size)
        self.learning_rate, self.seed = float(learning_rate), int(seed)
        if self.recent_half_life is not None and self.recent_half_life <= 0:
            raise ValueError('recent_half_life must be positive or None')
        if min(self.min_context, self.examples_per_machine, self.max_epochs,
               self.patience, self.batch_size) < 1:
            raise ValueError('context, sample, epoch, patience, and batch sizes must be positive')
        if self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive')
        self.network = None
        self.training_diagnostics = None
        self._torch = None

    @staticmethod
    def _signed_lags(x):
        return 2.0 * np.asarray(x, dtype=np.float64) - 1.0

    def _query_features(self, x):
        signed = self._signed_lags(x)
        return np.concatenate((signed, np.asarray([
            signed[a] * signed[b] for a, b in combinations(range(self.lags), 2)
        ], dtype=np.float64)))

    def _sequence_rows(self, seq, cutoff=None):
        ticks = sorted(t for t in seq if cutoff is None or t < cutoff)
        X, y, kept_ticks = [], [], []
        for tick in ticks:
            x = _lag_vector(seq, tick, self.lags)
            if x is None:
                continue
            X.append(self._query_features(x))
            y.append(int(seq[tick] > 0))
            kept_ticks.append(tick)
        if not X:
            return (np.empty((0, self.lags + self.lags * (self.lags - 1) // 2)),
                    np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        return np.asarray(X), np.asarray(y, dtype=np.int64), np.asarray(kept_ticks, dtype=np.int64)

    def _prefix_summaries(self, Z, y, ticks, terminal_query_tick=None):
        """Return all-history and optional recent prefix summaries, exclusive of row i."""
        n, width = Z.shape
        signed_y = (2 * y - 1).astype(np.float64)
        stats = np.zeros((n + 1, 2 * width + 2), dtype=np.float64)
        if n:
            counts = np.arange(1, n + 1, dtype=np.float64)
            stats[1:, :width] = np.cumsum(Z, axis=0) / counts[:, None]
            stats[1:, width:2 * width] = (
                np.cumsum(signed_y[:, None] * Z, axis=0) / counts[:, None]
            )
            stats[1:, 2 * width] = np.cumsum(y, dtype=np.float64) / counts
            stats[1:, -1] = np.log1p(np.arange(1, n + 1)) / log(257.0)
        if self.recent_half_life is None:
            return stats
        recent = np.zeros_like(stats)
        if n:
            # Anchor at the first tick so prefix weights do not depend on any
            # later row appended to the sequence.
            weights = np.exp2((ticks.astype(float) - float(ticks[0])) /
                              self.recent_half_life)
            weight_sum = np.concatenate(([0.0], np.cumsum(weights)))
            weighted_z = np.vstack((np.zeros((1, width)),
                                    np.cumsum(weights[:, None] * Z, axis=0)))
            weighted_yz = np.vstack((np.zeros((1, width)),
                                     np.cumsum(weights[:, None] * signed_y[:, None] * Z,
                                               axis=0)))
            weighted_y = np.concatenate(([0.0], np.cumsum(weights * y)))
            denom = np.where(weight_sum > 0.0, weight_sum, 1.0)
            recent[:, :width] = weighted_z / denom[:, None]
            recent[:, width:2 * width] = weighted_yz / denom[:, None]
            recent[:, 2 * width] = weighted_y / denom
            anchors = np.empty(n + 1, dtype=np.float64)
            anchors[:n] = ticks
            anchors[n] = (ticks[-1] + 1 if terminal_query_tick is None
                          else terminal_query_tick)
            # Convert masses scaled at ticks[0] into weights relative to each
            # prefix's known query tick (or the explicit final cutoff).
            query_scale = np.exp2((float(ticks[0]) - (anchors - 1.0)) /
                                  self.recent_half_life)
            query_weight_sum = weight_sum * query_scale
            recent[:, -1] = np.log1p(query_weight_sum) / log(257.0)
        return np.concatenate((stats, recent), axis=1)

    def context_statistics(self, seq, cutoff):
        """Compute summary of valid exact-lag examples at ticks strictly before cutoff."""
        Z, y, ticks = self._sequence_rows(seq, cutoff=cutoff)
        if len(Z) < self.min_context:
            return None, len(Z)
        prefixes = self._prefix_summaries(Z, y, ticks, terminal_query_tick=cutoff)
        return prefixes[-1].astype(np.float32), len(Z)

    @staticmethod
    def _episode_is_holdout(episode_id):
        digest = hashlib.sha256(episode_id.encode('utf-8')).digest()
        return int.from_bytes(digest[:4], 'big') % 5 == 0

    def _make_training_examples(self, queries, history):
        query_map = unique_queries(queries)
        indexed = index_history(queries, history)
        examples = {False: [], True: []}
        rng = np.random.default_rng(self.seed)
        for query in query_map.values():
            seq = {p.tick: p.payout
                   for p in indexed.get((query.episode_id, query.machine_id), ())}
            Z, y, ticks = self._sequence_rows(seq)
            if len(Z) <= self.min_context:
                continue
            prefixes = self._prefix_summaries(
                Z, y, ticks, terminal_query_tick=query.cutoff
            )
            eligible = np.arange(self.min_context, len(Z), dtype=np.int64)
            if len(eligible) > self.examples_per_machine:
                eligible = np.sort(rng.choice(eligible, self.examples_per_machine,
                                              replace=False))
            holdout = self._episode_is_holdout(query.episode_id)
            for index in eligible:
                examples[holdout].append((prefixes[index], Z[index], y[index]))
        return examples

    def fit(self, queries, history, labels=None):
        started = perf_counter()
        queries, history = tuple(queries), tuple(history)
        super().fit(queries, history, labels)
        examples = self._make_training_examples(queries, history)
        train_rows, holdout_rows = examples[False], examples[True]
        episodes = {q.episode_id for q in queries}
        if len(episodes) > 1 and (not train_rows or not holdout_rows):
            # Preserve a deterministic episode holdout even in small synthetic runs.
            ordered = sorted(episodes, key=lambda e: hashlib.sha256(e.encode()).hexdigest())
            fallback_episode = ordered[0]
            train_rows, holdout_rows = [], []
            all_examples = self._make_training_examples_with_episode_map(queries, history)
            for episode, row in all_examples:
                (holdout_rows if episode == fallback_episode else train_rows).append(row)
        if not train_rows:
            raise ValueError('TRAIN histories do not provide causal neural training examples')
        if not holdout_rows:
            raise ValueError('TRAIN episodes do not provide internal holdout examples')

        import torch
        torch.set_num_threads(min(2, torch.get_num_threads()))
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        stats_width = len(train_rows[0][0])
        self.network = torch.nn.Sequential(
            torch.nn.Linear(stats_width, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.lags + self.lags * (self.lags - 1) // 2 + 1),
        )
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()

        def tensors(rows):
            stats = torch.as_tensor(np.asarray([r[0] for r in rows]), dtype=torch.float32)
            query_z = torch.as_tensor(np.asarray([r[1] for r in rows]), dtype=torch.float32)
            target = torch.as_tensor(np.asarray([r[2] for r in rows]), dtype=torch.float32)
            return stats, query_z, target

        train_stats, train_z, train_y = tensors(train_rows)
        hold_stats, hold_z, hold_y = tensors(holdout_rows)
        best_loss, best_epoch, best_state, stale = float('inf'), 0, None, 0
        for epoch in range(1, self.max_epochs + 1):
            self.network.train()
            order = torch.randperm(len(train_y))
            for start in range(0, len(order), self.batch_size):
                index = order[start:start + self.batch_size]
                output = self.network(train_stats[index])
                logits = (output[:, :-1] * train_z[index]).sum(dim=1) + output[:, -1]
                loss = criterion(logits, train_y[index])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.network.eval()
            with torch.no_grad():
                output = self.network(hold_stats)
                logits = (output[:, :-1] * hold_z).sum(dim=1) + output[:, -1]
                hold_loss = float(criterion(logits, hold_y).item())
            if hold_loss < best_loss - 1e-6:
                best_loss, best_epoch = hold_loss, epoch
                best_state, stale = deepcopy(self.network.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.network.load_state_dict(best_state)
        self.network.eval()
        self._torch = torch
        self.training_diagnostics = {
            'epochs': epoch,
            'best_epoch': best_epoch,
            'internal_train_holdout_loss': best_loss,
            'train_examples': len(train_rows),
            'holdout_examples': len(holdout_rows),
            'training_seconds': perf_counter() - started,
            'seed': self.seed,
            'recent_half_life': self.recent_half_life,
        }
        return self

    def _make_training_examples_with_episode_map(self, queries, history):
        query_map = unique_queries(queries)
        indexed = index_history(queries, history)
        rng = np.random.default_rng(self.seed)
        result = []
        for query in query_map.values():
            seq = {p.tick: p.payout
                   for p in indexed.get((query.episode_id, query.machine_id), ())}
            Z, y, ticks = self._sequence_rows(seq)
            if len(Z) <= self.min_context:
                continue
            prefixes = self._prefix_summaries(
                Z, y, ticks, terminal_query_tick=query.cutoff
            )
            eligible = np.arange(self.min_context, len(Z), dtype=np.int64)
            if len(eligible) > self.examples_per_machine:
                eligible = np.sort(rng.choice(eligible, self.examples_per_machine,
                                              replace=False))
            for index in eligible:
                result.append((query.episode_id,
                               (prefixes[index], Z[index], y[index])))
        return result

    def win_probability(self, seq, cutoff):
        clean_seq = {tick: payout for tick, payout in seq.items() if tick < cutoff}
        xq = _lag_vector(clean_seq, cutoff, self.lags)
        if xq is None or self.network is None:
            return super().win_probability(clean_seq, cutoff)
        stats, count = self.context_statistics(clean_seq, cutoff)
        if stats is None or count < self.min_context:
            return super().win_probability(clean_seq, cutoff)
        zq = self._query_features(xq).astype(np.float32)
        torch = self._torch
        with torch.no_grad():
            context = torch.as_tensor(stats[None, :], dtype=torch.float32)
            query = torch.as_tensor(zq[None, :], dtype=torch.float32)
            output = self.network(context)
            logit = (output[:, :-1] * query).sum(dim=1) + output[:, -1]
            probability = torch.sigmoid(logit)[0].item()
        return float(np.clip(probability, self.clip, 1.0 - self.clip))

    def _artifact_configuration(self):
        return {
            'recent_half_life': self.recent_half_life,
            'min_context': self.min_context,
            'examples_per_machine': self.examples_per_machine,
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'seed': self.seed,
            'lags': self.lags,
            'C': self.C,
            'half_life': self.half_life,
            'min_rows': self.min_rows,
            'base_half_life': self.base.half_life,
            'clip': self.clip,
        }

    def save_artifact(self, path, metadata=None):
        """Save weights, constructor configuration, and fitted shape priors."""
        if self.network is None or self.training_diagnostics is None:
            raise ValueError('fit the forecaster before saving an artifact')
        import torch

        payload = {
            'format_version': 1,
            'configuration': self._artifact_configuration(),
            'network_state': self.network.state_dict(),
            'training_diagnostics': self.training_diagnostics,
            'base_forecaster': {
                'method': self.base.method,
                'prior_strength': self.base.prior_strength,
                'half_life': self.base.half_life,
                'global_prior': self.base.global_prior,
                'context_priors': self.base.context_priors,
            },
            'metadata': {} if metadata is None else metadata,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        manifest = {key: value for key, value in payload.items()
                    if key != 'network_state'}
        path.with_suffix(path.suffix + '.json').write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
        )

    @classmethod
    def load_artifact(cls, path):
        """Restore a fitted forecaster from ``save_artifact`` without training."""
        import torch

        try:
            payload = torch.load(path, map_location='cpu', weights_only=True)
        except TypeError:  # PyTorch versions before the weights_only argument.
            payload = torch.load(path, map_location='cpu')
        if not isinstance(payload, dict) or payload.get('format_version') != 1:
            raise ValueError('unsupported or malformed neural context artifact')
        configuration = payload.get('configuration')
        base_state = payload.get('base_forecaster')
        if not isinstance(configuration, dict) or not isinstance(base_state, dict):
            raise ValueError('artifact is missing configuration or base forecaster state')
        model = cls(**configuration)
        lag_width = model.lags + model.lags * (model.lags - 1) // 2
        stats_width = 2 * lag_width + 2
        if model.recent_half_life is not None:
            stats_width *= 2
        model.network = torch.nn.Sequential(
            torch.nn.Linear(stats_width, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, lag_width + 1),
        )
        model.network.load_state_dict(payload['network_state'])
        model.network.eval()
        model._torch = torch
        model.training_diagnostics = payload.get('training_diagnostics')
        model.base.method = base_state['method']
        model.base.prior_strength = float(base_state['prior_strength'])
        model.base.half_life = float(base_state['half_life'])
        model.base.global_prior = tuple(base_state['global_prior'])
        model.base.context_priors = {
            key: tuple(values) for key, values in base_state['context_priors'].items()
        }
        return model
