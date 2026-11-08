"""Episode-fold evaluation and bootstrap; evaluation answers stay outside model.py."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'mainFiles'))
from research_exact.model import ExactModel, fit, load_machines, size_predict
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.contest import contest_score
from slot_forecasting.io import read_rows, read_submission, read_targets, write_submission
from slot_forecasting.schema import CONDITIONS


class Evaluator:
    def __init__(self, public):
        self.public = Path(public)
        self.machines, self.conditions, self.queries, self.history, _ = load_machines(public, 'validation')
        self.targets = read_targets(self.public / 'validation' / 'targets.csv')
        self.truth = {t.query_id: np.array(t.probabilities) for t in self.targets}
        self.reference = read_submission(self.public / 'validation' / 'reference.csv')
        self.folds = {}
        self.episodes = {}
        for c in CONDITIONS:
            eids = sorted(set(m.episode_id for m in self.machines if self.conditions[m.query_id] == c))
            for j, eid in enumerate(eids):
                self.folds[eid] = j % 2
                self.episodes[eid] = c
        self.ids = [m.query_id for m in self.machines]
        self.type_by_id = {m.query_id: m.kind for m in self.machines}
        self.kind = {m.query_id: m.kind for m in self.machines}
        self.eid = {m.query_id: m.episode_id for m in self.machines}

    def score(self, pred, fold=None):
        allowed = {eid for eid in self.episodes if fold is None or self.folds[eid] == fold}
        targets = [t for t in self.targets if t.episode_id in allowed]
        ids = {t.query_id for t in targets}
        r = contest_score({i: pred[i] for i in ids}, targets,
                          {i: self.conditions[i] for i in ids}, {i: self.reference[i] for i in ids}, max_points=60)
        return {'points': r['technical_points'], 'R': r['mean_loss_ratio'],
                'ratios': {c: r['by_condition'][c]['loss_ratio'] for c in CONDITIONS}}

    def ci(self, pred, repeats=5000):
        rng = np.random.default_rng(314159)
        ratios = []
        for c in CONDITIONS:
            rows = []
            for eid in sorted(eid for eid, cc in self.episodes.items() if c == cc):
                ids = [i for i in self.ids if self.eid[i] == eid]
                pe = np.mean([np.sum((pred[i]-self.truth[i])**2) for i in ids])
                be = np.mean([np.sum((np.array(self.reference[i])-self.truth[i])**2) for i in ids])
                rows.append((pe, be))
            rows = np.array(rows)
            sample = rng.integers(len(rows), size=(repeats, len(rows)))
            mean = rows[sample].mean(1)
            ratios.append(mean[:, 0]/np.maximum(mean[:, 1], 1e-9))
        points = 60/(1+np.mean(ratios, axis=0))
        return np.quantile(points, [.025, .975]).tolist()

    def atoms(self, fold):
        atoms = {}
        for kind in ('steady', 'balanced', 'jackpot'):
            rows = []
            for t in self.targets:
                if self.kind[t.query_id] != kind or self.folds[t.episode_id] != fold:
                    continue
                p = np.array(t.probabilities)
                if 1-p[0] > .02:
                    rows.append(p[1:]/(1-p[0]))
            # Exact vertices repeated in released target distributions; no held-out rows.
            unique = np.unique(np.round(rows, 10), axis=0)
            if len(unique) != 2:
                raise ValueError(f'{kind}: expected two vertices, got {len(unique)}')
            atoms[kind] = unique[np.argsort(-unique[:, 0])].tolist()
        return atoms

    def subgroup(self, pred, key):
        out = {}
        values = sorted(set(key.values()))
        for value in values:
            ids = {i for i in self.ids if key[i] == value}
            targets = [t for t in self.targets if t.query_id in ids]
            refs = {i: self.reference[i] for i in ids}
            groups = {i: self.conditions[i] for i in ids}
            ratios = {}
            for c in CONDITIONS:
                ep = defaultdict(list)
                for t in targets:
                    if groups[t.query_id] == c:
                        ep[t.episode_id].append(t.query_id)
                if not ep:
                    continue
                pe = []
                be = []
                for qids in ep.values():
                    pe.append(np.mean([np.sum((pred[i]-self.truth[i])**2) for i in qids]))
                    be.append(np.mean([np.sum((np.array(refs[i])-self.truth[i])**2) for i in qids]))
                ratios[c] = float(np.mean(pe)/np.mean(be))
            R = float(np.mean(list(ratios.values())))
            out[value] = {'points': 60/(1+R), 'R': R, 'ratios': ratios}
        return out

    def decomposition(self, pred):
        win, size, full = {}, {}, {}
        for c in CONDITIONS:
            rows = []
            for i in pred:
                if self.conditions[i] != c:
                    continue
                p = np.array(pred[i]); q = self.truth[i]
                pw, qw = 1-p[0], 1-q[0]
                ps, qs = p[1:]/pw, q[1:]/qw
                # Two counterfactual errors; they need not add (there is a cross-term).
                win_error = (pw-qw)**2*(1+np.sum(qs**2))
                size_error = qw**2*np.sum((ps-qs)**2)
                rows.append((win_error, size_error, np.sum((p-q)**2), (pw-qw)**2))
            win[c], size[c], full[c] = [float(v) for v in np.mean(rows, axis=0)[:3]]
        return {'win_only_error': win, 'size_only_error': size, 'full_error': full,
                'definition': 'win_only: replace predicted sizes by truth; size_only: replace predicted win probability by truth; counterfactuals do not sum'}


def combine(win, sizes):
    return {i: np.r_[1-p, p*np.array(sizes[i])] for i, p in win.items()}


def evaluate(public, out, grid=25, refit=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    params_path = out / 'params.json'
    if refit or not params_path.exists():
        params = fit(public, size=grid)
        params_path.write_text(json.dumps(params, indent=2))
    else:
        params = json.loads(params_path.read_text())
    ev = Evaluator(public)
    win_path = out / f'win_grid{grid}.json'
    truth_priors = json.loads((ROOT.parent / '_coord/fresh/levels_from_truth.json').read_text())
    win = {}
    diagnostics = []
    if win_path.exists() and not refit:
        saved = json.loads(win_path.read_text()); win, diagnostics = saved['win'], saved['diagnostics']
    else:
        for eval_fold in (0, 1):
            fold_params = json.loads(json.dumps(params))
            prior = truth_priors['foldB' if eval_fold == 0 else 'foldA']
            for kind in ('steady', 'balanced', 'jackpot'):
                row = prior[kind]
                par = fold_params['levels'][kind]
                par['mean'] = [row['dead']['a'], row['live']['a']]
                par['slope'] = [row['dead']['b'], row['live']['b']]
                corr = np.array(par['cov'])
                rho = corr[0,1]/np.sqrt(corr[0,0]*corr[1,1])
                sd0, sd1 = row['dead']['sd'], row['live']['sd']
                par['cov'] = [[sd0**2, rho*sd0*sd1], [rho*sd0*sd1, sd1**2]]
            model = ExactModel(fold_params, size=grid)
            fold_machines = [m for m in ev.machines if ev.folds[m.episode_id] == eval_fold]
            fold_win, fold_diag = model.predict_win(fold_machines)
            win.update(fold_win); diagnostics.extend(fold_diag)
        win_path.write_text(json.dumps({'win': win, 'diagnostics': diagnostics}, indent=2))
    all_preds = {}
    evidence = {}
    for atom_source in ('train', 'A', 'B'):
        atoms = params['atoms'] if atom_source == 'train' else ev.atoms(0 if atom_source == 'A' else 1)
        for family in ('all', 'static', 'lag'):
            sizes, lls = size_predict(ev.machines, atoms, params['size_prior'], family)
            name = atom_source + '_' + family
            all_preds[name] = combine(win, sizes)
            evidence[name] = sum(lls)
    cross = {}
    chosen = {}
    for evaluation_fold in (0, 1):
        training_fold = 1-evaluation_fold
        atom_source = 'B' if training_fold else 'A'
        candidates = [f'{atom_source}_{f}' for f in ('all', 'static', 'lag')]
        # This decision uses training_fold answers only, and its own fold's atoms.
        name = min(candidates, key=lambda n: ev.score(all_preds[n], training_fold)['R'])
        chosen[str(evaluation_fold)] = name
        for i in ev.ids:
            if ev.folds[ev.eid[i]] == evaluation_fold:
                cross[i] = all_preds[name][i]
    train = all_preds['train_all']
    write_submission(out / 'validation_crossfit.csv', cross)
    write_submission(out / 'validation_trainonly.csv', train)
    winbins = {m.query_id: ('0-0.2' if 1-ev.truth[m.query_id][0] < .2 else '0.2-0.5' if 1-ev.truth[m.query_id][0] < .5 else '0.5-1') for m in ev.machines}
    summary = {'crossfit': ev.score(cross), 'crossfit_ci95': ev.ci(cross),
               'crossfit_A': ev.score(cross, 0), 'crossfit_B': ev.score(cross, 1),
               'choices': chosen, 'train_only': ev.score(train), 'train_only_ci95': ev.ci(train),
               'variants': {n: ev.score(p) for n, p in all_preds.items()},
               'size_log_evidence_validation': evidence,
               'decomposition': ev.decomposition(cross),
               'by_machine_type': ev.subgroup(cross, ev.type_by_id),
               'by_true_win_probability': ev.subgroup(cross, winbins),
               'decomposition_by_type': {k: ev.decomposition({i: cross[i] for i in cross if ev.type_by_id[i] == k}) for k in sorted(set(ev.type_by_id.values()))},
               'decomposition_by_true_win_probability': {k: ev.decomposition({i: cross[i] for i in cross if winbins[i] == k}) for k in sorted(set(winbins.values()))}}
    _, _, qs, hs, labels = load_machines(public, 'train', labels=True)
    pooled = FrequencyForecaster('pooled').fit(qs, hs, labels).predict(ev.queries, ev.history)
    summary['pooled'] = ev.score(pooled)
    summary['pooled_A'] = ev.score(pooled, 0)
    summary['pooled_B'] = ev.score(pooled, 1)
    summary['seconds'] = time.perf_counter()-started
    (out / 'results.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(ROOT/'mainFiles/public'))
    parser.add_argument('--out', default=str(ROOT/'runs/exact'))
    parser.add_argument('--grid', type=int, default=25)
    parser.add_argument('--refit', action='store_true')
    args = parser.parse_args()
    evaluate(args.data, args.out, args.grid, args.refit)
