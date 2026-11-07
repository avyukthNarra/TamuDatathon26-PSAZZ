"""Generate pinned teammate forecasts without invoking their all-validation scorers.

Extract sources with git archive as documented in COMBOS_TOP10.md first.
Run each mode in a fresh process to isolate teammate module names.
"""
import argparse
import json
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'docs/sai'


def main():
    mode = argparse.ArgumentParser()
    mode.add_argument('member', choices=['mixer4', 'zpool', 'autotune', 'avycombo'])
    args = mode.parse_args()
    if args.member == 'avycombo':
        def timed_out(*_):
            raise TimeoutError('Optional adaptive combo exceeded the 15-minute limit.')
        signal.signal(signal.SIGALRM, timed_out)
        signal.alarm(900)
    sys.path.insert(0, str(ROOT / 'worktrees/_tm' / args.member / 'mainFiles'))
    sys.path.insert(1, str(HERE))
    import numpy as np
    from harness import load_split, read_pred_csv, evaluate, to_csv
    from harness.scoring import scope_episodes
    from slot_forecasting.io import load_inputs, read_labels, read_targets

    data = ROOT / 'mainFiles/public'
    tq, th = load_inputs(data / 'train')
    labels = read_labels(data / 'train/labels.csv', tq)
    vq, vh = load_inputs(data / 'validation')
    v = load_split('validation')
    extra = {}
    if args.member == 'mixer4':
        from slot_forecasting.pair_mixer import adapt, size_modes_from_targets, win_prior_from_targets
        targets = read_targets(data / 'validation/targets.csv')
        sizes = size_modes_from_targets(targets, vq)
        prior = win_prior_from_targets(targets)
        model = adapt(tq, th, labels, size_modes=sizes, win_prior=prior)
        mid = 'TM_MIXER4'
        leak = 'YES: size modes per type AND Beta-mixture win prior fit from all 96 validation targets; settings previously validation-tuned.'
        extra = {'size_modes': sizes, 'win_prior': prior, 'source': '39e5757 (81d4fff changes docs only)'}
    elif args.member == 'zpool':
        from slot_forecasting.zhill_blend import ZhillBlendForecaster
        from slot_forecasting.pair_mixer_episode import EpisodeMemoryMixer
        model = ZhillBlendForecaster(mixer_factory=EpisodeMemoryMixer).fit(tq, th, labels)
        mid = 'TM_ZPOOL'
        leak = 'No validation targets read at runtime; inherited blend/settings were selected on validation. EpisodeMemoryMixer gamma=0.5.'
    elif args.member == 'avycombo':
        from slot_forecasting.combo_e2_memory import ComboAdaptiveMemoryForecaster
        model = ComboAdaptiveMemoryForecaster().fit(tq, th, labels)
        mid = 'TM_AVYCOMBO'
        leak = 'No validation targets read at runtime; fits train priors and selects memory prequentially in pre-cutoff histories; upstream hyperparameters validation-tuned.'
    else:
        # Native auto_tune.py scores all 96 (including our reserved audit).
        # Reuse its candidates/grid, restrict ALL selection and confirmation to dev.
        import hashlib
        import auto_tune
        models = auto_tune.candidates(tq, th, labels, print)
        names = list(models)
        pp = []
        for name, model in models.items():
            forecast = model.predict(vq, vh)
            pp.append(np.array([forecast[k] for k in v.qid]))
            print('predicted', name, flush=True)
        dev = scope_episodes(v, 'dev')
        tune, confirm = [], []
        for condition in sorted(set(v.condition)):
            eps = [e for e in dev if v.condition[v.episode_rows(e)[0]] == condition]
            eps.sort(key=lambda e: hashlib.sha256(f'autotune-dev|{e}'.encode()).hexdigest())
            tune.extend(eps[::2]); confirm.extend(eps[1::2])
        def score(p, eps):
            rows = np.concatenate([v.episode_rows(e) for e in eps])
            return evaluate(p[rows], v.subset(rows), 'smoke', bootstrap=0)['points']
        candidates = []
        for weights in auto_tune.grid(names):
            p = sum(w * pp[names.index(n)] for n, w in weights.items())
            candidates.append({'weights': weights, 'tune': score(p, tune), 'confirm': score(p, confirm)})
        best = max(candidates, key=lambda r: r['tune'])
        current = next(r for r in candidates if r['weights'] == {'rule': .2, 'mixer': .8})
        accepted = best['tune'] - current['tune'] >= .15 and best['confirm'] >= current['confirm'] - .05
        chosen = best if accepted else current
        p = sum(w * pp[names.index(n)] for n, w in chosen['weights'].items())
        mid = 'TM_AUTOTUNE'
        extra = {'native_not_run': 'Native script reads/scores all validation; would violate dev-only phase. Selection and confirmation restricted to disjoint 36-episode dev halves.', 'best': best, 'current': current, 'accepted': accepted, 'chosen': chosen, 'grid': candidates}
        leak = 'YES for blend selection: dev targets used in this scoped wrapper; native auto_tune uses all 96 validation targets, including its own tune/audit halves. Not a Phase B member.'
    if args.member != 'autotune':
        forecast = model.predict(vq, vh)
        p = np.array([forecast[k] for k in v.qid])
    to_csv(HERE / 'preds' / f'{mid}_validation.csv', v, p)
    report = {'id': mid, 'leakage': leak, 'dev': evaluate(p, v, 'dev'), **extra}
    if args.member == 'mixer4':
        rule = read_pred_csv(HERE / 'preds/TM_RULE_validation.csv', v)
        combined = .2 * rule + .8 * p
        to_csv(HERE / 'preds/TM_COMBINED4_validation.csv', v, combined)
        report['combined4'] = {'leakage': leak, 'dev': evaluate(combined, v, 'dev')}
    (HERE / 'combos' / f'{mid}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(mid, report['dev']['points'], leak, flush=True)


if __name__ == '__main__':
    main()
