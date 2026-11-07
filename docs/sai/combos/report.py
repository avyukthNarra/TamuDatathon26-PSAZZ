"""Render the frozen experiment record. Does not evaluate or inspect audit targets."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from harness import load_split, read_pred_csv
from harness.scoring import scope_episodes

SHORT = {'TM_RULE': 'Rule', 'TM_MIXER3': 'Mixer3', 'TM_MIXER4': 'Mixer4',
         'TM_ZPOOL': 'Zpool', 'TM_AVYCOMBO': 'AvyCombo', 'TM_COMBINED3': 'Ref',
         'TM_E67': 'AvyE67'}


def short(name):
    return SHORT.get(name, name)


def weights(r):
    def one(d):
        return ' + '.join(f'{100*w:.2f}% {short(n)}' for n, w in d.items() if w > 1e-8)
    w = r['weights']
    if r['family'] == 'factorised':
        return 'win: ' + one(w['win']) + '; size: ' + one(w['size'])
    return one(w)


def score(ev):
    return f"{ev['points']:.3f} [{ev['ci95'][0]:.3f}, {ev['ci95'][1]:.3f}]"


def difference(d):
    return f"{d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def main():
    data = json.loads((HERE / 'combos/results.json').read_text())
    by = {r['id']: r for r in data['results']}
    top = [by[name] for name in data['top10']]
    assert 'audit' in data, 'Render only after the one-shot audit has completed.'
    v = load_split('validation')
    rows = np.concatenate([v.episode_rows(e) for e in scope_episodes(v, 'dev')])
    reference = read_pred_csv(HERE / 'preds/TM_COMBINED3_validation.csv', v)
    def diagnostics(p):
        p, q = p[rows], v.truth[rows]
        pw = 1-p[:, 0]
        size = p[:, 1:] / pw[:, None]
        return np.mean((p[:, 0]-q[:, 0])**2), np.mean(((size*(1-q[:, 0])[:, None])-q[:, 1:])**2)*4
    ref_win, ref_size = diagnostics(reference)
    best = top[0]
    tied = [str(i) for i, r in enumerate(top, 1) if r['noise_band']]
    lines = [
        f"The best dev result is COMBO 1 at {score(best['dev'])} points out of 60, "
        f"a paired {difference(best['vs_reference'])} against the previous 20% rule + 80% mixer-v3 entry "
        f"(54.774 dev / 55.90 audit). All ten listed combinations fall in the best model's noise band; "
        f"their numerical order does not establish a winner. COMBO 7 is the cheaper stdlib linear alternative (24.05% Rule, 75.95% Mixer4). The main change is mixer v4's win prior, "
        f"which itself reads all validation targets. These are useful public-validation comparisons, not blind evidence of generalisation. "
        f"The {len(data['results'])}-structure search includes every teammate's available member; audit was opened once after freezing this ranking and all weights.",
        '',
        '| Rank | Name | Members + full-dev weights | Family | Dev OOF [95% CI] | Δ vs previous entry, dev [paired CI] | Audit [95% CI] | Stdlib? |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for rank, r in enumerate(top, 1):
        lines.append(f"| {rank} | COMBO {rank} | {weights(r)} | {r['family']} | {score(r['dev'])} | {difference(r['vs_reference'])} | {score(data['audit'][f'COMBO_{rank}'])} | {'yes' if r['stdlib_only'] else 'no'} |")
    lines += ['',
        'Weights are rounded for reading; exact vectors and per-fold weights are in [combos/results.json](combos/results.json). '
        'The delta column compares dev with dev; 55.90 names the reference run’s historical audit score, not the subtraction baseline. '
        'Stdlib describes prediction-time source requirements, after omitting components with fitted weight ≤ 1e-8; all selected structures still require their members for reproducing OOF fitting. '
        'The search/fitting code uses NumPy/SciPy in every case.',
        '',
        'Aliases: Rule = zhill’s TM_RULE; Mixer3/Mixer4 = Pranav’s TM_MIXER3/TM_MIXER4; '
        'Zpool = Avyukth’s pooled-memory zhill blend; AvyCombo = Avyukth’s adaptive tree/logistic combo; '
        'AvyE67 = his neural E6/E7 blend; Ref = TM_COMBINED3. C1, C2 and M9b_B4 are Sai/Claude models; S5_B4 is Codex’s soft-target learner.',
        '', '## New teammate members (Phase A, dev only)', '',
        '| ID | Dev [95% CI] | Validation-target use |', '| --- | --- | --- |']
    for name in ['TM_MIXER4', 'TM_ZPOOL', 'TM_AUTOTUNE', 'TM_AVYCOMBO']:
        d = json.loads((HERE / 'combos' / f'{name}.json').read_text())
        lines.append(f"| {name} | {score(d['dev'])} | {d['leakage']} |")
        if name == 'TM_MIXER4':
            lines.append(f"| TM_COMBINED4 | {score(d['combined4']['dev'])} | 20% saved Rule + 80% Mixer4; inherits Mixer4 leakage. |")
    auto = json.loads((HERE / 'combos/TM_AUTOTUNE.json').read_text())
    lines += ['',
        f"The dev-only autotuner's best candidate was {auto['best']['weights']}; it kept {auto['chosen']['weights']}. "
        f"Tune gain was {auto['best']['tune']-auto['current']['tune']:.3f}, below the native 0.15 threshold. "
        'The native auto_tune.py was deliberately not executed: it scores all 96 validation episodes, violating Phase A’s dev-only scope. '
        'The wrapper reuses its exact candidates and 0.1 weight grid with disjoint, condition-stratified 36/36-episode dev halves. '
        'That retained selection is an in-sample dev diagnostic, not an OOF score, and TM_AUTOTUNE is excluded from the combination pool. '
        'TM_AVYCOMBO completed in about four minutes with the already installed scikit-learn; no member was skipped.',
        '', '## Search and uncertainty', '',
        'All subsets of two or three of the eleven nominal members were enumerated: 220 subsets × three families = 660 structures. '
        'The reference is an allowed member but already contains Rule and Mixer3, so three nominal members can represent more than three underlying predictors. '
        'Linear and geometric pools have one simplex; factorised pools fit separate win and conditional-size simplices over the same subset (union ≤ 3). '
        'Geometric pooling uses log(max(p, 1e-12)) and renormalises. Factorised pooling uses the harness’s parts/combine functions.',
        '',
        'For each fixed structure, weights are fitted on five folds and evaluated on the sixth, repeating across six deterministic, '
        'condition-stratified episode folds (12 episodes held out each time). A machine never crosses its episode’s fold. '
        'Condition/reference loss weights are estimated only from the training folds. Full-dev fitted weights produce the saved deployment forecasts. '
        'Every table dev number uses held-out weights; it does not use the full-dev weights shown beside it. '
        'The 95% CIs use the shared evaluate/diff_ci harness with 1,000 condition-stratified episode bootstraps (seed 314159). '
        'Ranking is fixed on dev; numerical copies from zero-weight padding are suppressed only when full-fit dev predictions differ by <1e-5 in every probability. Among deployment-equivalent structures, the highest OOF structure represents the group. '
        'All 660 structures, including copies, remain in results.json.',
        '',
        f"A separate nested selector uses three inner episode folds to choose a structure within each of the six outer training sets, "
        f"then fits that structure there and scores the untouched outer fold. Its dev result is {score(data['selection']['dev'])}, "
        f"delta {difference(data['selection']['vs_reference'])}. This evaluates the selection procedure; it is not substituted for any fixed structure’s OOF number. "
        'Inherited member training/tuning is not rerun inside these folds, so neither number removes member-level leakage. '
        'Numerical fitting uses SLSQP with analytic gradients, a uniform simplex start, maxiter=200 and ftol=1e-11; any unsuccessful fit aborts the run.',
        '',
        f"Noise band: {data['noise_band_count']} of 660 structures have a paired-difference CI against the best that includes zero. "
        f"Listed ranks in the band: {', '.join(tied)}. Treat them as ties. This is a descriptive band without a multiple-comparison correction. "
        'Bootstrap intervals condition on fitted OOF predictions and do not refit the search within bootstrap replicates.',
        '', '## The ten combinations']
    explanations = {
        'linear': 'A convex average lets the members offset probability errors directly across all five payout classes.',
        'geometric': 'A renormalised geometric average rewards agreement on classes and can damp probabilities supported by only one member.',
        'factorised': 'Independent win and size weights allow one model to correct p0 while another supplies the payout mix after a win.'}
    for rank, r in enumerate(top, 1):
        p = read_pred_csv(HERE / 'combos' / f'COMBO_{rank}_oof_validation.csv', v)
        win, size = diagnostics(p)
        shares = r['dev']['class_error_shares']
        reduction = lambda value, baseline: f'{100*(baseline-value)/baseline:+.1f}%'
        cost = ('All listed model sources are standard-library code; port only the pooling arithmetic and exact fitted constants. '
                'Mixer4 requires its validation-derived size modes and Beta-mixture prior, beyond the currently packaged v3 entry.' if r['stdlib_only'] else
                'Package or port the non-stdlib member(s); Sai/Claude/Codex members need NumPy/SciPy and AvyE67 needs its neural environment. '
                'Zero-weight components may be omitted only after checking deployment parity; no new entry is packaged here.')
        lines += ['', f'### {rank}. COMBO {rank} — {r["family"]}', '',
                  f"Search subset: {', '.join(r['members'])}. Full-dev deployment: {weights(r)}. {explanations[r['family']]} Measured on OOF dev, "
                  f"p0 MSE falls by {reduction(win, ref_win)} and conditional-size squared error at the true win rate falls by {reduction(size, ref_size)} "
                  f"relative to Ref (negative means an increase). The size diagnostic isolates size error by substituting the true win probability; "
                  f"it is not an additive decomposition of total loss.", '',
                  'Class-error shares (0, 1, 2, 5, 20): ' + ', '.join(f'{100*shares[k]:.1f}%' for k in ['p_0', 'p_1', 'p_2', 'p_5', 'p_20']) +
                  f". Paired delta to COMBO 1: {difference(r['vs_best'])}; {'tied within noise' if r['noise_band'] else 'outside its noise band'}.", '',
                  'Packaging cost: ' + cost]
    lines += ['', '## Reproduce and verify', '',
        'Run from the repository root with the existing Python environment. Never switch this shared checkout’s branch. '
        'Archive the pinned sources into worktrees/_tm; 81d4fff changes only documentation relative to mixer4’s 39e5757 model.', '',
        '```sh',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
        'mkdir -p worktrees/_tm/{mixer4,zpool,autotune,avycombo}',
        'git archive 39e5757 | tar -x -C worktrees/_tm/mixer4',
        'git archive 9573666 | tar -x -C worktrees/_tm/zpool',
        'git archive 3bdf336 | tar -x -C worktrees/_tm/autotune',
        'git archive 7c506e2 | tar -x -C worktrees/_tm/avycombo',
        'nice -n 10 python3 docs/sai/combos/members.py mixer4',
        'nice -n 10 python3 docs/sai/combos/members.py zpool',
        'nice -n 10 python3 docs/sai/combos/members.py autotune',
        'nice -n 10 python3 docs/sai/combos/members.py avycombo',
        "nice -n 10 python3 -m unittest discover -s docs/sai/combos -p 'test_*.py' -v",
        'nice -n 10 python3 docs/sai/combos/search.py dev --out worktrees/_coord/combo-reproduce',
        '```', '',
        'The original dev run used the default output directory, then search.py audit once. '
        'Do not rerun the audit or delete its marker. The audit command verifies source/prediction SHA-256s and exclusively creates audit.once.json before reading the split. '
        'Exactly eleven evaluate calls use who=COMBO_TOP10: ten combinations and TM_COMBINED3. '
        'audit.once.json records the frozen pre-audit results hash, ranking and forecast hashes. '
        'Dev replay uses a new output directory so it cannot rewrite the frozen record. '
        'Saved COMBO_<n>_validation.csv files use full-dev weights for all 768 queries; *_oof_validation.csv files use OOF predictions on dev and inert 0.2 placeholders on audit and must never be submitted.', '',
        'Validation performed: five synthetic search tests; ten existing copied mixer tests; eight existing verification tests; '
        'twelve pinned v4 mixer tests; six pinned pooled-memory tests (41 total passed); Python compilation, whitespace and forecast integrity checks. '
        'An independent full dev replay reproduced results.json and all ten deployment/OOF CSVs byte for byte. '
        'There is no configured lint command or CI. No dependencies or configuration were changed.',
        '', '## Caveats and next steps', '',
        '- Teammate models/settings were tuned using all 96 validation episodes. Mixer3 reads all validation size targets; Mixer4 additionally fits a win prior from them. '
        'The audit numbers are not blind. The new weights alone are dev-only.',
        '- There are only 24 audit episodes (six per condition). Audit is shown in the frozen dev order; no weights, ranks or candidate membership changed after seeing it. '
        'The reference’s one-shot re-score is ' + score(data['audit']['TM_COMBINED3']) + '.',
        '- The many structures share most of their predictors. Small paired differences inside the band do not identify a better entry; prefer lower packaging cost among tied models. '
        'The table is a list of combinations, not ten independent discoveries.',
        '- Before deployment, reproduce the selected model on test data, verify determinism and the contest contract, and package it separately. '
        'This experiment changes no submission predictor. For round two, relearn the priors and lag structure from permitted data and use fresh episode-held-out evaluation. '
        'The main remaining opportunity is an independently estimated win-probability component, with payout-size calibration checked separately.',
        '']
    (HERE / 'COMBOS_TOP10.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
