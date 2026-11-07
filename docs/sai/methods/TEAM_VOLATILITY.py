"""TEAM_VOLATILITY: predeclared causal mixer score-memory/noise gate experiment.
Pinned source is 6a8098c; size modes are NEVER supplied. Gate observes only own
past outcomes, subtracts predicted Bernoulli noise from squared innovation,
and compares EWMA16/64 excess and signed innovation with variance normalization.
"""
import math
from _team_mixer_source import PairMixer, CLASSES

class VolatilityMixer(PairMixer):
    def __init__(self, score_half_life=None, adaptive=False, sensitivity=1.0, **kwargs):
        super().__init__(**kwargs)
        self.score_half_life = score_half_life
        self.adaptive = adaptive
        self.sensitivity = sensitivity
        self.base_alpha = self.alpha

    def walk(self, own, machine_type, stop, emit):
        """own[t] = payout class index or None. Calls emit(t, probs) for t in 0..stop before absorbing t.

        Win chance: mixture over lag-pair experts (decayed win table per pair cell).
        Payout size given a win: separate mixture over size experts keyed by
        () = machine-level, (a,) = one lag, (a, b) = a lag pair.
        """
        size_decay = 2 ** (-1.0 / self.size_half_life)
        sizes = CLASSES - 1
        size_tab = [dict() for _ in self.size_keys]      # cell -> [4 size counts]
        base_decay = 2 ** (-1.0 / self.half_lives[0])
        rate = [0.0, 0.0]
        machine_size = [0.0] * sizes
        # Win experts = (pair, mode, decay). Modes:
        #   full   - 4 cells (won/lost at lag a x won/lost at lag b)
        #   parity - 2 cells: did the two lags agree? (the simulator's rule is XOR/XNOR-like)
        #   sym    - 1 number: evidence for "win iff agree" vs "win iff differ", pooled over both cells
        experts, prior = [], []
        for pair, weight in zip(self.pairs, self.pair_prior):
            for mode in self.cell_mode:
                for half_life in self.half_lives:
                    experts.append((pair, mode, 2 ** (-1.0 / half_life)))
                    prior.append(weight)
        win_tab = [dict() for _ in experts]              # key -> [wins (or xnor evidence), n]
        logw = list(prior)
        size_logw = [0.0] * len(self.size_keys)
        type_size = self._size_prior(machine_type)
        modes = self.size_modes.get(machine_type)
        mode_llr = [math.log(max(b, 1e-3) / max(a, 1e-3)) for a, b in zip(*modes)] if modes else None
        a0, b0 = self.rate_prior

        def cell_of(t, lags, how="win"):
            values = []
            for lag in lags:
                v = own[t - lag] if t - lag >= 0 else None
                if v is None:
                    return None
                if how == "win":
                    values.append(v > 0)
                elif how == "big":
                    values.append(0 if v == 0 else 1 if v <= 2 else 2)  # lost / 1-2x / 5-20x
                else:
                    values.append(v)
            return tuple(values)

        fast_e = slow_e = fast_x = slow_x = fast_v = slow_v = 0.0
        gate = 0.0
        seen = 0
        df, ds = 2 ** (-1.0 / 16), 2 ** (-1.0 / 64)
        for t in range(stop + 1):
            base = (rate[0] + a0) / (rate[1] + b0)
            msum = math.fsum(machine_size)
            msize = [(c + self.k_machine * p) / (msum + self.k_machine) for c, p in zip(machine_size, type_size)]
            pair_cells = {pair: cell_of(t, pair) for pair in self.pairs}
            win_keys, win_preds = [], []
            for i, (pair, mode, _) in enumerate(experts):
                cell = pair_cells[pair]
                if cell is None:
                    win_keys.append(None)
                    win_preds.append(base)
                    continue
                agree = cell[0] == cell[1]
                if mode == "sym":
                    e, n = win_tab[i].get("sym", (0.0, 0.0))
                    q = (e + self.k_win * 0.5) / (n + self.k_win)
                    win_keys.append(("sym", agree))
                    win_preds.append(q if agree else 1.0 - q)
                    continue
                key = (agree,) if mode == "parity" else cell
                w, n = win_tab[i].get(key, (0.0, 0.0))
                win_keys.append(key)
                win_preds.append((w + self.k_win * base) / (n + self.k_win))
            size_cells = [cell_of(t, key, how) for key, how in zip(self.size_keys, self.size_cell)]
            size_preds = []
            for i, cell in enumerate(size_cells):
                sc = size_tab[i].get(cell) if cell is not None else None
                if modes is not None:
                    counts = sc if sc is not None else machine_size
                    llr = self.mode_temp * math.fsum(c * r for c, r in zip(counts, mode_llr))
                    big = 1.0 / (1.0 + math.exp(-max(min(llr, 50.0), -50.0)))
                    size_preds.append([(1.0 - big) * a + big * b for a, b in zip(*modes)])
                elif sc is None:
                    size_preds.append(msize)
                else:
                    ssum = math.fsum(sc)
                    size_preds.append([(c + self.k_size * m) / (ssum + self.k_size) for c, m in zip(sc, msize)])
            top = max(logw)
            weights = [math.exp(v - top) for v in logw]
            pw = math.fsum(w * p for w, p in zip(weights, win_preds)) / math.fsum(weights)
            top = max(size_logw)
            sweights = [math.exp(v - top) for v in size_logw]
            stotal = math.fsum(sweights)
            size = [math.fsum(w * sp[k] for w, sp in zip(sweights, size_preds)) / stotal for k in range(sizes)]
            emit(t, [1.0 - pw] + [pw * s for s in size])
            if t == stop:
                break
            value = own[t]
            if value is not None:
                won = 1.0 if value > 0 else 0.0
                # Forget only at observed outcomes. Gate was determined by earlier residuals.
                forget = self.score_half_life
                if self.adaptive:
                    forget = None if gate == 0 else 100.0 / max(gate, 1e-9)
                if forget is not None:
                    fade = 2 ** (-1.0 / forget)
                    logw = [v * fade for v in logw]
                self.alpha = self.base_alpha * gate if self.adaptive else self.base_alpha
                innovation = won - pw
                noise = pw * (1.0 - pw)
                excess = innovation * innovation - noise
                # Variance under a calibrated Bernoulli, with a floor for p near .5.
                variance = max(noise * (1.0 - 2.0 * pw) ** 2, 0.01)
                fast_e = df * fast_e + (1 - df) * innovation
                slow_e = ds * slow_e + (1 - ds) * innovation
                fast_x = df * fast_x + (1 - df) * excess
                slow_x = ds * slow_x + (1 - ds) * excess
                fast_v = df * fast_v + (1 - df) * variance
                slow_v = ds * slow_v + (1 - ds) * noise
                seen += 1
                z_excess = max(0.0, fast_x - slow_x) / math.sqrt(max(fast_v / 32, 1e-5))
                z_mean = abs(fast_e - slow_e) / math.sqrt(max(slow_v / 32, 1e-5))
                new_gate = min(1.0, max(0.0, max(z_excess, z_mean) - self.sensitivity)) if seen >= 32 else 0.0
                for i, p in enumerate(win_preds):
                    logw[i] -= self.eta * (p - won) ** 2
                if self.alpha > 0:
                    top = max(logw)
                    raw = [math.exp(v - top) for v in logw]
                    total = math.fsum(raw)
                    ptop = max(prior)
                    praw = [math.exp(v - ptop) for v in prior]
                    ptotal = math.fsum(praw)
                    logw = [math.log((1 - self.alpha) * r / total + self.alpha * q / ptotal)
                            for r, q in zip(raw, praw)]
                if value > 0:
                    for i, sp in enumerate(size_preds):
                        size_logw[i] -= self.size_eta * math.fsum(
                            (q - (1.0 if k == value - 1 else 0.0)) ** 2 for k, q in enumerate(sp))
            if value is not None:
                gate = new_gate
            if value is None and not self.decay_missing:
                continue
            for (_, _, decay), tab in zip(experts, win_tab):
                for c in tab.values():
                    c[0] *= decay
                    c[1] *= decay
            for tab in size_tab:
                for c in tab.values():
                    for k in range(sizes):
                        c[k] *= size_decay
            rate[0] *= base_decay
            rate[1] *= base_decay
            for k in range(sizes):
                machine_size[k] *= size_decay
            if value is None:
                continue
            won = 1.0 if value > 0 else 0.0
            rate[0] += won
            rate[1] += 1.0
            for i, key in enumerate(win_keys):
                if key is None:
                    continue
                if key[0] == "sym":
                    c = win_tab[i].setdefault("sym", [0.0, 0.0])
                    c[0] += 1.0 if (won == 1.0) == key[1] else 0.0
                else:
                    c = win_tab[i].setdefault(key, [0.0, 0.0])
                    c[0] += won
                c[1] += 1.0
            if value > 0:
                machine_size[value - 1] += 1.0
                for i, cell in enumerate(size_cells):
                    if cell is not None:
                        size_tab[i].setdefault(cell, [0.0] * sizes)[value - 1] += 1.0

