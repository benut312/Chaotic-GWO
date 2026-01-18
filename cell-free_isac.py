import random
import math
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) SYSTEM PARAMETERS
# =========================
M = 4          # number of APs
K = 2          # number of users

P_MAX = 1.0    # power constraint per AP
SIGMA2 = 0.1   # noise at users
SIGMA_R2 = 0.1 # radar noise
GAMMA = 5.0    # radar SNR threshold

LAMBDA1 = 10.0 # penalty for radar constraint violation
LAMBDA2 = 10.0 # penalty for power constraint violation

DIM = M * (K + 1)  # per AP: K user weights + 1 sensing weight

# =========================
# 2) CHANNELS (fixed)
# =========================
def make_channels(seed=1):
    rng = random.Random(seed)
    h = [[rng.uniform(0.5, 1.5) for _ in range(K)] for _ in range(M)]  # AP->user
    g = [rng.uniform(0.5, 1.5) for _ in range(M)]                     # AP->target
    return h, g

h, g = make_channels(seed=1)

# =========================
# 3) METRICS + FITNESS
# =========================
def unpack(x):
    # x: length DIM
    w_user = [[x[m*(K+1)+k] for k in range(K)] for m in range(M)]
    w_s = [x[m*(K+1)+K] for m in range(M)]
    return w_user, w_s

def compute_metrics(x):
    w_user, w_s = unpack(x)

    sinrs = []
    for k in range(K):
        signal = sum(h[m][k] * w_user[m][k] for m in range(M)) ** 2

        interf = 0.0
        for j in range(K):
            if j != k:
                interf += sum(h[m][k] * w_user[m][j] for m in range(M)) ** 2

        sensing_int = sum(h[m][k] * w_s[m] for m in range(M)) ** 2
        sinr = signal / (interf + sensing_int + SIGMA2)
        sinrs.append(sinr)

    min_sinr = min(sinrs)

    snr_radar = (sum(g[m] * w_s[m] for m in range(M)) ** 2) / SIGMA_R2

    power_viol = 0.0
    for m in range(M):
        p = sum(w_user[m][k]**2 for k in range(K)) + w_s[m]**2
        power_viol += max(0.0, p - P_MAX)

    radar_viol = max(0.0, GAMMA - snr_radar)

    return min_sinr, snr_radar, power_viol, radar_viol

def fitness(x):
    min_sinr, snr_radar, power_viol, radar_viol = compute_metrics(x)
    # maximize:
    return min_sinr - LAMBDA1 * radar_viol - LAMBDA2 * power_viol

# =========================
# 4) CHAOTIC SEQUENCE (Logistic map)
# =========================
class LogisticChaos:
    def __init__(self, x0=0.7):
        # x0 in (0,1)
        self.x = x0

    def next(self):
        self.x = 4.0 * self.x * (1.0 - self.x)
        # keep in (0,1)
        if self.x <= 0.0: self.x = 1e-6
        if self.x >= 1.0: self.x = 1.0 - 1e-6
        return self.x

# =========================
# 5) GWO / CGWO
# =========================
def run_gwo(iters=100, n_wolves=30, lb=-1.0, ub=1.0, seed=42, use_chaos=False, chaos_x0=0.7):
    rng = random.Random(seed)
    chaos = LogisticChaos(chaos_x0)

    def rand01():
        return chaos.next() if use_chaos else rng.random()

    # init wolves
    wolves = []
    for _ in range(n_wolves):
        x = [lb + (ub - lb) * rand01() for _ in range(DIM)]
        wolves.append(x)

    alpha = beta = delta = None
    fa = fb = fd = -1e18

    hist = []
    best_x = None

    for t in range(iters):
        # evaluate
        for w in wolves:
            f = fitness(w)
            if f > fa:
                delta, fd = beta, fb
                beta, fb = alpha, fa
                alpha, fa = w[:], f
                best_x = alpha[:]
            elif f > fb:
                delta, fd = beta, fb
                beta, fb = w[:], f
            elif f > fd:
                delta, fd = w[:], f

        a = 2.0 - 2.0 * (t / max(1, iters - 1))

        new_wolves = []
        for w in wolves:
            new = []
            for d in range(DIM):
                def update(best):
                    r1 = rand01()
                    r2 = rand01()
                    A = 2*a*r1 - a
                    C = 2*r2
                    D = abs(C*best[d] - w[d])
                    return best[d] - A * D

                x1 = update(alpha)
                x2 = update(beta)
                x3 = update(delta)
                xnew = (x1 + x2 + x3) / 3.0

                # bounds
                xnew = max(lb, min(ub, xnew))
                new.append(xnew)

            new_wolves.append(new)

        wolves = new_wolves
        hist.append(fa)

    return best_x, hist

# =========================
# 6) RUN + PLOT (GWO vs CGWO)
# =========================
if __name__ == "__main__":
    iters = 100
    n_wolves = 30

    best_gwo, hist_gwo = run_gwo(iters=iters, n_wolves=n_wolves, seed=1, use_chaos=False)
    best_cgwo, hist_cgwo = run_gwo(iters=iters, n_wolves=n_wolves, seed=1, use_chaos=True, chaos_x0=0.7)

    # Print metrics
    ms, sr, pv, rv = compute_metrics(best_gwo)
    print("GWO best fitness:", fitness(best_gwo))
    print("GWO  minSINR:", ms, " radarSNR:", sr, " powerViol:", pv, " radarViol:", rv)

    ms, sr, pv, rv = compute_metrics(best_cgwo)
    print("CGWO best fitness:", fitness(best_cgwo))
    print("CGWO minSINR:", ms, " radarSNR:", sr, " powerViol:", pv, " radarViol:", rv)

    # Plot
    plt.figure()
    plt.plot(hist_gwo, label="GWO")
    plt.plot(hist_cgwo, label="CGWO")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Convergence: GWO vs CGWO (Cell-Free ISAC)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("gwo_vs_cgwo.png", dpi=200)
    plt.show()
    print("Saved plot: gwo_vs_cgwo.png")
