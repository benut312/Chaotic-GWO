import random
import math
import matplotlib.pyplot as plt

# ================= SYSTEM PARAMETERS =================
M = 4          # number of APs
K = 2          # number of users
P_MAX = 1.0
SIGMA2 = 0.1
SIGMA_R2 = 0.1
GAMMA = 5.0

LAMBDA1 = 10.0
LAMBDA2 = 10.0

DIM = M * (K + 1)   # w_{m,k} + w_{m,s}

# ================= CHANNELS =================
random.seed(1)
h = [[random.uniform(0.5, 1.5) for _ in range(K)] for _ in range(M)]
g = [random.uniform(0.5, 1.5) for _ in range(M)]

# ================= FITNESS FUNCTION =================
def fitness(x):
    w_user = [[x[m*(K+1)+k] for k in range(K)] for m in range(M)]
    w_s = [x[m*(K+1)+K] for m in range(M)]

    # SINR computation
    sinrs = []
    for k in range(K):
        signal = sum(h[m][k]*w_user[m][k] for m in range(M))**2
        interf = 0.0
        for j in range(K):
            if j != k:
                interf += sum(h[m][k]*w_user[m][j] for m in range(M))**2
        sensing_int = sum(h[m][k]*w_s[m] for m in range(M))**2
        sinr = signal / (interf + sensing_int + SIGMA2)
        sinrs.append(sinr)

    min_sinr = min(sinrs)

    # Radar SNR
    snr_radar = (sum(g[m]*w_s[m] for m in range(M))**2) / SIGMA_R2

    # Power constraint penalty
    power_pen = 0.0
    for m in range(M):
        p = sum(w_user[m][k]**2 for k in range(K)) + w_s[m]**2
        power_pen += max(0.0, p - P_MAX)

    return (
        min_sinr
        - LAMBDA1 * max(0.0, GAMMA - snr_radar)
        - LAMBDA2 * power_pen
    )

# ================= GWO =================
def gwo():
    N_WOLVES = 30
    ITERS = 100
    LB, UB = -1.0, 1.0

    wolves = [[random.uniform(LB, UB) for _ in range(DIM)] for _ in range(N_WOLVES)]
    history = []

    alpha = beta = delta = None
    fa = fb = fd = -1e9

    for t in range(ITERS):
        for w in wolves:
            f = fitness(w)
            if f > fa:
                delta, fd = beta, fb
                beta, fb = alpha, fa
                alpha, fa = w[:], f
            elif f > fb:
                delta, fd = beta, fb
                beta, fb = w[:], f
            elif f > fd:
                delta, fd = w[:], f

        a = 2 - 2 * t / ITERS
        new_wolves = []

        for w in wolves:
            new = []
            for d in range(DIM):
                def update(best):
                    r1, r2 = random.random(), random.random()
                    A = 2*a*r1 - a
                    C = 2*r2
                    return best[d] - A * abs(C*best[d] - w[d])

                new.append((update(alpha) + update(beta) + update(delta)) / 3)

            new = [max(LB, min(UB, v)) for v in new]
            new_wolves.append(new)

        wolves = new_wolves
        history.append(fa)

    return history

# ================= RUN & PLOT =================
history = gwo()

plt.figure()
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("Fitness value")
plt.title("GWO Convergence for Cell-Free ISAC")
plt.grid(True)
plt.show()
