import numpy as np

##Logistic chaotic map used to generate chaotic sequences.
def logistic_map(x: float) -> float:
    return 4 * x * (1 - x)

#This function generates a sequence of chaotic numbers using the logistic map.
def chaotic_sequence(length: int, seed: float = 0.7) -> np.ndarray:
     x = seed
    seq = []
    for _ in range(length):
        x = logistic_map(x)
        seq.append(x)
    return np.array(seq)


def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def chaotic_gwo(
    objective_func,
    dim: int,
    bounds: tuple,
    num_wolves: int = 30,
    max_iter: int = 100,
    seed: float = 0.7,
) -> tuple:

    # Extract lower and upper bounds as arrays
    lb = np.array(bounds[0], dtype=float)
    ub = np.array(bounds[1], dtype=float)

    # Initialize wolves randomly within bounds
    wolves = lb + (ub - lb) * np.random.rand(num_wolves, dim)
    fitness = np.apply_along_axis(objective_func, 1, wolves)

    # Identify the first three best wolves: alpha, beta, delta
    sorted_idx = np.argsort(fitness)
    alpha = wolves[sorted_idx[0]].copy()
    alpha_score = fitness[sorted_idx[0]]
    beta = wolves[sorted_idx[1]].copy()
    delta = wolves[sorted_idx[2]].copy()

    # Pre-generate a chaotic sequence for the entire run (four numbers per iteration)
    chaos = chaotic_sequence(max_iter * 4, seed)
    chaos_idx = 0

    convergence = []

    # Main optimization loop
    for t in range(max_iter):
        # Linearly decreasing parameter 'a' controls exploration vs exploitation
        a = 2 - 2 * t / float(max_iter - 1)

        for i in range(num_wolves):
            # Draw four successive chaotic numbers (r1, r2, r3, r4)
            r1 = chaos[chaos_idx % len(chaos)]
            r2 = chaos[(chaos_idx + 1) % len(chaos)]
            r3 = chaos[(chaos_idx + 2) % len(chaos)]
            r4 = chaos[(chaos_idx + 3) % len(chaos)]
            chaos_idx += 4

            # Coefficients for alpha wolf influence
            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            # Coefficients for beta wolf influence
            A2 = 2 * a * r2 - a
            C2 = 2 * r3

            # Coefficients for delta wolf influence
            A3 = 2 * a * r3 - a
            C3 = 2 * r4

            # Position update relative to alpha wolf
            D_alpha = np.abs(C1 * alpha - wolves[i])
            X1 = alpha - A1 * D_alpha

            # Position update relative to beta wolf
            D_beta = np.abs(C2 * beta - wolves[i])
            X2 = beta - A2 * D_beta

            # Position update relative to delta wolf
            D_delta = np.abs(C3 * delta - wolves[i])
            X3 = delta - A3 * D_delta

            # New position is the average of the three influences
            wolves[i] = (X1 + X2 + X3) / 3.0

            # Enforce bounds
            wolves[i] = np.clip(wolves[i], lb, ub)

        # Evaluate fitness after updating all wolves
        fitness = np.apply_along_axis(objective_func, 1, wolves)
        sorted_idx = np.argsort(fitness)

        # Update alpha, beta, delta positions if improved
        if fitness[sorted_idx[0]] < alpha_score:
            alpha_score = fitness[sorted_idx[0]]
            alpha = wolves[sorted_idx[0]].copy()

        beta = wolves[sorted_idx[1]].copy()
        delta = wolves[sorted_idx[2]].copy()

        # Store the best score for convergence analysis
        convergence.append(alpha_score)

    return alpha, alpha_score, np.array(convergence)


if __name__ == "__main__":
    # Example usage: optimize the Rastrigin function in 10 dimensions
    DIMENSION = 10
    BOUNDS = ([-5.12] * DIMENSION, [5.12] * DIMENSION)
    NUM_WOLVES = 25
    MAX_ITER = 80
    SEED = 0.7  # Seed for the chaotic map

    best_position, best_value, convergence_history = chaotic_gwo(
        objective_func=rastrigin,
        dim=DIMENSION,
        bounds=BOUNDS,
        num_wolves=NUM_WOLVES,
        max_iter=MAX_ITER,
        seed=SEED,
    )

    # Print results for demonstration purposes
    print("===== Chaotic GWO Demo =====")
    print(f"Objective function : Rastrigin")
    print(f"Number of wolves   : {NUM_WOLVES}")
    print(f"Dimensions         : {DIMENSION}")
    print(f"Iterations         : {MAX_ITER}")
    print("-----------------------------------")
    print("Best position found (alpha):")
    print(best_position)
    print(f"Best objective value        : {best_value}")