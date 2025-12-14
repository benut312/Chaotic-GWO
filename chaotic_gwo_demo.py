import numpy as np
import matplotlib.pyplot as plt


def logistic_map(x: float) -> float:
    """Bản đồ hỗn độn Logistic: x_next = 4*x*(1-x)"""
    return 4 * x * (1 - x)


def chaotic_sequence(length: int, seed: float = 0.7) -> np.ndarray:
    """Sinh chuỗi số chaotic từ logistic map"""
    x = seed
    seq = []
    for _ in range(length):
        x = logistic_map(x)
        seq.append(x)
    return np.array(seq)


def rastrigin(x: np.ndarray) -> float:
    """Hàm Rastrigin - Global minimum tại x=[0,...,0] với f(x)=0"""
    A = 10.0
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def chaotic_gwo(
    objective_func,
    dim: int,
    bounds: tuple,
    num_wolves: int = 30,
    max_iter: int = 100,
    chaos_seed: float = 0.7,
    random_seed: int = 42,
    verbose: bool = True
) -> tuple:
    """
    Grey Wolf Optimizer với chaotic map (chuẩn).
    
    Improvements:
    - Vector chaos cho từng chiều (không dùng scalar broadcast)
    - Cập nhật alpha/beta/delta đúng chuẩn (top-3 mỗi iteration)
    - Reproducible với random_seed
    - Công thức a giảm về 0 ở iteration cuối
    
    Returns: (best_position, best_score, convergence_history)
    """
    
    # Validation
    assert dim > 0 and num_wolves >= 3, "dim > 0 và num_wolves >= 3"
    assert 0 < chaos_seed < 1, "chaos_seed phải trong (0, 1)"
    
    lb = np.array(bounds[0], dtype=float)
    ub = np.array(bounds[1], dtype=float)
    
    # Random generator với seed để reproducible
    rng = np.random.default_rng(random_seed)
    
    # Khởi tạo đàn sói ngẫu nhiên
    wolves = lb + (ub - lb) * rng.random((num_wolves, dim))
    fitness = np.apply_along_axis(objective_func, 1, wolves)
    
    # Pre-generate chaos sequence theo shape (max_iter, num_wolves, dim, 6)
    # Mỗi wolf mỗi iteration cần 6 vector (r1, r2 cho alpha; r3, r4 cho beta; r5, r6 cho delta)
    total_chaos_needed = max_iter * num_wolves * dim * 6
    chaos_flat = chaotic_sequence(total_chaos_needed, chaos_seed)
    chaos = chaos_flat.reshape(max_iter, num_wolves, dim, 6)
    
    convergence = []
    
    if verbose:
        print("="*50)
        print("Bắt đầu tối ưu hóa với Chaotic GWO")
        print("="*50)
    
    # Main loop
    for t in range(max_iter):
        # Tham số 'a' giảm tuyến tính từ 2→0 (về đúng 0 ở iteration cuối)
        a = 2 - 2 * t / (max_iter - 1)
        
        # Lấy top-3 tốt nhất hiện tại làm alpha, beta, delta (chuẩn GWO)
        sorted_idx = np.argsort(fitness)
        alpha = wolves[sorted_idx[0]].copy()
        alpha_score = fitness[sorted_idx[0]]
        beta = wolves[sorted_idx[1]].copy()
        delta = wolves[sorted_idx[2]].copy()
        
        for i in range(num_wolves):
            # Lấy chaos cho wolf thứ i, iteration t: shape (dim, 6)
            chaos_i = chaos[t, i]
            
            # r1, r2 là vector theo dim (không phải scalar)
            r1 = chaos_i[:, 0]  # shape (dim,)
            r2 = chaos_i[:, 1]
            r3 = chaos_i[:, 2]
            r4 = chaos_i[:, 3]
            r5 = chaos_i[:, 4]
            r6 = chaos_i[:, 5]
            
            # Hệ số từ alpha (vector theo dim)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            
            # Hệ số từ beta
            A2 = 2 * a * r3 - a
            C2 = 2 * r4
            
            # Hệ số từ delta
            A3 = 2 * a * r5 - a
            C3 = 2 * r6
            
            # Cập nhật vị trí theo công thức GWO (vector operations)
            D_alpha = np.abs(C1 * alpha - wolves[i])
            X1 = alpha - A1 * D_alpha
            
            D_beta = np.abs(C2 * beta - wolves[i])
            X2 = beta - A2 * D_beta
            
            D_delta = np.abs(C3 * delta - wolves[i])
            X3 = delta - A3 * D_delta
            
            wolves[i] = (X1 + X2 + X3) / 3.0
            wolves[i] = np.clip(wolves[i], lb, ub)
        
        # Đánh giá fitness
        fitness = np.apply_along_axis(objective_func, 1, wolves)
        
        convergence.append(alpha_score)
        
        if verbose and (t + 1) % 10 == 0:
            print(f"Iteration {t+1:3d}/{max_iter} | Best fitness: {alpha_score:.6f} | a: {a:.4f}")
    
    # Lấy nghiệm tốt nhất cuối cùng
    sorted_idx = np.argsort(fitness)
    final_best = wolves[sorted_idx[0]].copy()
    final_score = fitness[sorted_idx[0]]
    
    if verbose:
        print("="*50)
        print("Hoàn thành tối ưu hóa!")
        print("="*50)
    
    return final_best, final_score, np.array(convergence)


def plot_convergence(convergence_history, title="Đường Cong Hội Tụ - Chaotic GWO"):
    """Vẽ đồ thị hội tụ"""
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_history, linewidth=2, color='#2E86AB')
    plt.xlabel('Vòng Lặp', fontsize=12)
    plt.ylabel('Giá Trị Fitness Tốt Nhất (Alpha)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" "*15 + "CHAOTIC GWO DEMONSTRATION")
    print("="*60 + "\n")
    
    DIMENSION = 10
    BOUNDS = ([-5.12] * DIMENSION, [5.12] * DIMENSION)
    NUM_WOLVES = 30
    MAX_ITER = 100
    CHAOS_SEED = 0.7
    RANDOM_SEED = 42
    
    best_position, best_value, convergence_history = chaotic_gwo(
        objective_func=rastrigin,
        dim=DIMENSION,
        bounds=BOUNDS,
        num_wolves=NUM_WOLVES,
        max_iter=MAX_ITER,
        chaos_seed=CHAOS_SEED,
        random_seed=RANDOM_SEED,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("KẾT QUẢ TỐI ƯU HÓA")
    print("="*60)
    print(f"Hàm mục tiêu      : Rastrigin")
    print(f"Số lượng sói      : {NUM_WOLVES}")
    print(f"Số chiều          : {DIMENSION}")
    print(f"Số iterations     : {MAX_ITER}")
    print(f"Chaotic map seed  : {CHAOS_SEED}")
    print(f"Random seed       : {RANDOM_SEED}")
    print("-"*60)
    print("Vị trí tốt nhất tìm được (alpha):")
    print(best_position)
    print(f"\nGiá trị hàm mục tiêu: {best_value:.8f}")
    print(f"Sai số so với tối ưu toàn cục (0): {abs(best_value):.8f}")
    print("="*60 + "\n")
    
    plot_convergence(convergence_history)