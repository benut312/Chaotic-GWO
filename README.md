# 🐺 Chaotic Grey Wolf Optimizer (CGWO) cho hàm Rastrigin

 **Chaotic Grey Wolf Optimizer (CGWO)** – một biến thể cải tiến của Grey Wolf Optimizer sử dụng **Logistic chaotic map** và **vector chaos theo từng chiều**, áp dụng cho **bài toán tối ưu hàm Rastrigin**.


---

## ✨ Đặc điểm chính

- Grey Wolf Optimizer (GWO) chuẩn
- Thay số ngẫu nhiên bằng **Logistic chaotic map**
- **Chaos dạng vector theo từng chiều** (không dùng scalar)
- Cập nhật alpha / beta / delta theo **top-3 mỗi vòng lặp**
- Tham số điều khiển `a` giảm tuyến tính từ `2 → 0`
- Tách riêng `chaos_seed` và `random_seed`
- Áp dụng cho bài toán Rastrigin đa cực trị

---

## 🎯 Hàm mục tiêu: Rastrigin

Hàm Rastrigin được định nghĩa như sau: f(x) = 10D + Σ(x_i² − 10cos(2πx_i)) 
- Miền tìm kiếm: `[-5.12, 5.12]`
- Cực tiểu toàn cục: `f(0, ..., 0) = 0`
- Hàm có nhiều cực tiểu cục bộ, phù hợp để đánh giá thuật toán metaheuristic

---

## ⚙️ Thông số mặc định trong code

| Tham số | Giá trị |
|------|------|
| Số chiều (D) | 10 |
| Số lượng sói | 30 |
| Số vòng lặp | 100 |
| Chaotic map | Logistic |
| Chaos seed | 0.7 |
| Random seed | 42 |

---

## ▶️ Cách chạy chương trình

### 1. Cài đặt thư viện cần thiết

```bash
pip install numpy matplotlib
---
 2. Chạy thuật toán
python chaotic_gwo.py

