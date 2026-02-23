# 📘 Hướng Dẫn Dataset ECG — Phân Loại Stress

## Tổng Quan

Dataset này được tạo ra để huấn luyện mô hình phân loại mức độ stress từ tín hiệu ECG.
Dữ liệu được chia thành **2 tầng (Two-Tier)**:

| Tầng | Mục Đích | File Output |
|------|----------|-------------|
| **Tier 1 – Raw ECG** | Minh họa, kiểm tra nhiễu | `tier1_raw_samples/*.csv` |
| **Tier 2 – HRV Features** | Huấn luyện mô hình | `tier2_features_24h/*.csv` |

---

## 🏷️ Bảng Mức Độ Stress (Level 0–5)

| Level | Mô Tả | HR (bpm) | SDNN (ms) |
|-------|--------|----------|-----------|
| 0 | Nghỉ ngơi / Ngủ sâu | ~65 | ~41 |
| 1 | Hoạt động nhẹ / Thư giãn | ~68 | ~37 |
| 2 | Làm việc bình thường | ~75 | ~33 |
| 3 | Stress vừa / Tập trung cao | ~85 | ~25 |
| 4 | Stress cao (deadline, áp lực) | ~98 | ~16 |
| 5 | Stress cực cao (khủng hoảng) | ~111 | ~10 |

> Giá trị tham chiếu cho nam 25 tuổi, tính theo hồi quy tuyến tính của nghiên cứu HRV người Châu Á.

---

##  Ba Kịch Bản 24 Giờ

###  Kịch Bản 1 — Normal Day (Ngày Bình Thường)

**Mục tiêu:** Mô phỏng người bình thường, stress thấp, nhịp sinh học ổn định.

| Thời Gian | Hoạt Động | Level |
|-----------|-----------|-------|
| 00:00–06:00 | Ngủ sâu | 0 |
| 06:00–08:00 | Thức dậy, chuẩn bị | 1 |
| 08:00–12:00 | Làm việc bình thường | 2 |
| 12:00–13:00 | Nghỉ trưa | 1 |
| 13:00–17:00 | Làm việc buổi chiều | 2 |
| 17:00–22:00 | Thư giãn tối | 1 |
| 22:00–24:00 | Chuẩn bị ngủ | 0 |

**Phân bố:** Level 0–1: 65% | Level 2: 35%

---

###  Kịch Bản 2 — Acute Stress Day (Stress Đột Biến)

**Mục tiêu:** Có giai đoạn stress cao ngắn hạn, có hồi phục sau đó — đúng với sinh lý.

| Thời Gian | Hoạt Động | Level |
|-----------|-----------|-------|
| 00:00–06:00 | Ngủ | 0 |
| 06:00–08:00 | Buổi sáng | 1 |
| 08:00–10:00 | Làm việc | 2 |
| 10:00–12:00 | **Họp khẩn / Deadline** | **4** |
| 12:00–13:00 | Nghỉ trưa (hồi phục) | 1 |
| 13:00–16:00 | Áp lực tăng dần | 3 |
| 16:00–17:00 | **Đỉnh stress (khủng hoảng)** | **5** |
| 17:00–18:00 | Hạ nhiệt | 2 |
| 18:00–22:00 | Thư giãn | 1 |
| 22:00–24:00 | Ngủ | 0 |

**Phân bố:** Level 0–1: 40% | Level 2–3: 35% | Level 4–5: 25%

>  **Quan trọng:** Level 5 chỉ xuất hiện 1 giờ (16–17h) để đảm bảo đúng sinh lý học. Sau peak phải có recovery.

---

### Kịch Bản 3 — Chronic Stress Day (Stress Mãn Tính)

**Mục tiêu:** Stress kéo dài suốt ngày, không có đủ thời gian hồi phục, ngủ kém.

| Thời Gian | Hoạt Động | Level |
|-----------|-----------|-------|
| 00:00–06:00 | Ngủ kém, trằn trọc | 1 |
| 06:00–08:00 | Lo lắng buổi sáng | 2 |
| 08:00–12:00 | Làm việc căng thẳng | 4 |
| 12:00–13:00 | Ăn trưa lo lắng | 3 |
| 13:00–18:00 | Stress kiệt sức | 5 |
| 18:00–22:00 | Không thể thư giãn | 4 |
| 22:00–24:00 | Mất ngủ / lo lắng | 3 |

**Phân bố:** Level 3–5: >70% | Level 0–2: <30%

---

## 🔬 Clean vs Noisy

Mỗi kịch bản được tạo ra **2 phiên bản**:

| Phiên Bản | Mô Tả | Jitter |
|-----------|--------|--------|
| **Clean** | Tín hiệu lý tưởng, không nhiễu | ±5% |
| **Noisy** | Thêm nhiễu: sensor noise, chuyển động | ±12% |

> **Lý do cần Noisy version:** Để kiểm tra mô hình có đủ robust khi đối mặt với dữ liệu thực tế không hoàn hảo.

---

## 🛠️ Cách Gen Data (Quy Trình)

### Tier 1 — Raw ECG Samples
1. Gọi `nk.ecg_simulate()` với tham số HR, HRV tính từ `LEVEL_TRANSFORMS`.
2. Nếu `Noisy`: cộng thêm baseline wander + powerline noise + white noise.
3. Lưu ra CSV: `Time (s)`, `Voltage (mV)`.
4. Mỗi file = 5 phút, 250 Hz = **75,000 dòng**.

### Tier 2 — HRV Feature Dataset (24h)
1. **Không lưu raw ECG** (tránh file hàng trăm MB).
2. Với từng window 30 giây (2880 windows/ngày):
   - Xác định `Stress_Level` theo giờ từ kịch bản.
   - **Tính thẳng các feature HRV từ công thức** (dựa trên hồi quy tuyến tính + LEVEL_TRANSFORMS).
   - Cộng jitter ngẫu nhiên để mô phỏng biến động sinh lý tự nhiên.
3. Lưu 1 dòng / window → **2880 dòng/file**.

**Công thức tính feature:**
```
HR  = HR_base × hr_mul × gender_factor × N(1, σ)
SDNN = SDNN_base × hr_std_mul × N(1, σ)
RMSSD ≈ SDNN × (1.2 − level × 0.12)
pNN50 ≈ pnn50_base × (1 − level × 0.18)
```

Trong đó:
- `HR_base = 65 bpm`, `SDNN_base = 41ms` (nam 25 tuổi, từ nghiên cứu người Châu Á).
- `σ = 5%` (Clean) hoặc `σ = 12%` (Noisy).

---

## 📂 Cấu Trúc File Output

```
data/raw/datagen/final_dataset/
├── tier1_raw_samples/
│   ├── Raw_Normal_Morning_Clean.csv       # Level 2, không nhiễu
│   ├── Raw_Normal_Morning_Noisy.csv       # Level 2, có nhiễu
│   ├── Raw_Acute_Peak_Stress_Clean.csv    # Level 5, không nhiễu
│   ├── Raw_Acute_Peak_Stress_Noisy.csv    # Level 5, có nhiễu
│   ├── Raw_Chronic_Exhaustion_Clean.csv   # Level 4, không nhiễu
│   └── Raw_Chronic_Exhaustion_Noisy.csv   # Level 4, có nhiễu
│
└── tier2_features_24h/
    ├── Dataset_Features_24h_Normal_Clean.csv
    ├── Dataset_Features_24h_Normal_Noisy.csv
    ├── Dataset_Features_24h_Acute_Stress_Clean.csv
    ├── Dataset_Features_24h_Acute_Stress_Noisy.csv
    ├── Dataset_Features_24h_Chronic_Stress_Clean.csv
    └── Dataset_Features_24h_Chronic_Stress_Noisy.csv
```

### Cột trong Tier 2 Feature CSV

| Cột | Ý Nghĩa |
|-----|---------|
| `Time_Sec` | Thời điểm (giây) trong ngày |
| `Hour` | Giờ trong ngày (0.0 – 24.0) |
| `Stress_Level` | Nhãn stress (0–5) |
| `Scenario` | Tên kịch bản |
| `Is_Noisy` | True nếu là Noisy version |
| `HR_Target` | HR mục tiêu (bpm) |
| `SDNN_Target` | SDNN mục tiêu (ms) |
| `HR_Extracted` | HR tính được (có jitter) |
| `SDNN_Extracted` | SDNN tính được (có jitter) |
| `RMSSD` | RMSSD tính được (ms) |
| `pNN50` | pNN50 tính được (%) |

---

## ✅ Kiểm Tra Chất Lượng (verify_script.py)

Script `verify_script.py` kiểm tra:

1. **Tier 1 Raw:** Duration, sampling rate, có NaN hay không.
2. **Tier 2 Features:**
   - **Monotonicity:** HR tăng theo Level? SDNN giảm theo Level?
   - **Circadian:** HR ban ngày > HR ban đêm?
   - **Acute Peak:** Có tồn tại Level 5 trong kịch bản Acute không?

---

## 📌 Lưu Ý Quan Trọng

- **Tier 2 không dùng ECG simulation** cho từng window vì quá chậm (~3 giờ cho 6 files). Feature được tính trực tiếp từ công thức → chạy trong vài giây.
- File Tier 2 có thể dùng trực tiếp làm **training dataset** cho model phân loại stress.
- Để thêm kịch bản mới, chỉnh sửa hàm `get_stress_level_at_hour()` trong `datagen_script.py`.
