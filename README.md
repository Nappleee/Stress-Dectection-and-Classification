# Nhận diện và phân loại stress

Dự án này là khóa luận về xác định và phân loại mức độ stress từ tín hiệu ECG. Trọng tâm là trích xuất đặc trưng HRV và sử dụng các mô hình học máy có giám sát để dự đoán mức độ stress từ dữ liệu sinh lý.

## Mục tiêu

- Trích xuất đặc trưng HRV từ tín hiệu ECG
- Xây dựng mô hình phân loại mức độ stress
- Đánh giá hiệu năng mô hình trên dữ liệu đã chuẩn bị

## Cấu trúc thư mục

- data/: dữ liệu và đặc trưng đã trích xuất
- reports/: báo cáo và kết quả đánh giá
- src/: mã nguồn xử lý dữ liệu và huấn luyện mô hình
- *.ipynb: notebook khám phá dữ liệu và pipeline

## Cài đặt (Windows)

1) Tạo môi trường ảo
   - python -m venv .venv
2) Kích hoạt môi trường
   - .venv\Scripts\Activate.ps1
3) Cài đặt thư viện
   - pip install -r requirements.txt

## Ghi chú

- Kho lưu trữ này phục vụ nghiên cứu học thuật và thử nghiệm.
- Nếu cần giao diện Jupyter đầy đủ, cài riêng:
  - pip install jupyterlab

## Bắt đầu nhanh

- Mở notebook trong VS Code và chọn đúng Python kernel.
- Dùng các script trong src/ để xử lý đặc trưng và huấn luyện mô hình.

## Thử chạy nhanh

- Chạy chương trình mẫu:
  - python test_flow.py
