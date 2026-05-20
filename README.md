# Nhận diện và phân loại stress

Khóa luận tập trung vào việc tạo dữ liệu và trích xuất các đặc trưng HRV từ tín hiệu ECG và huấn luyên mô hình học máy để phân loại mức độ căng thẳng

## Mục tiêu
- Xây dựng một bộ dữ liệu mô phỏng từ tín hiệu ECG 
- Trích xuất đặc trưng HRV từ tín hiệu ECG
- Xây dựng mô hình phần loại mức độ stress
- Đánh giá kết quả

## Cau truc thu muc

- data/: dữ liệu đầu vào và đặc trưng đã được trích dẫn
  - raw_gen/: dữ liệu ECG được tạo từ công cụ gen data
  - concatenated/: dữ liệu được ghép từ các đoạn tín hiệu ECG
  - features/: đặc trưng của tín hiệu ECG
- notebooks/: Notebook xử lý và pipeline
  - concateFile.ipynb: ghép đoạn ECG và tạo feature 
  - extract.ipynb: trích suất các đặc trưng để kiểm tra file raw_gen được tạo từ công cụ gen data có khớp với mức độ căng thẳng được define. 
- src/: Nã nguồn xử lý
  - data_prep.py: chuan hoa dau vao va nhan
  - features.py: trích xuát đặc trưng của sổ, concateFile gọi sau khi ghép các file
  - custom_rf.py: mô hình Random Forest 
  - test_flow.py: demo huấn luyện 


## Cai dat (Windows)

1) Tạo môi trường ảo
   - python -m venv .venv
2) Kích hoạt môi trường
   - .venv\Scripts\Activate.ps1
3) Cài đặt thư viện
   - pip install -r requirements.txt

Neu can giao dien Jupyter day du:
- pip install jupyterlab

## Cach su dung nhanh

1) Tạo đặc trưng từ notebook
   - Mở notebooks/concateFile.ipynb trong VS Code
   - Chạy từ trên xuống dưới để tạo data/concatenated và data/features
2) Huấn luyện và đánh giá nhanh
   - python src/test_flow.py

