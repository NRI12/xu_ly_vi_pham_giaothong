
# Hệ Thống Xử Lý Vi Phạm Giao Thông Tự Động

Hệ thống này được thiết kế để phát hiện và xử lý các vi phạm giao thông từ dữ liệu camera, bao gồm việc không đội mũ bảo hiểm, đi sai làn đường, vượt đèn đỏ, và các lỗi vi phạm khác.

## Tính Năng

- **Phát hiện phương tiện**: Sử dụng các mô hình như YOLO và Faster R-CNN để nhận diện và theo dõi các phương tiện.
- **Phát hiện mũ bảo hiểm**: Nhận diện việc đeo hoặc không đeo mũ bảo hiểm của người lái xe.
- **Phát hiện vi phạm đèn đỏ và làn đường**: Xác định các hành vi vượt đèn đỏ và đi sai làn đường.
- **Xử lý vi phạm**: Tự động tạo báo cáo và gửi thông báo vi phạm qua email.
- **Thống kê**: Tổng hợp và hiển thị số liệu thống kê về các vi phạm.

## Cài Đặt

Để cài đặt hệ thống, vui lòng tham khảo chi tiết trong tài liệu hướng dẫn đính kèm.

### Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- Các thư viện cần thiết: FastAPI, OpenCV, SQLite, Jinja2, và các thư viện khác được liệt kê trong `requirements.txt`

### Hướng Dẫn Cài Đặt

1. Clone repository:

    ```bash
    git clone https://github.com/your-repo/traffic-violation-detection.git
    cd traffic-violation-detection
    ```

2. Tạo môi trường ảo và cài đặt các gói cần thiết:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows sử dụng venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. Chạy ứng dụng:

    ```bash
    uvicorn main:app --reload
    ```

### Cài Đặt CUDA

Để sử dụng GPU cho việc tăng tốc xử lý, bạn cần cài đặt CUDA:

1. Tải và cài đặt [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) phù hợp với hệ thống của bạn.
2. Cài đặt các thư viện cần thiết cho PyTorch:

    ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    ```

## Sử Dụng

Hướng dẫn chi tiết và các ví dụ cụ thể về cách chạy hệ thống được mô tả trong tài liệu hướng dẫn sử dụng.

### Chạy Hệ Thống

1. Mở trình duyệt và truy cập vào `http://127.0.0.1:8000`.
2. Chọn video từ danh sách để bắt đầu phát hiện vi phạm.
3. Cấu hình các tham số phát hiện và bắt đầu quá trình giám sát.

## Đóng Góp

Chúng tôi hoan nghênh mọi đóng góp cho dự án này. Để đóng góp, vui lòng thực hiện các bước sau:

1. Fork repository.
2. Tạo một nhánh mới: `git checkout -b feature/your-feature`
3. Commit các thay đổi của bạn: `git commit -m 'Add some feature'`
4. Push lên nhánh: `git push origin feature/your-feature`
5. Tạo một pull request mới.

---

Nếu bạn có bất kỳ câu hỏi nào, vui lòng liên hệ với chúng tôi qua email tại [ctv55345@gmail.com](mailto:ctv55345@gmail.com). Chúng tôi sẽ rất vui lòng hỗ trợ bạn.
