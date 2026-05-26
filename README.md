
# DS102 - Lab Week 5: K-means and Gaussian Mixture Models (GMM)

Dự án này thực hiện các thuật toán học máy không giám sát (Unsupervised Learning) bao gồm K-means và GMM bằng thư viện Numpy để giải quyết các bài toán phân cụm và xử lý ảnh.

---

## Cấu trúc thư mục

* `assignments/`: Chứa mã nguồn thực thi cho 5 bài tập (3 K-means, 2 GMM).
* `data/`: Chứa dữ liệu đầu vào (ảnh `cow.jpg`).
* `results/`: Chứa các kết quả trực quan hóa sau khi chạy thuật toán.
* `src/`: Mã nguồn cốt lõi (Core implementation).
    * `kmeans.py`: Thuật toán K-means (EM method).
    * `gmm.py`: Thuật toán Gaussian Mixture Model (EM method).
    * `data_generator.py`: Tạo dữ liệu toy dataset theo phân phối chuẩn.
* `README.md`: Hướng dẫn và báo cáo kết quả.

---

## Hướng dẫn cài đặt và chạy

### 1. Cài đặt thư viện
```bash
pip install numpy matplotlib pillow

```

### 2. Hướng dẫn chạy
```bash
python assignments/kmeans_assign1.py, etc
```

### 3. Nhận xét
### a. K-means Assignment 1: Random Initialization

**Nhận xét:**  
Việc khởi tạo tâm cụm ngẫu nhiên (*randomly initializing centroids*) làm cho thuật toán nhạy cảm với điều kiện ban đầu.

**Ảnh hưởng:**  
Nếu các tâm cụm khởi tạo quá gần nhau hoặc rơi vào cùng một vùng dữ liệu, thuật toán có thể hội tụ về nghiệm tối ưu cục bộ (*local optima*) thay vì tối ưu toàn cục, dẫn đến kết quả phân cụm sai lệch.

---

### b. K-means Assignment 2: Cluster Sizes

**Nhận xét:**  
K-means có xu hướng chia không gian thành các vùng có diện tích tương đối bằng nhau (*Voronoi cells*).

**Ảnh hưởng:**  
Khi các cụm có kích thước khác biệt lớn (ví dụ: 1200 điểm so với 200 điểm), tâm cụm của cụm nhỏ sẽ bị kéo về phía cụm lớn hơn để giảm thiểu tổng bình phương khoảng cách. Kết quả là ranh giới phân chia thường lấn sang cụm lớn.

---

### c. K-means Assignment 3: Distribution $$\(N((3,6), \Sigma_2)\)$$

**Nhận xét:**  
K-means sử dụng khoảng cách Euclidean, vốn mặc định các cụm có dạng hình cầu (*spherical clusters*).

**Ảnh hưởng:**  
Với $\(\Sigma_2\)$ làm dữ liệu bị kéo dẹt (*non-isotropic*), K-means không thể bao phủ chính xác hình dạng elip. Các điểm ở vùng biên của cụm dẹt dễ bị gán sai nhãn sang các cụm hình cầu lân cận. Đây là lúc GMM thể hiện ưu thế vượt trội nhờ học được ma trận hiệp phương sai (*covariance matrix*).

---

### d. GMM Assignment 2: Background Filtering

**Nhiệm vụ:**  
Sử dụng GMM để tách vật thể ra khỏi nền trong ảnh `data/cow.jpg`.

**Phương pháp:**  
Mỗi pixel được coi là một điểm dữ liệu trong không gian màu RGB. GMM phân loại các màu sắc thành $\(k\)$ cụm. Cụm có trọng số $\(\pi\)$ lớn nhất (hoặc phổ biến nhất ở các góc ảnh) được xác định là nền và bị loại bỏ bằng cách đưa về màu đen hoặc trong suốt.

