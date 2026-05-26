import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gmm import GMM

def remove_background():
    image_path = os.path.join(os.path.dirname
    (__file__), '../data/cow.jpg')
    
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file tại {image_path}")
        return

    img = Image.open(image_path).convert('RGB')
    
    
    original_size = img.size
    img.thumbnail((200, 200)) # Giới hạn kích thước ảnh khoảng 200px
    img_np = np.array(img)
    
    h, w, c = img_np.shape
    X = img_np.reshape(-1, 3).astype(np.float64) / 255.0

    print(f"Đang xử lý ảnh kích thước: {h}x{w} ({h*w} pixels)...")
    k = 3 
    gmm = GMM(k=k, max_iters=50)
    gmm.fit(X)
    
    labels = gmm.predict(X)
    bg_cluster_idx = np.argmax(gmm.weights)
    


    print(f"Cụm được xác định là nền: {bg_cluster_idx} (Trọng số: {gmm.weights[bg_cluster_idx]:.2f})")
    foreground_pixels = X.copy()

    mask = (labels == bg_cluster_idx)
    foreground_pixels[mask] = [0, 0, 0] 
    result_img = (foreground_pixels.reshape(h, w, 3) * 255).astype(np.uint8)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Ảnh gốc")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(labels.reshape(h, w), cmap='viridis')
    plt.title("Bản đồ phân cụm màu")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(result_img)
    plt.title("Vật thể sau khi loại nền")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    remove_background()