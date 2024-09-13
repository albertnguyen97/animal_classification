import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
import warnings
from animals_models import CNN  # Đảm bảo file animals_models.py có định nghĩa mô hình CNN
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Animal classifier")
    parser.add_argument("-s", "--size", type=int, default=224, help="Kích thước ảnh đầu vào")
    parser.add_argument("-i", "--image_path", type=str, default="test_images/dog.jpg",
                        help="Đường dẫn đến ảnh cần dự đoán")
    parser.add_argument("-c", "--checkpoint_path", type=str, default="trained_models/best.pt",
                        help="Đường dẫn đến mô hình đã huấn luyện")
    args = parser.parse_args()
    return args


def test(args):
    categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=len(categories)).to(device)
    # Kiểm tra và tải mô hình từ checkpoint
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    else:
        print("A checkpoint must be provided")
        exit(0)

    # Kiểm tra xem ảnh có được cung cấp hay không
    if not args.image_path or not os.path.isfile(args.image_path):
        print("An image must be provided")
        exit(0)

    # Đọc và xử lý ảnh
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB
    image = cv2.resize(image, (args.size, args.size))  # Thay đổi kích thước ảnh
    image = np.transpose(image, (2, 0, 1))[None, :, :, :]  # Định dạng ảnh thành (1, C, H, W)
    image = image / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    image = torch.from_numpy(image).to(device).float()  # Chuyển đổi ảnh thành tensor PyTorch

    softmax = nn.Softmax(dim=1)  # Định nghĩa Softmax với dim=1
    with torch.no_grad():
        prediction = model(image)  # Dự đoán với mô hình đã tải
    probs = softmax(prediction)  # Tính xác suất dự đoán
    max_value, max_index = torch.max(probs, dim=1)  # Tìm giá trị và chỉ số dự đoán cao nhất

    # In ra kết quả dự đoán
    print("This image is about {} with probability of {:.2f}%".format(categories[max_index.item()],
                                                                      max_value[0].item() * 100))

    # Vẽ biểu đồ thanh hiển thị xác suất
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(categories, probs[0].cpu().numpy())
    ax.set_xlabel("Animal")
    ax.set_ylabel("Probability")
    ax.set_title(f"Prediction: {categories[max_index.item()]}")
    plt.savefig("animal_prediction.png")
    plt.show()


if __name__ == '__main__':
    args = get_args()
    test(args)
