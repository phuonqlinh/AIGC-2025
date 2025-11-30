import os
import cv2
from ultralytics import YOLO

# Tải mô hình YOLOv11
model = YOLO("model.pt")

# Ngưỡng confidence thấp để phát hiện nhiều đối tượng hơn
CONFIDENCE_THRESHOLD = 0.4

def detect_video(video_path: str):
    """Xử lý và hiển thị kết quả detection cho video"""
    if not os.path.isfile(video_path):
        print(f"Không tìm thấy video: {video_path}")
        return

    print(f"\nĐang xử lý video: {video_path}")
    print("Nhấn 'q' để thoát")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video!")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Thông tin video: {width}x{height}, {fps} FPS, {total_frames} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Đã hết video hoặc không thể đọc frame!")
            break

        results = model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        annotated_frame = results.plot()

        cv2.imshow("YOLO Detection - Video", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Thoát chương trình...")
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_webcam():
    """Xử lý và hiển thị kết quả detection cho webcam"""
    print("\nĐang khởi động webcam...")
    print("Nhấn 'q' để thoát")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam!")
        return

    # Thiết lập độ phân giải (tùy chọn)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Thông tin webcam: {width}x{height}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ webcam!")
            break

        results = model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        annotated_frame = results.plot()

        cv2.imshow("YOLO Detection - Webcam", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Thoát chương trình...")
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_image(image_path: str):
    """Xử lý và hiển thị kết quả detection cho ảnh"""
    if not os.path.isfile(image_path):
        print(f"Không tìm thấy ảnh: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Không thể đọc ảnh! Vui lòng kiểm tra đường dẫn.")
        return

    print(f"\nĐang xử lý ảnh: {image_path}")

    results = model(image, conf=CONFIDENCE_THRESHOLD)[0]
    annotated_image = results.plot()

    cv2.imshow("YOLO Detection - Image", annotated_image)
    print("Nhấn phím bất kỳ để đóng cửa sổ.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=" * 50)
    print("CHƯƠNG TRÌNH PHÁT HIỆN ĐỐI TƯỢNG YOLO")
    print("=" * 50)
    print("1. Sử dụng video (từ file)")
    print("2. Sử dụng webcam")
    print("3. Sử dụng ảnh tĩnh")
    print("=" * 50)
    
    choice = input("Chọn phương thức (1, 2 hoặc 3): ").strip()
    
    if choice == "1":
        video_path = input("Nhập đường dẫn video: ").strip()
        if video_path:
            detect_video(video_path)
        else:
            print("Bạn chưa nhập đường dẫn video!")
    elif choice == "2":
        detect_webcam()
    elif choice == "3":
        image_path = input("Nhập đường dẫn ảnh: ").strip()
        if image_path:
            detect_image(image_path)
        else:
            print("Bạn chưa nhập đường dẫn ảnh!")
    else:
        print("Lựa chọn không hợp lệ! Vui lòng chọn 1, 2 hoặc 3.")