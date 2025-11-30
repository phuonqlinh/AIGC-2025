import cv2
import face_recognition

# --- Chọn camera ---
cap = cv2.VideoCapture(0)   # 0 = webcam

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        break

    # face_recognition dùng RGB → convert
    rgb_frame = frame[:, :, 2::-1]

    # --- 1) Tìm vị trí khuôn mặt ---
    face_locations = face_recognition.face_locations(rgb_frame)

    # --- 2) Tìm 68 keypoints ---
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    # --- Vẽ bounding box + keypoints ---
    for (top, right, bottom, left), landmarks in zip(face_locations, face_landmarks_list):
        
        # Vẽ bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Vẽ 68 điểm
        for feature_points in landmarks.values():
            for (x, y) in feature_points:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # --- Show frame ---
    cv2.imshow("Face + 68 Keypoints", frame)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()