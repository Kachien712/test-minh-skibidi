import cv2
import torch
import numpy as np
import os

from torchvision import models, transforms
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ======================================================
# 1. DEEPLAB – ROAD SEGMENTATION
# ======================================================
deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_road_mask(frame):
    small = cv2.resize(frame, (512, 512))
    inp = preprocess(small).unsqueeze(0)
    with torch.no_grad():
        out = deeplab(inp)['out'][0]
    pred = out.argmax(0).byte().cpu().numpy()
    mask = (pred != 0).astype(np.uint8) * 255
    return cv2.resize(mask, (frame.shape[1], frame.shape[0]))

# ======================================================
# 2. BLACKWHITE POTHOLE DETECT (ĐÃ LOẠI TRỪ XE)
# ======================================================
def get_largest_connected_component(img_bin):
    contours, _ = cv2.findContours(
        img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return img_bin
    cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img_bin)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    return mask


def detect_and_draw_potholes(
    img_display,
    img_gray,
    road_mask,
    vehicle_mask,
    threshold_val
):
    # Vùng tối
    _, dark = cv2.threshold(
        img_gray, threshold_val, 255, cv2.THRESH_BINARY_INV
    )

    dark = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )

    # CHỈ GIỮ VÙNG ĐƯỜNG & KHÔNG PHẢI XE
    valid_area = cv2.bitwise_and(
        road_mask,
        cv2.bitwise_not(vehicle_mask)
    )

    potholes = cv2.bitwise_and(dark, dark, mask=valid_area)

    contours, _ = cv2.findContours(
        potholes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    count = 0
    result = img_display.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                result,
                "Pothole",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

    return potholes, result, count

# ======================================================
# 3. GHÉP 3 FRAME + COUNT
# ======================================================
def make_blackwhite_result(road_mask, pothole_binary, final_img, count):
    h, w = final_img.shape[:2]

    road_vis = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
    pot_vis = cv2.cvtColor(pothole_binary, cv2.COLOR_GRAY2BGR)

    road_vis = cv2.resize(road_vis, (w, h))
    pot_vis = cv2.resize(pot_vis, (w, h))

    cv2.putText(
        final_img,
        f"Potholes detected: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    return np.hstack([road_vis, pot_vis, final_img])

# ======================================================
# 4. YOLO + DEEPSORT
# ======================================================
yolo = YOLO("yolov8m.pt")
tracker = DeepSort(max_age=30)

# ======================================================
# 5. MAIN PIPELINE
# ======================================================
cap = cv2.VideoCapture("trafic.mp4") # Link video <==
if not cap.isOpened():
    print("❌ Không mở được video")
    exit()

os.makedirs("bw_results", exist_ok=True)

fps = 25
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("deep_result.avi", fourcc, fps, (W, H)) #Tên video đầu ra <==

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # ---------- DEEPLAB ----------
    road_mask_full = get_road_mask(frame)

    # ---------- VEHICLE MASK ----------
    vehicle_mask = np.zeros((H, W), dtype=np.uint8)

    # ---------- YOLO + DEEPSORT ----------
    results = yolo(frame)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if road_mask_full[cy, cx] == 0:
            continue

        detections.append((
            [x1, y1, x2 - x1, y2 - y1],
            float(box.conf[0]),
            int(box.cls[0])
        ))

    tracks = tracker.update_tracks(detections, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue

        x1, y1, x2, y2 = map(int, t.to_ltrb())

        # vẽ bbox xe
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {t.track_id}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        # tô mask xe
        cv2.rectangle(vehicle_mask, (x1, y1), (x2, y2), 255, -1)

    out.write(frame)

    # ---------- BLACKWHITE MỖI 10 FRAME ----------
    if frame_id % 10 == 0:
        roi_y = int(H / 3)
        roi = frame[roi_y:, :]
        vehicle_roi = vehicle_mask[roi_y:, :]

        blur = cv2.GaussianBlur(roi, (7, 7), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        otsu, _ = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        thresh = otsu * 0.85

        _, road_raw = cv2.threshold(
            gray, thresh, 255, cv2.THRESH_BINARY
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
        road_healed = cv2.morphologyEx(
            road_raw, cv2.MORPH_CLOSE, kernel
        )

        road_bw = get_largest_connected_component(road_healed)

        pothole_bin, final_img, count = detect_and_draw_potholes(
            roi,
            gray,
            road_bw,
            vehicle_roi,
            thresh
        )

        bw_img = make_blackwhite_result(
            road_bw,
            pothole_bin,
            final_img,
            count
        )

        cv2.imwrite(f"bw_results/frame_{frame_id}.jpg", bw_img)

    cv2.imshow("SYSTEM FINAL", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
