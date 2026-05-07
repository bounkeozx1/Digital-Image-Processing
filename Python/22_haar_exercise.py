import cv2
import easyocr
import os
import re
import time
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, 'test(28.jpg)')

# EasyOCR: รองรับ อังกฤษ + ไทย (เพิ่ม 'lo' ถ้าต้องการลาว แต่ model อาจไม่มี)
LANGUAGES = ['en', 'th']

# ความมั่นใจขั้นต่ำของ OCR (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.3

# ข้ามทุก N frame เพื่อความเร็ว (1 = ทุก frame, 3 = ทุก 3 frame)
PROCESS_EVERY_N_FRAMES = 3

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
print("[INFO] กำลังโหลด EasyOCR model... (ครั้งแรกอาจช้า)")
reader = easyocr.Reader(LANGUAGES, gpu=False)  # ตั้ง gpu=True ถ้ามี CUDA

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] เปิดวิดีโอไม่ได้: {VIDEO_PATH}")
    exit(1)

# เก็บประวัติทะเบียนที่ detect ได้
plate_log: dict[str, dict] = defaultdict(lambda: {"count": 0, "last_seen": 0, "confidence": 0.0})
frame_count = 0
detected_plates: list[tuple] = []  # cache ผลล่าสุด

print("[INFO] เริ่ม tracking ทะเบียนรถ — กด 'q' เพื่อหยุด")
print("=" * 55)


# ─────────────────────────────────────────────
# HELPER: ดึงกรอบทะเบียนด้วย contour + geometry
# ─────────────────────────────────────────────
def find_plate_candidates(frame: cv2.typing.MatLike) -> list[tuple]:
    """
    คืน list ของ (x, y, w, h) ที่น่าจะเป็นป้ายทะเบียน
    ใช้ edge detection + contour filtering ตาม aspect ratio ของป้ายทะเบียน
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    candidates = []
    h_frame, w_frame = frame.shape[:2]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)

        if len(approx) >= 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h) if h != 0 else 0
            area = w * h
            area_ratio = area / (w_frame * h_frame)

            # ป้ายทะเบียนมาตรฐาน: ratio ~2:1 ถึง 5:1, ขนาดไม่เล็กหรือใหญ่เกิน
            if 1.5 < ratio < 6.0 and 0.001 < area_ratio < 0.15:
                candidates.append((x, y, w, h))

    # กรอง box ที่ซ้อนทับกัน
    return non_max_suppression(candidates)


def non_max_suppression(boxes: list[tuple], overlap_thresh: float = 0.5) -> list[tuple]:
    """กรอง bounding box ที่ซ้อนทับกันออก"""
    if not boxes:
        return []
    result = []
    for box in boxes:
        x1, y1, w1, h1 = box
        dominated = False
        for rx, ry, rw, rh in result:
            ix = max(x1, rx)
            iy = max(y1, ry)
            iw = min(x1+w1, rx+rw) - ix
            ih = min(y1+h1, ry+rh) - iy
            if iw > 0 and ih > 0:
                inter = iw * ih
                union = w1*h1 + rw*rh - inter
                if inter / union > overlap_thresh:
                    dominated = True
                    break
        if not dominated:
            result.append(box)
    return result


def clean_plate_text(text: str) -> str:
    """ทำความสะอาด OCR output — เอาเฉพาะตัวอักษร ตัวเลข และช่องว่าง"""
    text = text.strip().upper()
    # เอา character พิเศษที่ไม่ใช่ตัวอักษร/ตัวเลข/ช่องว่างออก
    text = re.sub(r'[^A-Z0-9ก-ฮ\s]', '', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def log_plate(text: str, confidence: float, timestamp: float):
    """บันทึกทะเบียนที่ detect ได้ลง dict + print terminal"""
    entry = plate_log[text]
    entry["count"] += 1
    entry["last_seen"] = timestamp
    if confidence > entry["confidence"]:
        entry["confidence"] = confidence

    # Print เฉพาะครั้งแรก หรือทุก 30 ครั้งที่เจอซ้ำ
    if entry["count"] == 1 or entry["count"] % 30 == 0:
        ts = time.strftime('%H:%M:%S', time.localtime(timestamp))
        print(f"[{ts}]  🚗 ทะเบียน: {text:<20} | ความมั่นใจ: {confidence*100:.1f}%  | พบแล้ว {entry['count']} ครั้ง")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1

        # ─── ประมวลผลเฉพาะทุก N frame ───
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            candidates = find_plate_candidates(frame)
            detected_plates = []

            for (x, y, w, h) in candidates:
                roi = frame[y:y+h, x:x+w]

                # Pre-process ROI ให้ OCR อ่านง่ายขึ้น
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.resize(roi_gray, (roi_gray.shape[1]*2, roi_gray.shape[0]*2))
                roi_gray = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                results = reader.readtext(roi_gray)

                for (_, text, conf) in results:
                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    cleaned = clean_plate_text(text)
                    if len(cleaned) < 2:  # ข้ามถ้าสั้นเกิน
                        continue
                    detected_plates.append((x, y, w, h, cleaned, conf))
                    log_plate(cleaned, conf, time.time())

        # ─── วาดกรอบและข้อความบน frame ───
        for (x, y, w, h, text, conf) in detected_plates:
            label = f"{text}  ({conf*100:.0f}%)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # พื้นหลังข้อความ
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - th - 12), (x + tw + 6, y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x + 3, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # ─── แสดง frame count มุมซ้ายบน ───
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('License Plate Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] หยุดโดยผู้ใช้")
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted")

finally:
    cap.release()
    cv2.destroyAllWindows()

    # ─── สรุปผลทั้งหมด ───
    print("\n" + "=" * 55)
    print("📋  สรุปทะเบียนที่ตรวจพบทั้งหมด:")
    print("=" * 55)
    if plate_log:
        sorted_plates = sorted(plate_log.items(), key=lambda x: x[1]["count"], reverse=True)
        for plate, info in sorted_plates:
            print(f"  {plate:<22} | พบ {info['count']:>4} ครั้ง | confidence max {info['confidence']*100:.1f}%")
    else:
        print("  ไม่พบทะเบียนในวิดีโอนี้")
    print("=" * 55)