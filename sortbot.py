import cv2
import numpy as np
import time
import math
import random
import os
import csv
from datetime import datetime

from ultralytics import YOLO

EVENT_LOG_PATH = "dashboard/data/events.csv"


def log_event(category, confidence=1.0, source="sortbot-01", note="auto-yolo"):
    os.makedirs(os.path.dirname(EVENT_LOG_PATH), exist_ok=True)

    header_needed = (
        not os.path.exists(EVENT_LOG_PATH) or os.stat(EVENT_LOG_PATH).st_size == 0
    )

    with open(EVENT_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if header_needed:
            writer.writerow(["ts", "category", "confidence", "source", "note"])

        writer.writerow(
            [datetime.utcnow().isoformat(), category, confidence, source, note]
        )


# =================== PARAMETERS ===================

FPS_TARGET = 30

categories = ["Plastic", "Burnable", "Cans", "Bottles", "Others"]

comp_colors = {
    "Plastic": (230, 120, 90),
    "Burnable": (60, 160, 255),
    "Cans": (200, 200, 220),
    "Bottles": (100, 220, 110),
    "Others": (140, 140, 160),
}

comp_dark = {
    "Plastic": (170, 80, 60),
    "Burnable": (40, 120, 200),
    "Cans": (160, 160, 180),
    "Bottles": (70, 170, 70),
    "Others": (100, 100, 120),
}

BIN_RADIUS = 200

TIMINGS = {
    "detect_delay": 0.6,
    "rotate_time": 0.7,
    "lid_open_time": 0.35,
    "fall_time": 0.55,
    "lid_close_time": 0.35,
    "particle_time": 0.9,
}

# Better model than yolov8n.pt.
# First run will download this automatically.
YOLO_MODEL_PATH = "yolov8s.pt"

# Lower threshold helps detect small objects like bottles.
YOLO_CONF_THRESHOLD = 0.20
SORT_TRIGGER_CONFIDENCE = 0.20

YOLO_INTERVAL = 0.25
DETECTION_COOLDOWN = 1.8


# =================== YOLO CATEGORY MAPPING ===================


def map_yolo_class_to_sortbot(yolo_class_name):
    """
    Maps YOLO COCO object names to SortBot waste categories.

    Pretrained YOLO is not waste-specific, so this mapping is mainly for demo.
    For final future enhancement, train YOLO on waste classes directly:
    Plastic, Burnable, Cans, Bottles, Others.
    """

    name = yolo_class_name.lower().strip()

    if name in ["bottle", "wine glass"]:
        return "Bottles"

    if name in ["cup"]:
        return "Cans"

    if name in [
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "book",
    ]:
        return "Burnable"

    if name in ["backpack", "handbag", "suitcase"]:
        return "Others"

    return "Others"


# =================== EASING FUNCTIONS ===================


def ease_in_out_cubic(t):
    if t < 0.5:
        return 4 * t * t * t

    f = (2 * t) - 2
    return 0.5 * f * f * f + 1


def ease_bounce_out(t):
    if t <= 0:
        return 0

    if t >= 1:
        return 1

    if t < 1 / 2.75:
        return 7.5625 * t * t

    elif t < 2 / 2.75:
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75

    elif t < 2.5 / 2.75:
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375

    t -= 2.625 / 2.75
    return 7.5625 * t * t + 0.984375


def ease_elastic_out(t):
    if t == 0 or t == 1:
        return t

    return (2 ** (-10 * t)) * math.sin((t - 0.075) * (2 * math.pi) / 0.3) + 1


# =================== PARTICLE EFFECT ===================


class Particle:
    def __init__(self, x, y, color):
        self.x = x + random.uniform(-6, 6)
        self.y = y + random.uniform(-6, 6)

        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.5, 5)

        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - random.uniform(2, 5)

        self.life = TIMINGS["particle_time"]
        self.max_life = self.life

        self.color = color
        self.size = random.randint(2, 5)

    def update(self, dt):
        self.vy += 9.8 * dt
        self.x += self.vx * dt * 60
        self.y += self.vy * dt * 60
        self.life -= dt


# =================== DUSTBIN STATE MACHINE ===================

IDLE, DETECTED, ROTATING, OPENING, FALLING, CLOSING = range(6)


class CircularDustbin:
    def __init__(self, n=5):
        self.n = n
        self.state = IDLE

        self.angle = 0
        self.target_angle = 0

        self.start = 0

        self.hole_x = 0
        self.hole_y = 0

        self.trash_x = 0
        self.trash_y = -999

        self.target_idx = 0
        self.lid = 0

        self.last_trigger = 0
        self.particles = []

    def detect(self, idx, now, conf):
        if self.state == IDLE and now - self.last_trigger > DETECTION_COOLDOWN:
            self.target_idx = idx

            sector_center = (idx + 0.5) * (360 / self.n)
            self.target_angle = ((-90 - sector_center + 180) % 360) - 180

            self.state = DETECTED
            self.start = now
            self.last_trigger = now

    def update(self, now, cx, cy, r):
        dt = 1 / FPS_TARGET

        if self.state == DETECTED:
            if now - self.start > TIMINGS["detect_delay"]:
                self.state = ROTATING
                self.start = now

        elif self.state == ROTATING:
            t = min(1, (now - self.start) / TIMINGS["rotate_time"])

            diff = ((self.target_angle - self.angle + 180) % 360) - 180
            self.angle += diff * ease_in_out_cubic(t)

            if t >= 1:
                self.angle = self.target_angle
                self.state = OPENING
                self.start = now

        elif self.state == OPENING:
            t = min(1, (now - self.start) / TIMINGS["lid_open_time"])
            self.lid = ease_elastic_out(t)

            if t >= 1:
                angle = math.radians(
                    (self.target_idx + 0.5) * (360 / self.n) + self.angle
                )

                self.hole_x = int(cx + (r - 15) * math.cos(angle))
                self.hole_y = int(cy + (r - 15) * math.sin(angle))

                self.trash_x = self.hole_x
                self.trash_y = cy - r - 60

                self.state = FALLING
                self.start = now

        elif self.state == FALLING:
            t = min(1, (now - self.start) / TIMINGS["fall_time"])

            top = cy - r - 60
            bottom = cy

            self.trash_y = int(top + (bottom - top) * ease_bounce_out(t))

            if t >= 1:
                color = comp_colors[categories[self.target_idx]]

                for _ in range(18):
                    self.particles.append(Particle(self.hole_x, self.trash_y, color))

                self.state = CLOSING
                self.start = now

        elif self.state == CLOSING:
            t = min(1, (now - self.start) / TIMINGS["lid_close_time"])
            self.lid = (1 - ease_in_out_cubic(t)) * 0.05

            if t >= 1:
                self.state = IDLE
                self.trash_y = -999

        alive_particles = []

        for particle in self.particles:
            particle.update(dt)

            if particle.life > 0:
                alive_particles.append(particle)

        self.particles = alive_particles


# =================== UI DRAWING ===================


def draw_modern_header(frame):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 90), (15, 15, 20), -1)
    cv2.rectangle(frame, (0, 88), (w, 90), (0, 255, 200), 2)

    title = "SMART WASTE SORTING SYSTEM"
    title_font = cv2.FONT_HERSHEY_DUPLEX
    text_size = cv2.getTextSize(title, title_font, 1.2, 2)[0]
    title_x = (w - text_size[0]) // 2

    cv2.putText(
        frame,
        title,
        (title_x, 45),
        title_font,
        1.2,
        (0, 255, 200),
        2,
        cv2.LINE_AA,
    )

    subtitle = "YOLOv8 Object Detection + Automated Sorting"

    cv2.putText(
        frame,
        subtitle,
        (title_x + 20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 255, 200),
        1,
        cv2.LINE_AA,
    )


def draw_detection_panel(
    frame, yolo_name, sortbot_label, conf, state_names, current_state
):
    panel_x, panel_y = 30, 120
    panel_w, panel_h = 410, 420

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (20, 20, 25),
        -1,
    )

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    border_color = (0, 255, 200)

    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        border_color,
        2,
    )

    cv2.putText(
        frame,
        "YOLO DETECTION",
        (panel_x + 20, panel_y + 35),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        border_color,
        2,
        cv2.LINE_AA,
    )

    cv2.line(
        frame,
        (panel_x + 20, panel_y + 45),
        (panel_x + panel_w - 20, panel_y + 45),
        border_color,
        1,
    )

    result_y = panel_y + 90

    cv2.putText(
        frame,
        "YOLO OBJECT:",
        (panel_x + 30, result_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        yolo_name.upper() if yolo_name else "WAITING...",
        (panel_x + 30, result_y + 35),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (0, 255, 200),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "SORTBOT CATEGORY:",
        (panel_x + 30, result_y + 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    cat_color = comp_colors.get(sortbot_label, (0, 255, 200))

    cv2.rectangle(
        frame,
        (panel_x + 25, result_y + 105),
        (panel_x + panel_w - 25, result_y + 155),
        (30, 30, 35),
        -1,
    )

    cv2.rectangle(
        frame,
        (panel_x + 25, result_y + 105),
        (panel_x + panel_w - 25, result_y + 155),
        cat_color,
        2,
    )

    cv2.putText(
        frame,
        f"→ {sortbot_label.upper() if sortbot_label else 'NONE'}",
        (panel_x + 40, result_y + 140),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        cat_color,
        2,
        cv2.LINE_AA,
    )

    conf_y = result_y + 190

    cv2.putText(
        frame,
        "CONFIDENCE",
        (panel_x + 30, conf_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    bar_x = panel_x + 30
    bar_y = conf_y + 15
    bar_w = panel_w - 80
    bar_h = 22

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        (40, 40, 45),
        -1,
    )
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        border_color,
        1,
    )

    if conf > 0:
        fill_w = int(bar_w * conf)

        color = (
            (0, 255, 100)
            if conf > 0.7
            else (255, 200, 0) if conf > 0.5 else (255, 100, 100)
        )

        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + fill_w, bar_y + bar_h),
            color,
            -1,
        )

        cv2.putText(
            frame,
            f"{conf * 100:.1f}%",
            (bar_x + bar_w + 10, bar_y + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    state_y = conf_y + 80

    cv2.putText(
        frame,
        "STATUS",
        (panel_x + 30, state_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    state_color = (0, 255, 100) if current_state == 0 else (255, 200, 0)

    cv2.circle(frame, (panel_x + 120, state_y - 5), 6, state_color, -1)

    cv2.putText(
        frame,
        state_names[current_state],
        (panel_x + 135, state_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        state_color,
        2,
        cv2.LINE_AA,
    )


def sector_polygon(cx, cy, r, a1, a2):
    points = [(cx, cy)]
    step = 4

    if a2 < a1:
        a2 += 360

    for angle in range(int(a1), int(a2) + 1, step):
        rad = math.radians(angle)
        points.append((int(cx + r * math.cos(rad)), int(cy + r * math.sin(rad))))

    return np.array(points, np.int32)


def draw_bin(frame, dustbin, cx, cy, r):
    cv2.circle(frame, (cx, cy), r + 18, (25, 25, 30), -1)
    cv2.circle(frame, (cx, cy), r + 15, (0, 255, 200), 2)

    angle_step = 360 / dustbin.n

    for i, cat in enumerate(categories):
        a1 = i * angle_step + dustbin.angle
        a2 = (i + 1) * angle_step + dustbin.angle

        points = sector_polygon(cx, cy, r, a1, a2)

        cv2.fillPoly(frame, [points], comp_dark[cat])

        inner = sector_polygon(cx, cy, int(r * 0.75), a1, a2)

        mixed_color = tuple(
            int(comp_dark[cat][channel] * 0.6 + comp_colors[cat][channel] * 0.4)
            for channel in range(3)
        )

        cv2.fillPoly(frame, [inner], mixed_color)
        cv2.polylines(frame, [points], True, (20, 20, 20), 2)

        if i == dustbin.target_idx and dustbin.state != IDLE:
            cv2.polylines(frame, [points], True, (0, 255, 200), 2)


def draw_lid(frame, dustbin, cx, cy, r):
    if dustbin.state not in (OPENING, FALLING, CLOSING):
        return

    overlay = frame.copy()

    cv2.circle(overlay, (cx, cy), r + 18, (220, 220, 230), -1)
    cv2.circle(overlay, (cx, cy), r - 10, (0, 0, 0), -1)

    mask = overlay.sum(axis=2) > 0

    frame[mask] = frame[mask] * 0.25 + overlay[mask] * 0.75


def draw_trash(frame, dustbin):
    if dustbin.trash_y < -900:
        return

    x = int(dustbin.trash_x)
    y = int(dustbin.trash_y)

    cat = categories[dustbin.target_idx]
    color = comp_colors[cat]

    cv2.circle(frame, (x, y), 18, color, -1)
    cv2.circle(frame, (x, y), 15, (255, 255, 255), 2)
    cv2.circle(frame, (x, y), 12, color, -1)


def draw_particles(frame, dustbin):
    for particle in dustbin.particles:
        alpha = max(0, min(1, particle.life / particle.max_life))

        color = tuple(int(particle.color[i] * alpha) for i in range(3))
        size = max(1, int(particle.size * alpha))

        cv2.circle(frame, (int(particle.x), int(particle.y)), size, color, -1)

        if alpha > 0.7:
            cv2.circle(
                frame,
                (int(particle.x), int(particle.y)),
                size + 2,
                (255, 255, 255),
                1,
            )


def draw_labels(frame, dustbin, cx, cy, r):
    font = cv2.FONT_HERSHEY_DUPLEX
    step = 360 / dustbin.n

    for i, cat in enumerate(categories):
        angle = (i + 0.5) * step + dustbin.angle
        rad = math.radians(angle)

        label_x = int(cx + (r + 55) * math.cos(rad))
        label_y = int(cy + (r + 55) * math.sin(rad))

        text_size = cv2.getTextSize(cat, font, 0.6, 2)[0]

        cv2.rectangle(
            frame,
            (label_x - text_size[0] // 2 - 8, label_y - text_size[1] - 4),
            (label_x + text_size[0] // 2 + 8, label_y + 4),
            (20, 20, 25),
            -1,
        )

        cv2.rectangle(
            frame,
            (label_x - text_size[0] // 2 - 8, label_y - text_size[1] - 4),
            (label_x + text_size[0] // 2 + 8, label_y + 4),
            comp_colors[cat],
            1,
        )

        cv2.putText(
            frame,
            cat,
            (label_x - text_size[0] // 2, label_y),
            font,
            0.6,
            comp_colors[cat],
            2,
            cv2.LINE_AA,
        )


def draw_footer(frame):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, h - 60), (w, h), (15, 15, 20), -1)
    cv2.line(frame, (0, h - 60), (w, h - 60), (0, 255, 200), 2)

    controls = "CONTROLS: 1-5 Manual Test | Q Quit | YOLO Auto-Detection Active"

    cv2.putText(
        frame,
        controls,
        (30, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (150, 255, 200),
        1,
        cv2.LINE_AA,
    )

    timestamp = datetime.now().strftime("%H:%M:%S")

    cv2.putText(
        frame,
        timestamp,
        (w - 120, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (150, 255, 200),
        1,
        cv2.LINE_AA,
    )


def get_best_yolo_detection(model, roi):
    """
    Runs YOLO on ROI and returns:
    yolo_name, sortbot_category, confidence, box

    This version ignores irrelevant classes like person, mouse, keyboard, etc.
    """

    results = model.predict(
        roi,
        conf=YOLO_CONF_THRESHOLD,
        verbose=False,
    )

    if not results:
        return "", "", 0.0, None

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return "", "", 0.0, None

    ignored_classes = {
        "person",
        "mouse",
        "keyboard",
        "laptop",
        "cell phone",
        "remote",
        "tv",
        "chair",
        "couch",
        "bed",
        "dining table",
        "handbag",
        "backpack",
        "suitcase",
    }

    useful_classes = {
        "bottle",
        "cup",
        "wine glass",
        "bowl",
        "banana",
        "apple",
        "orange",
        "sandwich",
        "book",
    }

    best = None
    best_conf = 0.0
    best_name = ""

    for box in result.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        yolo_name = model.names[class_id].lower().strip()

        if yolo_name in ignored_classes:
            continue

        if yolo_name not in useful_classes:
            continue

        if confidence > best_conf:
            best_conf = confidence
            best = box
            best_name = yolo_name

    if best is None:
        return "", "", 0.0, None

    sortbot_category = map_yolo_class_to_sortbot(best_name)

    x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)

    return best_name, sortbot_category, best_conf, (x1, y1, x2, y2)


# =================== MAIN PROGRAM ===================


def main():
    print("=== SORTBOT YOLOv8 AUTOMATIC SYSTEM ===")
    print("[INFO] Loading YOLO model...")
    model = YOLO(YOLO_MODEL_PATH)
    print("[OK] YOLO model loaded.")
    print("Controls: 1-5 manual test, q quit")
    print("Tip: Put only the object inside the green ROI box.")
    print("Try bottle, cup, apple, banana, orange, or book for demo.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No camera found.")
        return

    cap.set(3, 1920)
    cap.set(4, 1080)

    ok, test = cap.read()

    if not ok:
        print("Could not read from camera.")
        return

    h, w = test.shape[:2]

    # ROI means Region of Interest.
    # Show trash inside this box.
    roi_x = 470
    roi_y = h // 2 - 170
    roi_s = 340

    dustbin = CircularDustbin(5)

    last_yolo_time = 0
    last_trigger = 0

    last_yolo_name = ""
    last_sortbot_label = ""
    last_confidence = 0.0
    last_box = None

    state_names = ["IDLE", "DETECTED", "ROTATING", "OPENING", "FALLING", "CLOSING"]

    while True:
        ok, camera_frame = cap.read()

        if not ok:
            continue

        raw_frame = cv2.flip(camera_frame, 1)
        fh, fw = raw_frame.shape[:2]

        roi = raw_frame[roi_y : roi_y + roi_s, roi_x : roi_x + roi_s]

        now = time.time()

        if roi.size > 0 and now - last_yolo_time >= YOLO_INTERVAL:
            (
                detected_yolo_name,
                detected_sortbot_label,
                detected_confidence,
                detected_box,
            ) = get_best_yolo_detection(model, roi)

            if detected_yolo_name:
                last_yolo_name = detected_yolo_name
                last_sortbot_label = detected_sortbot_label
                last_confidence = detected_confidence
                last_box = detected_box
            else:
                last_yolo_name = ""
                last_sortbot_label = ""
                last_confidence = 0.0
                last_box = None

            last_yolo_time = now

        if (
            last_sortbot_label
            and last_confidence >= SORT_TRIGGER_CONFIDENCE
            and dustbin.state == IDLE
            and now - last_trigger > DETECTION_COOLDOWN
        ):
            idx = categories.index(last_sortbot_label)

            dustbin.detect(idx, now, last_confidence)
            log_event(last_sortbot_label, last_confidence)

            print(
                f"[YOLO] {last_yolo_name} → {last_sortbot_label} "
                f"| Confidence: {last_confidence:.2f}"
            )

            last_trigger = now

        dark_overlay = np.zeros_like(raw_frame)
        dark_overlay[:] = (10, 10, 15)

        frame = cv2.addWeighted(raw_frame, 0.3, dark_overlay, 0.7, 0)

        cx = fw // 2 + 150
        cy = int(fh * 0.53)

        dustbin.update(now, cx, cy, BIN_RADIUS)

        draw_modern_header(frame)

        draw_detection_panel(
            frame,
            last_yolo_name,
            last_sortbot_label,
            last_confidence,
            state_names,
            dustbin.state,
        )

        # ROI box
        cv2.rectangle(
            frame,
            (roi_x, roi_y),
            (roi_x + roi_s, roi_y + roi_s),
            (0, 255, 200),
            2,
        )

        cv2.putText(
            frame,
            "YOLO DETECTION ROI",
            (roi_x, roi_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )

        # Bounding box inside ROI
        if last_box is not None:
            x1, y1, x2, y2 = last_box

            gx1 = roi_x + x1
            gy1 = roi_y + y1
            gx2 = roi_x + x2
            gy2 = roi_y + y2

            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 100), 2)

            label_text = f"{last_yolo_name} {last_confidence:.2f}"

            cv2.putText(
                frame,
                label_text,
                (gx1, max(gy1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 100),
                2,
                cv2.LINE_AA,
            )

        draw_bin(frame, dustbin, cx, cy, BIN_RADIUS)
        draw_lid(frame, dustbin, cx, cy, BIN_RADIUS)
        draw_trash(frame, dustbin)
        draw_particles(frame, dustbin)
        draw_labels(frame, dustbin, cx, cy, BIN_RADIUS)
        draw_footer(frame)

        cv2.imshow("SortBot YOLOv8 Automatic System", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Manual keys are backup/demo only.
        if key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5")]:
            i = key - ord("1")

            dustbin.detect(i, time.time(), 1)
            log_event(categories[i], 1, "manual", "key")

            print(f"[MANUAL] Triggered: {categories[i]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
