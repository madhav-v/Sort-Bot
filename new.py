import cv2
import numpy as np
import time
import math
import random
import threading
import os
import csv
from datetime import datetime

# Optional AI imports
import torch
from torchvision import models, transforms
import requests
import json

EVENT_LOG_PATH = "dashboard/data/events.csv"


def log_event(category, confidence=1.0, source="sortbot-01", note="auto"):
    os.makedirs(os.path.dirname(EVENT_LOG_PATH), exist_ok=True)
    header_needed = (
        not os.path.exists(EVENT_LOG_PATH) or os.stat(EVENT_LOG_PATH).st_size == 0
    )
    with open(EVENT_LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["ts", "category", "confidence", "source", "note"])
        w.writerow([datetime.utcnow().isoformat(), category, confidence, source, note])


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

CLASSIFY_INTERVAL = 0.45
CONFIDENCE_THRESHOLD = 0.55
DETECTION_COOLDOWN = 1.2


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


# =================== PARTICLE ===================
class Particle:
    def __init__(self, x, y, color):
        self.x = x + random.uniform(-6, 6)
        self.y = y + random.uniform(-6, 6)
        ang = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.5, 5)
        self.vx = math.cos(ang) * speed
        self.vy = math.sin(ang) * speed - random.uniform(2, 5)
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
                ang = math.radians(
                    (self.target_idx + 0.5) * (360 / self.n) + self.angle
                )
                self.hole_x = int(cx + (r - 15) * math.cos(ang))
                self.hole_y = int(cy + (r - 15) * math.sin(ang))
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
                col = comp_colors[categories[self.target_idx]]
                for _ in range(18):
                    self.particles.append(Particle(self.hole_x, self.trash_y, col))
                self.state = CLOSING
                self.start = now

        elif self.state == CLOSING:
            t = min(1, (now - self.start) / TIMINGS["lid_close_time"])
            self.lid = (1 - ease_in_out_cubic(t)) * 0.05
            if t >= 1:
                self.state = IDLE
                self.trash_y = -999

        alive = []
        for p in self.particles:
            p.update(dt)
            if p.life > 0:
                alive.append(p)
        self.particles = alive


# =================== MODERN UI DRAWING ===================
def draw_modern_header(frame):
    """Draw futuristic header"""
    h, w = frame.shape[:2]

    # Dark overlay for header
    cv2.rectangle(frame, (0, 0), (w, 90), (15, 15, 20), -1)
    cv2.rectangle(frame, (0, 88), (w, 90), (0, 255, 200), 2)

    # Title with glow effect
    title = "SMART WASTE SORTING SYSTEM"
    title_font = cv2.FONT_HERSHEY_DUPLEX
    text_size = cv2.getTextSize(title, title_font, 1.2, 2)[0]
    title_x = (w - text_size[0]) // 2

    # Glow effect
    for offset in [4, 3, 2, 1]:
        alpha = 0.3 / offset
        cv2.putText(
            frame,
            title,
            (title_x, 45),
            title_font,
            1.2,
            (0, 255, 200),
            offset + 2,
            cv2.LINE_AA,
        )

    # Main title
    cv2.putText(
        frame, title, (title_x, 45), title_font, 1.2, (0, 255, 200), 2, cv2.LINE_AA
    )

    # Subtitle
    subtitle = "AI-Powered Waste Classification"
    cv2.putText(
        frame,
        subtitle,
        (title_x + 40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 255, 200),
        1,
        cv2.LINE_AA,
    )


def draw_modern_detection_panel(
    frame, label, conf, state_names, current_state, dustbin_obj
):
    """Draw AI detection panel with modern styling"""
    h, w = frame.shape[:2]

    # Panel dimensions
    panel_x, panel_y = 30, 120
    panel_w, panel_h = 380, 420

    # Dark panel background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (20, 20, 25),
        -1,
    )
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Cyan border with corner accents
    border_color = (0, 255, 200)
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        border_color,
        2,
    )

    # Corner accents
    corner_len = 20
    cv2.line(
        frame, (panel_x, panel_y), (panel_x + corner_len, panel_y), border_color, 4
    )
    cv2.line(
        frame, (panel_x, panel_y), (panel_x, panel_y + corner_len), border_color, 4
    )
    cv2.line(
        frame,
        (panel_x + panel_w, panel_y),
        (panel_x + panel_w - corner_len, panel_y),
        border_color,
        4,
    )
    cv2.line(
        frame,
        (panel_x + panel_w, panel_y),
        (panel_x + panel_w, panel_y + corner_len),
        border_color,
        4,
    )
    cv2.line(
        frame,
        (panel_x, panel_y + panel_h),
        (panel_x + corner_len, panel_y + panel_h),
        border_color,
        4,
    )
    cv2.line(
        frame,
        (panel_x, panel_y + panel_h),
        (panel_x, panel_y + panel_h - corner_len),
        border_color,
        4,
    )
    cv2.line(
        frame,
        (panel_x + panel_w, panel_y + panel_h),
        (panel_x + panel_w - corner_len, panel_y + panel_h),
        border_color,
        4,
    )
    cv2.line(
        frame,
        (panel_x + panel_w, panel_y + panel_h),
        (panel_x + panel_w, panel_y + panel_h - corner_len),
        border_color,
        4,
    )

    # Header
    cv2.putText(
        frame,
        "AI DETECTION",
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

    # Detection visualization area
    det_y = panel_y + 70
    det_center_x = panel_x + panel_w // 2
    det_center_y = det_y + 100

    # Scanning circle animation
    scan_radius = int(80 + 10 * math.sin(time.time() * 3))
    cv2.circle(frame, (det_center_x, det_center_y), scan_radius, (0, 200, 150), 1)
    cv2.circle(frame, (det_center_x, det_center_y), scan_radius + 5, (0, 150, 100), 1)

    # Crosshair
    cv2.line(
        frame,
        (det_center_x - 60, det_center_y),
        (det_center_x - 30, det_center_y),
        border_color,
        2,
    )
    cv2.line(
        frame,
        (det_center_x + 30, det_center_y),
        (det_center_x + 60, det_center_y),
        border_color,
        2,
    )
    cv2.line(
        frame,
        (det_center_x, det_center_y - 60),
        (det_center_x, det_center_y - 30),
        border_color,
        2,
    )
    cv2.line(
        frame,
        (det_center_x, det_center_y + 30),
        (det_center_x, det_center_y + 60),
        border_color,
        2,
    )

    # Detection result box - Show current dustbin target
    result_y = det_y + 200
    cv2.putText(
        frame,
        "ITEM DETECTED:",
        (panel_x + 30, result_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    # Show detected category from dustbin state
    if dustbin_obj.state != IDLE:
        # Get the target category from dustbin
        detected_cat = categories[dustbin_obj.target_idx]
        cat_y = result_y + 25
        cat_color = comp_colors[detected_cat]

        # Glowing box
        cv2.rectangle(
            frame,
            (panel_x + 25, cat_y - 25),
            (panel_x + panel_w - 25, cat_y + 15),
            (30, 30, 35),
            -1,
        )
        cv2.rectangle(
            frame,
            (panel_x + 25, cat_y - 25),
            (panel_x + panel_w - 25, cat_y + 15),
            cat_color,
            2,
        )

        # Category text with icon
        cv2.putText(
            frame,
            f"→ {detected_cat.upper()}",
            (panel_x + 40, cat_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            cat_color,
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "Waiting for item...",
            (panel_x + 40, result_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )

    # Confidence bar
    conf_y = result_y + 70
    cv2.putText(
        frame,
        "CONFIDENCE",
        (panel_x + 30, conf_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    bar_x = panel_x + 30
    bar_y = conf_y + 10
    bar_w = panel_w - 60
    bar_h = 20

    # Background bar
    cv2.rectangle(
        frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 45), -1
    )
    cv2.rectangle(
        frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), border_color, 1
    )

    # Fill bar
    if conf > 0:
        fill_w = int(bar_w * conf)
        color = (
            (0, 255, 100)
            if conf > 0.7
            else (255, 200, 0) if conf > 0.5 else (255, 100, 100)
        )
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

        # Percentage text
        cv2.putText(
            frame,
            f"{conf*100:.1f}%",
            (bar_x + bar_w + 10, bar_y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    # System state
    state_y = conf_y + 60
    cv2.putText(
        frame,
        "STATUS",
        (panel_x + 30, state_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
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
    pts = [(cx, cy)]
    step = 4
    if a2 < a1:
        a2 += 360
    for ang in range(int(a1), int(a2) + 1, step):
        rad = math.radians(ang)
        pts.append((int(cx + r * math.cos(rad)), int(cy + r * math.sin(rad))))
    return np.array(pts, np.int32)


def draw_bin(frame, bin: CircularDustbin, cx, cy, r):
    """Draw the rotating dustbin"""
    # Outer rim with glow
    cv2.circle(frame, (cx, cy), r + 18, (25, 25, 30), -1)
    cv2.circle(frame, (cx, cy), r + 15, (0, 255, 200), 2)

    angle_step = 360 / bin.n
    for i, cat in enumerate(categories):
        a1 = i * angle_step + bin.angle
        a2 = (i + 1) * angle_step + bin.angle
        pts = sector_polygon(cx, cy, r, a1, a2)
        cv2.fillPoly(frame, [pts], comp_dark[cat])
        inner = sector_polygon(cx, cy, int(r * 0.75), a1, a2)
        mix = tuple(
            int(comp_dark[cat][c] * 0.6 + comp_colors[cat][c] * 0.4) for c in range(3)
        )
        cv2.fillPoly(frame, [inner], mix)
        cv2.polylines(frame, [pts], True, (20, 20, 20), 2)

        # Glowing sector borders
        if i == bin.target_idx and bin.state != IDLE:
            cv2.polylines(frame, [pts], True, (0, 255, 200), 2)


def draw_lid(frame, bin: CircularDustbin, cx, cy, r):
    """Draw animated lid"""
    if bin.state not in (OPENING, FALLING, CLOSING):
        return
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), r + 18, (220, 220, 230), -1)
    cv2.circle(overlay, (cx, cy), r - 10, (0, 0, 0), -1)
    mask = overlay.sum(axis=2) > 0
    frame[mask] = frame[mask] * 0.25 + overlay[mask] * 0.75


def draw_trash(frame, bin: CircularDustbin):
    """Draw falling trash with glow"""
    if bin.trash_y < -900:
        return
    x, y = int(bin.trash_x), int(bin.trash_y)
    cat = categories[bin.target_idx]
    col = comp_colors[cat]

    # Glow effect
    cv2.circle(frame, (x, y), 18, col, -1)
    cv2.circle(frame, (x, y), 15, (255, 255, 255), 2)
    cv2.circle(frame, (x, y), 12, col, -1)


def draw_particles(frame, bin: CircularDustbin):
    """Draw particle effects"""
    for p in bin.particles:
        alpha = max(0, min(1, p.life / p.max_life))
        c = tuple(int(p.color[i] * alpha) for i in range(3))
        size = max(1, int(p.size * alpha))
        cv2.circle(frame, (int(p.x), int(p.y)), size, c, -1)
        # Add sparkle
        if alpha > 0.7:
            cv2.circle(frame, (int(p.x), int(p.y)), size + 2, (255, 255, 255), 1)


def draw_labels(frame, bin: CircularDustbin, cx, cy, r):
    """Draw category labels with modern styling"""
    font = cv2.FONT_HERSHEY_DUPLEX
    step = 360 / bin.n
    for i, cat in enumerate(categories):
        ang = (i + 0.5) * step + bin.angle
        rad = math.radians(ang)
        lx = int(cx + (r + 55) * math.cos(rad))
        ly = int(cy + (r + 55) * math.sin(rad))

        # Background
        text_size = cv2.getTextSize(cat, font, 0.6, 2)[0]
        cv2.rectangle(
            frame,
            (lx - text_size[0] // 2 - 8, ly - text_size[1] - 4),
            (lx + text_size[0] // 2 + 8, ly + 4),
            (20, 20, 25),
            -1,
        )
        cv2.rectangle(
            frame,
            (lx - text_size[0] // 2 - 8, ly - text_size[1] - 4),
            (lx + text_size[0] // 2 + 8, ly + 4),
            comp_colors[cat],
            1,
        )

        # Text
        cv2.putText(
            frame,
            cat,
            (lx - text_size[0] // 2, ly),
            font,
            0.6,
            comp_colors[cat],
            2,
            cv2.LINE_AA,
        )


def draw_modern_footer(frame):
    """Draw modern footer with controls"""
    h, w = frame.shape[:2]

    # Footer background
    cv2.rectangle(frame, (0, h - 60), (w, h), (15, 15, 20), -1)
    cv2.line(frame, (0, h - 60), (w, h - 60), (0, 255, 200), 2)

    # Controls
    controls = "CONTROLS: 1-5 Manual | Q Quit | AI: Auto-Detection Active"
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

    # Timestamp
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


# =================== CLASSIFIER THREAD ===================
class ClassifierWorker(threading.Thread):
    def __init__(self, roi_size, interval):
        super().__init__(daemon=True)
        self.interval = interval
        self.roi_size = roi_size
        self.model = None
        self.transform = None
        self.labels = None
        self._lock = threading.Lock()
        self._roi = None
        self.last = ("", 0, 0)
        self.load()

    def load(self):
        try:
            w = models.MobileNet_V2_Weights.DEFAULT
            self.model = models.mobilenet_v2(weights=w)
            self.model.eval()
            idx_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
            self.labels = [
                json.loads(requests.get(idx_url).text)[str(i)][1] for i in range(1000)
            ]
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            print("[OK] Classifier ready.")
        except:
            print("[ERR] classifier load fail")

    def update_roi(self, roi):
        with self._lock:
            self._roi = roi.copy()

    def run(self):
        while True:
            t0 = time.time()
            roi = None
            with self._lock:
                if self._roi is not None:
                    roi = self._roi.copy()
            if roi is not None and self.model is not None:
                try:
                    x = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    inp = self.transform(x).unsqueeze(0)
                    with torch.no_grad():
                        out = self.model(inp)
                    probs = torch.nn.functional.softmax(out[0], dim=0)
                    p, i = torch.topk(probs, 1)
                    self.last = (self.labels[int(i)], float(p), time.time())
                except:
                    pass
            time.sleep(max(0.01, self.interval - (time.time() - t0)))


def map_label(l):
    if l is None:
        return None
    l = l.lower()
    if "can" in l or "tin" in l:
        return "Cans"
    if "bottle" in l or "flask" in l:
        return "Bottles"
    if "plastic" in l or "bag" in l or "wrapper" in l:
        return "Plastic"
    if any(x in l for x in ("paper", "tissue", "cardboard")):
        return "Burnable"
    return "Others"


# =================== MAIN ===================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera")
        return
    cap.set(3, 1920)
    cap.set(4, 1080)

    _, test = cap.read()
    h, w = test.shape[:2]
    roi_x, roi_y, roi_s = 470, h // 2 - 170, 340

    dust = CircularDustbin(5)
    clf = ClassifierWorker(roi_s, CLASSIFY_INTERVAL)
    clf.start()

    last_trigger = 0
    state_names = ["IDLE", "DETECTED", "ROTATING", "OPENING", "FALLING", "CLOSING"]

    print("=== PROFESSIONAL SORTBOT SYSTEM ===")
    print("Controls: 1-5 manual, q quit")

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        # Create dark background overlay
        dark_overlay = np.zeros_like(frame)
        dark_overlay[:] = (10, 10, 15)
        frame = cv2.addWeighted(frame, 0.3, dark_overlay, 0.7, 0)

        cx, cy = fw // 2 + 150, int(fh * 0.53)

        roi = frame[roi_y : roi_y + roi_s, roi_x : roi_x + roi_s]
        if roi.size > 0:
            clf.update_roi(roi)

        label, conf, ts = clf.last
        now = time.time()

        if conf > CONFIDENCE_THRESHOLD and now - last_trigger > DETECTION_COOLDOWN:
            c = map_label(label)
            if c:
                idx = categories.index(c)
                dust.detect(idx, now, conf)
                log_event(c, conf)
                last_trigger = now

        dust.update(now, cx, cy, BIN_RADIUS)

        # Draw all UI elements
        draw_modern_header(frame)
        draw_modern_detection_panel(frame, label, conf, state_names, dust.state, dust)
        draw_bin(frame, dust, cx, cy, BIN_RADIUS)
        draw_lid(frame, dust, cx, cy, BIN_RADIUS)
        draw_trash(frame, dust)
        draw_particles(frame, dust)
        draw_labels(frame, dust, cx, cy, BIN_RADIUS)
        draw_modern_footer(frame)

        cv2.imshow("SortBot Professional System", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5")]:
            i = k - ord("1")
            dust.detect(i, time.time(), 1)
            log_event(categories[i], 1, "manual", "key")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
