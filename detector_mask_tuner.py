# Copyright (c) 2026 Thomas Zimmerman — MIT License
"""
Interactive parameter tuner for tracking one or more Stentor (plankton) swimming
inside a circular well.

What it does:
- Loads frames from a video (or camera feed) of a circular well.
- Builds and applies a circular mask (center XC/YC, radius RADIUS) to suppress
  bright reflections and wall artifacts outside the well boundary.
- Converts the masked frame to a binary image using a brightness threshold (THRESH).
- Finds candidate blobs/objects in the binary image and filters them by size/shape
  constraints (MIN_A/MAX_A for area; MIN_WH/MAX_WH for width/height).
- Displays a Raw view and a Binary view plus an on-screen Controls panel.
- Lets you adjust parameters live (buttons / mouse interaction) to quickly find
  settings that reliably isolate Stentor while rejecting glare and noise.
- Optionally accumulates and draws motion trails, and throttles playback via FPS.

Primary goal:
Make it fast to dial in mask geometry + detection thresholds so the tracker works
robustly across different wells, lighting conditions, and reflection patterns.
"""

import cv2
import numpy as np

# --- Initial tuning parameters (startup defaults) ---

# Directory containing your input videos (raw string keeps Windows backslashes literal).
# Update this to point at the folder where your well recordings live.
VIDEO_DIR  = r"C:/Users/MyVideoDir//"

# Full path to the specific video file to load and tune against.
# Change "MyVid.mov" to whichever clip you want to analyze.
VIDEO_PATH = VIDEO_DIR + "MyVid.mov"

# Circle mask geometry (in *source-frame pixels*):
# Center of the well (XC, YC) and radius of the well (RADIUS). Used to build the
# circular mask that excludes wall reflections outside the well boundary.
XC_INIT     = 349   # initial mask center X (pixels in original frame coordinates)
YC_INIT     = 226   # initial mask center Y (pixels in original frame coordinates)
RADIUS_INIT = 200   # initial well radius (pixels in original frame coordinates)

# Playback / UI pacing:
# Frames-per-second throttle for the display loop. 30 ≈ realtime; lower slows down,
# 0 means "no delay / run as fast as possible" if your loop interprets FPS that way.
FPS_INIT    = 30    # initial FPS limit for display/processing loop

# Binary thresholding:
# Pixel intensity threshold used to create the binary image for detection.
# Higher = fewer bright pixels pass; lower = more pixels pass.
THRESH_INIT = 10    # initial threshold value (0–255)

# Detection size filters (reject blobs that are too small/large):
# MIN_A/MAX_A are blob area limits in pixels^2, used to filter connected components
# (or contours) so only plausible stentor-sized detections remain.
MIN_A_INIT  = 10    # minimum blob area (pixels^2)
MAX_A_INIT  = 824   # maximum blob area (pixels^2)

# Detection shape filters (width/height constraints):
# MIN_WH/MAX_WH bound the blob width and height (in pixels) to reject noise specks
# and large glare patches.
MIN_WH_INIT = 2     # minimum blob width/height (pixels)
MAX_WH_INIT = 52    # maximum blob width/height (pixels)

# --- UI toggles (startup defaults) ---

# Mask display/usage toggle:
# 0 = mask off at startup; 1 = mask on at startup (apply circular mask).
MASK_INIT   = 0     # initial mask toggle state

# Trajectory trails toggle:
# 0 = trails off at startup; 1 = trails on at startup (accumulate and draw tracks).
TRAILS_INIT = 0     # initial trails toggle state

# --- Display resolution (for OpenCV windows) ---

# Output display size for the Raw/Binary windows (pixels).
# This affects ONLY what you see on screen; processing may still use full frame size.
HREZ, VREZ  = 640, 512   # display width, display height

GAUSSIAN_BLUR = (5, 5)
BG_SAMPLES    = 100

WIN_RAW = "Raw"
WIN_BIN = "Binary"
WIN_CTL = "Controls"

# ==================================================
# UI + STATE
# ==================================================
BTN_W, BTN_H, BTN_PAD = 140, 44, 6

ROW1 = ["XC", "YC", "RADIUS", "MASK", "TRAILS"]
ROW2 = ["FPS", "THRESH", "MIN_A", "MAX_A", "MIN_WH", "MAX_WH"]
ROW3 = ["-", "+", "1", "10", "100"]

RADIO_PARAMS = ["XC", "YC", "RADIUS", "FPS", "THRESH",
                "MIN_A", "MAX_A", "MIN_WH", "MAX_WH"]

EXIT_ORDER = ["XC", "YC", "RADIUS", "FPS",
              "THRESH", "MIN_A", "MAX_A", "MIN_WH", "MAX_WH"]

GRAY   = (80, 80, 80)
GREEN  = (0, 150, 0)
YELLOW = (0, 255, 255)
WHITE  = (240, 240, 240)
BORDER = (30, 30, 30)

STATE = {
    "selected": "FPS",
    "step": 1,
    "mask": False,
    "trails": False,
}

PARAMS = {}
BUTTONS = {}
_BG_MEDIAN_V = None


# ==================================================
# HELPERS
# ==================================================
def clamp(v, lo, hi):
    return int(max(lo, min(hi, int(v))))


def enforce_pairs():
    if PARAMS["MIN_A"]["value"] > PARAMS["MAX_A"]["value"]:
        PARAMS["MAX_A"]["value"] = PARAMS["MIN_A"]["value"]
    if PARAMS["MIN_WH"]["value"] > PARAMS["MAX_WH"]["value"]:
        PARAMS["MAX_WH"]["value"] = PARAMS["MIN_WH"]["value"]


def build_median_background(cap):
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, total - 1),
                       min(BG_SAMPLES, total)).astype(int)

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if not ret:
            continue
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:, :, 2])

    return np.median(np.stack(frames), axis=0).astype(np.uint8)


# ==================================================
# CONTROL PANEL UI
# ==================================================
def draw_button(img, rect, text, active, active_color):
    x, y, w, h = rect
    bg = active_color if active else GRAY
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), BORDER, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(img, text,
                (x + (w - tw) // 2, y + (h + th) // 2 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

def raw_mouse(event, x, y, flags, userdata):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Only allow mouse interaction when MASK is enabled
    if not STATE["mask"]:
        return

    w = userdata["w"]
    h = userdata["h"]

    # Convert display coords -> native video coords
    sx = w / HREZ
    sy = h / VREZ

    xc = int(x * sx)
    yc = int(y * sy)

    PARAMS["XC"]["value"] = clamp(xc, 0, w)
    PARAMS["YC"]["value"] = clamp(yc, 0, h)



def render_controls():
    global BUTTONS
    BUTTONS = {}

    rows = [ROW1, ROW2, ROW3]
    cols = max(len(r) for r in rows)

    img = np.zeros(
        (BTN_PAD + 3 * (BTN_H + BTN_PAD),
         BTN_PAD + cols * (BTN_W + BTN_PAD), 3),
        np.uint8
    )
    img[:] = (20, 20, 20)

    y = BTN_PAD
    for row in rows:
        x = BTN_PAD
        for name in row:
            rect = (x, y, BTN_W, BTN_H)
            BUTTONS[name] = rect

            if name in RADIO_PARAMS:
                draw_button(img, rect,
                            f"{name}: {PARAMS[name]['value']}",
                            STATE["selected"] == name, GREEN)
            elif name in ("MASK", "TRAILS"):
                active = STATE["mask"] if name == "MASK" else STATE["trails"]
                draw_button(img, rect, name, active, YELLOW)
            elif name in ("1", "10", "100"):
                draw_button(img, rect, name,
                            STATE["step"] == int(name), GREEN)
            else:
                draw_button(img, rect, name, False, GREEN)

            x += BTN_W + BTN_PAD
        y += BTN_H + BTN_PAD

    return img


def hit_test(x, y):
    for k, (bx, by, bw, bh) in BUTTONS.items():
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return k
    return None


def ctl_mouse(event, x, y, flags, userdata):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    key = hit_test(x, y)
    if not key:
        return

    if key in RADIO_PARAMS:
        STATE["selected"] = key
    elif key in ("MASK", "TRAILS"):
        if key == "MASK":
            STATE["mask"] = not STATE["mask"]
        else:
            STATE["trails"] = not STATE["trails"]
            if not STATE["trails"]:
                userdata["trails"][:] = 0
    elif key in ("1", "10", "100"):
        STATE["step"] = int(key)
    elif key == "+":
        p = STATE["selected"]
        PARAMS[p]["value"] = clamp(
            PARAMS[p]["value"] + STATE["step"],
            PARAMS[p]["min"], PARAMS[p]["max"]
        )
        enforce_pairs()
    elif key == "-":
        p = STATE["selected"]
        PARAMS[p]["value"] = clamp(
            PARAMS[p]["value"] - STATE["step"],
            PARAMS[p]["min"], PARAMS[p]["max"]
        )
        enforce_pairs()


# ==================================================
# MAIN
# ==================================================
def main():
    global PARAMS, _BG_MEDIAN_V

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame0 = cap.read()
    if not ret or frame0 is None:
        raise RuntimeError(f"Could not read first frame from VIDEO_PATH={VIDEO_PATH!r}")

    h, w = frame0.shape[:2]

    # Use user-constant initial values if provided; fall back to auto defaults if not.
    # (Define XC_INIT, YC_INIT, RADIUS_INIT up in USER CONSTANTS.
    #  If you set them to 0, this will auto-center/auto-radius.)
    xc0 = XC_INIT if (isinstance(XC_INIT, int) and XC_INIT > 0) else (w // 2)
    yc0 = YC_INIT if (isinstance(YC_INIT, int) and YC_INIT > 0) else (h // 2)
    r0  = RADIUS_INIT if (isinstance(RADIUS_INIT, int) and RADIUS_INIT > 0) else (h // 4)

    # Clamp to valid ranges (helps if constants are out of bounds for a new video size)
    xc0 = max(0, min(xc0, w))
    yc0 = max(0, min(yc0, h))
    r0  = max(1, min(r0, h // 2))

    PARAMS = {
        "XC":     {"value": xc0, "min": 0, "max": w},
        "YC":     {"value": yc0, "min": 0, "max": h},
        "RADIUS": {"value": r0,  "min": 1, "max": h // 2},
        "FPS":    {"value": FPS_INIT,    "min": 0, "max": 30},
        "THRESH": {"value": THRESH_INIT, "min": 0, "max": 255},
        "MIN_A":  {"value": MIN_A_INIT,  "min": 0, "max": 2000},
        "MAX_A":  {"value": MAX_A_INIT,  "min": 0, "max": 2000},
        "MIN_WH": {"value": MIN_WH_INIT, "min": 0, "max": 100},
        "MAX_WH": {"value": MAX_WH_INIT, "min": 0, "max": 100},
    }
    enforce_pairs()

    STATE["mask"] = bool(MASK_INIT)
    STATE["trails"] = bool(TRAILS_INIT)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _BG_MEDIAN_V = build_median_background(cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    trail_accum = np.zeros((VREZ, HREZ), np.uint8)
    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow(WIN_RAW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_BIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_CTL, cv2.WINDOW_NORMAL)

    # Force RAW/BIN window client areas to match the displayed image size
    cv2.resizeWindow(WIN_RAW, HREZ, VREZ)
    cv2.resizeWindow(WIN_BIN, HREZ, VREZ)

    # Optional: place them predictably (side-by-side)
    cv2.moveWindow(WIN_RAW, 40, 40)
    cv2.moveWindow(WIN_BIN, 40 + HREZ + 20, 40)

    cv2.setMouseCallback(
        WIN_RAW,
        raw_mouse,
        {"w": w, "h": h}
    )

    ctl_img = render_controls()
    cv2.resizeWindow(WIN_CTL, ctl_img.shape[1], ctl_img.shape[0])
    cv2.moveWindow(WIN_CTL, 40, 40 + VREZ + 80)
    cv2.setMouseCallback(WIN_CTL, ctl_mouse, {"trails": trail_accum})


    frame = None
    while True:
        fps = PARAMS["FPS"]["value"]
        new_frame = False

        if fps > 0 or frame is None:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                ret, frame = cap.read()
            frame_idx += 1
            new_frame = True

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2]
        Fg = cv2.subtract(V, _BG_MEDIAN_V)

        if GAUSSIAN_BLUR:
            Fg = cv2.GaussianBlur(Fg, GAUSSIAN_BLUR, 0)

        if STATE["mask"]:
            gate = np.zeros_like(V)
            cv2.circle(gate,
                       (PARAMS["XC"]["value"], PARAMS["YC"]["value"]),
                       PARAMS["RADIUS"]["value"], 255, -1)
            Fg = cv2.bitwise_and(Fg, Fg, mask=gate)

        _, bin_img = cv2.threshold(
            Fg, PARAMS["THRESH"]["value"], 255, cv2.THRESH_BINARY
        )

        raw_disp = cv2.resize(frame, (HREZ, VREZ))
        bin_disp = cv2.resize(
            cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR),
            (HREZ, VREZ)
        )
        
        # -----------------------------------------------
        # Draw MASK overlay on Raw (visualization only)
        # -----------------------------------------------
        if STATE["mask"]:
            sx = HREZ / w
            sy = VREZ / h
        
            cx = int(PARAMS["XC"]["value"] * sx)
            cy = int(PARAMS["YC"]["value"] * sy)
            r  = int(PARAMS["RADIUS"]["value"] * sy)
        
            cv2.circle(raw_disp, (cx, cy), r, YELLOW, 2)


        contours, _ = cv2.findContours(
            bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        sx, sy = HREZ / w, VREZ / h
        for c in contours:
            area = cv2.contourArea(c)
            if area < PARAMS["MIN_A"]["value"] or area > PARAMS["MAX_A"]["value"]:
                continue
            rect = cv2.minAreaRect(c)
            (_, _), (rw, rh), _ = rect
            bw, bh = min(rw, rh), max(rw, rh)
            if bw < PARAMS["MIN_WH"]["value"] or bw > PARAMS["MAX_WH"]["value"]:
                continue
            if bh < PARAMS["MIN_WH"]["value"] or bh > PARAMS["MAX_WH"]["value"]:
                continue

            box = cv2.boxPoints(rect)
            box[:, 0] *= sx
            box[:, 1] *= sy
            cv2.polylines(raw_disp, [box.astype(int)], True, YELLOW, 2)

        cv2.putText(bin_disp, f"{frame_idx}/{total}",
                    (10, VREZ - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        if STATE["trails"] and new_frame:
            small = cv2.resize(bin_img, (HREZ, VREZ),
                               interpolation=cv2.INTER_NEAREST)
            np.maximum(trail_accum, small, out=trail_accum)

        if STATE["trails"]:
            overlay = cv2.cvtColor(trail_accum, cv2.COLOR_GRAY2BGR)
            raw_disp = cv2.max(raw_disp, overlay)

        cv2.imshow(WIN_RAW, raw_disp)
        cv2.imshow(WIN_BIN, bin_disp)
        cv2.imshow(WIN_CTL, render_controls())

        key = cv2.waitKey(1 if fps == 0 else int(1000 / max(1, fps))) & 0xFF
        if key == 27 or key == ord('q'):
            break

    print("\nFinal values:")
    for k in EXIT_ORDER:
        print(f"{k}: {PARAMS[k]['value']}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
