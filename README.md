# Plankton-Mask-Tuner
Interactive OpenCV tuner for tracking Stentor (plankton) in circular wells: adjust circle mask (XC/YC/RADIUS), thresholding, and blob filters live to suppress wall reflections and improve detection.

# stentor-mask-tuner

Interactive OpenCV tuner for tracking **Stentor (plankton)** swimming in **circular wells**. Dial in a **circular mask** to suppress wall reflections, then tune **thresholding** and **blob filters** live until detections are stable.

## Why this exists

Stentor-in-well videos often have strong reflections and glare near the well wall. A circular mask (center + radius) removes most of that clutter so simple threshold + blob detection works reliably.

## What it does

* Loads frames from a video (or camera feed)
* Applies a **circular well mask** (XC/YC/RADIUS) to ignore pixels outside the well
* Creates a **binary image** with an intensity threshold (THRESH)
* Detects candidate blobs and filters them by:

  * **Area** (MIN_A / MAX_A)
  * **Width/Height** bounds (MIN_WH / MAX_WH)
* Shows:

  * **Raw** view
  * **Binary** view
  * **Controls** panel for interactive tuning
* Optional:

  * **TRAILS** overlay for motion
  * **FPS** throttling for comfortable tuning

## Requirements

* Python 3.9+ (3.10/3.11 recommended)
* opencv-python
* numpy

Install:
pip install opencv-python numpy

## Run

1. Set VIDEO_PATH in the script to your input video.
2. Start the tuner:
   python "detector_mask_tuner v4.py"

## Key parameters

Mask geometry (in source-frame pixels)

* XC, YC: center of the well
* RADIUS: radius of the well

Segmentation

* THRESH (0–255): brightness threshold for binary image
  Higher = stricter; Lower = more permissive (more noise)

Blob filtering

* MIN_A, MAX_A: blob area limits (pixels^2)
* MIN_WH, MAX_WH: blob width/height limits (pixels)

UI / tuning

* MASK: toggle applying the circular mask
* TRAILS: toggle motion trail accumulation
* FPS: throttle the update loop
* HREZ, VREZ: display resolution for Raw/Binary windows (UI only)

## Suggested tuning workflow

1. Enable MASK and adjust XC/YC/RADIUS until the circle matches the well boundary.
2. Adjust THRESH until stentor(s) are clearly visible in Binary without too much glare.
3. Tighten filters:

   * raise MIN_A / MIN_WH to remove speckles
   * lower MAX_A / MAX_WH to remove large glare blobs
4. Toggle MASK off briefly to confirm you’re not hiding real signal.
5. Enable TRAILS to sanity-check that motion paths look real.
6. When you exit the program, it prints the last-used values of key parameters (e.g., XC, YC, RADIUS, THRESH, MIN_A/MAX_A, MIN_WH/MAX_WH) so you can copy them into your *_INIT defaults for future runs.

## Notes

If Raw/Binary window sizes look different on your OS, force the window sizes with:
cv2.resizeWindow(WIN_RAW, HREZ, VREZ)
cv2.resizeWindow(WIN_BIN, HREZ, VREZ)

## License

MIT License
Copyright (c) 2026 Thomas G. Zimmerman
