import cv2
import numpy as np
import time
from scipy import ndimage
from picamera2 import Picamera2
from picarx import Picarx
import matplotlib.pyplot as plt
import os

# ── Constants ────────────────────────────────────────────────────────
FRAME_W, FRAME_H = 640, 480
ROT_ANGLE = 0
F_MM = 3.6
H_MM = 100.0
SENSOR_H_MM = 2.74
P_MM = SENSOR_H_MM / FRAME_H
H_FOV_DEG = 54.05
MIN_AREA = 50

SWEEP_PANS = [-30, 0, 30]
SWEEP_DELAY = 0.4
PAN_GAIN = 0.03
PAN_LIM = 35
FORWARD_STEP = 0.3
APPROACH_SPEED = 0.5
CLOSE_HEIGHT_THRESH = 120
CENTER_TOLERANCE = 100
FINAL_DRIVE_DURATION = 3.0

SAVE_DIR = "maps"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────
def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))

def rotate_frame(frame, angle):
    return ndimage.rotate(frame, angle, reshape=False, order=1, mode='reflect')

def detect_colors_bgr(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0,100,100), (10,255,255))
    red2 = cv2.inRange(hsv, (170,100,100), (180,255,255))
    mask_red = cv2.bitwise_or(red1, red2)
    mask_blue = cv2.inRange(hsv, (100,150,80), (130,255,255))

    cv2.imshow("Red Mask", mask_red)
    cv2.imshow("Blue Mask", mask_blue)

    out = []
    for name, mask in (("red", mask_red), ("blue", mask_blue)):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        print(f"{name} contour area: {area}")
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append((name, x, y, w, h))
    return sorted(out, key=lambda b: b[1])

def estimate_distance_mm(h_px):
    return (F_MM * H_MM) / (h_px * P_MM)

def estimate_bearing_rad(cx):
    offset = cx - FRAME_W / 2
    deg = offset / (FRAME_W / 2) * (H_FOV_DEG / 2)
    return np.deg2rad(deg)

def plot_and_save_map(map_pts):
    fig, ax = plt.subplots()
    ax.scatter(0, 0, marker="^", color="k", label="Robot (0,0)")
    for name, X, Y in map_pts:
        ax.scatter(X, Y, color=name, s=50, label=f"{name}")
        ax.text(X, Y, f"{X:.2f},{Y:.2f}", fontsize=8, ha="right", va="bottom")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Top‐down Object Map")
    ax.axis("equal")
    ax.legend()
    plt.savefig(f"{SAVE_DIR}/object_map.png")
    plt.show()

# ── Main ─────────────────────────────────────────────────────────────
def main():
    px = Picarx()
    px.stop()

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}))
    cam.start()
    time.sleep(2.0)

    px.set_cam_pan_angle(0)
    px.set_cam_tilt_angle(0)
    time.sleep(0.2)

    print("🔍 Looking forward to find objects...")
    frame = cam.capture_array()
    frame = rotate_frame(frame, ROT_ANGLE)
    dets = detect_colors_bgr(frame)

    if len(dets) >= 2:
        print("🎯 Found 2 objects. Saving map...")

        map_pts = []
        for name, x, y, w, h in dets:
            Z_mm = estimate_distance_mm(h)
            theta = estimate_bearing_rad(x + w / 2)
            X = (Z_mm * np.sin(theta)) / 1000.0
            Y = (Z_mm * np.cos(theta)) / 1000.0
            map_pts.append((name, X, Y))
        plot_and_save_map(map_pts)
    else:
        print("❌ Could not detect 2 objects. Aborting.")
        cam.stop()
        return

    print("🚗 Starting careful approach loop...")
    state = "APPROACH_LOOP"

    while True:
        frame = cam.capture_array()
        frame = rotate_frame(frame, ROT_ANGLE)
        dets = detect_colors_bgr(frame)

        cv2.imshow("View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(dets) < 2:
            print("🔎 Lost objects. Sweeping...")
            state = "SWEEP"

        if state == "SWEEP":
            found = False
            for pan in SWEEP_PANS:
                px.set_cam_pan_angle(pan)
                time.sleep(SWEEP_DELAY)
                frame = cam.capture_array()
                frame = rotate_frame(frame, ROT_ANGLE)
                dets = detect_colors_bgr(frame)
                if len(dets) >= 2:
                    print("✅ Reacquired targets.")
                    state = "APPROACH_LOOP"
                    found = True
                    break
            if not found:
                continue  # go back to top of loop

        if state == "APPROACH_LOOP":
            _, x1, y1, w1, h1 = dets[0]
            _, x2, y2, w2, h2 = dets[1]

            cx1 = x1 + w1 / 2
            cx2 = x2 + w2 / 2
            cx_mid = (cx1 + cx2) / 2
            pan_err = cx_mid - FRAME_W / 2
            pan = clamp(pan_err * PAN_GAIN, -PAN_LIM, PAN_LIM)

            px.set_cam_pan_angle(pan)
            px.set_dir_servo_angle(pan)
            time.sleep(0.3)

            px.forward(APPROACH_SPEED)
            time.sleep(FORWARD_STEP)
            px.forward(0)

            px.set_cam_pan_angle(0)
            px.set_dir_servo_angle(0)
            time.sleep(0.2)

            # Are we close and well centered?
            centered = abs(pan_err) < CENTER_TOLERANCE
            close_enough = h1 >= CLOSE_HEIGHT_THRESH or h2 >= CLOSE_HEIGHT_THRESH

            if centered and close_enough:
                print("✅ Close & centered → final forward drive.")
                px.forward(APPROACH_SPEED)
                time.sleep(FINAL_DRIVE_DURATION)
                px.forward(0)
                print("🏁 Finished passing between objects.")
                break  # exit the loop immediately, do not change state


    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()