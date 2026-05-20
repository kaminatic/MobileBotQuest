import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from picamera2 import Picamera2
from picarx import Picarx
import time

# === CONFIG ===
homography_path = "homography_og.npy"
mask_output_path = "mask_output.png"
debug_output_path = "detected_overlay.png"
min_area = 1000
exclude_top_ratio = 0.2
pixel_to_mm = 1.0

# === Initialize Camera ===
cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"format": "BGR888"}))
cam.start()
time.sleep(0.5)

# === Initialize Robot ===
px = Picarx()
px.forward(0)
px.stop()

# === Capture background ===
input("Place empty scene and press ENTER to capture background...")
bg_array = cam.capture_array()
background = cv2.cvtColor(bg_array, cv2.COLOR_RGB2BGR)
cv2.imwrite("background_robot.jpg", background)

# === Capture scene with objects ===
input("Place objects and press ENTER to capture scene...")
scene_array = cam.capture_array()
scene = cv2.cvtColor(scene_array, cv2.COLOR_RGB2BGR)
cv2.imwrite("scene_robot.jpg", scene)

# === Load homography ===
H_mm_to_px = np.load(homography_path)
H_px_to_mm = np.linalg.inv(H_mm_to_px)

# === SSIM Difference ===
gray_bg = cv2.GaussianBlur(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY), (7, 7), 0)
gray_scene = cv2.GaussianBlur(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), (7, 7), 0)
_, diff = ssim(gray_scene, gray_bg, full=True)
diff = (1 - diff) * 255
diff = diff.astype(np.uint8)

# === Threshold and morphology ===
_, mask = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)
kernel = np.ones((7, 7), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# === Exclude top region ===
height, width = mask.shape
cutoff_y = int(height * exclude_top_ratio)
cv2.rectangle(mask, (0, 0), (width, cutoff_y), 0, thickness=cv2.FILLED)
cv2.imwrite(mask_output_path, mask)

# === Contour detection and filtering ===
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

# === Analyze objects ===
overlay = scene.copy()
object_coords = []
for i, cnt in enumerate(contours):
    bottom_point = tuple(cnt[cnt[:, :, 1].argmax()][0])
    cv2.circle(overlay, bottom_point, 6, (0, 0, 255), -1)
    cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
    cv2.putText(overlay, f"Obj {i+1}", (bottom_point[0] + 10, bottom_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    px_point = np.array([[bottom_point[0]], [bottom_point[1]], [1]], dtype=np.float32)
    world_pt = np.dot(H_px_to_mm, px_point)
    world_pt /= world_pt[2]
    x_mm = world_pt[0][0] * pixel_to_mm
    y_mm = world_pt[1][0] * pixel_to_mm
    object_coords.append((x_mm, y_mm))

    print(f"[INFO] Object {i+1} world position: ({x_mm:.2f}, {y_mm:.2f}) mm")

cv2.imwrite(debug_output_path, overlay)

# === Require two objects ===
if len(object_coords) < 2:
    raise ValueError("[ERROR] Less than two objects detected. Cannot compute path.")

# === Triangle Construction ===
A, B = np.array(object_coords)
midpoint = (A + B) / 2
base_vec = B - A
side_length = np.linalg.norm(base_vec)
h = (np.sqrt(3) / 2) * side_length
perp_vec = np.array([-base_vec[1], base_vec[0]])
perp_unit = perp_vec / np.linalg.norm(perp_vec)
C1 = midpoint + h * perp_unit
C2 = midpoint - h * perp_unit
apex = C1 if np.linalg.norm(C1) < np.linalg.norm(C2) else C2

# === Save clean object map ===
plt.figure(figsize=(8, 8))
plt.title("Reference Map with Objects")
plt.scatter(0, 0, c='red', label='Camera (Start)')
plt.scatter(*A + 120, c='black', label='Object 1')
plt.scatter(*B + 120, c='black', label='Object 2')
plt.plot([0, 500], [0, 0], 'k:', label='500 mm ref line')
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("clean_object_map.png", dpi=300)
plt.show()

# === Triangle Route Map ===
plt.figure(figsize=(8, 8))
plt.title("Mapped Route with Triangle Apex")
plt.scatter(0, 0, c='red', label='Camera (Start)')
plt.scatter(*A, c='black', label='Object 1')
plt.scatter(*B, c='black', label='Object 2')
plt.scatter(*midpoint, c='blue', label='Midpoint')
plt.scatter(*apex, c='orange', label='Triangle Apex')
plt.plot([A[0], B[0]], [A[1], B[1]], 'k--', label='Base')
plt.plot([A[0], apex[0]], [A[1], apex[1]], 'gray', linestyle='dotted')
plt.plot([B[0], apex[0]], [B[1], apex[1]], 'gray', linestyle='dotted')
plt.plot([0, apex[0]], [0, apex[1]], 'm-.', label='Path to Apex')
plt.plot([apex[0], midpoint[0]], [apex[1], midpoint[1]], 'g--', label='Drive to Midpoint')
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mapped_triangle_route.png", dpi=300)
plt.show()

# === Drive to Apex ===
arc_displacement = np.array([120.0, 90.0])
vec_to_apex = apex
arc_radius = np.linalg.norm(arc_displacement)
arc_steps = int(np.round(np.linalg.norm(vec_to_apex) / arc_radius))
angle_to_apex = np.degrees(np.arctan2(vec_to_apex[1], vec_to_apex[0]))
steer_dir = 'left' if angle_to_apex > 0 else 'right'
arc_delta = arc_displacement if steer_dir == 'left' else np.array([-120.0, 90.0])

print(f"[INFO] Driving to apex ({steer_dir}) for {arc_steps} step(s)...")
for _ in range(arc_steps):
    px.set_dir_servo_angle(- 1.5 if steer_dir == 'left' else 1.5)
    px.forward(0.2)
    time.sleep(0.5)
    px.stop()
    time.sleep(0.2)

# === Align to midpoint and drive through with actual turn then drive ===
vec_to_mid = midpoint - apex
angle_to_mid = np.degrees(np.arctan2(vec_to_mid[1], vec_to_mid[0]))
steer_to_mid = np.clip(angle_to_mid, -90.0, 90.0)
#drive_duration = np.linalg.norm(vec_to_mid) / 100.0  # calibrated speed

print(f"vec_to_mid = {vec_to_mid}")
print(f"angle_to_mid = {angle_to_mid:.2f}°")

print(f"[INFO] Turning toward midpoint with steering {steer_to_mid:.2f} (scaled to degrees)")
px.set_dir_servo_angle(steer_to_mid)
px.forward(0.3)
time.sleep(1.0)
px.stop()
time.sleep(0.5)
px.set_dir_servo_angle(0)

print(f"[INFO] Driving forward for {6.0:.2f} seconds...")
px.forward(1.0)
time.sleep(6.0)
px.stop()
print("[INFO] Autonomous navigation complete.")