import cv2
import numpy as np
import math
import subprocess
from pathlib import Path

# Resolution matching 16:9 or reference aspect ratio ~ 1024x532 or 1280x670
W, H = 1024, 532
FPS = 30
DURATION_SEC = 6
TOTAL_FRAMES = FPS * DURATION_SEC

# Retro Phosphor Colors
BG_COLOR = (8, 14, 8)         # Deep CRT Black/Green
GRID_COLOR = (0, 195, 30)      # Bright CRT Phosphor Line
BRIGHT_GREEN = (0, 245, 45)    # Highlight Phosphor
DIM_GREEN = (0, 120, 20)       # Shadow Phosphor
TEXT_COLOR = (0, 235, 40)

def draw_isometric_bars(img, ox, oy, t, is_receiving=False, width=420, height=200):
    # 3D Isometric grid of signal plates/fins
    # Rows and cols
    rows, cols = 8, 12
    spacing_x = 18
    spacing_y = 10
    fin_w = 12
    fin_h = 16

    # Center offset
    start_x = ox + (width - (cols * spacing_x + rows * spacing_y)) // 2 + 30
    start_y = oy + (height - (rows * spacing_y)) // 2 - 10

    direction = -1 if is_receiving else 1

    for r in range(rows):
        for c in range(cols):
            # Isometric projection coordinates:
            # x' = start_x + (c * spacing_x) - (r * spacing_y * 1.2)
            # y' = start_y + (r * spacing_y * 0.8) + (c * spacing_y * 0.3)
            iso_x = int(start_x + c * 22 - r * 14)
            iso_y = int(start_y + r * 14 + c * 5)

            # Wave motion
            wave_phase = (c * 0.5 + r * 0.4) * direction + t * 4.0
            wave_h = math.sin(wave_phase) * 10 + math.cos(wave_phase * 0.7) * 4

            h_fin = max(4, int(fin_h + wave_h))
            alpha = 0.5 + 0.5 * math.sin(wave_phase)

            color = (
                int(DIM_GREEN[0] + (BRIGHT_GREEN[0] - DIM_GREEN[0]) * alpha),
                int(DIM_GREEN[1] + (BRIGHT_GREEN[1] - DIM_GREEN[1]) * alpha),
                int(DIM_GREEN[2] + (BRIGHT_GREEN[2] - DIM_GREEN[2]) * alpha)
            )

            # Draw vertical fin (isometric parallelogram/rectangle)
            pt1 = (iso_x, iso_y - h_fin)
            pt2 = (iso_x + fin_w, iso_y - h_fin - 3)
            pt3 = (iso_x + fin_w, iso_y - 3)
            pt4 = (iso_x, iso_y)

            pts = np.array([pt1, pt2, pt3, pt4], np.int32)
            cv2.fillPoly(img, [pts], color)
            cv2.polylines(img, [pts], True, BRIGHT_GREEN, 1, cv2.LINE_AA)

def draw_celestial_sphere(img, cx, cy, radius, t):
    # Rotating 3D wireframe sphere with equatorial rings and 3-axis reticles
    # Outer circle
    cv2.circle(img, (cx, cy), radius, BRIGHT_GREEN, 2, cv2.LINE_AA)

    # 1. Rotating equator ellipse
    # Pitch/roll angle modulates with t
    pitch = math.sin(t * 1.5) * 0.4 + 0.35
    axes_eq = (radius, max(4, int(radius * math.cos(pitch))))
    cv2.ellipse(img, (cx, cy), axes_eq, int(math.degrees(t * 0.4)), 0, 360, BRIGHT_GREEN, 1, cv2.LINE_AA)

    # 2. Polar ring ellipse
    roll = math.cos(t * 1.5) * 0.4 + 0.35
    axes_pol = (max(4, int(radius * math.sin(roll))), radius)
    cv2.ellipse(img, (cx, cy), axes_pol, int(math.degrees(t * 0.6)), 0, 360, DIM_GREEN, 1, cv2.LINE_AA)

    # 3. Rotating 3-axis tri-blade reticle (120 degree offsets)
    angle_base = t * 2.0
    for i in range(3):
        a = angle_base + i * (2 * math.pi / 3)
        # Project 3D vector onto 2D
        vx = math.cos(a) * radius * 0.95
        vy = math.sin(a) * radius * 0.95 * math.cos(pitch)
        ex = int(cx + vx)
        ey = int(cy + vy)
        cv2.line(img, (cx, cy), (ex, ey), BRIGHT_GREEN, 2, cv2.LINE_AA)

    # Inner concentric ring
    cv2.circle(img, (cx, cy), int(radius * 0.88), DIM_GREEN, 1, cv2.LINE_AA)

def render_frame(frame_idx, total_frames):
    t = (frame_idx / total_frames) * (2 * math.pi)
    time_sec = frame_idx / FPS

    # Canvas
    img = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)

    # Mid lines (Divider Grid)
    mid_x = W // 2
    mid_y = H // 2

    cv2.line(img, (0, mid_y), (W, mid_y), GRID_COLOR, 2)
    cv2.line(img, (mid_x, 0), (mid_x, H), GRID_COLOR, 2)

    # Top border header
    cv2.putText(img, "CH.003-V2-CP-ROHIT // KOI MIL GAYA PROTOCOL", (mid_x + 50, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, DIM_GREEN, 1, cv2.LINE_AA)

    # 1. TOP-LEFT: SENDING
    cv2.putText(img, "SENDING", (45, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, TEXT_COLOR, 2, cv2.LINE_AA)
    draw_isometric_bars(img, 20, 60, t, is_receiving=False, width=mid_x - 40, height=mid_y - 80)

    # 2. TOP-RIGHT: RECEIVING
    cv2.putText(img, "RECEIVING", (mid_x + 45, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, TEXT_COLOR, 2, cv2.LINE_AA)
    draw_isometric_bars(img, mid_x + 20, 60, t, is_receiving=True, width=mid_x - 40, height=mid_y - 80)

    # 3. BOTTOM-LEFT: CELESTIAL GYROSCOPE SPHERE
    draw_celestial_sphere(img, mid_x // 2, mid_y + (H - mid_y) // 2, radius=80, t=t)

    # 4. BOTTOM-RIGHT: TELEMETRY DATA STREAM
    start_tx = mid_x + 45
    start_ty = mid_y + 60
    line_h = 32

    # Dynamic values simulating the film's terminal
    axis_val = 40 + int(math.sin(t * 3) * 3)
    inc_val = 48 + int(math.cos(t * 2) * 2)
    dec_val = 57 + int(math.sin(t * 2.5) * 2)
    signal_val = 80 + int(math.sin(t * 5) * 6)
    uplink_state = "YES" if (int(t * 4) % 6 != 0) else "SYNC"

    reg_b = 66 + (int(t * 2) % 4)
    reg_c = 67 + (int(t * 3) % 4)
    reg_d = 68 + (int(t * 2.5) % 4)
    reg_e = 69 + (int(t * 3.5) % 4)
    reg_f = 70 + (int(t * 1.5) % 4)

    telemetry_left = [
        f"UPLINK = {uplink_state}",
        f"AXIS.  = {axis_val}",
        f"INC.   = {inc_val} POS",
        f"DEC    = {dec_val} NEG",
        f"SIGNAL = {signal_val} B",
    ]

    telemetry_right = [
        f"B:0{reg_b}",
        f"C:0{reg_c}",
        f"D:0{reg_d}",
        f"E:0{reg_e}",
        f"F:0{reg_f}",
    ]

    for idx, (tl, tr) in enumerate(zip(telemetry_left, telemetry_right)):
        y_pos = start_ty + idx * line_h
        cv2.putText(img, tl, (start_tx, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(img, tr, (mid_x + 360, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT_COLOR, 2, cv2.LINE_AA)

    # CRT Scanlines effect
    img[::3, :, :] = (img[::3, :, :].astype(np.float32) * 0.72).astype(np.uint8)

    return img

print(f"Generating {TOTAL_FRAMES} frames of Koi Mil Gaya Computer Terminal...")
raw_avi_path = Path(r"C:\Users\saket\.gemini\antigravity-ide\brain\f1d926f8-0a3b-4858-bc4b-540e53c67d07\scratch\koi_terminal_raw.avi")
out_mp4_path = Path(r"C:\Users\saket\OneDrive\Desktop\DEEPSHIELD-X\Frontend\public\koi_mil_gaya_terminal.mp4")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter(str(raw_avi_path), fourcc, FPS, (W, H))

for i in range(TOTAL_FRAMES):
    frame = render_frame(i, TOTAL_FRAMES)
    writer.write(frame)

writer.release()
print("Raw video written. Converting to optimized H.264 MP4 with ffmpeg...")

cmd = [
    "ffmpeg", "-y",
    "-i", str(raw_avi_path),
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    str(out_mp4_path)
]
res = subprocess.run(cmd, capture_output=True, text=True)
print("FFmpeg return code:", res.returncode)
if res.returncode == 0:
    print(f"Successfully generated: {out_mp4_path} ({out_mp4_path.stat().st_size / 1024:.1f} KB)")
else:
    print("FFmpeg stderr:", res.stderr)
