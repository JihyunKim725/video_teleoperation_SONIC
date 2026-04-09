"""
Video Teleoperation for G1 Robot — Protocol v1 (Joint-Based)
Intel RealSense D435f → MediaPipe Pose → G1 29-DOF → SONIC Packed Binary → ZMQ

사용법:
  python3 video_teleop_v1.py
  python3 video_teleop_v1.py --no-show
"""
import argparse
import json
import time

import cv2
import numpy as np
import zmq
import mediapipe as mp
import pyrealsense2 as rs

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SONIC Packed Binary Format
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HEADER_SIZE = 1280

def build_packed_message(topic, version, fields):
    field_meta = []
    binary_parts = []
    for name, array, dtype_str in fields:
        field_meta.append({"name": name, "dtype": dtype_str, "shape": list(array.shape)})
        binary_parts.append(array.tobytes())

    header = {"v": version, "endian": "le", "count": 1, "fields": field_meta}
    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    if len(header_bytes) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_bytes)} > {HEADER_SIZE}")
    header_padded = header_bytes + b'\x00' * (HEADER_SIZE - len(header_bytes))
    return topic.encode('utf-8') + header_padded + b''.join(binary_parts)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# G1 관절 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

JOINT_LIMITS_LOW = np.array([
    -2.53, -0.52, -2.76, -0.09, -0.87, -0.26,
    -2.53, -2.97, -2.76, -0.09, -0.87, -0.26,
    -2.62, -0.52, -0.52,
    -3.09, -1.59, -2.62, -1.05, -1.97, -1.61, -1.61,
    -3.09, -2.25, -2.62, -1.05, -1.97, -1.61, -1.61,
], dtype=np.float32)

JOINT_LIMITS_HIGH = np.array([
    2.88,  2.97,  2.76,  2.88,  0.52,  0.26,
    2.88,  0.52,  2.76,  2.88,  0.52,  0.26,
    2.62,  0.52,  0.52,
    2.67,  2.25,  2.62,  2.09,  1.97,  1.61,  1.61,
    2.67,  1.59,  2.62,  2.09,  1.97,  1.61,  1.61,
], dtype=np.float32)

DEFAULT_POSE = np.array([
    0.0, 0.0, -0.1, 0.3, -0.2, 0.0,
    0.0, 0.0, -0.1, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, -0.3, 0.0, 1.0, 0.0, 0.0, 0.0,
], dtype=np.float32)

ALPHA = 0.5

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MediaPipe 유틸리티
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def get_lm(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

def angle_between(v1, v2):
    v1n = v1 / (np.linalg.norm(v1) + 1e-8)
    v2n = v2 / (np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(np.dot(v1n, v2n), -1.0, 1.0))

def retarget(landmarks):
    ls = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    rs_ = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    le = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
    re = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
    lw = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
    rw = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
    lh = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    rh = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
    lk = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
    rk = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
    la = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
    ra = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)

    l_upper = le - ls;  l_fore = lw - le
    l_sh_pitch = -np.arctan2(-l_upper[2], l_upper[1])
    l_sh_roll = np.arctan2(-l_upper[0], l_upper[1])
    l_sh_yaw = np.arctan2(l_fore[0], -l_fore[2] + 1e-8) * 0.3
    l_elbow = np.clip(np.pi - angle_between(l_upper, l_fore), 0.0, 2.09)

    r_upper = re - rs_;  r_fore = rw - re
    r_sh_pitch = -np.arctan2(-r_upper[2], r_upper[1])
    r_sh_roll = -np.arctan2(r_upper[0], r_upper[1])
    r_sh_yaw = -np.arctan2(r_fore[0], -r_fore[2] + 1e-8) * 0.3
    r_elbow = np.clip(np.pi - angle_between(r_upper, r_fore), 0.0, 2.09)

    sv = rs_ - ls;  hv = rh - lh
    waist_yaw = np.arctan2(sv[2] - hv[2], sv[0] - hv[0]) * 0.3

    l_knee_angle = angle_between(lk - lh, la - lk) * 0.4
    r_knee_angle = angle_between(rk - rh, ra - rk) * 0.4
    l_hip_pitch = np.arctan2(-lk[2] + lh[2], lk[1] - lh[1]) * 0.3
    r_hip_pitch = np.arctan2(-rk[2] + rh[2], rk[1] - rh[1]) * 0.3

    return np.array([
        0.0, 0.0, np.clip(l_hip_pitch, -2.76, 2.76),
        np.clip(l_knee_angle, 0.0, 2.88), -l_knee_angle * 0.3, 0.0,
        0.0, 0.0, np.clip(r_hip_pitch, -2.76, 2.76),
        np.clip(r_knee_angle, 0.0, 2.88), -r_knee_angle * 0.3, 0.0,
        waist_yaw, 0.0, 0.0,
        l_sh_pitch, l_sh_roll, l_sh_yaw, l_elbow, 0.0, 0.0, 0.0,
        r_sh_pitch, r_sh_roll, r_sh_yaw, r_elbow, 0.0, 0.0, 0.0,
    ], dtype=np.float32)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--topic", type=str, default="pose")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    # ZMQ
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"✅ ZMQ Publisher: tcp://*:{args.port} (topic: {args.topic})")

    # MediaPipe
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # RealSense D435f
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(rs_config)
    print("✅ RealSense D435f 시작 (640x480 @ 30fps)")
    print("   q: 종료 | Terminal 2: ] → 9 → ENTER 또는 # (ZMQ 전환)")

    prev = DEFAULT_POSE.copy()
    frame_idx = 0
    t_start = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                if not args.no_show:
                    mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                raw = retarget(result.pose_landmarks.landmark)
                raw = np.clip(raw, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
                smoothed = ALPHA * raw + (1.0 - ALPHA) * prev
                prev = smoothed.copy()

                fields = [
                    ("joint_pos",   smoothed.reshape(1, 29).astype(np.float32), "f32"),
                    ("joint_vel",   np.zeros((1, 29), dtype=np.float32),        "f32"),
                    ("body_quat",   np.array([[1,0,0,0]], dtype=np.float32),    "f32"),
                    ("frame_index", np.array([frame_idx], dtype=np.int32),      "i32"),
                ]
                sock.send(build_packed_message(args.topic, version=1, fields=fields))

                if not args.no_show:
                    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, f"L_sh: {np.degrees(smoothed[15]):.1f}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(frame, f"R_sh: {np.degrees(smoothed[22]):.1f}", (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"  Frame {frame_idx} | FPS: {frame_idx/(time.time()-t_start):.1f}")
            else:
                if not args.no_show:
                    cv2.putText(frame, "No pose detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if not args.no_show:
                cv2.imshow("G1 Video Teleop (v1)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        sock.close()
        ctx.term()
        print("✅ 종료")

if __name__ == "__main__":
    main()
