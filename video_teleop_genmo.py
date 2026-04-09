"""
video_teleop_genmo.py
GENMO(GEM-SMPL) + Intel RealSense D435F → G1 Video Teleoperation

★ v3 프로토콜 사용 ★
  v3 = Joint + SMPL Combined (Encode Mode 2)
  SMPL fields가 primary motion data, wrist joints만 joint_pos에서 사용

파이프라인:
  RealSense D435F (640×480@30fps)
      ↓ BGR 프레임
  GENMO (diffusion, sliding window)
      ↓ SMPL: global_orient(3) + body_pose(63) + transl(3) + smpl_joints(24,3)
  SMPL → G1 리타겟팅 (v3: wrist만)
      ↓ joint_pos(29) + joint_vel(29) + smpl_joints(24,3) + smpl_pose(21,3) + body_quat(4)
  ZMQ Publisher — SONIC Packed Binary Format (v3)
      ↓ [topic][1280-byte JSON header null-padded][binary data]
  g1_deploy (--input-type zmq)
      ↓
  MuJoCo / 실제 G1 로봇

사용법:
  python3 video_teleop_genmo.py
  python3 video_teleop_genmo.py --ckpt ~/GENMO/inputs/pretrained/gem_smpl.ckpt
  python3 video_teleop_genmo.py --scale 0.6 --alpha 0.7 --no-show
"""

import argparse
import json
import time
from pathlib import Path

import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np
import zmq
import pyrealsense2 as rs

from smpl_to_g1_retarget import smpl_to_g1_v3
from genmo_streaming_inference import GENMOStreamingInference


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SONIC Packed Binary Format — v3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v3 Required fields:
#   joint_pos:   [N, 29]     f32   (wrist만 meaningful)
#   joint_vel:   [N, 29]     f32   (전부 0 가능)
#   smpl_joints: [N, 24, 3]  f32   (SMPL FK joint positions)
#   smpl_pose:   [N, 21, 3]  f32   (SMPL body pose, axis-angle)
# Common fields:
#   body_quat:   [N, 1, 4]   f32   (w, x, y, z)
#   frame_index: [N]          i32   (monotonically increasing)
#
# 출처: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/zmq.html
#       #protocol-v3-joint-smpl-combined-encode-mode-2

HEADER_SIZE = 1280


def build_packed_message(topic, version, fields):
    """
    SONIC packed binary message 생성

    Args:
        topic:   str — ZMQ 토픽 (예: "pose")
        version: int — 프로토콜 버전 (3)
        fields:  list of (name, np_array, dtype_str)
    Returns:
        bytes — topic + 1280-byte header + binary data
    """
    field_meta = []
    binary_parts = []
    for name, array, dtype_str in fields:
        field_meta.append({
            "name": name,
            "dtype": dtype_str,
            "shape": list(array.shape),
        })
        binary_parts.append(array.tobytes())

    header_json = json.dumps({
        "v": version,
        "endian": "le",
        "count": 1,
        "fields": field_meta,
    }, separators=(',', ':'))

    header_bytes = header_json.encode('utf-8')
    if len(header_bytes) > HEADER_SIZE:
        raise ValueError(f"Header overflow: {len(header_bytes)} > {HEADER_SIZE}")

    header_padded = header_bytes + b'\x00' * (HEADER_SIZE - len(header_bytes))

    return topic.encode('utf-8') + header_padded + b''.join(binary_parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 기본 자세 (GENMO 결과 대기 중 사용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_JP = np.zeros(29, dtype=np.float32)
DEFAULT_JV = np.zeros(29, dtype=np.float32)
DEFAULT_SP = np.zeros((21, 3), dtype=np.float32)
DEFAULT_SJ = np.zeros((24, 3), dtype=np.float32)
DEFAULT_BQ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RealSense D435F 카메라
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RealSenseCamera:
    """Intel RealSense D435F 래퍼"""

    def __init__(self, width=640, height=480, fps=30, warmup=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)
        self.width = width
        self.height = height
        self.fps = fps

        print(f"   워밍업 {warmup}프레임 ...", end="", flush=True)
        for _ in range(warmup):
            self.pipeline.wait_for_frames()
        print(" 완료")

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return False, None
        return True, np.asanyarray(color.get_data()).copy()

    def release(self):
        self.pipeline.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="GENMO + RealSense D435F → G1 Video Teleoperation (v3)")
    parser.add_argument("--ckpt", type=str,
                        default="~/GENMO/inputs/pretrained/gem_smpl.ckpt")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--topic", type=str, default="pose")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    ckpt = str(Path(args.ckpt).expanduser())

    if not args.no_show:
        cv2.namedWindow("GENMO Teleop (G1 v3) — RealSense D435F", cv2.WINDOW_NORMAL)
        cv2.waitKey(1)

    # ── ZMQ Publisher ─────────────────────────────────
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"✅ ZMQ Publisher: tcp://*:{args.port}  topic={args.topic}  protocol=v3")

    # ── GENMO 모델 ────────────────────────────────────
    genmo = GENMOStreamingInference(
        ckpt_path=ckpt,
        window_size=args.window,
        stride=args.stride,
    )
    
    # T-pose joints를 기본값으로 설정 (zero joints 대신)
    if genmo.smpl_fk is not None and hasattr (genmo, 'tpose_joints'):
        DEFAULT_SJ[:] = genmo.tpose_joints
        print(f"✅ T-pose joints 로드됨: pelvis_y={DEFAULT_SJ[0,1]:.3f}m")

    # ── RealSense D435F ───────────────────────────────
    print("✅ RealSense D435F 초기화 중...")
    cam = RealSenseCamera(width=640, height=480, fps=30)
    print(f"✅ RealSense D435F 시작 (640×480 @ 30fps)")

    # ── 상태 ──────────────────────────────────────────
    prev_jp = DEFAULT_JP.copy()
    prev_jv = DEFAULT_JV.copy()
    prev_sp = DEFAULT_SP.copy()
    prev_sj = DEFAULT_SJ.copy()
    prev_bq = DEFAULT_BQ.copy()
    fidx = 0
    sent = 0
    genmo_cnt = 0
    t0 = time.time()
    queue = []

    print()
    print("━" * 60)
    print("  GENMO Video Teleoperation for G1 — Protocol v3")
    print("━" * 60)
    print(f"  카메라:     Intel RealSense D435F (640×480@30fps)")
    print(f"  모델:       GENMO (GEM-SMPL, diffusion)")
    print(f"  프로토콜:   v3 (Joint + SMPL Combined, encode_mode=2)")
    print(f"  윈도우:     {args.window}프레임 ({args.window/30:.1f}초)")
    print(f"  스트라이드: {args.stride}프레임 ({args.stride/30:.1f}초)")
    print(f"  스케일:     {args.scale}  스무딩: α={args.alpha}")
    print(f"  ZMQ:        tcp://*:{args.port} topic={args.topic}")
    print()
    print("  q: 종료")
    print("━" * 60)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            # ── GENMO에 프레임 전달 ───────────────────
            new_results = genmo.process_frame(frame)
            queue.extend(new_results)

            # ── SMPL → G1 리타겟팅 (v3) ──────────────
            if queue:
                d = queue.pop(0)
                jp, jv, sp, sj, bq = smpl_to_g1_v3(
                    body_pose=d['body_pose'],
                    global_orient=d['global_orient'],
                    smpl_joints=d['smpl_joints'],
                    scale=args.scale,
                )
                # EMA 스무딩
                jp = args.alpha * jp + (1 - args.alpha) * prev_jp
                jv = args.alpha * jv + (1 - args.alpha) * prev_jv
                sp = args.alpha * sp + (1 - args.alpha) * prev_sp
                sj = args.alpha * sj + (1 - args.alpha) * prev_sj
                bq = args.alpha * bq + (1 - args.alpha) * prev_bq
                # quaternion 정규화
                norm = np.linalg.norm(bq)
                if norm > 1e-8:
                    bq /= norm
                prev_jp = jp.copy()
                prev_jv = jv.copy()
                prev_sp = sp.copy()
                prev_sj = sj.copy()
                prev_bq = bq.copy()
                genmo_cnt += 1
                status, color = "GENMO v3", (0, 255, 0)
            else:
                jp = prev_jp.copy()
                jv = prev_jv.copy()
                sp = prev_sp.copy()
                sj = prev_sj.copy()
                bq = prev_bq.copy()
                status = f"대기중 ({genmo.frame_count}/{args.window})"
                color = (0, 165, 255)

            # ── ZMQ 전송 (SONIC Packed Binary — v3) ──
            fields = [
                ("joint_pos",   jp.reshape(1, 29).astype(np.float32),      "f32"),
                ("joint_vel",   jv.reshape(1, 29).astype(np.float32),      "f32"),
                ("smpl_joints", sj.reshape(1, 24, 3).astype(np.float32),   "f32"),
                ("smpl_pose",   sp.reshape(1, 21, 3).astype(np.float32),   "f32"),
                ("body_quat",   bq.reshape(1, 1, 4).astype(np.float32),    "f32"),
                ("frame_index", np.array([fidx], dtype=np.int32),           "i32"),
            ]
            msg = build_packed_message(args.topic, version=3, fields=fields)
            sock.send(msg)
            sent += 1

            # ── 화면 표시 ────────────────────────────
            if not args.no_show:
                elapsed = time.time() - t0
                fps = sent / max(elapsed, 0.001)

                with genmo._bbox_lock:
                    bbox = genmo.last_bbox
                if bbox is not None:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "YOLO", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                cv2.putText(frame, f"[{status}] F:{fidx} FPS:{fps:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(frame, f"Q:{len(queue)} GENMO:{genmo_cnt} v3",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.putText(frame,
                            f"Lwr:{np.degrees(jp[19]):.0f}/{np.degrees(jp[20]):.0f}/{np.degrees(jp[21]):.0f} "
                            f"Rwr:{np.degrees(jp[26]):.0f}/{np.degrees(jp[27]):.0f}/{np.degrees(jp[28]):.0f}",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(frame,
                            f"bq:[{bq[0]:.2f},{bq[1]:.2f},{bq[2]:.2f},{bq[3]:.2f}]",
                            (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
                cv2.putText(frame,
                            f"sj[0]:{sj[0,0]:.3f},{sj[0,1]:.3f},{sj[0,2]:.3f} "
                            f"msg:{len(msg)}B",
                            (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

                cv2.imshow("GENMO Teleop (G1 v3) — RealSense D435F", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            fidx += 1
            if fidx % 100 == 0:
                elapsed = time.time() - t0
                print(f"  F:{fidx:5d} | Sent:{sent:5d} | GENMO:{genmo_cnt:4d} | "
                      f"FPS:{sent/elapsed:.1f} | Q:{len(queue)} | "
                      f"msg:{len(msg)}B | v3 | "
                      f"Lwr:{np.degrees(jp[19]):.0f}/{np.degrees(jp[20]):.0f} "
                      f"Rwr:{np.degrees(jp[26]):.0f}/{np.degrees(jp[27]):.0f} "
                      f"bq_w:{bq[0]:.3f} "
                      f"sj0:{sj[0,0]:.2f},{sj[0,1]:.2f},{sj[0,2]:.2f}")

    except KeyboardInterrupt:
        print("\n⚠️ Ctrl+C")
    finally:
        elapsed = time.time() - t0
        print()
        print("━" * 60)
        print(f"  프레임: {fidx}  전송: {sent}  GENMO: {genmo_cnt}")
        print(f"  평균FPS: {sent/max(elapsed,.001):.1f}  시간: {elapsed:.1f}s")
        print(f"  프로토콜: v3 (encode_mode=2)")
        print("━" * 60)
        cam.release()
        cv2.destroyAllWindows()
        sock.close()
        ctx.term()
        print("✅ 종료")


if __name__ == "__main__":
    main()
