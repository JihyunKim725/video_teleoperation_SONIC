"""
genmo_streaming_inference.py
GENMO(GEM-SMPL) 슬라이딩 윈도우 추론 모듈 — v3 프로토콜 지원

변경사항 (v3):
  - smplx 라이브러리의 SMPL 모델을 별도 로드하여 FK 실행
  - _run_inference() 반환값에 'smpl_joints' (T, 24, 3), 'betas' (T, 10) 추가
  - SmplxLiteV437Coco17은 COCO-17 joints(17개)만 반환하므로,
    표준 SMPL 모델(24 joints)을 별도 로드하여 FK 수행

전처리 파이프라인: YOLO(bbox) → ViTPose(kp2d) → HMR2(imgfeat) → GEM-SMPL(SMPL)
"""

import builtins
import os
import sys
import threading
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch


class GENMOStreamingInference:
    def __init__(self, ckpt_path, window_size=30, stride=15, device="cuda:0"):
        """
        Args:
            ckpt_path:   gem_smpl.ckpt 경로
            window_size: 프레임 버퍼 크기 (기본 30 = 1초@30fps)
            stride:      추론 간격 (기본 15 = 0.5초)
            device:      GPU 디바이스
        """
        self.window_size = window_size
        self.stride = stride
        self.device = device
        self.frame_buffer = deque(maxlen=window_size)
        self.prev_overlap = None
        self.frame_count = 0

        # 백그라운드 추론용
        self._result_queue = deque()
        self._infer_lock = threading.Lock()
        self._infer_running = False

        # 최신 YOLO bbox (메인 루프에서 시각화용) — (x1,y1,x2,y2) or None
        self.last_bbox = None
        self._bbox_lock = threading.Lock()

        self._load_model(ckpt_path)

        # ── v3: 표준 SMPL 모델 로드 (24 joints FK용) ──
        self._load_smpl_fk()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # v3 추가: SMPL FK 모델 로딩 (표준 SMPL 우선)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _load_smpl_fk(self):
        """
        표준 SMPL 모델(SMPL_NEUTRAL.pkl)을 우선 로드하여 FK를 수행.
        SONIC encoder는 표준 SMPL(24 joints, body_pose=23×3=69)로 학습되었으므로,
        SMPL-X가 아닌 표준 SMPL을 사용해야 joint positions가 encoder 기대값과 일치.

        GENMO 출력: body_pose = 21×3 = 63D (SMPL-X 기반)
        표준 SMPL:  body_pose = 23×3 = 69D
        → GENMO 63D + 6D zero padding (hand joints) = SMPL 69D

        SMPL FK 출력: joints (T, 24, 3) — 바로 사용 가능 (매핑 불필요)
        """
        try:
            import smplx as smplx_lib
        except ImportError:
            print("[SMPL FK] ⚠️ smplx 라이브러리 미설치. pip install smplx 필요")
            self.smpl_fk = None
            self._use_smpl_standard = False
            return

        genmo_root = Path.home() / "GENMO"
        body_models_root = genmo_root / "inputs" / "checkpoints" / "body_models"

        # ── 1차: 표준 SMPL (SMPL_NEUTRAL.pkl) 탐색 ──
        smpl_dir = body_models_root / "smpl"
        smpl_file = smpl_dir / "SMPL_NEUTRAL.pkl"

        if smpl_file.exists():
            print(f"[SMPL FK] 표준 SMPL 모델 발견: {smpl_file}")
            try:
                self.smpl_fk = smplx_lib.create(
                    model_path=str(body_models_root),
                    model_type="smpl",
                    gender="neutral",
                    batch_size=1,
                ).to(self.device)
                self.smpl_fk.eval()
                self._use_smpl_standard = True

                # T-pose joints 캐시
                with torch.no_grad():
                    tpose_out = self.smpl_fk(
                        body_pose=torch.zeros(1, 69, device=self.device),
                        global_orient=torch.zeros(1, 3, device=self.device),
                        transl=torch.zeros(1, 3, device=self.device),
                        betas=torch.zeros(1, 10, device=self.device),
                    )
                    tpose_joints = tpose_out.joints[0, :24, :].cpu().numpy()
                    self.tpose_joints = tpose_joints.astype(np.float32)

                print(f"[SMPL FK] ✅ 표준 SMPL 모델 로드 성공 (24 joints, 69D body_pose)")
                print(f"  T-pose pelvis: ({tpose_joints[0,0]:.4f}, {tpose_joints[0,1]:.4f}, {tpose_joints[0,2]:.4f})")
                print(f"  T-pose head:   ({tpose_joints[15,0]:.4f}, {tpose_joints[15,1]:.4f}, {tpose_joints[15,2]:.4f})")
                return
            except Exception as e:
                print(f"[SMPL FK] ⚠️ 표준 SMPL 로드 실패: {e}")
                import traceback; traceback.print_exc()

        # ── 2차: SMPL-X fallback ──
        print("[SMPL FK] ⚠️ SMPL_NEUTRAL.pkl 없음 → SMPL-X fallback")
        smplx_dir = body_models_root / "smplx"
        if not smplx_dir.exists():
            print(f"[SMPL FK] ⚠️ SMPL-X 모델도 없음: {smplx_dir}")
            self.smpl_fk = None
            self._use_smpl_standard = False
            return

        try:
            self.smpl_fk = smplx_lib.create(
                model_path=str(body_models_root),
                model_type="smplx",
                gender="neutral",
                batch_size=1,
                use_face_contour=False,
                use_pca=False,
                flat_hand_mean=True,
            ).to(self.device)
            self.smpl_fk.eval()
            self._use_smpl_standard = False
            self._smplx_to_smpl24_idx = list(range(22)) + [22, 37]

            # T-pose joints 캐시 (SMPL-X)
            with torch.no_grad():
                T = 1
                tpose_out = self.smpl_fk(
                    body_pose=torch.zeros(T, 63, device=self.device),
                    global_orient=torch.zeros(T, 3, device=self.device),
                    transl=torch.zeros(T, 3, device=self.device),
                    betas=torch.zeros(T, 10, device=self.device),
                    left_hand_pose=torch.zeros(T, 45, device=self.device),
                    right_hand_pose=torch.zeros(T, 45, device=self.device),
                    jaw_pose=torch.zeros(T, 3, device=self.device),
                    leye_pose=torch.zeros(T, 3, device=self.device),
                    reye_pose=torch.zeros(T, 3, device=self.device),
                    expression=torch.zeros(T, 10, device=self.device),
                )
                tpose_joints = tpose_out.joints[0, self._smplx_to_smpl24_idx, :].cpu().numpy()
                self.tpose_joints = tpose_joints.astype(np.float32)

            print(f"[SMPL FK] ✅ SMPL-X fallback 로드 (⚠️ SONIC과 skeleton 불일치 가능)")
            print(f"  T-pose pelvis: ({tpose_joints[0,0]:.4f}, {tpose_joints[0,1]:.4f}, {tpose_joints[0,2]:.4f})")
        except Exception as e:
            print(f"[SMPL FK] ⚠️ SMPL-X 로드 실패: {e}")
            import traceback; traceback.print_exc()
            self.smpl_fk = None
            self._use_smpl_standard = False

    @torch.no_grad()
    def _compute_smpl_joints(self, body_pose, global_orient, transl, betas):
        """
        SMPL FK → 24 joint positions

        GENMO 출력: body_pose (T, 63) = SMPL-X 21 joints × 3
        표준 SMPL:  body_pose (T, 69) = SMPL 23 joints × 3
        → 63D 뒤에 6D zero padding 추가 (hand joints 2개 × 3)

        Returns:
            smpl_joints: (T, 24, 3) numpy float32
        """
        if self.smpl_fk is None:
            T = body_pose.shape[0]
            return np.zeros((T, 24, 3), dtype=np.float32)

        T = body_pose.shape[0]

        if self._use_smpl_standard:
            # 표준 SMPL: body_pose 63D → 69D (뒤에 hand 2×3=6D zero padding)
            bp_63 = body_pose.to(self.device)  # (T, 63)
            bp_69 = torch.cat([
                bp_63,
                torch.zeros(T, 6, device=self.device),  # L_hand(3) + R_hand(3)
            ], dim=1)  # (T, 69)

            output = self.smpl_fk(
                body_pose=bp_69,
                global_orient=global_orient.to(self.device),
                transl=transl.to(self.device),
                betas=betas[:, :10].to(self.device),
            )
            # 표준 SMPL: joints[:, :24] = 24 body joints
            smpl_24 = output.joints[:, :24, :]  # (T, 24, 3)
        else:
            # SMPL-X fallback
            output = self.smpl_fk(
                body_pose=body_pose.to(self.device),
                global_orient=global_orient.to(self.device),
                transl=transl.to(self.device),
                betas=betas[:, :10].to(self.device),
                left_hand_pose=torch.zeros(T, 45, device=self.device),
                right_hand_pose=torch.zeros(T, 45, device=self.device),
                jaw_pose=torch.zeros(T, 3, device=self.device),
                leye_pose=torch.zeros(T, 3, device=self.device),
                reye_pose=torch.zeros(T, 3, device=self.device),
                expression=torch.zeros(T, 10, device=self.device),
            )
            smpl_24 = output.joints[:, self._smplx_to_smpl24_idx, :]

        return smpl_24.cpu().numpy().astype(np.float32)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 모델 로딩
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _load_model(self, ckpt_path):
        genmo_root = Path.home() / "GENMO"
        scripts_dir = genmo_root / "scripts" / "demo"
        for p in [str(genmo_root), str(scripts_dir)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from omegaconf import OmegaConf
        import hydra
        from hydra import compose, initialize_config_dir
        from gem.utils.net_utils import load_pretrained_model

        OmegaConf.register_new_resolver("eval", builtins.eval, replace=True)

        config_dir = str(genmo_root / "configs")
        overrides = ["exp=gem_smpl", "ckpt_path=null", "video_name=demo"]

        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name="demo", overrides=overrides)

        _orig_cwd = os.getcwd()
        os.chdir(str(genmo_root))
        try:
            print("[GENMO] Instantiating GEM model ...")
            self.model = hydra.utils.instantiate(cfg.model, _recursive_=False)
            load_pretrained_model(self.model, ckpt_path)
            self.model.cuda().eval()
            print(f"✅ GENMO 로드: {ckpt_path}")
            print(f"   윈도우={self.window_size} 스트라이드={self.stride} 디바이스={self.device}")

            print("[GENMO] Loading YOLOv8 ...")
            from ultralytics import YOLO
            self.yolo = YOLO(str(genmo_root / "inputs/checkpoints/yolo/yolov8x.pt"))

            print("[GENMO] Loading ViTPose-H ...")
            from demo_utils import CocoPoseExtractor
            self.vitpose = CocoPoseExtractor(device=self.device)

            print("[GENMO] Loading HMR2 ...")
            from gem.utils.hmr2_extractor import HMR2FeatureExtractor
            from gem.network.hmr2.utils.preproc import IMAGE_MEAN, IMAGE_STD, crop_and_resize

            hmr2_ckpt = str(genmo_root / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt")
            self.hmr2 = HMR2FeatureExtractor(hmr2_ckpt, device=self.device)
            self._hmr2_mean = IMAGE_MEAN
            self._hmr2_std = IMAGE_STD
            self._crop_and_resize = crop_and_resize

            from gem.utils.cam_utils import estimate_K
            from gem.utils.geo_transform import compute_cam_angvel, get_bbx_xys_from_xyxy
            from gem.utils.net_utils import get_valid_mask, moving_average_smooth

            self._estimate_K = estimate_K
            self._compute_cam_angvel = compute_cam_angvel
            self._get_bbx_xys_from_xyxy = get_bbx_xys_from_xyxy
            self._get_valid_mask = get_valid_mask
            self._moving_average_smooth = moving_average_smooth
        finally:
            os.chdir(_orig_cwd)

        print("✅ 전처리 모델 모두 로드 완료")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 추론
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    @torch.no_grad()
    def _run_inference(self, frames):
        T = len(frames)
        H, W = frames[0].shape[:2]

        frames_rgb = np.stack([f[..., ::-1].copy() for f in frames])

        # ── YOLO ──
        bbx_xyxy_list = []
        yolo_detected = 0
        yolo_confs = []
        for frame_bgr in frames:
            results = self.yolo(frame_bgr, classes=[0], conf=0.3, verbose=False)
            best = None
            best_area = -1
            best_conf = 0.0
            for r in results:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    if area > best_area:
                        best_area = area
                        best = xyxy
                        best_conf = float(box.conf[0].cpu())
            if best is None:
                if bbx_xyxy_list:
                    best = bbx_xyxy_list[-1]
                else:
                    best = np.array([W / 4, H / 4, 3 * W / 4, 3 * H / 4], dtype=np.float32)
            else:
                yolo_detected += 1
                yolo_confs.append(best_conf)
                with self._bbox_lock:
                    self.last_bbox = best.copy()
            bbx_xyxy_list.append(best)

        avg_conf = float(np.mean(yolo_confs)) if yolo_confs else 0.0
        print(f"[GENMO] YOLO: {yolo_detected}/{T}프레임 감지, 평균conf={avg_conf:.2f}")

        if yolo_detected < T * 0.5:
            raise RuntimeError(f"YOLO 감지율 낮음 ({yolo_detected}/{T}) — 이 윈도우 스킵, 이전 결과 유지")

        bbx_xyxy = torch.tensor(np.stack(bbx_xyxy_list), dtype=torch.float32)
        bbx_xyxy[:, [0, 2]] = bbx_xyxy[:, [0, 2]].clamp(0, W - 1)
        bbx_xyxy[:, [1, 3]] = bbx_xyxy[:, [1, 3]].clamp(0, H - 1)
        bbx_xys = self._get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2)
        bbx_xys = self._moving_average_smooth(bbx_xys, window_size=5, dim=0)

        # ── ViTPose ──
        kp2d = self.vitpose.extract(frames_rgb, bbx_xys, batch_size=T)
        avg_kp_conf = float(kp2d[:, :, 2].mean())
        min_kp_conf = float(kp2d[:, :, 2].min())
        print(f"[GENMO] ViTPose: 평균conf={avg_kp_conf:.3f}, 최소conf={min_kp_conf:.3f}")

        # ── HMR2 ──
        f_imgseq = self._extract_hmr2_features(frames_rgb, bbx_xys)
        has_img_mask = torch.ones(T, dtype=torch.bool)

        # ── 카메라 ──
        R_w2c = torch.eye(3).unsqueeze(0).expand(T, -1, -1).clone()
        cam_angvel = self._compute_cam_angvel(R_w2c, padding_last=True)
        cam_tvel = torch.zeros(T, 3)
        K = self._estimate_K(W, H)
        K_fullimg = K.unsqueeze(0).expand(T, -1, -1).clone()

        has_2d_mask = torch.ones(T, dtype=torch.bool)
        has_cam_mask = torch.zeros(T, dtype=torch.bool)

        data = {
            "kp2d": kp2d,
            "bbx_xys": bbx_xys,
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "cam_tvel": cam_tvel,
            "R_w2c": R_w2c,
            "f_imgseq": f_imgseq,
            "has_text": torch.tensor([True]),
            "caption": "",
            "mask": {
                "has_img_mask": has_img_mask,
                "has_2d_mask": has_2d_mask,
                "has_cam_mask": has_cam_mask,
                "has_audio_mask": self._get_valid_mask(T, 0),
                "has_music_mask": self._get_valid_mask(T, 0),
            },
            "length": torch.tensor(T),
            "meta": [{"mode": "default"}],
        }

        # ── 추론 ──
        pred = self.model.predict(data, static_cam=True)
        # body_params_global: 중력 정렬된 세계 좌표계 (Z-up)
        # body_params_incam: 카메라 좌표계 (Y-down, Z-forward)
        # bp = pred["body_params_incam"]
        bp = pred["body_params_global"]

        # ── v3: SMPL FK → 24 joints ──
        smpl_joints = self._compute_smpl_joints(
            body_pose=bp["body_pose"],
            global_orient=bp["global_orient"],
            transl=bp["transl"],
            betas=bp["betas"],
        )

        return {
            "global_orient": bp["global_orient"].cpu().numpy(),  # (T,3)
            "body_pose":     bp["body_pose"].cpu().numpy(),      # (T,63)
            "transl":        bp["transl"].cpu().numpy(),         # (T,3)
            "betas":         bp["betas"].cpu().numpy(),          # (T,10)
            "smpl_joints":   smpl_joints,                        # (T,24,3)
        }

    @torch.no_grad()
    def _extract_hmr2_features(self, frames_rgb, bbx_xys):
        T = len(frames_rgb)
        img_dst_size = 256

        gt_center = bbx_xys[:, :2]
        gt_bbx_size = bbx_xys[:, 2]

        imgs_list = []
        for i in range(T):
            frame = frames_rgb[i]
            cx = float(gt_center[i, 0])
            cy = float(gt_center[i, 1])
            sz = float(gt_bbx_size[i])

            ds_factor = sz / img_dst_size / 2.0
            if ds_factor > 1.1:
                frame = cv2.GaussianBlur(frame, (5, 5), (ds_factor - 1) / 2)

            img_crop, _ = self._crop_and_resize(
                frame,
                np.array([cx, cy]),
                sz,
                img_dst_size,
                enlarge_ratio=1.0,
            )
            imgs_list.append(img_crop)

        imgs = np.stack(imgs_list)
        imgs_t = torch.from_numpy(imgs)
        imgs_t = ((imgs_t / 255.0 - self._hmr2_mean) / self._hmr2_std).permute(0, 3, 1, 2).float()

        imgs_t = imgs_t.to(self.device)
        features = []
        batch_size = 16
        for j in range(0, T, batch_size):
            feat = self.hmr2.model({"img": imgs_t[j: j + batch_size]})
            features.append(feat.detach().cpu())
        return torch.cat(features, dim=0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 프레임 처리
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def process_frame(self, frame):
        self.frame_buffer.append(frame.copy())
        self.frame_count += 1

        if (len(self.frame_buffer) == self.window_size
                and self.frame_count % self.stride == 0
                and not self._infer_running):

            frames_snapshot = list(self.frame_buffer)
            self._infer_running = True

            def _worker(frames_snap, prev_overlap_snap):
                import traceback
                overlap = self.window_size - self.stride
                try:
                    out = self._run_inference(frames_snap)
                except Exception as e:
                    print(f"[GENMO] 추론 오류: {e}")
                    traceback.print_exc()
                    self._infer_running = False
                    return

                if prev_overlap_snap is not None:
                    for k in ('global_orient', 'body_pose', 'transl',
                              'betas', 'smpl_joints'):
                        if k in out and k in prev_overlap_snap:
                            out[k][:overlap] = prev_overlap_snap[k]

                start = overlap if prev_overlap_snap is not None else 0
                new_results = [{k: out[k][i] for k in out}
                               for i in range(start, self.window_size)]

                with self._infer_lock:
                    self._result_queue.extend(new_results)
                    self.prev_overlap = {
                        k: out[k][-overlap:].copy() for k in out
                    }
                self._infer_running = False

            threading.Thread(
                target=_worker,
                args=(frames_snapshot, self.prev_overlap),
                daemon=True,
            ).start()

        with self._infer_lock:
            results = list(self._result_queue)
            self._result_queue.clear()
        return results


if __name__ == "__main__":
    g = GENMOStreamingInference("~/GENMO/inputs/pretrained/gem_smpl.ckpt")
    total = 0
    for i in range(60):
        r = g.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        if r:
            total += len(r)
            print(f"  frame {i}: {len(r)}개 → body_pose {r[0]['body_pose'].shape}"
                  f" smpl_joints {r[0]['smpl_joints'].shape}")
    print(f"✅ 총 {total}개 SMPL 프레임")
