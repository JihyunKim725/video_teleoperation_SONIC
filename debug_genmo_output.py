"""
debug_genmo_output.py — GENMO predict() 반환값 탐색 패치

사용법:
  genmo_streaming_inference.py 의 _run_inference() 메서드에서
  `pred = self.model.predict(data, static_cam=True)` 직후에
  아래 함수를 호출하세요.

삽입 위치: genmo_streaming_inference.py line ~226
"""


def debug_predict_output(pred, model):
    """
    GENMO predict() 반환값과 모델 내부 SMPL 접근 경로를 탐색.
    한 번만 실행되도록 호출 측에서 제어.

    Args:
        pred:  model.predict() 반환 dict
        model: self.model (GEM 모델 인스턴스)
    """
    print("\n" + "=" * 70)
    print("[DEBUG] ===== GENMO predict() 반환값 탐색 =====")
    print("=" * 70)

    # ── 1. pred 최상위 키 ──
    print(f"\n[1] pred 최상위 키: {list(pred.keys())}")

    # ── 2. body_params_incam 내부 키 ──
    if "body_params_incam" in pred:
        bp = pred["body_params_incam"]
        print(f"\n[2] body_params_incam 키: {list(bp.keys())}")
        for k, v in bp.items():
            if hasattr(v, 'shape'):
                print(f"    '{k}': shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"    '{k}': type={type(v)}, value={v}")

        # joints 키가 있는지 확인
        if "joints" in bp:
            print(f"\n  ★★★ 'joints' 발견! shape={bp['joints'].shape}")
            print(f"      → 이것을 직접 smpl_joints로 사용 가능!")

    # ── 3. pred에 직접 joints가 있는지 ──
    for key in ["joints", "smpl_joints", "pred_joints", "body_joints",
                 "kp_3d", "kp3d", "joints_3d"]:
        if key in pred:
            v = pred[key]
            shape = v.shape if hasattr(v, 'shape') else "N/A"
            print(f"\n  ★ pred['{key}'] 발견! shape={shape}")

    # ── 4. body_params_global이 있으면 확인 ──
    if "body_params_global" in pred:
        bg = pred["body_params_global"]
        print(f"\n[3] body_params_global 키: {list(bg.keys())}")
        for k, v in bg.items():
            if hasattr(v, 'shape'):
                print(f"    '{k}': shape={v.shape}")

    # ── 5. 모델 내부 SMPL body model 접근 경로 탐색 ──
    print(f"\n[4] model 내부 SMPL 접근 경로 탐색:")
    smpl_candidates = [
        "smpl", "body_model", "smplx", "smpl_model",
        "smplx_lite", "smpl_layer", "bm",
    ]
    for attr in smpl_candidates:
        if hasattr(model, attr):
            obj = getattr(model, attr)
            print(f"  ★ model.{attr} 존재! type={type(obj).__name__}")
            # forward 메서드 시그니처 확인
            if hasattr(obj, 'forward'):
                import inspect
                sig = inspect.signature(obj.forward)
                print(f"    forward() 파라미터: {list(sig.parameters.keys())}")

    # 모델의 모든 nn.Module 서브모듈 중 'smpl' 포함하는 것 탐색
    print(f"\n[5] model 서브모듈 중 'smpl' 포함:")
    import torch.nn as nn
    for name, module in model.named_modules():
        if 'smpl' in name.lower():
            print(f"  model.{name} → {type(module).__name__}")

    # ── 6. betas 확인 (FK에 필요) ──
    if "body_params_incam" in pred:
        bp = pred["body_params_incam"]
        if "betas" in bp:
            print(f"\n[6] betas 발견: shape={bp['betas'].shape}")
        else:
            print(f"\n[6] betas 없음 — FK 시 zeros(10) 사용 필요")

    print("\n" + "=" * 70)
    print("[DEBUG] ===== 탐색 완료 =====")
    print("=" * 70 + "\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# genmo_streaming_inference.py에 삽입할 패치 코드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH_INSTRUCTIONS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
genmo_streaming_inference.py 수정 방법:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: 파일 상단에 import 추가
  from debug_genmo_output import debug_predict_output

Step 2: _run_inference() 메서드에서 predict() 호출 직후에 삽입 (line ~226):

    # ── 7. 추론 ───────────────────────────────────
    pred = self.model.predict(data, static_cam=True)

    # ★ 디버그: 첫 실행 시에만 출력
    if not hasattr(self, '_debug_done'):
        debug_predict_output(pred, self.model)
        self._debug_done = True

    bp = pred["body_params_incam"]
    ...

Step 3: 실행 후 터미널 출력을 확인하여:
  - pred에 'joints' 키가 있으면 → 방법 1 (직접 추출)
  - model.smpl 또는 model.body_model이 있으면 → 방법 2 (내부 FK)
  - 둘 다 없으면 → 방법 3 (외부 smplx 라이브러리)

Step 4: 결과를 Claude에게 공유하면 다음 단계 코드를 생성합니다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

if __name__ == "__main__":
    print(PATCH_INSTRUCTIONS)
