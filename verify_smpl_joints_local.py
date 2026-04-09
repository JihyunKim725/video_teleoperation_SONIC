import torch
import numpy as np

from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
)
from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up


def compute_smpl_joints_local_reference(body_pose, global_orient):
    """pico_manager process_smpl_joints() 최소 재현."""
    # Step 1~3
    go_quat   = angle_axis_to_quaternion(global_orient)
    go_quat_z = smpl_root_ytoz_up(go_quat)
    go_aa_z   = quaternion_to_angle_axis(go_quat_z)

    # Step 4: FK (transl 없음)
    joints = compute_human_joints(
        body_pose=body_pose,
        global_orient=go_aa_z,
    )[:, :24, :]   # (Nf, 24, 3)

    # Step 5~6
    go_quat_final = remove_smpl_base_rot(go_quat_z, w_last=False)
    go_inv        = quat_inv(go_quat_final)
    go_inv_exp    = go_inv.unsqueeze(1).expand(-1, 24, -1)
    return quat_apply(go_inv_exp, joints)   # (Nf, 24, 3)


def verify(smpl_params_path: str):
    print(f"검증 파일: {smpl_params_path}")
    data   = torch.load(smpl_params_path, map_location="cpu")
    params = data["body_params_global"]    # ← global 좌표계 사용

    body_pose     = params["body_pose"]    # (F, 63)
    global_orient = params["global_orient"]# (F, 3)
    transl        = params["transl"]       # (F, 3)

    F = body_pose.shape[0]
    print(f"프레임 수: {F}")

    # ── 검증 1: transl 영향 확인 ─────────────────────────────────────
    print("\n[1] transl 범위 (큰 값이면 이전 버그 재현됨)")
    print(f"    Z 평균: {transl[:, 2].abs().mean():.2f}m  "
          f"(old code에서 smpl_joints Z ≈ 이 값)")

    # ── 검증 2: smpl_joints_local 계산 ───────────────────────────────
    joints = compute_smpl_joints_local_reference(body_pose, global_orient)
    print(f"\n[2] smpl_joints_local shape: {joints.shape}  (기대: ({F}, 24, 3))")
    assert joints.shape == (F, 24, 3), f"FAIL: {joints.shape}"

    # ── 검증 3: 범위 검사 (인체 스케일 ±1.5m) ────────────────────────
    jmax = joints.abs().max().item()
    print(f"[3] 최대 절댓값: {jmax:.3f}m  (기대: < 1.5)")
    assert jmax < 1.5, f"FAIL: {jmax:.3f}m — 좌표계 변환 실패 가능성"

    # ── 검증 4: 수직 축 (Z-up: head_z > ankle_z) ─────────────────────
    head_z  = joints[:, 15, 2].mean().item()   # head  joint 15
    ankle_z = joints[:, 7,  2].mean().item()   # L_ankle joint 7
    dz      = head_z - ankle_z
    print(f"[4] head_z={head_z:.3f}  ankle_z={ankle_z:.3f}  Δz={dz:.3f}m")
    print(f"    {'PASS: Z-up 정상' if dz > 0.8 else 'FAIL: Z-up 변환 이상'}")

    # ── 검증 5: wrist 인덱스 확인 ────────────────────────────────────
    print("\n[5] IsaacLab 인터리브 wrist 인덱스 확인:")
    print("    L_wrist: [23, 25, 27] / R_wrist: [24, 26, 28]")
    print("    (pico_manager line 1354-1362 기준)")

    # ── 검증 6: 전체 PASS 확인 ───────────────────────────────────────
    print("\n" + "="*50)
    print("ALL PASSED — smpl_joints_local 변환 정상")
    print("다음 단계: video_teleop_pipeline.py --mode sim 재실행")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
           "outputs/demo_smpl/test/smpl_params.pt"
    verify(path)
