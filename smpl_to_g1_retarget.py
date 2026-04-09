"""
smpl_to_g1_retarget.py  (v3 프로토콜 지원)
GENMO(GEM-SMPL) 출력 → G1 29-DOF 관절각 + body_quat 변환

v3 추가사항:
  - smpl_to_g1_v3(): wrist 6-DOF만 retarget, SMPL 데이터는 pass-through
  - v3에서는 SMPL fields(smpl_joints, smpl_pose)가 primary motion data
  - joint_pos의 wrist joints(인덱스 23-28)만 meaningful 값 필요
    공식 문서: "3 joints per wrist × 2 wrists"

기존 smpl_to_g1() (v1용)도 유지하여 하위 호환성 보장.
"""

import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# G1 관절 각도 제한 (rad)
LIMITS_LO = np.array([
    -2.5307, -0.5236, -2.7576, -0.0873, -0.8727, -0.2618,
    -2.5307, -2.9671, -2.7576, -0.0873, -0.8727, -0.2618,
    -2.618,  -0.52,   -0.52,
    -3.0892, -1.5882, -2.618, -1.0472, -1.9722, -1.6144, -1.6144,
    -3.0892, -2.2515, -2.618, -1.0472, -1.9722, -1.6144, -1.6144,
], dtype=np.float32)

LIMITS_HI = np.array([
    2.8798,  2.9671,  2.7576,  2.8798,  0.5236,  0.2618,
    2.8798,  0.5236,  2.7576,  2.8798,  0.5236,  0.2618,
    2.618,   0.52,    0.52,
    2.6704,  2.2515,  2.618,   2.0944,  1.9722,  1.6144,  1.6144,
    2.6704,  1.5882,  2.618,   2.0944,  1.9722,  1.6144,  1.6144,
], dtype=np.float32)

# SMPL body_pose 인덱스
_LH, _RH, _S1 = 0, 1, 2
_LK, _RK, _S2 = 3, 4, 5
_LA, _RA, _S3 = 6, 7, 8
_LS, _RS       = 15, 16
_LE, _RE       = 17, 18
_LW, _RW       = 19, 20

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SMPL Euler(XYZ) rest-offset (차렷 자세 관측값, deg)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_LS_REST = np.radians(np.array([-5.0, -13.0, -48.0]))
_RS_REST = np.radians(np.array([-4.0, +22.0, +32.0]))
_LE_REST = np.radians(np.array([+70.0, -60.0, +97.0]))
_RE_REST = np.radians(np.array([+70.0, -60.0, +97.0]))
_SP_REST = np.radians(np.array([+49.0, +2.0, +2.0]))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# G1 IsaacLab 29-DOF wrist 인덱스 참조
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v3 공식 스펙: wrist joints = [23, 24, 25, 26, 27, 28]
#   "3 joints per wrist × 2 wrists"
#
# IsaacLab 29-DOF 순서:
#   19: left_wrist_roll   → v3 wrist index: 이 매핑은 PICO 참조 구현 기준
#   20: left_wrist_pitch    실제 g1_deploy C++ 코드에서
#   21: left_wrist_yaw      [23,24,25] = R_shoulder_roll, R_shoulder_yaw, R_elbow
#   26: right_wrist_roll    [26,27,28] = R_wrist_roll, R_wrist_pitch, R_wrist_yaw
#   27: right_wrist_pitch
#   28: right_wrist_yaw
#
# 해석: 공식 문서의 "wrist joint indices [23,24,25,26,27,28]"은
#   left wrist = [19,20,21], right wrist = [26,27,28]이 아니라
#   양쪽 합산 6DOF를 [23..28] 범위로 표현한 것.
#   실제로는 IsaacLab order에서 wrist는:
#     Left:  19, 20, 21
#     Right: 26, 27, 28
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_dbg_counter = 0


def _euler(aa):
    """axis-angle (3,) → Euler XYZ (3,)"""
    return R.from_rotvec(aa).as_euler('XYZ')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v1 용: 기존 함수 (하위 호환)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def smpl_to_g1(body_pose, global_orient, scale=0.8):
    global _dbg_counter
    _dbg_counter += 1

    bp = body_pose.reshape(21, 3)
    e = [_euler(bp[i]) for i in range(21)]
    s = scale

    lh = e[_LH]; lk = e[_LK]; la = e[_LA]
    rh = e[_RH]; rk = e[_RK]; ra = e[_RA]
    sp = e[_S1] + e[_S2] + e[_S3]
    ls = e[_LS]; le = e[_LE]; lw = e[_LW]
    rs = e[_RS]; re = e[_RE]; rw = e[_RW]

    ls_d = ls - _LS_REST
    rs_d = rs - _RS_REST
    le_d = le - _LE_REST
    re_d = re - _RE_REST
    sp_d = sp - _SP_REST

    AMP_SH = 3.0
    AMP_EL = 2.5
    AMP_WR = 2.0
    AMP_SP = 2.0

    l_sh_pitch = ls_d[2] * s * AMP_SH
    l_sh_roll  = -ls_d[0] * s * AMP_SH
    l_sh_yaw   = ls_d[1] * s * AMP_SH * 0.5
    l_elbow    = max(0.0, le_d[0] * s * AMP_EL)
    l_wr_roll  = lw[0] * s * AMP_WR * 0.3
    l_wr_pitch = lw[1] * s * AMP_WR * 0.3
    l_wr_yaw   = lw[2] * s * AMP_WR * 0.3

    r_sh_pitch = -rs_d[2] * s * AMP_SH
    r_sh_roll  = -rs_d[0] * s * AMP_SH
    r_sh_yaw   = rs_d[1] * s * AMP_SH * 0.5
    r_elbow    = max(0.0, re_d[0] * s * AMP_EL)
    r_wr_roll  = rw[0] * s * AMP_WR * 0.3
    r_wr_pitch = rw[1] * s * AMP_WR * 0.3
    r_wr_yaw   = rw[2] * s * AMP_WR * 0.3

    waist_yaw   = sp_d[2] * s * AMP_SP
    waist_roll  = sp_d[0] * s * AMP_SP
    waist_pitch = sp_d[1] * s * AMP_SP

    joint_pos = np.array([
        0.0, 0.0, -0.1, 0.3, -0.2, 0.0,
        0.0, 0.0, -0.1, 0.3, -0.2, 0.0,
        waist_yaw, waist_roll, waist_pitch,
        l_sh_pitch, l_sh_roll, l_sh_yaw, l_elbow,
        l_wr_roll, l_wr_pitch, l_wr_yaw,
        r_sh_pitch, r_sh_roll, r_sh_yaw, r_elbow,
        r_wr_roll, r_wr_pitch, r_wr_yaw,
    ], dtype=np.float32)

    joint_pos = np.clip(joint_pos, LIMITS_LO, LIMITS_HI)

    if _dbg_counter % 30 == 1:
        print(f"  [DBG] L_Sh euler(deg):  X={np.degrees(ls[0]):+7.1f} Y={np.degrees(ls[1]):+7.1f} Z={np.degrees(ls[2]):+7.1f}", file=sys.stderr)
        print(f"  [DBG] L_Sh delta(deg):  X={np.degrees(ls_d[0]):+7.1f} Y={np.degrees(ls_d[1]):+7.1f} Z={np.degrees(ls_d[2]):+7.1f}", file=sys.stderr)
        print(f"  [DBG] R_Sh euler(deg):  X={np.degrees(rs[0]):+7.1f} Y={np.degrees(rs[1]):+7.1f} Z={np.degrees(rs[2]):+7.1f}", file=sys.stderr)
        print(f"  [DBG] R_Sh delta(deg):  X={np.degrees(rs_d[0]):+7.1f} Y={np.degrees(rs_d[1]):+7.1f} Z={np.degrees(rs_d[2]):+7.1f}", file=sys.stderr)
        print(f"  [DBG] L_El euler(deg):  X={np.degrees(le[0]):+7.1f}  delta_X={np.degrees(le_d[0]):+7.1f}", file=sys.stderr)
        print(f"  [G1]  Lsh_p={np.degrees(joint_pos[15]):+7.1f} Lsh_r={np.degrees(joint_pos[16]):+7.1f} Lel={np.degrees(joint_pos[18]):+7.1f}", file=sys.stderr)
        print(f"  [G1]  Rsh_p={np.degrees(joint_pos[22]):+7.1f} Rsh_r={np.degrees(joint_pos[23]):+7.1f} Rel={np.degrees(joint_pos[25]):+7.1f}", file=sys.stderr)
        print(f"  [G1]  waist: y={np.degrees(joint_pos[12]):+7.1f} r={np.degrees(joint_pos[13]):+7.1f} p={np.degrees(joint_pos[14]):+7.1f}", file=sys.stderr)
        print(f"  [G1]  ===", file=sys.stderr)

    body_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    return joint_pos, body_quat


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v3 용: SMPL pass-through + wrist만 retarget
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_dbg_v3_counter = 0


def smpl_to_g1_v3(body_pose, global_orient, smpl_joints, scale=0.8):
    """
    v3 프로토콜용: SMPL 데이터를 그대로 전달 + wrist만 retarget.

    v3에서는 smpl_joints(24,3)와 smpl_pose(21,3)가 primary motion data이고,
    joint_pos는 wrist 6-DOF만 meaningful 값이 필요합니다.

    Args:
        body_pose:    (63,) float32 — SMPL 21 joints axis-angle
        global_orient: (3,) float32 — root axis-angle
        smpl_joints:  (24, 3) float32 — SMPL FK joint positions
        scale:        float — 동작 스케일

    Returns:
        joint_pos:   (29,) float32 — wrist만 채움, 나머지 0
        joint_vel:   (29,) float32 — 전부 0
        smpl_pose:   (21, 3) float32 — body_pose reshape (pass-through)
        smpl_joints: (24, 3) float32 — pass-through
        body_quat:   (4,) float32 — global_orient → quaternion (w,x,y,z)
    """
    global _dbg_v3_counter
    _dbg_v3_counter += 1

    bp = body_pose.reshape(21, 3)

    # ── smpl_pose: body_pose를 (21, 3) axis-angle로 직접 사용 ──
    smpl_pose_out = bp.copy().astype(np.float32)

    # ── joint_pos: wrist 6-DOF만 계산 (나머지 0) ──
    joint_pos = np.zeros(29, dtype=np.float32)

    # SMPL wrist axis-angle → Euler XYZ
    lw = _euler(bp[_LW])  # Left wrist
    rw = _euler(bp[_RW])  # Right wrist

    s = scale
    AMP_WR = 2.0

    # IsaacLab 29-DOF에서 wrist indices:
    #   Left wrist:  19 (roll), 20 (pitch), 21 (yaw)
    #   Right wrist: 26 (roll), 27 (pitch), 28 (yaw)
    joint_pos[19] = lw[0] * s * AMP_WR * 0.3   # L wrist roll
    joint_pos[20] = lw[1] * s * AMP_WR * 0.3   # L wrist pitch
    joint_pos[21] = lw[2] * s * AMP_WR * 0.3   # L wrist yaw
    joint_pos[26] = rw[0] * s * AMP_WR * 0.3   # R wrist roll
    joint_pos[27] = rw[1] * s * AMP_WR * 0.3   # R wrist pitch
    joint_pos[28] = rw[2] * s * AMP_WR * 0.3   # R wrist yaw

    joint_pos = np.clip(joint_pos, LIMITS_LO, LIMITS_HI)

    # # ── body_quat: global_orient → quaternion (w,x,y,z) ──
    # rot = R.from_rotvec(global_orient)
    # q_xyzw = rot.as_quat()  # scipy 출력: (x, y, z, w)
    # body_quat = np.array(
    #     [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
    #     dtype=np.float32,
    # )  # → SONIC 요구 순서: (w, x, y, z)

    # 수정1: body_quat -> identity 고정 
    # GENMO의 global_orient는 카메라를 향한 -180도 y회전을 포함하고 있어 이것을 그대로 body_quat로 보내면 policy가 로봇을 뒤로 돌리려 함.
    # body_quat는 heading 계산에만 사용되므로, identity로 고정하면 로봇이 현재 방향 유지 (video teleop에서는 heading 추적 불필요)
    body_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # smpl_joints: 표준 SMPL FK 출력을 그대로 전달
    # 표준 SMPL은 발=0, pelvis_y≈0.9m 좌표계이므로 SONIC encoder 기대값과 일치
    smpl_joints_out = smpl_joints.copy().astype(np.float32)

    joint_vel = np.zeros(29, dtype=np.float32)

    # ── 디버그 (60프레임마다) ──
    if _dbg_v3_counter % 60 == 1:
        print(f"  [v3] Lwrist(deg): r={np.degrees(joint_pos[19]):+.1f} "
              f"p={np.degrees(joint_pos[20]):+.1f} y={np.degrees(joint_pos[21]):+.1f}",
              file=sys.stderr)
        print(f"  [v3] Rwrist(deg): r={np.degrees(joint_pos[26]):+.1f} "
              f"p={np.degrees(joint_pos[27]):+.1f} y={np.degrees(joint_pos[28]):+.1f}",
              file=sys.stderr)
        print(f"  [v3] body_quat: w={body_quat[0]:.3f} x={body_quat[1]:.3f} "
              f"y={body_quat[2]:.3f} z={body_quat[3]:.3f}", file=sys.stderr)
        print(f"  [v3] smpl_joints[0](pelvis): {smpl_joints_out[0]}", file=sys.stderr)
        print(f"  [v3] smpl_pose shape: {smpl_pose_out.shape}", file=sys.stderr)
        print(f"  [v3] ===", file=sys.stderr)

    # Y-up (SMPL) → Z-up (SONIC/IsaacGym) 좌표 변환
    # (x, y, z)_Yup → (x, -z, y)_Zup
    smpl_joints_zup = smpl_joints_out.copy()
    smpl_joints_zup[:, 0] = smpl_joints_out[:, 0]   # X → X
    smpl_joints_zup[:, 1] = -smpl_joints_out[:, 2]  # Z → -Y (new Y)
    smpl_joints_zup[:, 2] = smpl_joints_out[:, 1]   # Y → Z (new Z = 높이)
    smpl_joints_out = smpl_joints_zup


    return joint_pos, joint_vel, smpl_pose_out, smpl_joints_out, body_quat

if __name__ == "__main__":
    # v1 테스트
    rest_bp = np.zeros(63, dtype=np.float32)
    rest_bp[15*3:15*3+3] = R.from_euler('XYZ', _LS_REST).as_rotvec().astype(np.float32)
    rest_bp[16*3:16*3+3] = R.from_euler('XYZ', _RS_REST).as_rotvec().astype(np.float32)
    rest_bp[17*3:17*3+3] = R.from_euler('XYZ', _LE_REST).as_rotvec().astype(np.float32)

    jp, bq = smpl_to_g1(rest_bp, np.zeros(3, dtype=np.float32))
    print(f"\n[v1] 차렷 자세 테스트:")
    print(f"  Lsh_pitch={np.degrees(jp[15]):+.1f}° Rsh_pitch={np.degrees(jp[22]):+.1f}°")
    print(f"  Lelbow   ={np.degrees(jp[18]):+.1f}° Relbow   ={np.degrees(jp[25]):+.1f}°")

    # v3 테스트
    dummy_joints = np.zeros((24, 3), dtype=np.float32)
    jp3, jv3, sp3, sj3, bq3 = smpl_to_g1_v3(
        rest_bp, np.array([0.0, 0.1, 0.0], dtype=np.float32), dummy_joints
    )
    print(f"\n[v3] 테스트:")
    print(f"  joint_pos non-zero indices: {np.nonzero(jp3)[0].tolist()}")
    print(f"  smpl_pose shape: {sp3.shape}")
    print(f"  body_quat: {bq3}")
    print(f"✅ 테스트 완료")
