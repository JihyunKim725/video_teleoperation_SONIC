"""
더미 ZMQ Publisher — SONIC Packed Binary Format
1280-byte JSON header + concatenated binary fields

사용법:
  python3 dummy_publisher.py
"""
import json
import struct
import time
import numpy as np
import zmq

HEADER_SIZE = 1280  # 고정 헤더 크기

def build_packed_message(topic: str, version: int, fields: dict) -> bytes:
    """
    SONIC packed binary message 생성.
    
    fields: { "name": (np_array, "dtype_str") }
      예: { "joint_pos": (np.zeros((1,29)), "f32") }
    """
    # 필드 메타데이터 + 바이너리 데이터
    field_meta = []
    binary_parts = []
    
    for name, (array, dtype_str) in fields.items():
        data = array.tobytes()
        field_meta.append({
            "name": name,
            "dtype": dtype_str,
            "shape": list(array.shape),
        })
        binary_parts.append(data)
    
    # JSON 헤더 (1280 bytes, null-padded)
    header = {
        "v": version,
        "endian": "le",
        "count": int(fields[list(fields.keys())[0]][0].shape[0]),
        "fields": field_meta,
    }
    header_json = json.dumps(header, separators=(',', ':'))
    header_bytes = header_json.encode('utf-8')
    
    if len(header_bytes) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_bytes)} > {HEADER_SIZE}")
    
    # null-padding to 1280 bytes
    header_padded = header_bytes + b'\x00' * (HEADER_SIZE - len(header_bytes))
    
    # topic + header + binary data
    payload = header_padded + b''.join(binary_parts)
    return topic.encode('utf-8') + payload


def main():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5556")
    topic = "pose"

    print(f"✅ 더미 Publisher 시작: tcp://*:5556 (topic: {topic})")
    print("   SONIC Packed Binary Format (1280-byte JSON header)")
    print("   Terminal 2에서 ENTER → ZMQ 모드 활성화 후 확인")
    time.sleep(1)

    frame_idx = 0
    while True:
        t = frame_idx * 0.02

        # 기본 직립 자세 + 사인파 어깨 움직임
        joint_pos = np.array([[
            0.0, 0.0, -0.1, 0.3, -0.2, 0.0,     # left leg
            0.0, 0.0, -0.1, 0.3, -0.2, 0.0,     # right leg
            0.0, 0.0, 0.0,                        # waist
            0.5*np.sin(t*2), 0.3, 0.0, 0.8+0.3*np.sin(t*3), 0.0, 0.0, 0.0,  # left arm
            0.5*np.sin(t*2+np.pi), -0.3, 0.0, 0.8+0.3*np.sin(t*3+np.pi), 0.0, 0.0, 0.0,  # right arm
        ]], dtype=np.float32)  # [1, 29]

        joint_vel = np.zeros((1, 29), dtype=np.float32)
        body_quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # [1, 4]
        frame_index = np.array([frame_idx], dtype=np.int32)  # [1]

        # Protocol v1 필드
        fields = {
            "joint_pos":   (joint_pos, "f32"),
            "joint_vel":   (joint_vel, "f32"),
            "body_quat":   (body_quat, "f32"),
            "frame_index": (frame_index, "i32"),
        }

        msg = build_packed_message(topic, version=1, fields=fields)
        socket.send(msg)  # 단일 메시지 (send_multipart 아님!)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx} sent ({len(msg)} bytes)")

        time.sleep(0.02)  # 50Hz

if __name__ == "__main__":
    main()
