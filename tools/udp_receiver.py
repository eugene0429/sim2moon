#!/usr/bin/env python3
"""UDP receiver/decoder for S2R_ICD ISSA-ROVER-DATA packets.

Usage:
    python3 tools/udp_receiver.py [--port 7777]

Listens on the specified port and decodes incoming S2R_ICD packets,
printing the rover data fields in real-time.
"""

import argparse
import socket
import struct
import sys
import time
import zlib


def decode_packet(data: bytes) -> dict:
    """Decode a S2R_ICD packet and return fields as a dict."""
    if len(data) < 19:  # minimum: header (15B) + checksum (4B)
        return {"error": f"Packet too short: {len(data)} bytes"}

    # STX
    stx = data[0]
    if stx != 0xFE:
        return {"error": f"Invalid STX: 0x{stx:02X}"}

    # LEN (2B LE)
    payload_len = struct.unpack_from('<H', data, 1)[0]

    # SEQ (1B)
    seq = data[3]

    # MSG ID (3B)
    msg_id = data[4:7]
    msg_id_hex = '0x' + msg_id.hex()

    # TIMESTAMP (8B LE) — ms since 2015-01-01 UTC
    timestamp_ms = struct.unpack_from('<Q', data, 7)[0]
    epoch_2015 = 1420070400  # 2015-01-01 00:00:00 UTC
    unix_ts = epoch_2015 + timestamp_ms / 1000.0
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(unix_ts))

    # Payload starts at offset 15
    payload = data[15:15 + payload_len]

    # CRC-32 verification
    expected_crc = struct.unpack_from('<I', data, 15 + payload_len)[0]
    actual_crc = zlib.crc32(data[:15 + payload_len]) & 0xFFFFFFFF
    crc_ok = expected_crc == actual_crc

    result = {
        "stx": f"0x{stx:02X}",
        "length": payload_len,
        "seq": seq,
        "msg_id": msg_id_hex,
        "timestamp": time_str,
        "timestamp_ms": timestamp_ms,
        "crc_ok": crc_ok,
        "total_bytes": len(data),
    }

    # Decode ISSA-ROVER-DATA (0x030101)
    if msg_id == b'\x03\x01\x01' and payload_len == 40:
        fields = struct.unpack_from('<10f', payload)
        result["rover_data"] = {
            "pos_x": fields[0],
            "pos_y": fields[1],
            "pos_z": fields[2],
            "roll_deg": fields[3],
            "pitch_deg": fields[4],
            "yaw_deg": fields[5],
            "wheel_fl": fields[6],
            "wheel_fr": fields[7],
            "wheel_rl": fields[8],
            "wheel_rr": fields[9],
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="S2R_ICD UDP Receiver")
    parser.add_argument("--port", type=int, default=7777, help="Listen port")
    parser.add_argument("--bind", default="0.0.0.0", help="Bind address")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    print(f"Listening on {args.bind}:{args.port}...")
    print("-" * 80)

    count = 0
    try:
        while True:
            data, addr = sock.recvfrom(65536)
            count += 1
            result = decode_packet(data)

            if "error" in result:
                print(f"[{count}] ERROR from {addr}: {result['error']}")
                continue

            rover = result.get("rover_data", {})
            crc_str = "OK" if result["crc_ok"] else "FAIL"

            print(f"[{count}] SEQ={result['seq']:3d} | MSG={result['msg_id']} | "
                  f"CRC={crc_str} | {result['timestamp']}")

            if rover:
                print(f"  Position: ({rover['pos_x']:8.3f}, {rover['pos_y']:8.3f}, {rover['pos_z']:8.3f})")
                print(f"  Rotation: R={rover['roll_deg']:7.2f}° P={rover['pitch_deg']:7.2f}° Y={rover['yaw_deg']:7.2f}°")
                print(f"  Wheels:   FL={rover['wheel_fl']:7.3f} FR={rover['wheel_fr']:7.3f} "
                      f"RL={rover['wheel_rl']:7.3f} RR={rover['wheel_rr']:7.3f}")
                print()

    except KeyboardInterrupt:
        print(f"\nReceived {count} packets total.")

    finally:
        sock.close()


if __name__ == "__main__":
    main()
