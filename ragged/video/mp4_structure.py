import struct
from typing import List


def create_mp4_box(box_type: str, data: bytes) -> bytes:
    size = len(data) + 8
    return struct.pack('>I', size) + box_type.encode('ascii') + data


def create_minimal_video_track() -> bytes:
    return bytes([
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x0a, 0x8d, 0x68, 0x05, 0x8b,
        0x00, 0x00, 0x00, 0x01, 0x68, 0xce, 0x06, 0xe2, 0x00, 0x00, 0x00, 0x01,
        0x65, 0x88, 0x80, 0x10, 0x00, 0x02, 0x00, 0x08
    ])


def create_video_boxes(total_vectors: int) -> List[bytes]:
    duration = min(max(total_vectors // 100, 1), 60) * 1000

    mvhd_data = struct.pack('>IIIIII', 0, 0, 0, 1000, duration, 0x00010000) + b'\x00' * 76
    tkhd_data = struct.pack('>IIIIII', 0x0000000F, 0, 0, 1, 0, duration) + b'\x00' * 60
    mdhd_data = struct.pack('>IIIIII', 0, 0, 0, 1000, duration, 0)
    hdlr_data = struct.pack('>IIII', 0, 0, 0x76696465, 0) + b'VideoHandler\x00'

    return [
        create_mp4_box('moov',
                       create_mp4_box('mvhd', mvhd_data) +
                       create_mp4_box('trak',
                                      create_mp4_box('tkhd', tkhd_data) +
                                      create_mp4_box('mdia',
                                                     create_mp4_box('mdhd', mdhd_data) +
                                                     create_mp4_box('hdlr', hdlr_data)
                                                     )
                                      )
                       ),
        create_mp4_box('mdat', create_minimal_video_track())
    ]