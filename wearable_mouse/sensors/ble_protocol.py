"""
ESP32 / BLE HID Deterministic Event Protocol and Transport Layer.
Provides binary packet serialization, error detection, sequence tracking, and mock BLE transport.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List, Tuple
import struct


class BLEEventCode(IntEnum):
    MOVE = 0
    LEFT_CLICK = 1
    RIGHT_CLICK = 2
    DOUBLE_CLICK = 3
    BUTTON_DOWN = 4
    BUTTON_UP = 5
    SCROLL = 6
    ZOOM = 7
    PAUSE = 8
    RESUME = 9


class MalformedPacketError(ValueError):
    """Raised when binary packet header, size, or checksum validation fails."""
    pass


@dataclass
class BLEHIDPacket:
    """
    Deterministic Binary Packet Contract for ESP32 Firmware <-> Wearable Mouse Interface.
    Format: [Header:0xAA55 (2B), Seq:uint16 (2B), Time:float32 (4B), Code:uint8 (1B), dx:int16 (2B), dy:int16 (2B), buttons:uint8 (1B), wheel:int8 (1B), Checksum:uint8 (1B)]
    Total size: 16 bytes (15 bytes payload + 1 byte checksum).
    """
    seq_num: int
    timestamp: float
    event_code: BLEEventCode
    dx: int = 0
    dy: int = 0
    buttons: int = 0
    wheel: int = 0

    MAGIC_HEADER: int = 0xAA55
    PACKET_SIZE: int = 16

    def compute_checksum(self, body_bytes: bytes) -> int:
        """Calculate 8-bit XOR checksum over header and body bytes."""
        chk = 0
        for b in body_bytes:
            chk ^= b
        return chk & 0xFF

    def encode(self) -> bytes:
        """Serialize BLEHIDPacket into 15-byte binary buffer."""
        # Pack header and body fields
        body_bytes = struct.pack(
            ">HHfBhhBb",
            self.MAGIC_HEADER,
            self.seq_num & 0xFFFF,
            float(self.timestamp),
            int(self.event_code),
            max(-32768, min(32767, int(self.dx))),
            max(-32768, min(32767, int(self.dy))),
            self.buttons & 0xFF,
            max(-128, min(127, int(self.wheel)))
        )
        checksum = self.compute_checksum(body_bytes)
        return body_bytes + struct.pack("B", checksum)

    @classmethod
    def decode(cls, data: bytes) -> 'BLEHIDPacket':
        """Deserialize 15-byte binary buffer into BLEHIDPacket with checksum validation."""
        if len(data) != cls.PACKET_SIZE:
            raise MalformedPacketError(f"Invalid packet size {len(data)}, expected {cls.PACKET_SIZE} bytes")

        body_bytes = data[:-1]
        received_checksum = data[-1]

        header, seq, ts, code_val, dx, dy, buttons, wheel = struct.unpack(">HHfBhhBb", body_bytes)

        if header != cls.MAGIC_HEADER:
            raise MalformedPacketError(f"Invalid magic header 0x{header:04X}, expected 0x{cls.MAGIC_HEADER:04X}")

        # Compute checksum
        chk = 0
        for b in body_bytes:
            chk ^= b
        chk &= 0xFF

        if chk != received_checksum:
            raise MalformedPacketError(f"Checksum validation failed: calculated 0x{chk:02X}, received 0x{received_checksum:02X}")

        try:
            event_code = BLEEventCode(code_val)
        except ValueError:
            raise MalformedPacketError(f"Invalid BLE event code {code_val}")

        return cls(
            seq_num=seq,
            timestamp=ts,
            event_code=event_code,
            dx=dx,
            dy=dy,
            buttons=buttons,
            wheel=wheel
        )


class BLEPacketDecoder:
    """
    Stateful Packet Stream Decoder with resilience against:
    - Malformed packets
    - Dropped packets (sequence gaps)
    - Duplicate packets
    - Timestamp ordering violations
    """

    def __init__(self):
        self.last_seq_num: Optional[int] = None
        self.last_timestamp: float = -1.0
        self.dropped_packet_count: int = 0
        self.duplicate_packet_count: int = 0
        self.out_of_order_count: int = 0
        self.malformed_count: int = 0
        self.processed_packets: List[BLEHIDPacket] = []

    def reset(self) -> None:
        self.last_seq_num = None
        self.last_timestamp = -1.0
        self.dropped_packet_count = 0
        self.duplicate_packet_count = 0
        self.out_of_order_count = 0
        self.malformed_count = 0
        self.processed_packets.clear()

    def process_raw_packet(self, data: bytes) -> Optional[BLEHIDPacket]:
        """
        Decode raw bytes and evaluate stream integrity.
        Returns decoded packet if valid and unique, or None if duplicate/malformed.
        """
        try:
            packet = BLEHIDPacket.decode(data)
        except MalformedPacketError:
            self.malformed_count += 1
            return None

        # Sequence gap check (dropped frame detection)
        if self.last_seq_num is not None:
            expected_seq = (self.last_seq_num + 1) & 0xFFFF
            if packet.seq_num != expected_seq:
                if packet.seq_num <= self.last_seq_num and (self.last_seq_num - packet.seq_num) < 1000:
                    # Duplicate packet
                    self.duplicate_packet_count += 1
                    return None
                else:
                    # Dropped frame gap
                    gap = (packet.seq_num - self.last_seq_num) & 0xFFFF
                    if gap > 1:
                        self.dropped_packet_count += (gap - 1)

        # Timestamp ordering check
        if packet.timestamp < self.last_timestamp:
            self.out_of_order_count += 1

        self.last_seq_num = packet.seq_num
        self.last_timestamp = max(self.last_timestamp, packet.timestamp)
        self.processed_packets.append(packet)
        return packet


class MockBLETransport:
    """
    In-memory Mock Bluetooth LE Transport simulating RF PHY transmission and packet buffers.
    """

    def __init__(self, drop_rate: float = 0.0):
        self.drop_rate = drop_rate
        self.tx_buffer: List[bytes] = []
        self.rx_buffer: List[bytes] = []
        self.total_sent = 0
        self.total_received = 0

    def send(self, data: bytes) -> bool:
        """Transmit binary packet over mock BLE PHY."""
        self.total_sent += 1
        import random
        if self.drop_rate > 0.0 and random.random() < self.drop_rate:
            # Simulate packet loss
            return False

        self.tx_buffer.append(data)
        self.rx_buffer.append(data)
        self.total_received += 1
        return True

    def receive(self) -> Optional[bytes]:
        """Fetch next received byte packet from BLE RX buffer."""
        if self.rx_buffer:
            return self.rx_buffer.pop(0)
        return None

    def clear(self) -> None:
        self.tx_buffer.clear()
        self.rx_buffer.clear()
        self.total_sent = 0
        self.total_received = 0
