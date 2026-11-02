"""
Phase 4 Computer Output / BLE HID Abstraction Test Suite.
Verifies all 5 concrete output adapters, BLE HID report encoding, binary packet contracts,
error recovery (malformed, dropped, duplicate, out-of-order packets), mock BLE transport,
and end-to-end pipeline multi-adapter routing.
"""

import pytest
import io
import numpy as np
from wearable_mouse.config import Config
from wearable_mouse.pipeline import WearableMousePipeline
from wearable_mouse.sensors.imu_data import IMUSample
from wearable_mouse.gestures.gesture_map import GestureType
from wearable_mouse.interaction.interaction_state import UnifiedState
from wearable_mouse.interaction.base_output_adapter import BaseOutputAdapter
from wearable_mouse.interaction.adapters.virtual_output_adapter import VirtualOutputAdapter
from wearable_mouse.interaction.adapters.ble_hid_adapter import BLEHIDOutputAdapter, BLEHIDReport
from wearable_mouse.interaction.adapters.test_output_adapter import TestOutputAdapter, OutputLogEntry
from wearable_mouse.interaction.adapters.debug_output_adapter import DebugOutputAdapter
from wearable_mouse.interaction.output_adapter import OutputAdapter, SemanticEvent, SemanticEventType
from wearable_mouse.sensors.ble_protocol import (
    BLEHIDPacket,
    BLEEventCode,
    BLEPacketDecoder,
    MalformedPacketError,
    MockBLETransport
)


# Helper function to create calibrated pipeline
def create_calibrated_pipeline() -> WearableMousePipeline:
    config = Config()
    pipeline = WearableMousePipeline(config)
    # Warmup calibrator to CALIBRATED state (250 stationary samples @ 100Hz)
    for i in range(250):
        t = i * 0.01
        ax = 1.0 + 0.8 * np.sin(t * 20.0)
        ay = 1.0 + 0.8 * np.cos(t * 20.0)
        az = 9.81 + 0.5 * np.sin(t * 15.0)
        sample = IMUSample(timestamp=t, accel=np.array([ax, ay, az]), gyro=np.array([0.0, 0.0, 0.0]))
        pipeline.process_frame(sample)
    return pipeline


# -----------------------------------------------------------------------------
# Test 1: VirtualOutputAdapter Movement
# -----------------------------------------------------------------------------
def test_1_virtual_output_adapter_movement():
    adapter = VirtualOutputAdapter(1920, 1080, 960, 540)
    adapter.move(50.0, -30.0)
    assert adapter.x == 1010.0
    assert adapter.y == 510.0

    # Test screen bounding clamping
    adapter.move(-2000.0, -2000.0)
    assert adapter.x == 0.0
    assert adapter.y == 0.0


# -----------------------------------------------------------------------------
# Test 2: VirtualOutputAdapter Clicks
# -----------------------------------------------------------------------------
def test_2_virtual_output_adapter_clicks():
    adapter = VirtualOutputAdapter()
    adapter.left_click()
    assert adapter.click_count == 1
    assert "LEFT_CLICK" in adapter.last_action

    adapter.right_click()
    assert adapter.click_count == 2
    assert "RIGHT_CLICK" in adapter.last_action

    adapter.double_click()
    assert adapter.click_count == 4
    assert "DOUBLE_CLICK" in adapter.last_action


# -----------------------------------------------------------------------------
# Test 3: VirtualOutputAdapter Drag
# -----------------------------------------------------------------------------
def test_3_virtual_output_adapter_drag():
    adapter = VirtualOutputAdapter()
    adapter.button_down("left")
    assert adapter.buttons_held["left"] is True
    assert "BUTTON_DOWN (left)" in adapter.last_action

    adapter.button_up("left")
    assert adapter.buttons_held["left"] is False
    assert "BUTTON_UP (left)" in adapter.last_action


# -----------------------------------------------------------------------------
# Test 4: VirtualOutputAdapter Scroll & Zoom
# -----------------------------------------------------------------------------
def test_4_virtual_output_adapter_scroll_and_zoom():
    adapter = VirtualOutputAdapter()
    adapter.scroll(0.0, 5.0)
    assert "SCROLL (dx=0.0, dy=5.0)" in adapter.last_action

    adapter.zoom(2.0)
    assert "ZOOM (2.0)" in adapter.last_action


# -----------------------------------------------------------------------------
# Test 5: TestOutputAdapter Call Logging
# -----------------------------------------------------------------------------
def test_5_test_output_adapter_logging():
    adapter = TestOutputAdapter()
    adapter.move(10.0, 20.0)
    adapter.left_click()
    adapter.button_down("right")
    adapter.button_up("right")

    assert len(adapter.log_entries) == 4
    assert adapter.log_entries[0].operation == "move"
    assert adapter.log_entries[1].operation == "left_click"
    assert adapter.log_entries[2].operation == "button_down"
    assert adapter.log_entries[2].button == "right"


# -----------------------------------------------------------------------------
# Test 6: DebugOutputAdapter Formatting
# -----------------------------------------------------------------------------
def test_6_debug_output_adapter_formatting():
    buf = io.StringIO()
    adapter = DebugOutputAdapter(stream=buf)
    adapter.move(15.5, -10.2)
    adapter.left_click()

    output = buf.getvalue()
    assert "[DEBUG_OUTPUT] MOVE dx=15.50, dy=-10.20" in output
    assert "[DEBUG_OUTPUT] LEFT_CLICK" in output


# -----------------------------------------------------------------------------
# Test 7: BLE HID Report Encoding (Movement)
# -----------------------------------------------------------------------------
def test_7_ble_hid_report_encoding_move():
    report = BLEHIDReport(buttons=0, dx=15, dy=-20, wheel=0)
    data = report.encode()
    assert len(data) == 4

    decoded = BLEHIDReport.decode(data)
    assert decoded.buttons == 0
    assert decoded.dx == 15
    assert decoded.dy == -20
    assert decoded.wheel == 0


# -----------------------------------------------------------------------------
# Test 8: BLE HID Report Encoding (Clicks)
# -----------------------------------------------------------------------------
def test_8_ble_hid_report_encoding_clicks():
    received_reports: List[BLEHIDReport] = []

    def on_report(raw_bytes: bytes):
        received_reports.append(BLEHIDReport.decode(raw_bytes))

    adapter = BLEHIDOutputAdapter(transport_callback=on_report)
    adapter.left_click()

    assert len(received_reports) == 2
    assert received_reports[0].buttons == 0x01  # Button press (bit 0)
    assert received_reports[1].buttons == 0x00  # Button release


# -----------------------------------------------------------------------------
# Test 9: BLE HID Report Encoding (Drag)
# -----------------------------------------------------------------------------
def test_9_ble_hid_report_encoding_drag():
    received_reports: List[BLEHIDReport] = []

    def on_report(raw_bytes: bytes):
        received_reports.append(BLEHIDReport.decode(raw_bytes))

    adapter = BLEHIDOutputAdapter(transport_callback=on_report)
    adapter.button_down("left")
    adapter.move(10, 5)
    adapter.button_up("left")

    assert len(received_reports) == 3
    assert received_reports[0].buttons == 0x01  # Left down
    assert received_reports[1].buttons == 0x01  # Move while held
    assert received_reports[1].dx == 10
    assert received_reports[2].buttons == 0x00  # Left up


# -----------------------------------------------------------------------------
# Test 10: BLE HID Report Encoding (Scroll)
# -----------------------------------------------------------------------------
def test_10_ble_hid_report_encoding_scroll():
    received_reports: List[BLEHIDReport] = []

    def on_report(raw_bytes: bytes):
        received_reports.append(BLEHIDReport.decode(raw_bytes))

    adapter = BLEHIDOutputAdapter(transport_callback=on_report)
    adapter.scroll(0, 5)

    assert len(received_reports) == 1
    assert received_reports[0].wheel == 5


# -----------------------------------------------------------------------------
# Test 11: BLE Protocol Packet Serialization
# -----------------------------------------------------------------------------
def test_11_ble_protocol_packet_serialization():
    pkt = BLEHIDPacket(seq_num=101, timestamp=2.50, event_code=BLEEventCode.MOVE, dx=25, dy=-15, buttons=0, wheel=0)
    raw = pkt.encode()
    assert len(raw) == BLEHIDPacket.PACKET_SIZE


# -----------------------------------------------------------------------------
# Test 12: BLE Protocol Packet Deserialization
# -----------------------------------------------------------------------------
def test_12_ble_protocol_packet_deserialization():
    pkt = BLEHIDPacket(seq_num=202, timestamp=3.75, event_code=BLEEventCode.LEFT_CLICK, dx=0, dy=0, buttons=1, wheel=0)
    raw = pkt.encode()

    decoded = BLEHIDPacket.decode(raw)
    assert decoded.seq_num == 202
    assert abs(decoded.timestamp - 3.75) < 1e-4
    assert decoded.event_code == BLEEventCode.LEFT_CLICK
    assert decoded.buttons == 1


# -----------------------------------------------------------------------------
# Test 13: BLE Protocol Malformed Packet Detection
# -----------------------------------------------------------------------------
def test_13_ble_protocol_malformed_packet_detection():
    pkt = BLEHIDPacket(seq_num=1, timestamp=1.0, event_code=BLEEventCode.MOVE, dx=5, dy=5)
    raw = bytearray(pkt.encode())

    # Corrupt checksum byte
    raw[-1] ^= 0xFF

    with pytest.raises(MalformedPacketError):
        BLEHIDPacket.decode(bytes(raw))

    # Corrupt header byte
    raw = bytearray(pkt.encode())
    raw[0] = 0x00
    with pytest.raises(MalformedPacketError):
        BLEHIDPacket.decode(bytes(raw))


# -----------------------------------------------------------------------------
# Test 14: BLE Protocol Dropped Packet Detection
# -----------------------------------------------------------------------------
def test_14_ble_protocol_dropped_packet_detection():
    decoder = BLEPacketDecoder()

    pkt1 = BLEHIDPacket(seq_num=10, timestamp=1.0, event_code=BLEEventCode.MOVE).encode()
    pkt2 = BLEHIDPacket(seq_num=13, timestamp=1.03, event_code=BLEEventCode.MOVE).encode()  # Gaps: 11, 12 dropped

    decoder.process_raw_packet(pkt1)
    decoder.process_raw_packet(pkt2)

    assert decoder.dropped_packet_count == 2


# -----------------------------------------------------------------------------
# Test 15: BLE Protocol Duplicate Packet Handling
# -----------------------------------------------------------------------------
def test_15_ble_protocol_duplicate_packet_handling():
    decoder = BLEPacketDecoder()

    pkt1 = BLEHIDPacket(seq_num=10, timestamp=1.0, event_code=BLEEventCode.MOVE).encode()

    res1 = decoder.process_raw_packet(pkt1)
    res2 = decoder.process_raw_packet(pkt1)  # Duplicate

    assert res1 is not None
    assert res2 is None
    assert decoder.duplicate_packet_count == 1


# -----------------------------------------------------------------------------
# Test 16: BLE Protocol Timestamp Ordering Check
# -----------------------------------------------------------------------------
def test_16_ble_protocol_timestamp_ordering():
    decoder = BLEPacketDecoder()

    pkt1 = BLEHIDPacket(seq_num=1, timestamp=2.0, event_code=BLEEventCode.MOVE).encode()
    pkt2 = BLEHIDPacket(seq_num=2, timestamp=1.5, event_code=BLEEventCode.MOVE).encode()  # Out-of-order timestamp

    decoder.process_raw_packet(pkt1)
    decoder.process_raw_packet(pkt2)

    assert decoder.out_of_order_count == 1


# -----------------------------------------------------------------------------
# Test 17: Mock BLE Transport Transmission
# -----------------------------------------------------------------------------
def test_17_mock_ble_transport_transmission():
    transport = MockBLETransport(drop_rate=0.0)
    pkt = BLEHIDPacket(seq_num=1, timestamp=1.0, event_code=BLEEventCode.LEFT_CLICK).encode()

    ok = transport.send(pkt)
    assert ok is True
    assert transport.total_sent == 1
    assert transport.total_received == 1

    rcvd = transport.receive()
    assert rcvd == pkt


# -----------------------------------------------------------------------------
# Test 18: End-to-End Pipeline with BLE Adapter
# -----------------------------------------------------------------------------
def test_18_end_to_end_pipeline_with_ble_adapter():
    pipeline = create_calibrated_pipeline()
    ble_adapter = BLEHIDOutputAdapter()
    pipeline.output_adapter.add_adapter(ble_adapter)

    # Move cursor right
    for i in range(10):
        t = 3.0 + i * 0.01
        ax = 1.0 + 0.8 * np.sin(t * 20.0)
        ay = 1.0 + 0.8 * np.cos(t * 20.0)
        az = 9.81 + 0.5 * np.sin(t * 15.0)
        pipeline.process_frame(IMUSample(t, np.array([ax, ay, az]), np.array([0.0, 0.0, -0.40])))

    assert len(ble_adapter.report_history) > 0


# -----------------------------------------------------------------------------
# Test 19: End-to-End Pipeline with Test Adapter
# -----------------------------------------------------------------------------
def test_19_end_to_end_pipeline_with_test_adapter():
    pipeline = create_calibrated_pipeline()
    test_adapter = TestOutputAdapter()
    pipeline.output_adapter.add_adapter(test_adapter)

    # Trigger click
    t = 3.0
    ax = 1.0 + 0.8 * np.sin(t * 20.0)
    ay = 1.0 + 0.8 * np.cos(t * 20.0)
    az = 9.81 + 0.5 * np.sin(t * 15.0)
    pipeline.process_frame(IMUSample(t, np.array([ax, ay, az]), np.array([0.0, 0.0, 0.0])), gesture_input=GestureType.LEFT_CLICK)

    click_ops = [entry for entry in test_adapter.log_entries if entry.operation == "left_click"]
    assert len(click_ops) > 0


# -----------------------------------------------------------------------------
# Test 20: Multi-Adapter Router Dispatch
# -----------------------------------------------------------------------------
def test_20_multi_adapter_router_dispatch():
    router = OutputAdapter()

    test_adapter_1 = TestOutputAdapter()
    test_adapter_2 = TestOutputAdapter()

    router.add_adapter(test_adapter_1)
    router.add_adapter(test_adapter_2)

    evt = SemanticEvent(
        event_type=SemanticEventType.LEFT_CLICK,
        timestamp=1.0,
        cursor_x=960.0,
        cursor_y=540.0,
        state=UnifiedState.LEFT_CLICK
    )
    router.dispatch(evt)

    assert len(test_adapter_1.log_entries) > 0
    assert len(test_adapter_2.log_entries) > 0
    assert test_adapter_1.log_entries[0].operation == "left_click"
    assert test_adapter_2.log_entries[0].operation == "left_click"
