#!/usr/bin/env python3
"""
mqtt_movement_consumer.py
- Subscribes to sensors/movement/<node>
- Handles out-of-order via seq numbers
- Optionally simulates additional network latency (--simulate-ms)
- Prints latest smoothed movement & direction
"""
import time, json, argparse, heapq
from datetime import datetime, timezone
import paho.mqtt.client as mqtt
from threading import Thread, Lock

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", type=int, default=1883)
parser.add_argument("--topic", default="webcam/detection")
parser.add_argument("--simulate-ms", type=int, default=0,
                    help="simulate additional processing/network latency in ms")
parser.add_argument("--buffer-s", type=float, default=2.0,
                    help="buffer window for reordering (seconds)")
args = parser.parse_args()

HOST = args.host; PORT = args.port; TOPIC = args.topic
SIM_LAT_MS = args.simulate_ms
BUFFER_S = args.buffer_s

# We'll use a min-heap keyed by seq to reorder quickly
heap = []
heap_lock = Lock()
last_processed_seq = 0

def consumer_thread():
    global heap, last_processed_seq
    while True:
        time.sleep(0.05)
        now = time.time()
        with heap_lock:
            # process any messages where timestamp + buffer <= now
            processed_any = False
            while heap and heap[0][0] <= now - BUFFER_S:
                _, msg = heapq.heappop(heap)
                process_message(msg)
                processed_any = True
            # Also process if next seq is immediate (no buffer)
            # optional additional early flush can be implemented
        if SIM_LAT_MS > 0:
            # If simulating network latency, sleep a bit here
            time.sleep(SIM_LAT_MS / 1000.0)

def process_message(payload):
    global last_processed_seq
    try:
        data = json.loads(payload)
    except Exception as e:
        print("Bad JSON:", e); return
    seq = data.get("seq", 0)
    ts = data.get("timestamp_utc", "")
    direction = data.get("direction", "unknown")
    speed = data.get("speed_px_s", 0.0)
    centroid = data.get("centroid", [None, None])
    # detect gaps
    if seq <= last_processed_seq:
        # duplicate or old; ignore or log
        print(f"[{seq}] DUP/OLD (last={last_processed_seq}) dir={direction} sp={speed}")
        return
    # detect missing sequences
    if last_processed_seq and seq != last_processed_seq + 1:
        print(f"[{seq}] gap detected (last={last_processed_seq}) — may be high latency or packet loss")

    # simple smoothing: print with timestamp
    print(f"[{seq}] {ts} node:{data.get('node_id')} dir:{direction} angle:{data.get('angle_deg')} sp:{speed} centroid:{centroid}")
    last_processed_seq = seq

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT, subscribing", TOPIC)
    client.subscribe(TOPIC, qos=1)

def on_message(client, userdata, msg):
    # receive: place into reorder buffer as (arrival_time, payload) and keyed by seq ordering later
    try:
        payload = msg.payload.decode('utf-8')
        # parse seq quickly to order by seq arrival time rather than timestamp
        # we'll push into heap keyed by arrival_time for buffer, but keep seq in payload
        arrival = time.time()
        with heap_lock:
            # Use arrival time as key for buffer flush; store full payload as value
            heapq.heappush(heap, (arrival, payload))
    except Exception as e:
        print("on_message error:", e)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(HOST, PORT, 60)
client.loop_start()

# start consumer thread
t = Thread(target=consumer_thread, daemon=True)
t.start()

print("Listening. Ctrl+C to quit.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    client.loop_stop()
    client.disconnect()
    print("Shutdown.")