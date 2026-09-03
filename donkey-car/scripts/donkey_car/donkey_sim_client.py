"""
donkey_sim_client.py
--------------------
Handles all communication with the Donkey Sim server.

The sim runs a TCP server on port 9091.
Messages are UTF-8-encoded JSON. The sim terminates each outgoing
packet with a newline '\n'. We buffer incoming data and split on '\n'
to reliably parse one message at a time.

All outgoing messages just need valid JSON — no newline required.
"""

import json
import socket
import threading
import time

class DonkeySimClient:
    """TCP client for the Donkey Sim server."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        self._socket               = None
        self._running              = False
        self._on_telemetry         = None   # callback registered by the caller
        self._on_node_position     = None   # callback for node_position replies

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self):
        """Open the TCP connection to the sim."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self.host, self.port))
        self._running = True
        print(f"[Client] Connected to Donkey Sim at {self.host}:{self.port}")

    def disconnect(self):
        """Close the connection and stop the receive thread."""
        self._running = False
        if self._socket:
            self._socket.close()
        print("[Client] Disconnected.")

    # ── Send helpers ──────────────────────────────────────────────────────────

    def _send(self, msg_dict: dict):
        """Serialize a dict to JSON and send it to the sim."""
        raw = json.dumps(msg_dict).encode("utf-8")
        self._socket.sendall(raw)

    def load_scene(self, scene_name: str):
        """
        Ask the sim to load a scene.
        Valid options: "generated_road", "generated_track",
                       "warehouse", "sparkfun_avc"
        """
        self._send({"msg_type": "load_scene", "scene_name": scene_name})
        print(f"[Client] Loading scene: {scene_name}")

    def send_control(self, steering: float, throttle: float, brake: float = 0.0):
        """
        Send a throttle/steering command to the car.

        steering : -1.0 (full left) to 1.0 (full right)
        throttle : -1.0 (full reverse) to 1.0 (full forward)
        brake    :  0.0 (none) to 1.0 (full)
        """
        self._send({
            "msg_type" : "control",
            "steering" : str(steering),
            "throttle" : str(throttle),
            "brake"    : str(brake),
        })

    def reset_car(self):
        """Return the car to the track start position."""
        self._send({"msg_type": "reset_car"})
        print("[Client] Car reset to start.")

    def request_node_position(self, index: int):
        """
        Ask the sim for the world position of one track waypoint node.

        The sim replies asynchronously with a "node_position" message
        containing pos_x, pos_y, pos_z (and optional quaternion fields).
        Register a callback via set_node_position_callback() to receive it.

        NOTE: responses carry no index field, so request nodes one at a
        time and wait for each reply before requesting the next.
        """
        self._send({"msg_type": "node_position", "index": str(index)})

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def set_telemetry_callback(self, callback):
        """
        Register a function to call each time a telemetry packet arrives.

        The callback receives the raw telemetry dict, which includes fields
        like 'cte', 'speed', 'steering_angle', 'hit', pos_x/z, yaw, etc.

        Signature: callback(msg: dict) -> None
        """
        self._on_telemetry = callback

    def set_node_position_callback(self, callback):
        """
        Register a function to call when a node_position reply arrives.

        The callback receives the raw message dict with keys:
            pos_x, pos_y, pos_z  (world coordinates as strings)
            Qx, Qy, Qz, Qw       (quaternion, optional)

        Signature: callback(msg: dict) -> None
        """
        self._on_node_position = callback

    # ── Receive loop ──────────────────────────────────────────────────────────

    def start_listening(self):
        """Start the background thread that reads messages from the sim."""
        t = threading.Thread(target=self._receive_loop, daemon=True)
        t.start()
        print("[Client] Listening for sim messages...")

    def _receive_loop(self):
        """
        Continuously read from the socket, split on newlines, and
        dispatch each complete JSON message to _handle_message().
        """
        buffer = ""
        while self._running:
            try:
                data = self._socket.recv(4096).decode("utf-8")
                if not data:
                    print("[Client] Sim closed the connection.")
                    break

                buffer += data

                # The sim ends every packet with '\n' — split on that
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        try:
                            msg = json.loads(line)
                            self._handle_message(msg)
                        except json.JSONDecodeError as e:
                            print(f"[Client] JSON parse error: {e} | raw: {line[:80]}")

            except OSError:
                if self._running:
                    print("[Client] Socket error in receive loop.")
                break

    def _handle_message(self, msg: dict):
        """Route an incoming sim message to the right handler."""
        msg_type = msg.get("msg_type")

        if msg_type == "telemetry":
            if self._on_telemetry:
                self._on_telemetry(msg)

        elif msg_type == "node_position":
            if self._on_node_position:
                self._on_node_position(msg)

        elif msg_type == "scene_selection_ready":
            print("[Sim] Scene selection ready.")

        elif msg_type == "scene_loaded":
            print("[Sim] Scene loaded.")

        elif msg_type == "car_loaded":
            print("[Sim] Car loaded — ready to drive.")

        elif msg_type == "protocol_version":
            print(f"[Sim] Protocol version {msg.get('version')}.")