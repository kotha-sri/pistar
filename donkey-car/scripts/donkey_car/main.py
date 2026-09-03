"""
main.py
-------
Entry point for the CTE + PID track follower.

Run with:
    python main.py

Make sure the Donkey Sim executable is already running before starting.

How it works
    Steering is a PID controller on the cross-track error (CTE) the sim
    reports each telemetry packet — no waypoints or world position needed.
    Throttle is a proportional speed controller. This is the algorithmic
    "expert" baseline that an RL agent can later refine.

Startup sequence:
    1. Connect to the sim, load the scene.
    2. Wait for the first telemetry packet (sanity check the sim is sending).
    3. Drive: feed each packet's CTE + speed into the controller.
"""

import time
import os
import ast
import msvcrt

from donkey_sim_client import DonkeySimClient
from path_follower     import PathFollower

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {

    # --- Sim connection ---
    "host" : "127.0.0.1",
    "port" : 9091,

    # Scene to load. Options: "generated_road", "generated_track",
    #                          "warehouse",      "sparkfun_avc"
    "scene_name" : "circuit_launch",

    # Seconds to wait after load_scene before expecting telemetry.
    # Increase on slower machines.
    "scene_load_wait" : 2.0,

    # --- Safety ---
    # Collision detection: if 'hit' is not 'none', reset the car.
    "reset_on_hit" : False,
    "reset_on_off_track" : True,
    "off_track_threshold" : 4.0,
}
# ──────────────────────────────────────────────────────────────────────────────

def load_config_from_file(filepath: str) -> dict:
    """Safely extract the CONFIG dictionary from a source file (hot-reload)."""
    print(f"\n[HotReload] Detected save on {os.path.basename(filepath)}. Attempting read...")
    try:
        # Give the text editor 200ms to finish writing the file to disk
        time.sleep(0.2)

        # Force UTF-8 so Windows doesn't choke on math symbols / dashes
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        if not source.strip():
            print(f"[HotReload] FAIL: File read as empty (editor race condition).")
            return None

        # Safely parse the text into an Abstract Syntax Tree
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if getattr(target, 'id', '') == 'CONFIG':
                        config = ast.literal_eval(node.value)
                        print(f"[HotReload] SUCCESS: Extracted config with {len(config)} keys.")
                        return config

    except Exception as e:
        print(f"[HotReload] ERROR: Parsing failed -> {type(e).__name__}: {e}")

    return None


def main():

    # ── Build the controller ───────────────────────────────────────────────────
    controller = PathFollower()
    controller_file_name = controller.get_file_name()

    # ── Connect to sim ─────────────────────────────────────────────────────────
    client = DonkeySimClient(CONFIG["host"], CONFIG["port"])
    client.connect()
    client.start_listening()

    time.sleep(1.0)   # let the sim send scene_selection_ready
    client.load_scene(CONFIG["scene_name"])
    time.sleep(CONFIG["scene_load_wait"])   # wait for scene + car to load

    # ── Drive: PID on CTE ──────────────────────────────────────────────────────
    def on_telemetry(msg: dict):
        speed = float(msg.get("speed", 0.0))
        cte   = float(msg.get("cte",   0.0))
        hit   = msg.get("hit", "none")

        print(
            f"cte={cte:6.2f}  "
            f"speed={speed:5.2f}  "
            f"hit={hit}"
        )

        # Reset on collision
        if CONFIG["reset_on_hit"] and hit != "none":
            print(f"\n[Main] Collision with '{hit}' — resetting car...")
            client.send_control(0.0, 0.0)
            time.sleep(0.2)
            client.reset_car()
            controller.reset()
            return

        # Reset if the car has wandered too far off the centerline
        if CONFIG["reset_on_off_track"] and abs(cte) > CONFIG["off_track_threshold"]:
            print(f"\n[Main] Car off track (cte={cte:.2f}) — resetting car...")
            client.send_control(0.0, 0.0)
            time.sleep(0.2)
            client.reset_car()
            controller.reset()
            return

        steering, throttle = controller.compute_controls(cte, speed)
        client.send_control(steering, throttle)

    client.set_telemetry_callback(on_telemetry)

    print("[Main] Waiting for first telemetry packet...")
    time.sleep(1.0)   # on_telemetry will start firing as packets arrive

    # ── Run until Ctrl-C (with hot reloading) ──────────────────────────────────
    print("\nCTE + PID follower running. Press Ctrl-C to stop, Ctrl-R to reset car.")
    print(f"Hot-reload enabled: save CONFIG in main.py or {controller_file_name} to update live.\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(base_dir, "main.py")
    controller_path = os.path.join(base_dir, controller_file_name)

    last_mtime_main = os.path.getmtime(main_path)
    last_mtime_controller = os.path.getmtime(controller_path)

    try:
        while True:
            time.sleep(1.0)

            try:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\x12':  # Ctrl+R (ASCII 18)
                        print("\n[Main] Manual reset triggered via Ctrl+R!")
                        client.send_control(0.0, 0.0)
                        time.sleep(0.2)
                        client.reset_car()
                        controller.reset()
                        while msvcrt.kbhit():   # drain queued keystrokes
                            msvcrt.getch()
            except OSError:
                pass   # no interactive console (e.g. output redirected) — skip key polling

            # --- Hot-reload main.py CONFIG ---
            current_mtime_main = os.path.getmtime(main_path)
            if current_mtime_main > last_mtime_main:
                new_config = load_config_from_file(main_path)
                if new_config:
                    CONFIG.update(new_config)
                    print("\n[HotReload] Updated main.py CONFIG!")
                last_mtime_main = current_mtime_main

            # --- Hot-reload controller (PID gains) ---
            current_mtime_controller = os.path.getmtime(controller_path)
            if current_mtime_controller > last_mtime_controller:
                new_controller_config = load_config_from_file(controller_path)
                if new_controller_config:
                    controller.update_config(new_controller_config)
                    print(f"\n[HotReload] Updated {controller_file_name} CONFIG!")
                last_mtime_controller = current_mtime_controller

    except KeyboardInterrupt:
        print("\nStopping — sending zero controls...")
        client.send_control(0.0, 0.0)
        time.sleep(0.2)
        client.disconnect()


if __name__ == "__main__":
    main()
