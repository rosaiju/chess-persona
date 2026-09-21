#!/usr/bin/env python3
"""
SenseRobot Chess Persona Client
================================
Run this on any machine connected to the SenseRobot to receive quips
from chess-persona and speak them through the robot/local speakers.

Usage:
    python senserobot_client.py [SERVER_URL]

    SERVER_URL defaults to http://localhost:8001
    Set it to the IP of the machine running chess-persona, e.g.:
        python senserobot_client.py http://192.168.1.42:8001

The client subscribes to the /events SSE stream and speaks each quip
using the best available TTS method on the current platform.
"""

import sys
import json
import platform
import subprocess
import time
import urllib.request

SERVER = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8001"
EVENTS_URL = f"{SERVER}/events"
TTS_URL = f"{SERVER}/tts"


def speak_via_server(text: str, personality: str = "Cocky") -> bool:
    """Fetch TTS audio from chess-persona server and play it locally."""
    try:
        url = f"{TTS_URL}?text={urllib.parse.quote(text)}&personality={urllib.parse.quote(personality)}"
        with urllib.request.urlopen(url, timeout=10) as r:
            audio = r.read()

        # Write to temp file and play
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio)
            tmp = f.name

        system = platform.system()
        if system == "Windows":
            subprocess.run(["powershell", "-c", f'(New-Object Media.SoundPlayer).PlaySync()'],
                           check=False)
            # Use Windows Media Player for mp3
            subprocess.run(["cmd", "/c", "start", "/wait", "", tmp], check=False)
        elif system == "Darwin":
            subprocess.run(["afplay", tmp], check=False)
        else:
            # Linux / NAO (try mpg123 or aplay)
            for player in ["mpg123", "mpg321", "cvlc --play-and-exit"]:
                if subprocess.run(["which", player.split()[0]],
                                   capture_output=True).returncode == 0:
                    subprocess.run(player.split() + [tmp], check=False)
                    break

        os.unlink(tmp)
        return True
    except Exception as e:
        print(f"[TTS server] {e}")
        return False


def speak_local(text: str) -> bool:
    """Fall back to local system TTS."""
    system = platform.system()
    try:
        if system == "Windows":
            script = (
                'Add-Type -AssemblyName System.Speech; '
                '$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                f'$s.Speak("{text.replace(chr(39), "")}")'
            )
            subprocess.run(["powershell", "-Command", script], check=True,
                           capture_output=True)
        elif system == "Darwin":
            subprocess.run(["say", text], check=True)
        else:
            subprocess.run(["espeak", "-v", "en", text], check=True)
        return True
    except Exception as e:
        print(f"[TTS local] {e}")
        return False


def speak(text: str, personality: str = "Cocky"):
    print(f"[QUIP] {text}")
    if not speak_via_server(text, personality):
        speak_local(text)


_robot_color = "black"   # updated on each "started" event
_current_fen = None      # updated on each "fen" event


def on_game_started(event: dict):
    global _robot_color, _current_fen
    _robot_color = event.get("color", "black")
    _current_fen = event.get("fen")
    print(f"[GAME] Started vs {event.get('opponent')} — robot is {_robot_color}")
    print(f"[GAME] Starting FEN: {_current_fen}")


def on_fen(event: dict):
    global _current_fen
    _current_fen = event.get("fen")


def on_robot_move(event: dict):
    uci = event.get("uci", "")
    move_num = event.get("moveNum", "?")
    print(f"[MOVE] Robot played: {uci}  (move {move_num})  fen={event.get('fen', '')}")


def connect_and_listen():
    print(f"[SenseRobot Client] Connecting to {EVENTS_URL} ...")
    while True:
        try:
            import urllib.parse
            req = urllib.request.Request(
                EVENTS_URL,
                headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
            )
            with urllib.request.urlopen(req, timeout=None) as resp:
                print("[SenseRobot Client] Connected. Waiting for game events...")
                buf = b""
                while True:
                    chunk = resp.read(1)
                    if not chunk:
                        break
                    buf += chunk
                    if buf.endswith(b"\n\n"):
                        lines = buf.decode("utf-8").strip().splitlines()
                        buf = b""
                        for line in lines:
                            if line.startswith("data: "):
                                try:
                                    event = json.loads(line[6:])
                                    etype = event.get("type")
                                    if etype == "started":
                                        on_game_started(event)
                                    elif etype == "fen":
                                        on_fen(event)
                                    elif etype == "move":
                                        on_robot_move(event)
                                    elif etype == "quip":
                                        # Browser handles all TTS; client only logs
                                        delay_ms = event.get("delay_ms", 0)
                                        cap = event.get("capture") is True
                                        print(f"[{time.strftime('%H:%M:%S')}][QUIP] delay_ms={delay_ms} capture={cap} text={event['text'][:40]!r} (TTS via browser)")
                                    elif etype == "done":
                                        print(f"[GAME] Over — {event.get('result', event.get('status'))}")
                                except json.JSONDecodeError:
                                    pass
        except KeyboardInterrupt:
            print("\n[SenseRobot Client] Stopped.")
            sys.exit(0)
        except Exception as e:
            print(f"[SenseRobot Client] Disconnected: {e}. Retrying in 3s...")
            time.sleep(3)


if __name__ == "__main__":
    import urllib.parse
    connect_and_listen()
