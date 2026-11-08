#!/usr/bin/env python3
"""Show sidecar alerts locally and collect one-time approval for pending calls."""
import json
import os
import platform
import subprocess
import sys
import urllib.request


def notify(title, message):
    system = platform.system()
    if system == "Darwin":
        esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.run(["osascript", "-e", f'display notification "{esc(message)}" with title "{esc(title)}"'], check=False)
    elif system == "Linux":
        if shutil_which("notify-send"):
            subprocess.run(["notify-send", title, message], check=False)
    elif system == "Windows":
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        messagebox.showinfo(title, message)
        root.destroy()


def choose(alert):
    title = "Zero Trust Sidecar approval"
    text = f"{alert['agent']} requested {alert['target']}\n{alert['reason']}\nInjection: {alert.get('injection_confidence')}  Intent: {alert.get('malicious_intent')}"
    system = platform.system()
    if system == "Darwin":
        esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
        script = f'display dialog "{esc(text)}" with title "{esc(title)}" buttons {{"Block", "Allow once"}} default button "Block"'
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        return "allow_once" if "Allow once" in result.stdout else "block"
    if system == "Linux" and shutil_which("zenity"):
        result = subprocess.run(["zenity", "--question", "--title", title, "--text", text, "--ok-label", "Allow once", "--cancel-label", "Block"])
        return "allow_once" if result.returncode == 0 else "block"
    if system == "Windows":
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        answer = messagebox.askyesno(title, text + "\n\nAllow this request once?", default="no")
        root.destroy()
        return "allow_once" if answer else "block"
    notify(title, text + "\nNo approval UI is available; request will be blocked.")
    return "block"


def shutil_which(name):
    import shutil
    return shutil.which(name)


def main():
    base = os.environ.get("SIDECAR_URL", "http://127.0.0.1:8080").rstrip("/")
    token = os.environ.get("JWT_SECRET")
    if not token:
        raise SystemExit("Set ALERT_TOKEN (or JWT_SECRET) to subscribe to sidecar alerts.")
    req = urllib.request.Request(base + "/v1/alerts", headers={"Authorization": "Bearer " + token})
    seen = set()
    with urllib.request.urlopen(req, timeout=None) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            try:
                alert = json.loads(raw[5:].strip())
                if alert["id"] in seen:
                    continue
                seen.add(alert["id"])
                text = f"{alert['target']}: {alert['reason']}"
                if alert.get("approval_required") and alert.get("action") == "pending":
                    decision = choose(alert)
                    payload = json.dumps({"decision": decision}).encode()
                    decision_req = urllib.request.Request(
                        base + "/v1/alerts/" + alert["id"] + "/decision", data=payload,
                        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"}, method="POST")
                    try:
                        urllib.request.urlopen(decision_req, timeout=5).read()
                    except Exception as exc:
                        print(f"Could not submit human decision: {exc}", file=sys.stderr)
                else:
                    notify("Zero Trust Sidecar blocked a request", text)
            except (ValueError, KeyError) as exc:
                print(f"Ignoring malformed alert: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
