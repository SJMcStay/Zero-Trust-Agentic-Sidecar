#!/usr/bin/env python3
"""Put a decision model (default: kev) on local disk.

Stdlib only, Python 3.8+ (works with macOS system python3).
Reads models.json; verifies against models.lock.json when present.

  python3 scripts/fetch_model.py                  # fetch kev from Hugging Face
  python3 scripts/fetch_model.py --lock           # fetch, record revision + sha256 in models.lock.json
  python3 scripts/fetch_model.py --from-dir DIR   # copy weights shared outside Hugging Face
  python3 scripts/fetch_model.py --repo org/name  # override repo (testing)

Private repo: export HF_TOKEN=hf_...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "models.json"
LOCK = ROOT / "models.lock.json"
HF = "https://huggingface.co"
CHUNK = 1 << 20


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


class StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Hugging Face redirects large files to a CDN. Never send our token to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


OPENER = urllib.request.build_opener(StripAuthOnRedirect)


def http_get(url: str, token: str | None):
    req = urllib.request.Request(url, headers={"User-Agent": "zt-sidecar-fetch/0.1"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return OPENER.open(req, timeout=60)


def resolve_revision(repo: str, revision: str, token: str | None) -> str:
    """Turn a branch/tag into a commit sha, so every teammate gets identical files."""
    url = f"{HF}/api/models/{repo}/revision/{urllib.parse.quote(revision, safe='')}"
    try:
        with http_get(url, token) as r:
            return json.load(r)["sha"]
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            die(f"cannot access {repo}@{revision} (HTTP {e.code}). Wrong repo id, or private: export HF_TOKEN=...")
        raise


def download(url: str, dest: Path, token: str | None) -> bool:
    """Stream url to dest via a .part file. Returns False on 404."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with http_get(url, token) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            for block in iter(lambda: r.read(CHUNK), b""):
                f.write(block)
                done += len(block)
                if total and sys.stdout.isatty():
                    print(f"\r  {dest.name}: {done * 100 // total}%", end="", flush=True)
        print(f"\r  {dest.name}: {done / 1e6:.1f} MB".ljust(60))
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        if e.code == 404:
            return False
        die(f"{url}: HTTP {e.code}")
    os.replace(tmp, dest)  # atomic: a crash never leaves a half-written model file
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", default="kev", help="entry in models.json (default: kev)")
    ap.add_argument("--lock", action="store_true", help="record revision + sha256 in models.lock.json")
    ap.add_argument("--from-dir", type=Path, help="copy files from a local directory instead of downloading")
    ap.add_argument("--repo", help="override Hugging Face repo id")
    ap.add_argument("--revision", help="override branch, tag or commit")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = ap.parse_args()

    models = json.loads(CONFIG.read_text())
    locks = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    if args.name not in models:
        die(f"{args.name!r} not in models.json")
    # Dependencies first (e.g. Kev's LoRA adapter needs its base model). Overrides apply to the named model only.
    for dep in models[args.name].get("requires", []):
        if dep not in models:
            die(f"{args.name} requires {dep!r}, not in models.json")
        fetch_one(dep, models[dep], locks, argparse.Namespace(lock=args.lock, from_dir=None, repo=None, revision=None, force=args.force))
    fetch_one(args.name, models[args.name], locks, args)


def fetch_one(name: str, cfg: dict, locks: dict, args: argparse.Namespace) -> None:
    pin = locks.get(name)
    dest = ROOT / cfg["dest"]
    wanted = [(f, True) for f in cfg["files"]] + [(f, False) for f in cfg.get("optional_files", [])]

    if args.from_dir:
        source = {"path": str(args.from_dir.resolve())}
        for f, required in wanted:
            src = args.from_dir / f
            if src.exists():
                (dest / f).parent.mkdir(parents=True, exist_ok=True)
                if src.resolve() != (dest / f).resolve():
                    shutil.copyfile(src, dest / f)
                print(f"  {f}: copied")
            elif required:
                die(f"{src} missing")
    else:
        repo = args.repo or cfg.get("repo")
        if not repo:
            die(f"set {name}.repo in models.json to the Hugging Face repo id, or use --from-dir")
        token = os.environ.get("HF_TOKEN")
        use_pin = pin and pin.get("repo") == repo and not (args.lock or args.repo or args.revision)
        revision = pin["revision"] if use_pin else resolve_revision(repo, args.revision or cfg.get("revision", "main"), token)
        source = {"repo": repo, "revision": revision}
        print(f"{name}: {repo}@{revision[:12]}")
        for f, required in wanted:
            path = dest / f
            if path.exists() and not args.force and (not use_pin or sha256(path) == pin["files"].get(f)):
                print(f"  {f}: present")
                continue
            if not download(f"{HF}/{repo}/resolve/{revision}/{f}", path, token) and required:
                die(f"{f} not found in {repo}@{revision}")

    present = {f: sha256(dest / f) for f, _ in wanted if (dest / f).exists()}
    weights = cfg.get("weights_any_of", [])
    if weights and not any(w in present for w in weights):
        die(f"no weights in {dest}; expected one of {weights}")

    if args.lock:
        locks[name] = {**source, "files": present}
        LOCK.write_text(json.dumps(locks, indent=2) + "\n")
        print(f"locked {len(present)} files in {LOCK.name}; commit it")
    elif pin:
        bad = [f for f, h in pin["files"].items() if present.get(f) != h]
        if bad:
            die(f"sha256 mismatch: {bad}. Delete {cfg['dest']} and re-fetch, or --lock if the model changed on purpose")
        print("verified against models.lock.json")
    else:
        print("warning: no lock entry, files unverified. Run once with --lock and commit models.lock.json")

    # Read by the sidecar at startup: model id for the audit log, file hashes for integrity.
    manifest = {"name": name, **source, "files": present}
    if cfg.get("positive_label"):  # label meaning "risky"; Rust defaults to index 1 for binary models
        manifest["positive_label"] = cfg["positive_label"]
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"ok: {cfg['dest']}")


if __name__ == "__main__":
    main()
