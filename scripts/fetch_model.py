#!/usr/bin/env python3
"""Download a model from models.json into models/, pinned and sha256-verified.

Stdlib only, Python 3.8+ (works with macOS system python3). Fetches the model's `requires`
(its base model) first. Checks free disk before downloading. Every file is checked against
models.lock.json when locked, and large files against Hugging Face's own sha256 always.

  python3 scripts/fetch_model.py                    # $KEV_MODEL, default kev-4b
  python3 scripts/fetch_model.py kev-0.8b           # smaller Kev for <32 GB machines
  python3 scripts/fetch_model.py kev-4b --lock-remote   # pin hashes from Hugging Face, no weights download
  python3 scripts/fetch_model.py kev-4b --lock      # download, then pin hashes of the local files
  python3 scripts/fetch_model.py kev-4b --from-dir DIR  # copy weights shared outside Hugging Face

Private repo: export HF_TOKEN=hf_...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
DISK_MARGIN = 1 << 30  # keep 1 GiB free after downloading
COMMIT = re.compile(r"^[0-9a-f]{40}$")
DEBUG = os.environ.get("ZT_DEBUG", "0") not in ("", "0")  # ZT_DEBUG=1: explain each step on stderr


def debug(msg: str) -> None:
    if DEBUG:
        print(f"[debug] {msg}", file=sys.stderr, flush=True)


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256_file(path: Path) -> str:
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
    req = urllib.request.Request(url, headers={"User-Agent": "zt-sidecar-fetch/0.2"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return OPENER.open(req, timeout=60)


def api_json(url: str, token: str | None, what: str):
    try:
        with http_get(url, token) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            die(f"cannot access {what} (HTTP {e.code}). Wrong repo id, or private: export HF_TOKEN=...")
        raise


def resolve_revision(repo: str, revision: str, token: str | None) -> str:
    """Commit sha for a branch or tag. Uses /refs: /revision/<tag> has returned main's sha for tags."""
    if COMMIT.match(revision):
        debug(f"{repo}: revision {revision[:12]} is a commit sha, used as-is")
        return revision
    debug(f"{repo}: resolving {revision!r} via {HF}/api/models/{repo}/refs")
    refs = api_json(f"{HF}/api/models/{repo}/refs", token, f"{repo} refs")
    for kind in ("branches", "tags"):
        for ref in refs.get(kind, []):
            if ref.get("name") == revision:
                return ref["targetCommit"]
    die(f"{repo} has no branch or tag {revision!r}")


def remote_files(repo: str, revision: str, token: str | None) -> dict:
    """{filename: {"size": bytes, "sha256": lfs sha256 or None}} at this commit."""
    debug(f"GET {HF}/api/models/{repo}/revision/{revision}?blobs=true (sizes + sha256)")
    meta = api_json(f"{HF}/api/models/{repo}/revision/{revision}?blobs=true", token, f"{repo}@{revision}")
    return {
        s["rfilename"]: {"size": s.get("size", 0), "sha256": (s.get("lfs") or {}).get("sha256")}
        for s in meta.get("siblings", [])
    }


def download(url: str, dest: Path, token: str | None) -> None:
    """Stream url to dest via a .part file, renamed only when complete."""
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
        print(f"\r  {dest.name}: {done / 1e6:.1f} MB".ljust(70))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dest)  # atomic: a crash never leaves a half-written model file


def remote_sha256(repo: str, revision: str, name: str, info: dict, token: str | None) -> str:
    """sha256 of a remote file: Hugging Face's LFS hash, or hash the (small) file in memory."""
    if info["sha256"]:
        return info["sha256"]
    with http_get(f"{HF}/{repo}/resolve/{revision}/{name}", token) as r:
        return hashlib.sha256(r.read()).hexdigest()


def fetch_one(name: str, cfg: dict, locks: dict, args: argparse.Namespace) -> None:
    dest = ROOT / cfg["dest"]
    wanted = [(f, True) for f in cfg["files"]] + [(f, False) for f in cfg.get("optional_files", [])]
    pin = locks.get(name) if not (args.lock or args.lock_remote) else None

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
        repo = cfg["repo"]
        token = os.environ.get("HF_TOKEN")
        revision = resolve_revision(repo, cfg.get("revision", "main"), token)
        if pin and pin.get("revision") != revision:
            die(f"{name}: models.json pins {revision[:12]} but models.lock.json has {pin.get('revision', '?')[:12]}; re-lock")
        source = {"repo": repo, "revision": revision}
        print(f"{name}: {repo}@{revision[:12]}")
        files = remote_files(repo, revision, token)
        for f, required in wanted:
            if f not in files and required:
                die(f"{f} not in {repo}@{revision[:12]}")
        wanted = [(f, r) for f, r in wanted if f in files]

        if args.lock_remote:
            locks[name] = {**source, "files": {f: remote_sha256(repo, revision, f, files[f], token) for f, _ in wanted}}
            LOCK.write_text(json.dumps(locks, indent=2) + "\n")
            print(f"  pinned {len(wanted)} files from Hugging Face metadata in {LOCK.name}; commit it")
            return

        def ok_on_disk(f: str) -> bool:
            path = dest / f
            if not path.exists() or args.force:
                return False
            expected = (pin or {}).get("files", {}).get(f) or files[f]["sha256"]
            return expected is None or path.stat().st_size == files[f]["size"] and sha256_file(path) == expected

        todo = [f for f, _ in wanted if not ok_on_disk(f)]
        need = sum(files[f]["size"] for f in todo)
        free = shutil.disk_usage(ROOT).free
        debug(f"{name}: {len(todo)} of {len(wanted)} files to download, {need / 1e9:.2f} GB; {free / 1e9:.1f} GB free")
        if need + DISK_MARGIN > free:
            die(f"{name} needs {need / 1e9:.1f} GB, {free / 1e9:.1f} GB free (plus 1 GB margin). "
                f"Free space (scripts/cleanup.sh) or pick a smaller model (kev-0.8b).")
        dest.mkdir(parents=True, exist_ok=True)
        for f, _ in wanted:
            if f not in todo:
                print(f"  {f}: present")
                continue
            debug(f"GET {HF}/{repo}/resolve/{revision[:12]}/{f} -> {dest / f}")
            download(f"{HF}/{repo}/resolve/{revision}/{f}", dest / f, token)
            lfs = files[f]["sha256"]
            if lfs and sha256_file(dest / f) != lfs:
                (dest / f).unlink()
                die(f"{f}: sha256 differs from Hugging Face's; deleted. Retry, or check your network.")

    present = {f: sha256_file(dest / f) for f, _ in wanted if (dest / f).exists()}
    for f, h in present.items():
        debug(f"{name}/{f}: sha256 {h[:16]}… (lock: {((pin or {}).get('files', {}).get(f) or 'none')[:16]})")
    if not any(w in present for w in cfg.get("weights_any_of", [])):
        die(f"no weights in {dest}; expected one of {cfg.get('weights_any_of')}")

    if args.lock:
        locks[name] = {**source, "files": present}
        LOCK.write_text(json.dumps(locks, indent=2) + "\n")
        print(f"  locked {len(present)} files in {LOCK.name}; commit it")
    elif pin:
        bad = [f for f, h in pin["files"].items() if present.get(f) != h]
        if bad:
            die(f"sha256 mismatch vs models.lock.json: {bad}. Delete {cfg['dest']} and re-fetch.")
        print("  verified against models.lock.json")
    else:
        print("  warning: no lock entry; run with --lock-remote once and commit models.lock.json")
    print(f"ok: {cfg['dest']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", default=os.environ.get("KEV_MODEL", "kev-4b"), help="entry in models.json")
    pin = ap.add_mutually_exclusive_group()
    pin.add_argument("--lock", action="store_true", help="download, then record sha256 of local files")
    pin.add_argument("--lock-remote", action="store_true", help="record sha256 from Hugging Face; no weights download")
    pin.add_argument("--from-dir", type=Path, help="copy files from a local directory instead of downloading")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = ap.parse_args()

    models = json.loads(CONFIG.read_text())
    locks = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    if args.name not in models:
        die(f"{args.name!r} not in models.json (have: {', '.join(models)})")
    # Dependencies first (Kev's LoRA adapter needs its base). --from-dir applies to the named model only.
    for dep in models[args.name].get("requires", []):
        if dep not in models:
            die(f"{args.name} requires {dep!r}, not in models.json")
        dep_args = argparse.Namespace(**{**vars(args), "from_dir": None})
        fetch_one(dep, models[dep], locks, dep_args)
    fetch_one(args.name, models[args.name], locks, args)


if __name__ == "__main__":
    main()
