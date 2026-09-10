"""Download the pretrained AlbuNet checkpoint used by this benchmark.

Upstream README points at `neat-fullxview-epoch75`, hosted on Google Drive
(file id below is transcribed from upstream README HEAD e1668a4). Google Drive
gives no HTTP checksum; we compute SHA-256 the first time and refuse to reuse
a mismatched file on subsequent runs, so a corrupted or swapped checkpoint
cannot silently poison a benchmark.

Weights are gitignored and never committed. This script is idempotent: it
skips the download if the destination file already matches the recorded
SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

# Google Drive file id transcribed from the pinned upstream README.
# See external/abd/README.md at SHA e1668a470b94252040f27d26098826c293fbb46d.
GDRIVE_FILE_ID = "1XvOX_2N_uMt8OWlHKvA2VzL6mvXaVKvY"  # neat-fullxview-epoch75
DEFAULT_DEST = Path("weights") / "neat-fullxview-epoch75.pth"

# On first successful download we write the observed SHA-256 here. On
# subsequent runs we verify against it. Delete this file to intentionally
# accept a new checkpoint.
CHECKSUM_FILE = Path("weights") / "neat-fullxview-epoch75.sha256.json"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            block = fp.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _gdrive_download(file_id: str, dest: Path) -> None:
    """Download a Google Drive public file to `dest`. Handles the confirm-token
    dance for files >100 MB (Drive interstitial HTML)."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    req = urllib.request.Request(url, headers={"User-Agent": "bfbench-weights-fetch/1.0"})
    with opener.open(req) as resp:
        head = resp.read(4096)
        if b"<html" in head.lower():
            token = None
            for line in head.split(b"\n"):
                if b"confirm=" in line:
                    token = line.split(b"confirm=", 1)[1].split(b"&", 1)[0].decode()
                    break
            if not token:
                raise RuntimeError(
                    "Google Drive returned an HTML interstitial without a confirm "
                    "token. The file may be restricted; fetch manually and place "
                    f"at {dest}."
                )
            url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
            req = urllib.request.Request(url, headers={"User-Agent": "bfbench-weights-fetch/1.0"})
            with opener.open(req) as resp2, dest.open("wb") as out:
                out.write(resp2.read())
        else:
            with dest.open("wb") as out:
                out.write(head)
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    out.write(block)


def _load_pinned_checksum() -> str | None:
    if not CHECKSUM_FILE.is_file():
        return None
    try:
        return json.loads(CHECKSUM_FILE.read_text(encoding="utf-8"))["sha256"]
    except (json.JSONDecodeError, KeyError):
        return None


def _write_pinned_checksum(digest: str) -> None:
    CHECKSUM_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUM_FILE.write_text(
        json.dumps({"sha256": digest, "source": f"gdrive:{GDRIVE_FILE_ID}"}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=None,
                        help=f"destination path (default: env BFBENCH_WEIGHTS or {DEFAULT_DEST})")
    parser.add_argument("--force", action="store_true",
                        help="ignore the pinned checksum and overwrite the file")
    args = parser.parse_args()

    dest = args.dest or Path(os.environ.get("BFBENCH_WEIGHTS", DEFAULT_DEST))
    pinned = _load_pinned_checksum()

    if dest.is_file() and not args.force:
        digest = _sha256(dest)
        if pinned is None:
            print(f"weights already at {dest}; recording sha256 {digest}")
            _write_pinned_checksum(digest)
            return 0
        if digest == pinned:
            print(f"weights ok: {dest} sha256={digest[:12]}...")
            return 0
        print(
            f"ERROR: existing {dest} sha256={digest} does not match pinned "
            f"{pinned}. Refusing to run with an unverified checkpoint. "
            "Pass --force to accept the new file, or delete "
            f"{CHECKSUM_FILE} to re-pin.",
            file=sys.stderr,
        )
        return 2

    print(f"downloading neat-fullxview-epoch75 -> {dest}")
    _gdrive_download(GDRIVE_FILE_ID, dest)
    digest = _sha256(dest)
    if pinned is None or args.force:
        _write_pinned_checksum(digest)
        print(f"pinned sha256={digest}")
    elif digest != pinned:
        print(
            f"ERROR: downloaded file sha256={digest} does not match pinned "
            f"{pinned}. Not overwriting {CHECKSUM_FILE}; delete it manually "
            "to accept the change.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
