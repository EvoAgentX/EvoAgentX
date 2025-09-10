import os
import time


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def unique_filename(prefix: str, ext: str = "png") -> str:
    ts = int(time.time())
    return f"{prefix}_{ts}.{ext}"



