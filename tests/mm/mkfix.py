#!/usr/bin/env python3
"""G5 fixtures: mkfix.py OUTDIR [COCO_JPG]. Real encoder output (PIL) for every accepted format + edge cases."""
import os, shutil, struct, sys, wave
from PIL import Image
F = sys.argv[1]; os.makedirs(F, exist_ok=True)
im = Image.new("RGB", (64, 48), (200, 30, 30))
im.save(f"{F}/a.png"); im.save(f"{F}/a.jpg", quality=90); im.save(f"{F}/a.gif")
im.save(f"{F}/a-lossy.webp", lossless=False); im.save(f"{F}/a-lossless.webp", lossless=True)
Image.new("RGBA", (33, 17), (1, 2, 3, 128)).save(f"{F}/a-alpha.webp")      # VP8X
Image.new("L", (9000, 8)).save(f"{F}/wide.png")                            # over the side cap
Image.new("RGB", (1, 1)).save(f"{F}/one.png")
Image.new("RGB", (320, 240), (10, 200, 10)).save(f"{F}/prog.jpg", progressive=True)
with wave.open(f"{F}/a.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(struct.pack("<800h", *range(800)))
open(f"{F}/a.mp3", "wb").write(b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" + bytes(400))
open(f"{F}/a.txt", "wb").write(b"hello, not an image\n")
open(f"{F}/big.bin", "wb").write(b"\x89PNG\r\n\x1a\n" + os.urandom(7 * 1024 * 1024))  # PNG magic, no IHDR, 7 MiB
coco = sys.argv[2] if len(sys.argv) > 2 else sys.exit("usage: mkfix.py OUT IMAGE")
shutil.copy(coco, f"{F}/coco142472.jpg")
