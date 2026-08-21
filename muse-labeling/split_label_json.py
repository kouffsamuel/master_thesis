#!/usr/bin/env python3
"""
Split a large MUSE labeling JSON into smaller top-level frame chunks.

The browser labeling tool loads the selected JSON file into memory. Very large
exports such as a multi-GB yolo_tracking_data.json should be split first, while
camera/ and rd_raw/ stay in the same folder and are reused by every chunk.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FRAME_RE = re.compile(r"frame_(\d+)\.jpeg$")
CHUNK_SIZE = 8 * 1024 * 1024


class JsonStream:
    def __init__(self, path):
        self.fp = path.open("r", encoding="utf-8")
        self.decoder = json.JSONDecoder()
        self.buffer = ""
        self.pos = 0
        self.eof = False

    def close(self):
        self.fp.close()

    def fill(self):
        if self.eof:
            return
        chunk = self.fp.read(CHUNK_SIZE)
        if chunk:
            self.buffer += chunk
        else:
            self.eof = True

    def compact(self):
        if self.pos > CHUNK_SIZE:
            self.buffer = self.buffer[self.pos:]
            self.pos = 0

    def ensure(self):
        while self.pos >= len(self.buffer) and not self.eof:
            self.fill()

    def skip_ws(self):
        while True:
            self.ensure()
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer) or self.eof:
                self.compact()
                return
            self.fill()

    def peek(self):
        self.skip_ws()
        if self.pos >= len(self.buffer):
            return ""
        return self.buffer[self.pos]

    def consume(self, expected=None):
        self.skip_ws()
        if self.pos >= len(self.buffer):
            raise ValueError("Unexpected EOF")
        ch = self.buffer[self.pos]
        if expected is not None and ch != expected:
            raise ValueError(f"Expected {expected!r}, got {ch!r}")
        self.pos += 1
        self.compact()
        return ch

    def raw_decode(self):
        self.skip_ws()
        while True:
            try:
                obj, end = self.decoder.raw_decode(self.buffer, self.pos)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.fill()
                continue
            self.pos = end
            self.compact()
            return obj


def frame_number(key):
    match = FRAME_RE.search(key)
    return int(match.group(1)) if match else None


def chunk_name(prefix, first_key, last_key, chunk_index):
    first = frame_number(first_key)
    last = frame_number(last_key)
    if first is not None and last is not None:
        return f"{prefix}_{first:05d}_{last:05d}.json"
    return f"{prefix}_part_{chunk_index:04d}.json"


def finalize_chunk(fp, tmp_path, output_dir, prefix, first_key, last_key, chunk_index):
    fp.write("\n}\n")
    fp.close()
    final_path = output_dir / chunk_name(prefix, first_key, last_key, chunk_index)
    tmp_path.replace(final_path)
    return final_path


def split_json(input_path, output_dir, frames_per_file, prefix, force):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    stream = JsonStream(input_path)
    try:
        stream.consume("{")
        chunk_fp = None
        chunk_tmp = None
        chunk_count = 0
        chunk_index = 0
        first_key = None
        last_key = None
        total = 0

        while True:
            ch = stream.peek()
            if ch == "}":
                stream.consume("}")
                break
            if ch == ",":
                stream.consume(",")

            key = stream.raw_decode()
            if not isinstance(key, str):
                raise ValueError(f"Expected frame key string, got {type(key).__name__}")
            stream.consume(":")
            value = stream.raw_decode()

            if chunk_fp is None:
                chunk_tmp = output_dir / f".{prefix}_part_{chunk_index:04d}.tmp"
                if chunk_tmp.exists() and not force:
                    raise FileExistsError(f"{chunk_tmp} already exists; use --force")
                chunk_fp = chunk_tmp.open("w", encoding="utf-8")
                chunk_fp.write("{\n")
                chunk_count = 0
                first_key = key

            if chunk_count:
                chunk_fp.write(",\n")
            chunk_fp.write("  ")
            json.dump(key, chunk_fp, ensure_ascii=False)
            chunk_fp.write(": ")
            json.dump(value, chunk_fp, ensure_ascii=False, separators=(",", ":"))
            chunk_count += 1
            total += 1
            last_key = key

            if chunk_count >= frames_per_file:
                final_path = output_dir / chunk_name(prefix, first_key, last_key, chunk_index)
                if final_path.exists() and not force:
                    raise FileExistsError(f"{final_path} already exists; use --force")
                written.append(finalize_chunk(chunk_fp, chunk_tmp, output_dir, prefix, first_key, last_key, chunk_index))
                chunk_fp = None
                chunk_tmp = None
                chunk_index += 1

        if chunk_fp is not None:
            final_path = output_dir / chunk_name(prefix, first_key, last_key, chunk_index)
            if final_path.exists() and not force:
                raise FileExistsError(f"{final_path} already exists; use --force")
            written.append(finalize_chunk(chunk_fp, chunk_tmp, output_dir, prefix, first_key, last_key, chunk_index))
    finally:
        stream.close()

    return total, written


def main():
    parser = argparse.ArgumentParser(description="Split a large MUSE yolo_tracking_data JSON into browser-friendly chunks.")
    parser.add_argument("input_json", type=Path, help="Path to yolo_tracking_data.json")
    parser.add_argument("--output-dir", type=Path, help="Where to write chunk JSON files; defaults to the input folder")
    parser.add_argument("--frames-per-file", type=int, default=500, help="Frames per chunk JSON")
    parser.add_argument("--prefix", default="yolo_tracking_data", help="Output filename prefix")
    parser.add_argument("--force", action="store_true", help="Overwrite existing chunk files")
    args = parser.parse_args()

    if args.frames_per_file <= 0:
        raise SystemExit("--frames-per-file must be positive")
    input_path = args.input_json.resolve()
    output_dir = (args.output_dir or input_path.parent).resolve()
    total, written = split_json(input_path, output_dir, args.frames_per_file, args.prefix, args.force)

    print(f"Read {total} frames from {input_path}")
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
