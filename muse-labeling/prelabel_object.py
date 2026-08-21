#!/usr/bin/env python3
"""Pre-label one tracked object in a yolo_tracking_data.json file.

Binds a radar track id to a YOLO box id across every frame where the radar track
is present, writing into each frame's ``labeling.object`` map.

The radar tracker drops and re-issues ids when it briefly loses a target. When
re-acquisition is enabled (default) the script follows the object across those id
switches: on frames where the tracked id is missing it adopts the nearest radar
detection (within a range/velocity gate) so the gap frames still get labeled.

Usage:
    python prelabel_object.py yolo_tracking_data.json --radar-id 1 --box-id 1
"""

import argparse
import json
import math
import os
import shutil
import sys


def frame_sort_key(item):
    """Sort frames by frame_index, falling back to the dict key."""
    key, frame = item
    idx = frame.get("frame_index")
    return (idx if isinstance(idx, int) else 0, key)


def centroid(det):
    c = det.get("centroid") or {}
    return c.get("range_m"), c.get("velocity_kmh")


def find_by_id(detections, track_id):
    for det in detections:
        if det.get("track_id") == track_id:
            return det
    return None


def resolve_box(frame, box_id):
    """The box value to store: box_id if that box exists this frame, else None.

    Mirrors the original prelabel: an object can be present (radar) with no
    associated camera box, recorded as ``{radar_id: null}``.
    """
    for box in frame.get("box_detections") or []:
        if box.get("track_id") == box_id:
            return box_id
    return None


def in_gate(det, last_pos, range_gate, vel_gate):
    r, v = centroid(det)
    if r is None or v is None:
        return False
    lr, lv = last_pos
    return abs(r - lr) <= range_gate and abs(v - lv) <= vel_gate


def gate_distance(det, last_pos, range_gate, vel_gate):
    """Normalized distance inside the elliptical gate (lower = closer)."""
    r, v = centroid(det)
    lr, lv = last_pos
    return math.hypot((r - lr) / range_gate, (v - lv) / vel_gate)


def pick_candidate(detections, last_pos, original_id, range_gate, vel_gate,
                   confirmed_only):
    """Choose the detection to re-acquire, or None.

    Preference: the originally-configured id (if itself in the gate) wins, so a
    returning original track beats a spurious one; then confirmed tracks; then
    smallest normalized distance.
    """
    candidates = [d for d in detections if in_gate(d, last_pos, range_gate, vel_gate)]
    if confirmed_only:
        candidates = [d for d in candidates if d.get("is_confirmed")]
    if not candidates:
        return None

    for d in candidates:
        if d.get("track_id") == original_id:
            return d

    def rank(d):
        # confirmed first (False sorts before True, so negate), then closest
        return (not d.get("is_confirmed", False),
                gate_distance(d, last_pos, range_gate, vel_gate))

    return min(candidates, key=rank)


def prelabel(data, radar_id, box_id, reacquire, range_gate, vel_gate, max_coast,
             confirmed_only):
    """Mutate ``data`` in place; return a summary dict."""
    frames = sorted(data.items(), key=frame_sort_key)

    current_id = radar_id
    last_pos = None
    frames_since_seen = 0

    labeled_direct = 0   # frames matched by the current id
    labeled_reacq = 0    # frames matched via re-acquisition (id switch)
    switches = []        # (frame_key, from_id, to_id)

    for key, frame in frames:
        detections = frame.get("radar_detections") or []
        labeling = frame.setdefault("labeling", {})
        obj = labeling.setdefault("object", {})

        hit = find_by_id(detections, current_id)
        if hit is not None:
            obj[str(current_id)] = resolve_box(frame, box_id)
            last_pos = centroid(hit)
            frames_since_seen = 0
            labeled_direct += 1
            continue

        if reacquire and last_pos is not None and frames_since_seen < max_coast:
            cand = pick_candidate(detections, last_pos, radar_id,
                                  range_gate, vel_gate, confirmed_only)
            if cand is not None:
                new_id = cand["track_id"]
                if new_id != current_id:
                    switches.append((key, current_id, new_id))
                current_id = new_id
                obj[str(current_id)] = resolve_box(frame, box_id)
                last_pos = centroid(cand)
                frames_since_seen = 0
                labeled_reacq += 1
                continue

        # target missing this frame
        frames_since_seen += 1

    return {
        "frames": len(frames),
        "labeled_direct": labeled_direct,
        "labeled_reacquired": labeled_reacq,
        "labeled_total": labeled_direct + labeled_reacq,
        "switches": switches,
    }


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json_path", help="path to yolo_tracking_data.json")
    p.add_argument("--radar-id", type=int, required=True,
                   help="radar track id of the object to label")
    p.add_argument("--box-id", type=int, required=True,
                   help="YOLO box id to bind the radar track to")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--reacquire", dest="reacquire", action="store_true",
                     help="follow the object across id switches (default)")
    grp.add_argument("--no-reacquire", dest="reacquire", action="store_false",
                     help="legacy behavior: only label frames where --radar-id exists")
    p.set_defaults(reacquire=True)
    p.add_argument("--range-gate", type=float, default=2.0,
                   help="re-acquire range tolerance, meters (default 2.0)")
    p.add_argument("--vel-gate", type=float, default=2.0,
                   help="re-acquire velocity tolerance, km/h (default 2.0)")
    p.add_argument("--max-coast", type=int, default=10,
                   help="max consecutive missing frames to keep re-acquiring (default 10)")
    p.add_argument("--confirmed-only", action="store_true",
                   help="only adopt is_confirmed detections when re-acquiring")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would change, write nothing")
    p.add_argument("--no-backup", action="store_true",
                   help="do not create a .bak before writing")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if not os.path.isfile(args.json_path):
        sys.exit("error: file not found: %s" % args.json_path)

    with open(args.json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    summary = prelabel(
        data,
        radar_id=args.radar_id,
        box_id=args.box_id,
        reacquire=args.reacquire,
        range_gate=args.range_gate,
        vel_gate=args.vel_gate,
        max_coast=args.max_coast,
        confirmed_only=args.confirmed_only,
    )

    print("frames:              %d" % summary["frames"])
    print("labeled (direct):    %d" % summary["labeled_direct"])
    print("labeled (re-acq):    %d" % summary["labeled_reacquired"])
    print("labeled (total):     %d" % summary["labeled_total"])
    if summary["switches"]:
        print("id switches (%d):" % len(summary["switches"]))
        for key, frm, to in summary["switches"]:
            print("  %s: %d -> %d" % (key, frm, to))

    if args.dry_run:
        print("dry-run: no files written")
        return

    if not args.no_backup:
        bak = args.json_path + ".bak"
        if not os.path.exists(bak):
            # Copy the untouched file on disk (still original until we overwrite).
            shutil.copy2(args.json_path, bak)
            print("backup: %s" % bak)

    with open(args.json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    print("wrote %s" % args.json_path)


if __name__ == "__main__":
    main()
