"""Command line interface: analyze / batch / compare."""

import argparse
import csv
import glob as globmod
import os
import sys

from .video.io import load_video

ALL_METRICS = ["drift", "jitter", "memory", "quality"]


def resolve_device(name: str) -> str:
    if name == "cpu":
        return "cpu"
    import torch
    if name == "cuda":
        if not torch.cuda.is_available():
            print("warning: cuda requested but not available, falling back to cpu")
            return "cpu"
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def analyze_video(video_path, pose_str=None, prompt=None, fps_override=None,
                  output="reports/run", device="auto", metrics=None,
                  force_loop=False):
    """Run the selected metrics on one video and write the report directory."""
    from .metrics import drift as drift_mod
    from .metrics import jitter as jitter_mod
    from .metrics import memory as memory_mod
    from .metrics import quality as quality_mod
    from .report.generator import write_report
    from .video.flow import flow_magnitudes

    metrics = metrics or ALL_METRICS
    device = resolve_device(device)

    frames, fps = load_video(video_path)
    if fps_override:
        fps = float(fps_override)
    print(f"{video_path}: {len(frames)} frames @ {fps:.2f} fps, device={device}")

    results = {}

    if "drift" in metrics:
        if pose_str:
            results["drift"] = drift_mod.drift_metrics(frames, pose_str)
            if results["drift"] is None:
                print("note: drift skipped, VO recovered too few frames")
        else:
            print("note: drift skipped, no --pose given")

    if "jitter" in metrics:
        mags, _ = flow_magnitudes(frames)
        results["jitter"] = jitter_mod.jitter_score(mags, fps)

    if "quality" in metrics:
        sharp = quality_mod.sharpness_series(frames)
        results["quality"] = quality_mod.quality_decay(sharp)
        if prompt:
            try:
                sims = quality_mod.clip_alignment(frames, prompt, device=device)
                import numpy as np
                x = np.arange(len(sims))
                slope = float(np.polyfit(x, sims, 1)[0]) if len(sims) > 2 else 0.0
                results["clip"] = {"sims": sims, "slope": slope}
            except ImportError:
                print("note: clip alignment skipped, install the [clip] extra")

    if "memory" in metrics:
        if pose_str and (memory_mod.is_loop(pose_str) or force_loop):
            results["memory"] = memory_mod.loop_closure_error(
                frames[0], frames[-1], device=device)
            results["memory"]["is_loop"] = True
        elif pose_str:
            results["memory"] = {"is_loop": False}
        else:
            print("note: memory skipped, no --pose given")

    return write_report(output, video_path, fps, len(frames), pose_str,
                        results, prompt=prompt)


def find_videos(pattern: str) -> list:
    if os.path.isdir(pattern):
        pattern = os.path.join(pattern, "*")
    paths = []
    for ext in ("mp4", "avi", "mov", "mkv", "webm"):
        paths.extend(globmod.glob(f"{pattern}.{ext}"))
        paths.extend(globmod.glob(f"{pattern}.{ext.upper()}"))
    return sorted(set(paths))


def cmd_analyze(args) -> int:
    metrics = args.metrics.split(",") if args.metrics else ALL_METRICS
    unknown = [m for m in metrics if m not in ALL_METRICS]
    if unknown:
        print(f"unknown metrics: {unknown}. valid: {ALL_METRICS}")
        return 2
    report = analyze_video(
        args.video, pose_str=args.pose, prompt=args.prompt,
        fps_override=args.fps, output=args.output, device=args.device,
        metrics=metrics, force_loop=args.force_loop)
    print(f"report written to {report}")
    return 0


def cmd_batch(args) -> int:
    videos = find_videos(args.pattern)
    if not videos:
        print(f"no videos found matching {args.pattern}")
        return 1
    os.makedirs(args.output, exist_ok=True)
    summary_rows = []
    for i, v in enumerate(videos):
        name = os.path.splitext(os.path.basename(v))[0]
        outdir = os.path.join(args.output, name)
        print(f"[{i + 1}/{len(videos)}] {name}")
        analyze_video(v, pose_str=args.pose, output=outdir,
                      device=args.device)
        with open(os.path.join(outdir, "metrics.csv")) as f:
            summary_rows.append(next(csv.DictReader(f)))
    summary_path = os.path.join(args.output, "summary.csv")
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    print(f"summary written to {summary_path}")
    return 0


def cmd_compare(args) -> int:
    def load_metrics(d):
        path = os.path.join(d, "metrics.csv")
        if not os.path.exists(path):
            print(f"no metrics.csv in {d}")
            sys.exit(1)
        with open(path) as f:
            return next(csv.DictReader(f))

    a = load_metrics(args.dir_a)
    b = load_metrics(args.dir_b)
    keys = sorted(set(a) | set(b))
    name_a = os.path.basename(os.path.normpath(args.dir_a))
    name_b = os.path.basename(os.path.normpath(args.dir_b))
    print(f"| metric | {name_a} | {name_b} |")
    print("|---|---|---|")
    for k in keys:
        print(f"| {k} | {a.get(k, '-')} | {b.get(k, '-')} |")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="wmdrift",
        description="Long-horizon drift diagnostics for video world model rollouts.")
    sub = p.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("analyze", help="analyze one video")
    pa.add_argument("video")
    pa.add_argument("--pose", default=None,
                    help='minWM pose string, e.g. "w*30" or "w*10,a*5"')
    pa.add_argument("--prompt", default=None,
                    help="generation prompt, enables CLIP alignment if [clip] installed")
    pa.add_argument("--fps", type=float, default=None,
                    help="override video fps (for the jitter frequency axis)")
    pa.add_argument("--output", default=os.path.join("reports", "run"))
    pa.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    pa.add_argument("--metrics", default=None,
                    help="comma-separated subset of drift,jitter,memory,quality")
    pa.add_argument("--force-loop", action="store_true",
                    help="run loop closure even if the pose does not return home")
    pa.set_defaults(func=cmd_analyze)

    pb = sub.add_parser("batch", help="analyze every video in a dir or glob")
    pb.add_argument("pattern")
    pb.add_argument("--output", default=os.path.join("reports", "batch"))
    pb.add_argument("--pose", default=None, help="pose string shared by all videos")
    pb.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    pb.set_defaults(func=cmd_batch)

    pc = sub.add_parser("compare", help="markdown diff of two report dirs")
    pc.add_argument("dir_a")
    pc.add_argument("dir_b")
    pc.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
