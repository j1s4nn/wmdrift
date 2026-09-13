"""Report writing: markdown + csvs + matplotlib PNGs.

Everything lands in one output directory so `compare` can diff two runs by
reading their metrics.csv.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _plot(x, y, xlabel, ylabel, title, path, extra=None):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, y)
    if extra is not None:
        ax.plot(extra[0], extra[1], linestyle="--", color="gray")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(outdir: str, video_path: str, fps: float, n_frames: int,
                 pose_str, results: dict, prompt=None) -> str:
    """Write report.md, metrics.csv, per_frame.csv and the PNGs. Returns report path.

    results keys are optional: drift / jitter / quality / memory / clip.
    Missing metrics are reported as skipped instead of crashing.
    """
    os.makedirs(outdir, exist_ok=True)
    lines = []
    lines.append(f"# wmdrift report: {os.path.basename(video_path)}")
    lines.append("")
    lines.append(f"- video: `{video_path}`")
    lines.append(f"- frames: {n_frames}, fps: {fps:.2f}")
    lines.append(f"- commanded pose: `{pose_str}`" if pose_str
                 else "- commanded pose: none given (drift/memory skipped)")
    if prompt:
        lines.append(f"- prompt: {prompt}")
    lines.append("")

    flat = {"video": os.path.basename(video_path), "n_frames": n_frames,
            "fps": round(fps, 3), "pose": pose_str or ""}
    per_frame = {"frame": list(range(n_frames))}

    drift = results.get("drift")
    lines.append("## Camera drift")
    lines.append("")
    if drift is None:
        lines.append("Skipped (no pose string, or VO found too few frames).")
    else:
        lines.append("VO recovered the trajectory and was Sim(3)-aligned to the")
        lines.append(f"commanded one. Fitted scale factor: **{drift['scale']:.4f}**. ")
        lines.append("Monocular VO cannot recover metric scale, so this factor is")
        lines.append("the recovered/commanded unit ratio, not an error (see minWM")
        lines.append("issue #18 on camera translation scale).")
        lines.append("")
        lines.append(f"- ATE mean: {drift['ate_mean']:.4f}")
        lines.append(f"- ATE max: {drift['ate_max']:.4f}")
        lines.append(f"- ATE final frame: {drift['ate_final']:.4f}")
        lines.append(f"- rotation error mean: {drift['rot_err_mean_deg']:.2f} deg")
        lines.append(f"- rotation error max: {drift['rot_err_max_deg']:.2f} deg")
        if drift["unreliable_pairs"]:
            pairs = ", ".join(str(p) for p in drift["unreliable_pairs"])
            lines.append(f"- unreliable frame pairs (low RANSAC inlier ratio, "
                         f"pose carried forward): {pairs}")
        lines.append("")
        lines.append("![drift curve](drift_curve.png)")
        lines.append("")
        lines.append("![rotation error](rotation_error.png)")
        x = np.arange(drift["n_frames"])
        _plot(x, drift["ate"], "frame", "ATE (aligned units)",
              "Drift: aligned translation error", os.path.join(outdir, "drift_curve.png"))
        _plot(x, drift["rot_err_deg"], "frame", "rotation error (deg)",
              "Rotation error vs commanded", os.path.join(outdir, "rotation_error.png"))
        flat.update({
            "drift_scale": round(drift["scale"], 5),
            "ate_mean": round(drift["ate_mean"], 5),
            "ate_max": round(drift["ate_max"], 5),
            "ate_final": round(drift["ate_final"], 5),
            "rot_err_mean_deg": round(drift["rot_err_mean_deg"], 4),
            "rot_err_max_deg": round(drift["rot_err_max_deg"], 4),
            "unreliable_pairs": len(drift["unreliable_pairs"]),
        })
        for i, v in enumerate(drift["ate"]):
            per_frame.setdefault("ate", [None] * n_frames)[i] = round(float(v), 5)
        for i, v in enumerate(drift["rot_err_deg"]):
            per_frame.setdefault("rot_err_deg", [None] * n_frames)[i] = round(float(v), 4)
    lines.append("")

    jit = results.get("jitter")
    lines.append("## Temporal jitter")
    lines.append("")
    if jit is None:
        lines.append("Skipped.")
    else:
        lines.append(f"Jitter score (spectral power above fs/4 over total power, "
                     f"linear trend removed): **{jit['score']:.4f}**")
        lines.append("")
        lines.append(f"High-frequency RMS of the detrended flow-magnitude series "
                     f"(flow units): **{jit['hf_rms']:.4f}**. The ratio alone barely "
                     f"separates a clean dolly from a jittered one -- a nearly flat "
                     f"residual still has a moderate ratio -- so this amplitude is "
                     f"the number I actually compare between runs.")
        if jit.get("note"):
            lines.append(f"({jit['note']})")
        lines.append("")
        lines.append("![jitter spectrum](jitter_spectrum.png)")
        if len(jit["freqs"]):
            _plot(jit["freqs"], jit["power"], "frequency (Hz)", "power",
                  "Flow-magnitude spectrum (detrended)",
                  os.path.join(outdir, "jitter_spectrum.png"))
        flat["jitter_score"] = round(jit["score"], 5) if jit["score"] == jit["score"] else ""
        flat["jitter_hf_rms"] = round(jit["hf_rms"], 5) if jit["hf_rms"] == jit["hf_rms"] else ""
    lines.append("")

    qual = results.get("quality")
    lines.append("## Quality decay")
    lines.append("")
    if qual is None:
        lines.append("Skipped.")
    else:
        lines.append("Sharpness = per-frame Laplacian variance (a noisy no-reference")
        lines.append("proxy, not a perceptual metric).")
        lines.append("")
        lines.append(f"- decay slope: {qual['slope']:.4f} per frame")
        lines.append(f"- fit r: {qual['rvalue']:.3f}")
        if qual["collapse_frame"] is not None:
            lines.append(f"- collapse heuristic fires at frame {qual['collapse_frame']} "
                         f"(sharpness below {qual['collapse_threshold']:.1f})")
        else:
            lines.append("- collapse heuristic never fires")
        lines.append("")
        lines.append("![quality trajectory](quality_trajectory.png)")
        _plot(np.arange(len(qual["sharpness"])), qual["sharpness"],
              "frame", "laplacian variance", "Sharpness over time",
              os.path.join(outdir, "quality_trajectory.png"))
        flat["quality_slope"] = round(qual["slope"], 5)
        flat["quality_r"] = round(qual["rvalue"], 4)
        flat["collapse_frame"] = (qual["collapse_frame"]
                                  if qual["collapse_frame"] is not None else "")
        per_frame["sharpness"] = [round(float(v), 3) for v in qual["sharpness"]]
    lines.append("")

    if results.get("clip") is not None:
        clip = results["clip"]
        lines.append("## CLIP prompt alignment")
        lines.append("")
        lines.append(f"- mean cosine similarity: {np.mean(clip['sims']):.4f}")
        lines.append(f"- final-frame similarity: {clip['sims'][-1]:.4f}")
        lines.append(f"- alignment slope: {clip['slope']:.6f} per frame")
        flat["clip_mean"] = round(float(np.mean(clip["sims"])), 5)
        flat["clip_slope"] = round(float(clip["slope"]), 7)
        per_frame["clip_sim"] = [round(float(v), 5) for v in clip["sims"]]
        lines.append("")

    mem = results.get("memory")
    lines.append("## Memory (loop closure)")
    lines.append("")
    if mem is None:
        lines.append("Skipped.")
    elif not mem.get("is_loop"):
        lines.append("Commanded trajectory does not return to the start "
                     "(and --force-loop was not given), so there is nothing to close.")
    else:
        lines.append(f"Trajectory ends near its start, comparing first and last frame "
                     f"({mem['backend']}):")
        lines.append("")
        if mem.get("lpips") is not None:
            lines.append(f"- LPIPS(first, last): {mem['lpips']:.4f} (lower = better memory)")
        lines.append(f"- SSIM(first, last): {mem['ssim']:.4f}")
        lines.append(f"- MSE(first, last): {mem['mse']:.1f}")
        if mem.get("note"):
            lines.append(f"- note: {mem['note']}")
        flat["loop_lpips"] = round(mem["lpips"], 5) if mem.get("lpips") is not None else ""
        flat["loop_ssim"] = round(mem["ssim"], 5)
        flat["loop_mse"] = round(mem["mse"], 2)
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("Monocular VO gives trajectory shape, not metric truth: ATE is")
    lines.append("reported after Sim(3) alignment, so it measures consistency with")
    lines.append("the commanded path up to a global scale. Laplacian variance is a")
    lines.append("crude quality proxy. The loop-closure check assumes the renderer")
    lines.append("and the model see the same scene at the same viewpoint. Numbers on")
    lines.append("synthetic fixtures say more about the pipeline than about any model.")
    lines.append("")

    report_path = os.path.join(outdir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(os.path.join(outdir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        w.writeheader()
        w.writerow(flat)

    cols = {k: v for k, v in per_frame.items()}
    n_rows = len(cols["frame"])
    with open(os.path.join(outdir, "per_frame.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(cols.keys()))
        for i in range(n_rows):
            w.writerow([cols[k][i] if i < len(cols[k]) else "" for k in cols])

    return report_path
