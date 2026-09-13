"""No-reference quality decay: Laplacian variance sharpness over time.

Laplacian variance is a noisy proxy for perceptual quality, but it is cheap,
dependency-free, and catches the blur-type collapse that autoregressive
rollouts tend to produce. BRISQUE/NIQE would be better; the brisque pip
package and cv2's contrib quality module are both fragile on Windows.
"""

import cv2
import numpy as np
from scipy import stats

# TODO: BRISQUE instead of laplacian variance once a reliable Windows build exists


def sharpness_series(frames: list) -> np.ndarray:
    """Per-frame Laplacian variance."""
    out = np.zeros(len(frames))
    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        out[i] = cv2.Laplacian(gray, cv2.CV_64F).var()
    return out


def quality_decay(sharp: np.ndarray) -> dict:
    """Fit a line to the sharpness series and find the collapse point.

    Collapse heuristic: first frame below mean(first 25%) - 2*std(first 25%).
    None if the series never drops that far.
    """
    n = len(sharp)
    x = np.arange(n)
    fit = stats.linregress(x, sharp)

    head = sharp[: max(1, n // 4)]
    thresh = head.mean() - 2 * head.std()
    below = np.where(sharp < thresh)[0]
    collapse = int(below[0]) if len(below) else None

    return {
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "rvalue": float(fit.rvalue),
        "collapse_frame": collapse,
        "collapse_threshold": float(thresh),
        "sharpness": sharp,
    }


def clip_alignment(frames: list, prompt: str, device: str = "cpu",
                   batch_size: int = 16) -> np.ndarray:
    """Per-frame CLIP cosine similarity to the prompt (needs the [clip] extra).

    Raises:
        ImportError: if open_clip_torch is not installed.
    """
    import torch
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_pretrained(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    with torch.no_grad():
        text = tokenizer([prompt]).to(device)
        text_feat = model.encode_text(text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

        sims = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            imgs = torch.stack([
                preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                for f in batch
            ]).to(device)
            img_feat = model.encode_image(imgs)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            sims.append((img_feat @ text_feat.T).squeeze(-1).cpu().numpy())
    return np.concatenate(sims)
