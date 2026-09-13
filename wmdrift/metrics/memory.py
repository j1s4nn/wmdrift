"""Memory attenuation: loop-closure error.

If the commanded trajectory returns the camera to its starting viewpoint,
the last frame should look like the first. A world model with weak spatial
memory will have drifted visually even though the pose came home.
"""

import numpy as np

from ..pose.parser import c2w_trajectory


def is_loop(pose_str: str, tol: float = 0.2) -> bool:
    """True if the commanded trajectory ends near where it started."""
    c2w = c2w_trajectory(pose_str)
    return bool(np.linalg.norm(c2w[-1][:3, 3] - c2w[0][:3, 3]) < tol)


def ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Mean SSIM over channels, manual implementation (cv2 has no builtin)."""
    import cv2

    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    vals = []
    for ch in range(a.shape[2]):
        x, y = a[..., ch], b[..., ch]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sig_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sig_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sig_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
        den = (mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2)
        vals.append((num / den).mean())
    return float(np.mean(vals))


def loop_closure_error(first: np.ndarray, last: np.ndarray,
                       device: str = "cpu") -> dict:
    """Compare first and last frame with LPIPS (fallback: SSIM + MSE)."""
    result = {"backend": None, "lpips": None, "ssim": None, "mse": None}
    mse = float(np.mean((first.astype(np.float64) - last.astype(np.float64)) ** 2))
    result["mse"] = mse
    result["ssim"] = ssim(first, last)

    try:
        import torch
        import lpips as lpips_pkg

        model = lpips_pkg.LPIPS(net="alex", verbose=False).to(device)

        # lpips wants NCHW float in [-1, 1], RGB
        def to_tensor(img):
            t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)
            return (t.float() / 127.5 - 1.0).unsqueeze(0).to(device)

        with torch.no_grad():
            d = model(to_tensor(first), to_tensor(last))
        result["lpips"] = float(d.item())
        result["backend"] = "lpips-alex"
    except Exception as e:  # no torch, no weights download, cuda oom...
        result["backend"] = "ssim-fallback"
        result["note"] = f"lpips unavailable ({e.__class__.__name__}), used ssim+mse"
    return result
