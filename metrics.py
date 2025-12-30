import os
import glob
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.transform import Rotation

from depth_anything_3.api import DepthAnything3


# -------------------------
# GT depth loading (Isaac)
# -------------------------
def load_depth_png_to_meters(path: str, png_depth_scale: float) -> np.ndarray:
    """
    Loads Isaac Sim depth.png (likely uint16) and converts to meters.
    """
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Could not read depth png: {path}")

    # If it's 16-bit single-channel, shape is (H, W) dtype=uint16
    # If it ends up 3-channel, take one channel (shouldn't happen for proper depth png)
    if d.ndim == 3:
        d = d[:, :, 0]

    d = d.astype(np.float32) * float(png_depth_scale)
    return d


def depth_to_colormap(depth: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """
    Convert depth map to colored visualization using jet colormap.
    Returns RGB image (H, W, 3) uint8.
    """
    # Handle invalid values
    valid_mask = np.isfinite(depth) & (depth > 0)
    
    if vmin is None:
        vmin = depth[valid_mask].min() if valid_mask.any() else 0
    if vmax is None:
        vmax = depth[valid_mask].max() if valid_mask.any() else 1
    
    # Normalize to 0-1
    depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    # Apply colormap
    cmap = cm.get_cmap('jet')
    colored = cmap(depth_norm)[:, :, :3]  # RGB, drop alpha
    
    # Convert to uint8
    colored = (colored * 255).astype(np.uint8)
    
    # Set invalid pixels to black
    colored[~valid_mask] = 0
    
    return colored


def save_depth_comparison(gt_depth: np.ndarray, pred_depth: np.ndarray, 
                         output_path: str, frame_name: str):
    """
    Save side-by-side comparison of GT and predicted depth.
    
    Args:
        gt_depth: Ground truth depth map (H, W) in meters
        pred_depth: Predicted depth map (H, W) in meters
        output_path: Directory to save results
        frame_name: Name of the frame for the filename
    """
    # Use same color scale for both
    valid_mask = np.isfinite(gt_depth) & (gt_depth > 0)
    if valid_mask.any():
        vmin = min(gt_depth[valid_mask].min(), pred_depth[valid_mask].min())
        vmax = max(gt_depth[valid_mask].max(), pred_depth[valid_mask].max())
    else:
        vmin, vmax = 0, 1
    
    # Convert to color
    gt_colored = depth_to_colormap(gt_depth, vmin, vmax)
    pred_colored = depth_to_colormap(pred_depth, vmin, vmax)
    
    # Create side-by-side image
    h, w = gt_colored.shape[:2]
    combined = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
    combined[:, :w] = gt_colored
    combined[:, w+20:] = pred_colored
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Ground Truth', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'DA3 Prediction', (w + 30, 30), font, 1, (255, 255, 255), 2)
    
    # Add depth range info
    info_text = f'Range: {vmin:.2f}m - {vmax:.2f}m'
    cv2.putText(combined, info_text, (10, h - 10), font, 0.6, (255, 255, 255), 1)
    
    # Save
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, f'{frame_name}_comparison.png')
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    
    return save_path


# -------------------------
# Pose Loading and Conversion
# -------------------------
def load_traj_txt(path: str) -> np.ndarray:
    """
    Load trajectory from traj.txt where each line is a flattened 4x4 c2w matrix.
    Returns: [N, 4, 4] array of c2w poses
    """
    poses = []
    with open(path, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if len(values) == 16:
                pose = np.array(values).reshape(4, 4)
                poses.append(pose)
    return np.stack(poses, axis=0)


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """
    Convert world-to-camera to camera-to-world.
    Works for both [3,4] and [4,4] matrices.
    """
    if w2c.shape == (3, 4):
        # Convert [3,4] to [4,4]
        w2c_full = np.eye(4)
        w2c_full[:3, :] = w2c
        w2c = w2c_full
    
    c2w = np.linalg.inv(w2c)
    return c2w


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """
    Convert camera-to-world to world-to-camera.
    """
    if c2w.shape == (3, 4):
        c2w_full = np.eye(4)
        c2w_full[:3, :] = c2w
        c2w = c2w_full
    
    w2c = np.linalg.inv(c2w)
    return w2c


# -------------------------
# Pose Alignment
# -------------------------
def align_trajectory_sim3(pred_c2w: np.ndarray, gt_c2w: np.ndarray):
    """
    Align predicted trajectory to GT using Sim(3) (rotation + translation + scale).
    
    Args:
        pred_c2w: [N, 4, 4] predicted c2w poses
        gt_c2w: [N, 4, 4] ground truth c2w poses
    
    Returns:
        aligned_pred_c2w: [N, 4, 4] aligned predicted poses
        scale: scalar scale factor
        R: [3, 3] rotation
        t: [3,] translation
    """
    # Extract positions (camera centers)
    pred_pos = pred_c2w[:, :3, 3]  # [N, 3]
    gt_pos = gt_c2w[:, :3, 3]  # [N, 3]
    
    # Center both
    pred_mean = pred_pos.mean(axis=0)
    gt_mean = gt_pos.mean(axis=0)
    
    pred_centered = pred_pos - pred_mean
    gt_centered = gt_pos - gt_mean
    
    # Compute scale
    scale = np.sqrt((gt_centered ** 2).sum() / (pred_centered ** 2).sum() + 1e-12)
    
    # Apply scale to centered pred
    pred_scaled = pred_centered * scale
    
    # Solve for rotation: R = argmin ||R @ pred_scaled.T - gt_centered.T||
    H = pred_scaled.T @ gt_centered  # [3, 3]
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Translation
    t = gt_mean - R @ (pred_mean * scale)
    
    # Apply transformation to all poses
    aligned_pred_c2w = np.zeros_like(pred_c2w)
    for i in range(len(pred_c2w)):
        # Scale and transform position
        new_pos = R @ (pred_c2w[i, :3, 3] * scale) + t
        
        # Keep rotation as is (only align positions)
        aligned_pred_c2w[i] = pred_c2w[i].copy()
        aligned_pred_c2w[i, :3, 3] = new_pos
    
    return aligned_pred_c2w, scale, R, t


def align_trajectory_se3(pred_c2w: np.ndarray, gt_c2w: np.ndarray):
    """
    Align predicted trajectory to GT using SE(3) (rotation + translation, no scale).
    Similar to Sim3 but without scale.
    """
    pred_pos = pred_c2w[:, :3, 3]
    gt_pos = gt_c2w[:, :3, 3]
    
    pred_mean = pred_pos.mean(axis=0)
    gt_mean = gt_pos.mean(axis=0)
    
    pred_centered = pred_pos - pred_mean
    gt_centered = gt_pos - gt_mean
    
    # Solve for rotation
    H = pred_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = gt_mean - R @ pred_mean
    
    # Apply transformation
    aligned_pred_c2w = np.zeros_like(pred_c2w)
    for i in range(len(pred_c2w)):
        new_pos = R @ pred_c2w[i, :3, 3] + t
        aligned_pred_c2w[i] = pred_c2w[i].copy()
        aligned_pred_c2w[i, :3, 3] = new_pos
    
    return aligned_pred_c2w, R, t


# -------------------------
# Pose Metrics
# -------------------------
def compute_ate(pred_c2w: np.ndarray, gt_c2w: np.ndarray, align_mode='sim3'):
    """
    Compute Absolute Trajectory Error (ATE).
    
    Args:
        pred_c2w: [N, 4, 4] predicted c2w poses
        gt_c2w: [N, 4, 4] ground truth c2w poses
        align_mode: 'sim3' (with scale) or 'se3' (no scale)
    
    Returns:
        dict with ATE metrics
    """
    if align_mode == 'sim3':
        aligned_pred, scale, R, t = align_trajectory_sim3(pred_c2w, gt_c2w)
    else:
        aligned_pred, R, t = align_trajectory_se3(pred_c2w, gt_c2w)
        scale = 1.0
    
    # Compute translation errors
    pred_pos = aligned_pred[:, :3, 3]
    gt_pos = gt_c2w[:, :3, 3]
    
    errors = np.linalg.norm(pred_pos - gt_pos, axis=1)
    
    ate_rmse = np.sqrt(np.mean(errors ** 2))
    ate_mean = np.mean(errors)
    ate_median = np.median(errors)
    ate_std = np.std(errors)
    
    return {
        'ATE_RMSE': float(ate_rmse),
        'ATE_mean': float(ate_mean),
        'ATE_median': float(ate_median),
        'ATE_std': float(ate_std),
        'scale': float(scale),
    }


def rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute rotation error in degrees between two rotation matrices.
    """
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    # Clamp to avoid numerical issues with arccos
    trace = np.clip((trace - 1) / 2, -1, 1)
    angle_rad = np.arccos(trace)
    return np.degrees(angle_rad)


def compute_rpe(pred_c2w: np.ndarray, gt_c2w: np.ndarray, delta=1):
    """
    Compute Relative Pose Error (RPE) for frame-to-frame motion.
    
    Args:
        pred_c2w: [N, 4, 4] predicted c2w poses
        gt_c2w: [N, 4, 4] ground truth c2w poses
        delta: frame step size (default 1 for consecutive frames)
    
    Returns:
        dict with RPE metrics
    """
    trans_errors = []
    rot_errors = []
    
    for i in range(len(pred_c2w) - delta):
        # Compute relative transforms
        # GT: T_i^{-1} @ T_{i+delta}
        gt_rel = np.linalg.inv(gt_c2w[i]) @ gt_c2w[i + delta]
        
        # Pred: T_i^{-1} @ T_{i+delta}
        pred_rel = np.linalg.inv(pred_c2w[i]) @ pred_c2w[i + delta]
        
        # Error: E = (gt_rel)^{-1} @ pred_rel
        E = np.linalg.inv(gt_rel) @ pred_rel
        
        # Translation error
        trans_error = np.linalg.norm(E[:3, 3])
        trans_errors.append(trans_error)
        
        # Rotation error
        rot_error = rotation_error(np.eye(3), E[:3, :3])
        rot_errors.append(rot_error)
    
    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    
    return {
        'RPE_trans_RMSE': float(np.sqrt(np.mean(trans_errors ** 2))),
        'RPE_trans_mean': float(np.mean(trans_errors)),
        'RPE_trans_median': float(np.median(trans_errors)),
        'RPE_rot_RMSE': float(np.sqrt(np.mean(rot_errors ** 2))),
        'RPE_rot_mean': float(np.mean(rot_errors)),
        'RPE_rot_median': float(np.median(rot_errors)),
    }


def compute_rra_rta_auc(pred_c2w: np.ndarray, gt_c2w: np.ndarray, 
                        thresholds_rot=None, thresholds_trans=None,
                        max_pairs=1000):
    """
    Compute AUC of Relative Rotation Accuracy (RRA) and Relative Translation Accuracy (RTA).
    
    Args:
        pred_c2w: [N, 4, 4] predicted c2w poses
        gt_c2w: [N, 4, 4] ground truth c2w poses
        thresholds_rot: rotation thresholds in degrees (default: [1,2,3,5,10,20,30])
        thresholds_trans: translation angle thresholds in degrees (default: [1,2,3,5,10,20,30])
        max_pairs: maximum number of random pairs to sample
    
    Returns:
        dict with AUC metrics
    """
    if thresholds_rot is None:
        thresholds_rot = np.array([1, 2, 3, 5, 10, 20, 30])
    if thresholds_trans is None:
        thresholds_trans = np.array([1, 2, 3, 5, 10, 20, 30])
    
    N = len(pred_c2w)
    
    # Sample pairs (avoid all N^2 pairs if N is large)
    pairs = []
    if N * (N - 1) // 2 <= max_pairs:
        # Use all pairs
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append((i, j))
    else:
        # Random sample
        np.random.seed(42)
        for _ in range(max_pairs):
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            if i != j:
                pairs.append((i, j))
    
    rot_errors = []
    trans_angle_errors = []
    
    for i, j in pairs:
        # Relative rotation GT
        R_gt_rel = gt_c2w[i, :3, :3].T @ gt_c2w[j, :3, :3]
        
        # Relative rotation Pred
        R_pred_rel = pred_c2w[i, :3, :3].T @ pred_c2w[j, :3, :3]
        
        # Rotation error
        rot_err = rotation_error(R_gt_rel, R_pred_rel)
        rot_errors.append(rot_err)
        
        # Relative translation direction GT
        t_gt_rel = gt_c2w[j, :3, 3] - gt_c2w[i, :3, 3]
        t_gt_rel_norm = t_gt_rel / (np.linalg.norm(t_gt_rel) + 1e-12)
        
        # Relative translation direction Pred
        t_pred_rel = pred_c2w[j, :3, 3] - pred_c2w[i, :3, 3]
        t_pred_rel_norm = t_pred_rel / (np.linalg.norm(t_pred_rel) + 1e-12)
        
        # Angular error between translation directions
        cos_angle = np.clip(np.dot(t_gt_rel_norm, t_pred_rel_norm), -1, 1)
        angle_err = np.degrees(np.arccos(cos_angle))
        trans_angle_errors.append(angle_err)
    
    rot_errors = np.array(rot_errors)
    trans_angle_errors = np.array(trans_angle_errors)
    
    # Compute accuracy at each threshold
    rra_accs = []
    for th in thresholds_rot:
        acc = np.mean(rot_errors < th)
        rra_accs.append(acc)
    
    rta_accs = []
    for th in thresholds_trans:
        acc = np.mean(trans_angle_errors < th)
        rta_accs.append(acc)
    
    # Compute AUC (trapezoidal)
    rra_auc = np.trapz(rra_accs, thresholds_rot) / (thresholds_rot[-1] - thresholds_rot[0])
    rta_auc = np.trapz(rta_accs, thresholds_trans) / (thresholds_trans[-1] - thresholds_trans[0])
    
    # Also report specific thresholds (3° and 30° as in DA3 paper)
    rra_3 = float(np.mean(rot_errors < 3))
    rra_30 = float(np.mean(rot_errors < 30))
    rta_3 = float(np.mean(trans_angle_errors < 3))
    rta_30 = float(np.mean(trans_angle_errors < 30))
    
    return {
        'RRA_AUC': float(rra_auc),
        'RTA_AUC': float(rta_auc),
        'RRA@3deg': rra_3,
        'RRA@30deg': rra_30,
        'RTA@3deg': rta_3,
        'RTA@30deg': rta_30,
        'num_pairs': len(pairs),
    }


def evaluate_pose_metrics(pred_c2w: np.ndarray, gt_c2w: np.ndarray, align_mode='sim3'):
    """
    Compute all pose metrics: ATE, RPE, and RRA/RTA AUC.
    
    Args:
        pred_c2w: [N, 4, 4] predicted c2w poses
        gt_c2w: [N, 4, 4] ground truth c2w poses
        align_mode: 'sim3' or 'se3' for ATE alignment
    
    Returns:
        dict with all pose metrics
    """
    metrics = {}
    
    # ATE
    ate_metrics = compute_ate(pred_c2w, gt_c2w, align_mode)
    metrics.update(ate_metrics)
    
    # RPE (using aligned trajectory for fairness)
    if align_mode == 'sim3':
        aligned_pred, _, _, _ = align_trajectory_sim3(pred_c2w, gt_c2w)
    else:
        aligned_pred, _, _ = align_trajectory_se3(pred_c2w, gt_c2w)
    
    rpe_metrics = compute_rpe(aligned_pred, gt_c2w, delta=1)
    metrics.update(rpe_metrics)
    
    # RRA/RTA AUC
    auc_metrics = compute_rra_rta_auc(aligned_pred, gt_c2w)
    metrics.update(auc_metrics)
    
    return metrics


# -------------------------
# Alignment helpers
# -------------------------
def fit_scale(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """
    Solve s = argmin || s*pred - gt ||^2 on valid pixels.
    """
    p = pred[mask].reshape(-1)
    g = gt[mask].reshape(-1)
    denom = np.dot(p, p) + 1e-12
    s = float(np.dot(p, g) / denom)
    return s


def fit_scale_shift(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    """
    Solve (s, b) = argmin || s*pred + b - gt ||^2 on valid pixels.
    Linear least squares.
    """
    p = pred[mask].reshape(-1)
    g = gt[mask].reshape(-1)
    A = np.stack([p, np.ones_like(p)], axis=1)  # [N, 2]
    x, *_ = np.linalg.lstsq(A, g, rcond=None)
    s, b = float(x[0]), float(x[1])
    return s, b


# -------------------------
# Metrics
# -------------------------
def depth_metrics(pred_m: np.ndarray, gt_m: np.ndarray, mask: np.ndarray):
    """
    pred_m, gt_m in meters. mask indicates valid pixels.
    Returns dict of common depth metrics.
    """
    p = pred_m[mask]
    g = gt_m[mask]

    eps = 1e-6
    p = np.clip(p, eps, None)
    g = np.clip(g, eps, None)

    abs_rel = np.mean(np.abs(p - g) / g)
    sq_rel  = np.mean(((p - g) ** 2) / g)

    rmse = np.sqrt(np.mean((p - g) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))

    log10 = np.mean(np.abs(np.log10(p) - np.log10(g)))

    ratio = np.maximum(p / g, g / p)
    d1 = np.mean(ratio < 1.25)
    d2 = np.mean(ratio < (1.25 ** 2))
    d3 = np.mean(ratio < (1.25 ** 3))

    return {
        "AbsRel": float(abs_rel),
        "SqRel": float(sq_rel),
        "RMSE": float(rmse),
        "RMSE_log": float(rmse_log),
        "log10": float(log10),
        "delta1": float(d1),
        "delta2": float(d2),
        "delta3": float(d3),
        "num_valid_px": int(mask.sum()),
    }


# -------------------------
# Main evaluation
# -------------------------
def evaluate_da3_depth_only(
    rgb_dir: str,
    gt_depth_dir: str,
    gt_traj_path: str = None,
    png_depth_scale: float = 0.00015244,
    min_depth_m: float = 1e-3,
    max_depth_m: float = 200.0,
    align_mode: str = "scale",  # "scale" or "scale_shift" for depth
    pose_align_mode: str = "sim3",  # "sim3" or "se3" for pose
    device: str = "cuda",
    model_id: str = "/home/maribjonov_mr/Depth-Anything-3/DA3NESTED-GIANT-LARGE-1.1",
    save_visualizations: bool = True,
    results_dir: str = "results",
    evaluate_pose: bool = True,
):
    # 1) Collect files
    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    if not rgb_paths:
        raise ValueError(f"No RGB .jpg files found in: {rgb_dir}")

    depth_paths = sorted(glob.glob(os.path.join(gt_depth_dir, "*.png")))
    if not depth_paths:
        raise ValueError(f"No GT depth .png files found in: {gt_depth_dir}")
    
    rgb_paths = rgb_paths
    depth_paths = depth_paths
    # 2) Run DA3 inference (batch over list of image paths)
    dev = torch.device(device)
    model = DepthAnything3.from_pretrained(model_id).to(device=dev)

    prediction = model.inference(rgb_paths, process_res=1280)
    pred_depth = prediction.depth  # [N,H,W] float32

    if isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.detach().cpu().numpy()
    pred_depth = pred_depth.astype(np.float32)

    # 3) Evaluate per-frame then average
    per_frame = []
    for i, dp in enumerate(depth_paths):
        gt_m = load_depth_png_to_meters(dp, png_depth_scale)

        pred_i = pred_depth[i]
        if pred_i.shape != gt_m.shape:
            print(f"[WARNING]: pred_i.shape ({pred_i.shape}) != gt_m.shape ({gt_m.shape}):")
            pred_i = cv2.resize(
                pred_i,
                (gt_m.shape[1], gt_m.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            # raise ValueError(
            #     f"Shape mismatch for {os.path.basename(dp)}: "
            #     f"pred {pred_i.shape} vs gt {gt_m.shape}. "
            #     f"(If DA3 resizes internally, you must resize pred back to GT size.)"
            # )

        # Validity mask
        mask = np.isfinite(gt_m) & (gt_m > min_depth_m) & (gt_m < max_depth_m)
        # Also require finite pred
        mask &= np.isfinite(pred_i)

        if mask.sum() < 100:
            # Not enough valid pixels; skip
            continue

        # Align prediction to GT
        if align_mode == "scale":
            s = fit_scale(pred_i, gt_m, mask)
            pred_m = s * pred_i
        elif align_mode == "scale_shift":
            s, b = fit_scale_shift(pred_i, gt_m, mask)
            pred_m = s * pred_i + b
        else:
            raise ValueError("align_mode must be 'scale' or 'scale_shift'")

        # Save visualization if requested
        if save_visualizations:
            frame_name = os.path.splitext(os.path.basename(dp))[0]
            save_path = save_depth_comparison(gt_m, pred_m, results_dir, frame_name)
            print(f"Saved comparison: {save_path}")

        m = depth_metrics(pred_m, gt_m, mask)
        m["frame"] = os.path.basename(dp)
        per_frame.append(m)

    if not per_frame:
        raise RuntimeError("No frames were evaluated (maybe masks filtered everything).")

    # Aggregate (simple mean over frames)
    keys = ["AbsRel", "SqRel", "RMSE", "RMSE_log", "log10", "delta1", "delta2", "delta3"]
    avg = {k: float(np.mean([f[k] for f in per_frame])) for k in keys}
    avg["num_frames"] = len(per_frame)
    
    # Evaluate pose metrics if requested
    pose_metrics = {}
    if evaluate_pose and gt_traj_path is not None:
        print("\n--- Evaluating pose metrics ---")
        
        # Load GT trajectory
        gt_c2w = load_traj_txt(gt_traj_path)
        print(f"Loaded GT trajectory with {len(gt_c2w)} poses")
        gt_c2w = gt_c2w
        
        # Convert DA3 extrinsics (w2c [3,4]) to c2w [4,4]
        pred_extrinsics = prediction.extrinsics  # [N, 3, 4] w2c
        pred_c2w = np.zeros((len(pred_extrinsics), 4, 4))
        for i in range(len(pred_extrinsics)):
            pred_c2w[i] = w2c_to_c2w(pred_extrinsics[i])
        
        # Make sure we have same number of poses
        n_poses = min(len(gt_c2w), len(pred_c2w))
        gt_c2w = gt_c2w[:n_poses]
        pred_c2w = pred_c2w[:n_poses]
        
        # Compute pose metrics
        pose_metrics = evaluate_pose_metrics(pred_c2w, gt_c2w, align_mode=pose_align_mode)
        print(f"Evaluated {n_poses} poses")

    return avg, per_frame, pose_metrics


if __name__ == "__main__":
    rgb_dir = "/home/maribjonov_mr/3D_SSGG_IsaacSim/complex_1/rgb"         # e.g., dataset/rgb
    gt_depth_dir = "/home/maribjonov_mr/3D_SSGG_IsaacSim/complex_1/depth"  # e.g., dataset/depth
    gt_traj_path = "/home/maribjonov_mr/3D_SSGG_IsaacSim/complex_1/traj.txt"  # GT trajectory (c2w)

    avg, per_frame, pose_metrics = evaluate_da3_depth_only(
        rgb_dir=rgb_dir,
        gt_depth_dir=gt_depth_dir,
        gt_traj_path=gt_traj_path,
        png_depth_scale=0.00015244,
        min_depth_m=1e-3,
        max_depth_m=200.0,
        align_mode="scale",  # try "scale_shift" too
        pose_align_mode="se3",  # or "se3" for no scale
        device="cuda",
        save_visualizations=True,
        results_dir="results",
        evaluate_pose=True,
    )

    print("\n" + "=" * 60)
    print("DEPTH METRICS (Average)")
    print("=" * 60)
    for k, v in avg.items():
        print(f"{k:>15}: {v}")
    
    if pose_metrics:
        print("\n" + "=" * 60)
        print("POSE METRICS")
        print("=" * 60)
        print("\n--- Absolute Trajectory Error (ATE) ---")
        print(f"{'ATE_RMSE':>20}: {pose_metrics['ATE_RMSE']:.4f} m")
        print(f"{'ATE_mean':>20}: {pose_metrics['ATE_mean']:.4f} m")
        print(f"{'ATE_median':>20}: {pose_metrics['ATE_median']:.4f} m")
        print(f"{'Scale':>20}: {pose_metrics['scale']:.4f}")
        
        print("\n--- Relative Pose Error (RPE) ---")
        print(f"{'RPE_trans_RMSE':>20}: {pose_metrics['RPE_trans_RMSE']:.4f} m")
        print(f"{'RPE_trans_mean':>20}: {pose_metrics['RPE_trans_mean']:.4f} m")
        print(f"{'RPE_rot_RMSE':>20}: {pose_metrics['RPE_rot_RMSE']:.4f} deg")
        print(f"{'RPE_rot_mean':>20}: {pose_metrics['RPE_rot_mean']:.4f} deg")
        
        print("\n--- Relative Accuracy (AUC) ---")
        print(f"{'RRA_AUC':>20}: {pose_metrics['RRA_AUC']:.4f}")
        print(f"{'RTA_AUC':>20}: {pose_metrics['RTA_AUC']:.4f}")
        print(f"{'RRA@3deg':>20}: {pose_metrics['RRA@3deg']:.4f}")
        print(f"{'RRA@30deg':>20}: {pose_metrics['RRA@30deg']:.4f}")
        print(f"{'RTA@3deg':>20}: {pose_metrics['RTA@3deg']:.4f}")
        print(f"{'RTA@30deg':>20}: {pose_metrics['RTA@30deg']:.4f}")
        print(f"{'Num pairs':>20}: {pose_metrics['num_pairs']}")
    
    print("\n" + "=" * 60)
    print(f"Visualizations saved to: results/")
    print(f"Total frames evaluated: {len(per_frame)}")
    print("=" * 60)
