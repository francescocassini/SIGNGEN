#!/usr/bin/env python3
import argparse
import glob
import os
import pickle

import imageio.v2 as imageio
import numpy as np
import torch

from mGPT.render.matplot.plot_3d_global import plot_3d_motion
from mGPT.utils.human_models import get_coord


def load_mean_std(mean_path, std_path, device):
    mean = torch.load(mean_path, map_location=device).float()
    std = torch.load(std_path, map_location=device).float()
    mean = mean[(3 + 3 * 11):]
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
    std = std[(3 + 3 * 11):]
    std = torch.cat([std[:-20], std[-10:]], dim=0)
    return mean.to(device), std.to(device)


def feats_to_joints22(feats_np, mean, std, device):
    feats = torch.from_numpy(feats_np).float().to(device)  # [T, 133]
    T = feats.shape[0]
    feats = feats * std.unsqueeze(0) + mean.unsqueeze(0)

    zero_pose = torch.zeros((T, 36), device=device)
    shape_param = torch.tensor(
        [[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
          0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]],
        dtype=torch.float32,
        device=device,
    ).repeat(T, 1)
    full = torch.cat([zero_pose, feats], dim=-1)  # [T, 169]

    _, joints = get_coord(
        root_pose=full[..., 0:3],
        body_pose=full[..., 3:66],
        lhand_pose=full[..., 66:111],
        rhand_pose=full[..., 111:156],
        jaw_pose=full[..., 156:159],
        shape=shape_param,
        expr=full[..., 159:169],
    )
    joints22 = joints[:, :22, :].detach().cpu().numpy()  # [T, 22, 3]
    return joints22


def render_skeleton_gif(joints, text, out_path, fps):
    frames_rgba = plot_3d_motion((joints, None, text), fps=fps)
    frames = frames_rgba[..., :3].cpu().numpy().astype(np.uint8)
    duration = 1.0 / max(fps, 1)
    imageio.mimsave(out_path, list(frames), duration=duration)


def concat_side_by_side(frames_a, frames_b):
    T = max(len(frames_a), len(frames_b))
    if len(frames_a) < T:
        pad = np.repeat(frames_a[-1][None, ...], T - len(frames_a), axis=0)
        frames_a = np.concatenate([frames_a, pad], axis=0)
    if len(frames_b) < T:
        pad = np.repeat(frames_b[-1][None, ...], T - len(frames_b), axis=0)
        frames_b = np.concatenate([frames_b, pad], axis=0)
    return np.concatenate([frames_a, frames_b], axis=2)


def resolve_sample(pred_dir, sample, index):
    if sample:
        p = sample
        if not p.endswith(".pkl"):
            p = p + ".pkl"
        if not os.path.isabs(p):
            p = os.path.join(pred_dir, p)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Sample not found: {p}")
        return p

    files = sorted(glob.glob(os.path.join(pred_dir, "*.pkl")))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {pred_dir}")
    if index < 0 or index >= len(files):
        raise IndexError(f"Index {index} out of range (0..{len(files)-1})")
    return files[index]


def main():
    default_data_root = os.environ.get("SOKE_DATA_ROOT", "../data")
    default_csl_root = os.environ.get("SOKE_CSL_ROOT", os.path.join(default_data_root, "CSL-Daily"))
    default_mean = os.environ.get("SOKE_CSL_MEAN_PATH", os.path.join(default_csl_root, "mean.pt"))
    default_std = os.environ.get("SOKE_CSL_STD_PATH", os.path.join(default_csl_root, "std.pt"))

    parser = argparse.ArgumentParser(description="Preview one inferred test sample as skeleton video.")
    parser.add_argument("--pred_dir", default="results/mgpt/SOKE_INFER/test_rank_0")
    parser.add_argument("--sample", default=None, help="sample filename stem or .pkl path")
    parser.add_argument("--index", type=int, default=0, help="sample index if --sample is not provided")
    parser.add_argument("--out_dir", default="visualize/preview_samples")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--mean_path", default=default_mean)
    parser.add_argument("--std_path", default=default_std)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_path = resolve_sample(args.pred_dir, args.sample, args.index)
    with open(sample_path, "rb") as f:
        data = pickle.load(f)

    text = data["text"]
    feats_rst = data["feats_rst"]
    feats_ref = data["feats_ref"]
    stem = os.path.splitext(os.path.basename(sample_path))[0]

    mean, std = load_mean_std(args.mean_path, args.std_path, device)
    joints_rst = feats_to_joints22(feats_rst, mean, std, device)
    joints_ref = feats_to_joints22(feats_ref, mean, std, device)

    out_pred = os.path.join(args.out_dir, f"{stem}_pred.gif")
    out_compare = os.path.join(args.out_dir, f"{stem}_compare_ref_pred.gif")
    out_txt = os.path.join(args.out_dir, f"{stem}_text.txt")

    # Pred-only
    render_skeleton_gif(joints_rst, text, out_pred, args.fps)

    # Compare ref vs pred
    frames_ref = plot_3d_motion((joints_ref, None, f"REF | {text}"), fps=args.fps)[..., :3].cpu().numpy().astype(np.uint8)
    frames_pred = plot_3d_motion((joints_rst, None, f"PRED | {text}"), fps=args.fps)[..., :3].cpu().numpy().astype(np.uint8)
    frames_cmp = concat_side_by_side(frames_ref, frames_pred)
    duration = 1.0 / max(args.fps, 1)
    imageio.mimsave(out_compare, list(frames_cmp), duration=duration)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"Sample: {sample_path}")
    print(f"Input text: {text}")
    print(f"Pred video: {out_pred}")
    print(f"Compare video: {out_compare}")
    print(f"Text file: {out_txt}")


if __name__ == "__main__":
    main()
