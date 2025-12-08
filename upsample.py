import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
import cv2

from torch.nn.functional import interpolate

def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

def tensor2uint(img_tensor):
    img = img_tensor.squeeze().cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255.0).round().astype(np.uint8)
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    return img

def bicubic_upsample_folder(input_dir, output_dir, scale=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        img = img.astype(np.float32) / 255.0
        img_tensor = uint2tensor3(img).unsqueeze(0).to(device)

        up_tensor = interpolate(img_tensor, scale_factor=scale, mode="bicubic", align_corners=False)
        up_img = tensor2uint(up_tensor)

        cv2.imwrite(out_path, up_img)
        print(f"✅ Upsampled image saved to: {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input image folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save upsampled images')
    parser.add_argument('--scale', type=int, default=2, help='Upsample scale')
    args = parser.parse_args()

    bicubic_upsample_folder(args.input_dir, args.output_dir, args.scale)