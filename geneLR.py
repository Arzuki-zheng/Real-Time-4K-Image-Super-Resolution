import os
import glob
import torch
import numpy as np
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

def downsample_to_lr(img_path, output_path, scale=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"⚠️ 讀取失敗：{img_path}")
        return

    h, w, _ = img.shape
    # 先確保 HR 尺寸能被 scale 整除，且是偶數
    # 目標：new_h, new_w 既是偶數，又能被 scale 整除
    def fix_dim(x):
        x = x - (x % scale)        # 先讓它能被 scale 整除
        if x % 2 != 0:             # 再確保是偶數
            x -=  scale            # 再往下減一個 scale（因為 scale 本身是 2 或 3 之類）
        return x

    new_h = fix_dim(h)
    new_w = fix_dim(w)

    if new_h <= 0 or new_w <= 0:
        print(f"尺寸太小無法處理：{img_path} ({h}x{w})")
        return

    if new_h != h or new_w != w:
        img = img[:new_h, :new_w, :]
        print(f"裁剪 HR：{h}x{w} -> {new_w}x{new_h}")

    img = img.astype(np.float32) / 255.0
    img_tensor = uint2tensor3(img).unsqueeze(0).to(device)

    # HR -> LR（縮小）
    lr_tensor = interpolate(
        img_tensor,
        scale_factor=1.0 / scale,
        mode="bicubic",
        align_corners=False
    )

    lr_img = tensor2uint(lr_tensor)

    # 再保險一次：確保 LR 也是偶數尺寸
    lh, lw, _ = lr_img.shape
    if lh % 2 != 0 or lw % 2 != 0:
        lr_img = lr_img[:lh - (lh % 2), :lw - (lw % 2), :]
        print(f"裁剪 LR：{lh}x{lw} -> {lr_img.shape[1]}x{lr_img.shape[0]}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, lr_img)
    print(f"✅ 產生 LR x{scale}: {output_path}")

def process_folder(input_folder, output_folder, scale=2):
    os.makedirs(output_folder, exist_ok=True)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    for ext in extensions:
        for img_file in glob.glob(os.path.join(input_folder, ext)):
            filename = os.path.basename(img_file)
            output_file = os.path.join(output_folder, filename)
            print(f"Processing {img_file} -> {output_file}")
            downsample_to_lr(img_file, output_file, scale)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='輸入 HR 圖片檔案或資料夾路徑')
    parser.add_argument('--output', type=str, required=True, help='輸出 LR 圖片檔案或資料夾路徑')
    parser.add_argument('--scale', type=int, default=2, help='縮小倍率 (例如 2, 3)')
    args = parser.parse_args()

    if os.path.isdir(args.input):
        process_folder(args.input, args.output, args.scale)
    else:
        downsample_to_lr(args.input, args.output, args.scale)
