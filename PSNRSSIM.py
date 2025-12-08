import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
import cv2
import csv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import interpolate

def calculate_psnr_ssim(img1, img2):
    # 確保兩張圖尺寸完全一致，如果不一致則報錯（理論上在外部已經處理過裁切了）
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same dimensions. Got {img1.shape} and {img2.shape}")

    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    psnr_val = psnr(img1, img2, data_range=1.0)

    # 判斷是否為彩色
    if img1.ndim == 3 and img1.shape[2] == 3:
        channel_axis = -1
    else:
        channel_axis = None

    # 自動調整 win_size
    min_dim = min(img1.shape[0], img1.shape[1])
    win_size = 7 if min_dim >= 7 else (min_dim if min_dim % 2 == 1 else min_dim - 1)

    ssim_val = ssim(img1, img2, data_range=1.0, channel_axis=channel_axis, win_size=win_size)
    return psnr_val, ssim_val

def crop_to_match(img1, img2):
    """
    將兩張圖片裁切到相同的最小尺寸。
    通常是裁切右邊和下面多出來的像素。
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    min_h = min(h1, h2)
    min_w = min(w1, w2)
    
    # 如果尺寸不同，就裁切
    if h1 != min_h or w1 != min_w:
        img1 = img1[:min_h, :min_w]
    if h2 != min_h or w2 != min_w:
        img2 = img2[:min_h, :min_w]
        
    return img1, img2

def evaluate_psnr_ssim(gt_folder, pred_folder):
    image_files = sorted([f for f in os.listdir(gt_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    psnr_list = []
    ssim_list = []
    results = []

    print(f"🔍 Start evaluation...\nGT: {gt_folder}\nPred: {pred_folder}\n")

    for fname in image_files:
        gt_path = os.path.join(gt_folder, fname)
        pred_path = os.path.join(pred_folder, fname)

        if not os.path.exists(pred_path):
            print(f"⚠️ Skipping {fname}, not found in prediction folder")
            continue

        gt_img = cv2.imread(gt_path)
        pred_img = cv2.imread(pred_path)

        if gt_img is None or pred_img is None:
            print(f"❌ Failed to load: {fname}")
            continue

        # === 修改點開始：自動裁切 ===
        orig_gt_shape = gt_img.shape
        orig_pred_shape = pred_img.shape

        if gt_img.shape != pred_img.shape:
            # 呼叫裁切函式讓兩者尺寸一致
            gt_img, pred_img = crop_to_match(gt_img, pred_img)
            
            # 選擇性印出警告，告訴你哪張圖被切了（為了除錯方便）
            # print(f"✂️ Cropped {fname}: GT{orig_gt_shape} / Pred{orig_pred_shape} -> Now {gt_img.shape}")
        # === 修改點結束 ===

        try:
            psnr_val, ssim_val = calculate_psnr_ssim(gt_img, pred_img)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            results.append((fname, psnr_val, ssim_val))
            print(f"{fname}: PSNR = {psnr_val:.2f}, SSIM = {ssim_val:.4f}")
        except Exception as e:
            print(f"❌ Error calculating metrics for {fname}: {e}")

    if results:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        print(f"\n📊 Average PSNR: {avg_psnr:.2f}")
        print(f"📊 Average SSIM: {avg_ssim:.4f}")

        csv_path = os.path.join(pred_folder, "psnr_ssim_report.csv")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Filename", "PSNR", "SSIM"])
            for row in results:
                writer.writerow(row)
        print(f"📁 Results saved to: {csv_path}")
    else:
        print("No valid image pairs found.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_folder', type=str, required=True, help='Ground truth image folder')
    parser.add_argument('--pred_folder', type=str, required=True, help='Predicted image folder')
    args = parser.parse_args()

    evaluate_psnr_ssim(args.gt_folder, args.pred_folder)
