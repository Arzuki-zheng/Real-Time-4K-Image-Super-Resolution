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

def simulate_up_down_sample(img_path, output_path, scale=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 讀取圖片並轉 tensor
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"⚠️ 讀取失敗：{img_path}")
        return
    img = img.astype(np.float32) / 255.0
    img_tensor = uint2tensor3(img).unsqueeze(0).to(device)

    # Upsample
    up_tensor = interpolate(img_tensor, scale_factor=scale, mode="bicubic", align_corners=False)

    # Downsample 回原解析度
    down_tensor = interpolate(up_tensor, scale_factor=1.0/scale, mode="bicubic", align_corners=False)

    # 存儲圖片
    down_img = tensor2uint(down_tensor)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, down_img)
    #print(f"✅ 模擬完成：{output_path}")

def process_folder(input_folder, output_folder, scale=2):
    os.makedirs(output_folder, exist_ok=True)
    # 定義支援的圖片格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    # 遍歷所有符合格式的檔案
    for ext in extensions:
        for img_file in glob.glob(os.path.join(input_folder, ext)):
            filename = os.path.basename(img_file)
            output_file = os.path.join(output_folder, filename)
            print(f"Processing {img_file} -> {output_file}")
            simulate_up_down_sample(img_file, output_file, scale)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='輸入圖片檔案或資料夾路徑')
    parser.add_argument('--output', type=str, required=True, help='輸出圖片檔案或資料夾路徑')
    parser.add_argument('--scale', type=int, default=2, help='Upsample/Downsample 的比例')
    args = parser.parse_args()

    # 判斷輸入是否為資料夾
    if os.path.isdir(args.input):
        process_folder(args.input, args.output, args.scale)
    else:
        simulate_up_down_sample(args.input, args.output, args.scale)
