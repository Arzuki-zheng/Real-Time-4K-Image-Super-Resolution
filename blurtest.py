import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
import cv2
import csv

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

def blur_score_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def edge_score_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.mean(edges)

def high_freq_energy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    h, w = magnitude_spectrum.shape
    center_region = magnitude_spectrum[h//4:3*h//4, w//4:3*w//4]
    outer_region = magnitude_spectrum.copy()
    outer_region[h//4:3*h//4, w//4:3*w//4] = 0
    return np.mean(outer_region)

def analyze_blur_in_folder(folder_path):
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    blur_scores = []
    edge_scores = []
    freq_scores = []
    results = []

    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"❌ Failed to load: {fname}")
            continue

        lap_var = blur_score_laplacian(img)
        edge_mean = edge_score_canny(img)
        freq_energy = high_freq_energy(img)
        blur_scores.append(lap_var)
        edge_scores.append(edge_mean)
        freq_scores.append(freq_energy)
        results.append((fname, lap_var, edge_mean, freq_energy))

        print(f"{fname}: Laplacian Var = {lap_var:.2f}, Canny Edge Mean = {edge_mean:.2f}, High-Freq Energy = {freq_energy:.2f}")

    if results:
        avg_blur = np.mean(blur_scores)
        avg_edge = np.mean(edge_scores)
        avg_freq = np.mean(freq_scores)
        print(f"\nAverage Laplacian Variance: {avg_blur:.2f}")
        print(f"Average Canny Edge Mean: {avg_edge:.2f}")
        print(f"Average High-Frequency Energy: {avg_freq:.2f}")

        csv_path = os.path.join(folder_path, "blur_analysis_DIV2K.csv")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Filename", "LaplacianVariance", "CannyEdgeMean", "HighFreqEnergy"])
            for row in results:
                writer.writerow(row)
        print(f"Results saved to: {csv_path}")
    else:
        print("No valid images found.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Folder path to analyze blur')
    args = parser.parse_args()

    analyze_blur_in_folder(args.folder)
