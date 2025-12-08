import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
import cv2

from torch.nn.functional import interpolate

import model
from utils import image, parser

def load_checkpoint(model, device, time_stamp=None):
    checkpoint = glob.glob(os.path.join("code/checkpoints", time_stamp + ".pth"))
    if isinstance(checkpoint, list):
        checkpoint = checkpoint.pop(0)
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model

def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

def reparameterize(config, net, device, save_rep_checkpoint=False):
    config.is_train = False
    rep_model = torch.nn.DataParallel(model.__dict__[config.arch](config)).to(device)
    rep_state_dict = rep_model.state_dict()
    pretrained_state_dict = net.state_dict()

    for k, v in rep_state_dict.items():
        if "rep_conv.weight" in k:
            k0 = pretrained_state_dict[k.replace("rep", "expand")]
            k1 = pretrained_state_dict[k.replace("rep", "fea")]
            k2 = pretrained_state_dict[k.replace("rep", "reduce")]

            bias_str = k.replace("weight", "bias")
            b0 = pretrained_state_dict[bias_str.replace("rep", "expand")]
            b1 = pretrained_state_dict[bias_str.replace("rep", "fea")]
            b2 = pretrained_state_dict[bias_str.replace("rep", "reduce")]

            mid_feats, n_feats = k0.shape[:2]

            for i in range(mid_feats):
                k1[i, i, 1, 1] += 1.0

            merged_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            merged_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, mid_feats, 3, 3).to(device)
            merged_b0b1 = F.conv2d(input=merged_b0b1, weight=k1, bias=b1)

            merged_k0k1k2 = F.conv2d(input=merged_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
            merged_b0b1b2 = F.conv2d(input=merged_b0b1, weight=k2, bias=b2).view(-1)

            for i in range(n_feats):
                merged_k0k1k2[i, i, 1, 1] += 1.0

            rep_state_dict[k] = merged_k0k1k2.float()
            rep_state_dict[bias_str] = merged_b0b1b2.float()

        elif "rep_conv.bias" in k:
            pass

        elif k in pretrained_state_dict.keys():
            rep_state_dict[k] = pretrained_state_dict[k]

    rep_model.load_state_dict(rep_state_dict, strict=True)
    if save_rep_checkpoint:
        torch.save(rep_state_dict, f"rep_model_{config.checkpoint_id}.pth")

    return rep_model


def test(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.nn.DataParallel(model.__dict__[config.arch](config)).to(device)
    net = load_checkpoint(net, device, config.checkpoint_id)

    if config.rep:
        net = reparameterize(config, net, device)

    net.eval()

    input_files = sorted([f for f in os.listdir(config.input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    os.makedirs(config.output_path, exist_ok=True)

    for file_name in input_files:
        input_img_path = os.path.join(config.input_path, file_name)
        output_img_path = os.path.join(config.output_path, file_name)

        img = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
        img = img.astype(np.float32) / 255.0
        lr_tensor = uint2tensor3(img).unsqueeze(0).to(device)

        with torch.no_grad():
            if config.bicubic:
                sr_tensor = interpolate(lr_tensor, scale_factor=config.scale, mode="bicubic", align_corners=False).clamp(0, 1)
            else:
                sr_tensor = net(lr_tensor).clamp(0, 1)

        sr_img = image.tensor2uint(sr_tensor * 255.0)
        cv2.imwrite(output_img_path, sr_img)
        print(f"✅ Saved SR image to: {output_img_path}")


if __name__ == "__main__":
    args = parser.base_parser()
    test(args)