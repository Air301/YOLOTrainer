import argparse
import os
import json
from server.detection import Yolo11Detector, Yolo11Config
import torch


def main():
    args = parse_opt()
    device = torch.device(args.device) if args.device != "auto" else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_config = Yolo11Config(
        weight=args.detector_weights_path,
        device=device,
        data=args.data_config
    )
    yoloe = Yolo11Detector(yolo_config)
    yoloe.train()


def parse_opt():
    parser = argparse.ArgumentParser(description="Benchmark script for LightGlue")
    parser.add_argument(
        "--config_file", type=str,
        help="path to a JSON config file"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="device to benchmark on",
    )
    parser.add_argument(
        "--save_path", type=str,
        help="path where figure should be saved"
    )
    parser.add_argument(
        "--detector_weights_path",
        default="weights/yolo11l.pt",
        type=str, help="path to detector weights file"
    )
    parser.add_argument(
        "--data_config",
        default="configs/data/DOTAv1.5.yaml",
        type=str, help="dataset config file"
    )
    args = parser.parse_args()
    # 如果提供了参数文件，则从文件加载参数
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, "r") as f:
            file_args = json.load(f)  # 读取 JSON 文件

        # 遍历文件中的参数，如果命令行参数未提供，则使用文件参数
        for key, value in file_args.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)
    return args


if __name__ == "__main__":
    main()