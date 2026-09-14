import os
import subprocess
import sys
import argparse
from pathlib import Path

def download_dataset():
    project_root = os.getcwd()
    target_dir = os.path.join(project_root, "data", "raw")

    os.makedirs(target_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="下载 ds004117 数据集")
    parser.add_argument("--subject", type=str, default=None, help="仅下载指定被试")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    target_dir = project_root / "data" / "raw"

    try:
        from openneuro import download
        test_file = os.path.join(target_dir, 'sub-{args.subject}')
        if os.path.exists(test_file):
            print(f"数据已存在: {test_file}")
            print("无需重复下载，直接运行预处理脚本即可。")
            return
        print(f"开始下载 ds004117 至: {target_dir}")
        include = f'sub-{args.subject}/eeg' if args.subject else None
        download(dataset='ds004117', target_dir=target_dir,include=include)
        print("下载完成！现在可以运行 pipeline1_empirical_rsa/scripts/01_preprocess_all_subjects.py 了")
    except ImportError:
        print("未安装 openneuro-py，正在尝试自动安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openneuro-py"])
        print("安装完成，请重新运行本脚本。")
    except Exception as e:
        print(f"下载失败: {e}")
        print("手动备选方案：")
        print("1. 浏览器打开 https://openneuro.org/datasets/ds004117/versions/1.0.1")
        print("2. 点击 'Download' 下载压缩包，解压后把里面的内容放进 data/raw/")


if __name__ == "__main__":
    download_dataset()