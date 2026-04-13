#!/usr/bin/env python3
import os
import requests
from tqdm import tqdm

MODEL_URLS = {
    "cutie-base-mega.pth": "https://github.com/baidu/CUTIE/releases/download/v1.0/cutie-base-mega.pth",
    "cutie-base-nomose.pth": "https://github.com/baidu/CUTIE/releases/download/v1.0/cutie-base-nomose.pth",
    "cutie-base-wmose.pth": "https://github.com/baidu/CUTIE/releases/download/v1.0/cutie-base-wmose.pth",
    "cutie-small-mega.pth": "https://github.com/baidu/CUTIE/releases/download/v1.0/cutie-small-mega.pth",
    "cutie-small-nomose.pth": "https://github.com/baidu/CUTIE/releases/download/v1.0/cutie-small-nomose.pth",
    "cutie-small-wmose.pth": "https://github.com/baidu/CUTIE/releases/download/v1.0/cutie-small-wmose.pth"
}

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def download_file(url, dest):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest, 'wb') as file, tqdm(
        desc=os.path.basename(dest),
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    for filename, url in MODEL_URLS.items():
        dest = os.path.join(MODEL_DIR, filename)
        if os.path.exists(dest):
            print(f"{filename} 已存在，跳过下载。")
            continue
        print(f"正在下载 {filename} ...")
        download_file(url, dest)
    print("所有模型下载完成。")

if __name__ == "__main__":
    main()
