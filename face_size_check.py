# -*- coding: utf-8 -*-
"""抽取视频前/中/后 3 帧，检查最大人脸是否至少为 128x128px。"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
POSITIONS = (0.1, 0.5, 0.9)
MIN_FACE_SIZE = 128
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2026may.onnx"
)
MODEL_PATH = Path(__file__).with_name(".face_size_check") / "face_detection_yunet.onnx"
KNOWN_YT_DLP = Path(r"D:\github好用工具\youtube抓视频\yt-dlp_win\yt-dlp.exe")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def require_tools():
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise SystemExit(f"缺少工具：{', '.join(missing)}")
    try:
        import cv2
    except ImportError:
        raise SystemExit("缺少 OpenCV，请运行：python -m pip install opencv-python-headless")
    return cv2


def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        return
    MODEL_PATH.parent.mkdir(exist_ok=True)
    print("首次运行：下载轻量 YuNet 人脸检测模型……")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, MODEL_PATH.open("wb") as output:
            shutil.copyfileobj(response, output)
    except Exception as error:
        MODEL_PATH.unlink(missing_ok=True)
        raise SystemExit(f"模型下载失败：{error}")


def get_duration(video):
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(video),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFprobe 读取失败")
    duration = float(json.loads(result.stdout)["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("视频时长无效")
    return duration


def extract_frame(video, timestamp, output):
    command = [
        "ffmpeg", "-v", "error", "-ss", f"{timestamp:.3f}", "-i", str(video),
        "-frames:v", "1", "-q:v", "2", "-y", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if result.returncode != 0 or not output.exists():
        raise RuntimeError(result.stderr.strip() or "FFmpeg 抽帧失败")


def detect_largest_face(cv2, detector, image_path):
    import numpy as np

    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("无法读取抽帧图片")
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        cv2.imencode(".jpg", image)[1].tofile(image_path)
        return 0, 0, 0, "未检出"

    largest = max(faces, key=lambda face: float(face[2] * face[3]))
    x, y, face_width, face_height = [int(round(value)) for value in largest[:4]]
    score = float(largest[-1])
    passed = face_width >= MIN_FACE_SIZE and face_height >= MIN_FACE_SIZE
    color = (0, 180, 0) if passed else (0, 0, 255)
    cv2.rectangle(image, (x, y), (x + face_width, y + face_height), color, 3)
    cv2.putText(
        image, f"{face_width}x{face_height}px", (max(0, x), max(25, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
    )
    cv2.imencode(".jpg", image)[1].tofile(image_path)
    return face_width, face_height, score, "合格" if passed else "不足128px"


def find_videos(source):
    if source.is_file():
        return [source] if source.suffix.lower() in VIDEO_EXTENSIONS else []
    return sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def safe_name(relative_path):
    return "__".join(relative_path.parts).replace(" ", "_")


def check_video(cv2, detector, video, display_name, evidence_prefix, evidence_dir):
    duration = get_duration(video)
    frame_results = []
    for label, position in zip(("前", "中", "后"), POSITIONS):
        timestamp = duration * position
        output = evidence_dir / f"{evidence_prefix}__{label}_{timestamp:.2f}s.jpg"
        extract_frame(video, timestamp, output)
        width, height, score, status = detect_largest_face(cv2, detector, output)
        frame_results.append((label, timestamp, width, height, score, status, output))

    statuses = [item[5] for item in frame_results]
    overall = "合格" if all(status == "合格" for status in statuses) else (
        "不合格" if "不足128px" in statuses else "需复核"
    )
    return display_name, duration, overall, frame_results


def read_tiktok_links(path):
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError("无法识别链接文件编码")

    links = []
    if path.suffix.lower() == ".csv":
        for row in csv.DictReader(text.splitlines()):
            values = [row.get("视频链接", "")] if "视频链接" in row else row.values()
            links.extend(value.strip() for value in values if value and "tiktok.com/" in value)
    else:
        links.extend(line.strip() for line in text.splitlines() if "tiktok.com/" in line)
    return list(dict.fromkeys(links))


def get_yt_dlp():
    command = shutil.which("yt-dlp")
    if command:
        return command
    if KNOWN_YT_DLP.exists():
        return str(KNOWN_YT_DLP)
    raise SystemExit("未找到 yt-dlp，无法读取 TikTok 视频链接")


def download_tiktok(yt_dlp, url, target_dir):
    target_dir.mkdir(parents=True)
    output_template = str(target_dir / "%(id)s.%(ext)s")
    command = [
        yt_dlp, "--no-playlist", "--no-warnings", "--quiet",
        "--format", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "--merge-output-format", "mp4", "--output", output_template,
        "--print", "after_move:filepath", url,
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "yt-dlp 下载失败")
    candidates = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    candidates.extend(target_dir.glob("*"))
    video = next((path for path in candidates if path.exists() and path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS), None)
    if not video:
        raise RuntimeError("yt-dlp 未生成视频文件")
    return video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", help="TikTok URL、CSV/TXT 链接清单，或本地视频路径")
    args = parser.parse_args()

    source_text = args.source or input("粘贴 TikTok URL、链接清单或视频路径：").strip().strip('"')
    is_url = "tiktok.com/" in source_text
    source = None if is_url else Path(source_text).resolve()
    if source is not None and not source.exists():
        raise SystemExit(f"路径不存在：{source}")

    cv2 = require_tools()
    ensure_model()
    remote_links = [source_text] if is_url else (
        read_tiktok_links(source) if source.is_file() and source.suffix.lower() in {".csv", ".txt"} else []
    )
    local_videos = [] if remote_links else find_videos(source)
    if not remote_links and not local_videos:
        raise SystemExit("没有找到 TikTok 链接或支持的视频文件")

    base_dir = Path.cwd() if is_url else source.parent
    report_name = "TikTok" if is_url else source.stem
    report_dir = base_dir / f"{report_name}_人脸尺寸检查_{datetime.now():%Y%m%d_%H%M%S_%f}"
    evidence_dir = report_dir / "证据帧"
    evidence_dir.mkdir(parents=True)
    csv_path = report_dir / "人脸尺寸检查.csv"
    detector = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (320, 320), 0.7, 0.3, 5000)

    counts = {"合格": 0, "不合格": 0, "需复核": 0, "处理失败": 0}
    items = [(link, link, True) for link in remote_links]
    items.extend(
        (str(video.relative_to(source)) if source.is_dir() else video.name, video, False)
        for video in local_videos
    )
    yt_dlp = get_yt_dlp() if remote_links else None
    print(f"报告：{csv_path}")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "视频", "TikTok链接", "时长秒", "抽帧位置", "时间戳秒", "最大人脸宽px", "最大人脸高px",
            "检测置信度", "该帧结论", "视频结论", "证据图", "错误",
        ])
        file.flush()
        with tempfile.TemporaryDirectory(prefix="tiktok-face-check-") as temp_root:
            for index, (display_name, item, remote) in enumerate(items, 1):
                print(f"[{index}/{len(items)}] {display_name}")
                rows = []
                try:
                    video = download_tiktok(yt_dlp, item, Path(temp_root) / str(index)) if remote else item
                    name, duration, overall, frames = check_video(
                        cv2, detector, video, display_name, f"video_{index:05d}", evidence_dir
                    )
                    counts[overall] += 1
                    for label, timestamp, width, height, score, status, image_path in frames:
                        rows.append([
                            name, item if remote else "", f"{duration:.3f}", label, f"{timestamp:.3f}",
                            width, height, f"{score:.4f}", status, overall, str(image_path), "",
                        ])
                except Exception as error:
                    counts["处理失败"] += 1
                    rows.append([
                        display_name, item if remote else "", "", "", "", "", "", "",
                        "处理失败", "处理失败", "", str(error),
                    ])
                writer.writerows(rows)
                file.flush()

    print(f"完成：{len(items)} 个视频\n报告：{csv_path}")
    print("，".join(f"{status}：{count}" for status, count in counts.items()))
    print("说明：仅检查三个采样帧中检出的人脸尺寸，不代表全片质量合格。")
    return 2 if counts["处理失败"] else (1 if counts["不合格"] or counts["需复核"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
