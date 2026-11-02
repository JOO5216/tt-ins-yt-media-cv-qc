#!/usr/bin/env python3
"""递归检查目录中的视频和图片是否满足交付规格。"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
import json
import shutil
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
    ".ts", ".mts", ".m2ts", ".wmv", ".flv", ".mpg", ".mpeg",
}
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff",
}
JPEG_SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="递归检查视频和图片规格；只读取文件，不修改媒体。"
    )
    parser.add_argument("folder", type=Path, help="需要递归检查的文件夹")
    return parser.parse_args()


def read_image_info(path: Path) -> tuple[str, int, int]:
    """从文件头读取 JPEG/PNG 的真实格式和原生尺寸。"""
    with path.open("rb") as file:
        signature = file.read(24)
        if signature.startswith(b"\x89PNG\r\n\x1a\n"):
            if len(signature) < 24 or signature[12:16] != b"IHDR":
                raise ValueError("PNG 文件头无效")
            width, height = struct.unpack(">II", signature[16:24])
            return "PNG", width, height

        if not signature.startswith(b"\xff\xd8"):
            raise ValueError("实际格式不是 JPEG/PNG")

        file.seek(2)
        while True:
            byte = file.read(1)
            if not byte:
                raise ValueError("JPEG 中未找到尺寸信息")
            if byte != b"\xff":
                continue

            marker_byte = file.read(1)
            while marker_byte == b"\xff":
                marker_byte = file.read(1)
            if not marker_byte:
                raise ValueError("JPEG 文件不完整")

            marker = marker_byte[0]
            if marker in {0x01, 0xD8, 0xD9}:
                continue

            length_bytes = file.read(2)
            if len(length_bytes) != 2:
                raise ValueError("JPEG 段长度无效")
            segment_length = struct.unpack(">H", length_bytes)[0]
            if segment_length < 2:
                raise ValueError("JPEG 段长度无效")

            if marker in JPEG_SOF_MARKERS:
                data = file.read(5)
                if len(data) != 5:
                    raise ValueError("JPEG 尺寸信息不完整")
                height, width = struct.unpack(">HH", data[1:5])
                return "JPEG", width, height

            file.seek(segment_length - 2, 1)


def check_image(path: Path) -> tuple[bool, str, list[str]]:
    try:
        image_format, width, height = read_image_info(path)
    except (OSError, ValueError, struct.error) as exc:
        return False, "无法读取图片规格", [str(exc)]

    errors = []
    short_edge = min(width, height)
    if short_edge < 512:
        errors.append(f"短边 {short_edge}px < 512px")

    detail = f"{image_format}, {width}x{height}, 短边 {short_edge}px"
    return not errors, detail, errors


def parse_frame_rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        raise ValueError("缺少有效帧率")
    fps = float(Fraction(value))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("缺少有效帧率")
    return fps


def check_video(path: Path, ffprobe: str) -> tuple[bool, str, list[str]]:
    command = [
        ffprobe,
        "-v", "error",
        "-show_entries",
        "format=format_name,duration:stream=codec_type,codec_name,width,height,avg_frame_rate,duration",
        "-of", "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        data = json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        return False, "ffprobe 读取超时", ["超过60秒，需复核"]
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        message = getattr(exc, "stderr", None) or str(exc)
        return False, "ffprobe 无法读取视频规格", [message.strip()]

    stream = next(
        (item for item in data.get("streams", []) if item.get("codec_type") == "video"),
        None,
    )
    if stream is None:
        return False, "未找到视频流", ["文件中没有可识别的视频流"]

    errors = []
    video_format = data.get("format", {}).get("format_name", "")
    codec = stream.get("codec_name", "unknown")
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    short_edge = min(width, height)

    duration_text = data.get("format", {}).get("duration") or stream.get("duration")
    try:
        duration = float(duration_text)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("无效时长")
    except (TypeError, ValueError):
        duration = 0.0
        errors.append("缺少有效时长")

    try:
        fps = parse_frame_rate(stream.get("avg_frame_rate"))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
        errors.append("缺少有效平均帧率")

    if path.suffix.lower() != ".mp4" or "mp4" not in video_format.split(","):
        errors.append("格式不是 MP4")
    if codec not in {"h264", "hevc"}:
        errors.append(f"编码 {codec} 不是 H.264/H.265")
    if short_edge < 720:
        errors.append(f"短边 {short_edge}px < 720px")
    if fps and fps < 24:
        errors.append(f"平均帧率 {fps:.3f}fps < 24fps")
    if duration and not 5 <= duration <= 60:
        errors.append(f"时长 {duration:.3f}s 不在 5～60s")

    detail = (
        f"{path.suffix.lower()}, {codec}, {width}x{height}, "
        f"{fps:.3f}fps, {duration:.3f}s"
    )
    return not errors, detail, errors


def main() -> int:
    args = parse_args()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"错误：文件夹不存在：{folder}", file=sys.stderr)
        return 2

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        print("错误：未找到 ffprobe，请先安装 FFmpeg 并加入 PATH。", file=sys.stderr)
        return 2

    media_files = [
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
    ]

    passed = 0
    failed = 0
    video_count = 0
    image_count = 0

    if not media_files:
        print("错误：未找到支持的媒体文件", file=sys.stderr)
        return 2

    report_path = folder / f"媒体规格检查_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    print(f"报告：{report_path}")
    with report_path.open("w", newline="", encoding="utf-8-sig") as report:
        writer = csv.writer(report)
        writer.writerow(["文件", "类型", "结论", "规格", "不通过原因"])
        report.flush()
        for path in sorted(media_files, key=lambda item: str(item).lower()):
            relative_path = path.relative_to(folder)
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                video_count += 1
                ok, detail, errors = check_video(path, ffprobe)
            else:
                image_count += 1
                ok, detail, errors = check_image(path)

            if ok:
                passed += 1
                print(f"[通过] {relative_path} | {detail}")
            else:
                failed += 1
                print(f"[不通过] {relative_path} | {detail} | {'；'.join(errors)}")

            writer.writerow([str(path), "视频" if path.suffix.lower() in VIDEO_EXTENSIONS else "图片",
                             "通过" if ok else "不通过", detail, "；".join(errors)])
            report.flush()

    print("\n检查完成")
    print(f"视频：{video_count}，图片：{image_count}，通过：{passed}，不通过：{failed}")
    print("说明：成品文件是否曾被放大或插帧，无法仅凭媒体元数据可靠判定。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
