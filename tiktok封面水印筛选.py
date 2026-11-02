# -*- coding: utf-8 -*-
"""TikTok 视频封面水印识别与筛选（轻量版，仅 OpenCV 启发式，无模型依赖）。"""

import argparse
import csv
import io
import json
import random
import re
import shutil
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

# Optional OCR (RapidOCR) for accurate text detection
try:
    from rapidocr_onnxruntime import RapidOCR
    _OCR_AVAILABLE = True
except Exception:
    RapidOCR = None
    _OCR_AVAILABLE = False

_OCR_INSTANCE = {"ocr": None}


def get_ocr():
    global _OCR_AVAILABLE
    if not _OCR_AVAILABLE:
        return None
    if _OCR_INSTANCE["ocr"] is not None:
        return _OCR_INSTANCE["ocr"]
    try:
        _OCR_INSTANCE["ocr"] = RapidOCR()
        return _OCR_INSTANCE["ocr"]
    except Exception:
        _OCR_AVAILABLE = False
        return None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COVER_COL_PRIORITY = [
    "cover_url", "cover", "封面url", "封面URL", "封面链接", "封面", "thumbnail",
    "thumb", "image_url", "image", "url", "链接",
]


def find_cover_column(headers):
    lower = [str(h).strip().lower() if h is not None else "" for h in headers]
    for name in COVER_COL_PRIORITY:
        if name.lower() in lower:
            return lower.index(name.lower())
    for i, h in enumerate(headers):
        s = str(h or "").lower()
        if any(k in s for k in ("cover", "thumb", "封面", "图片", "image")):
            return i
    return 0


def read_table(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
        if not rows:
            return [], None
        header = list(rows[0])
        col = find_cover_column(header)
        out = []
        for r in rows[1:]:
            if not r:
                continue
            val = r[col] if col < len(r) else None
            if val is not None and str(val).strip():
                out.append(str(val).strip())
        return out, col
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], None
    header = rows[0]
    col = find_cover_column(header)
    out = []
    for r in rows[1:]:
        if not r:
            continue
        val = r[col] if col < len(r) else None
        if val is not None and str(val).strip():
            out.append(str(val).strip())
    return out, col


def safe_name(s):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(s))


def load_image(path):
    try:
        img = Image.open(path)
        img.load()
        if img.mode not in ("RGB", "BGR"):
            img = img.convert("RGB")
        arr = np.array(img)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    except Exception:
        return cv2.imread(str(path))


def region_score(roi):
    if roi is None or roi.size == 0:
        return 0.0, {}
    h, w = roi.shape[:2]
    if h < 8 or w < 8:
        return 0.0, {}
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # semi-transparent white (watermark blends, not pure-white highlights)
    white_mask = (hsv[:, :, 1] < 60) & (hsv[:, :, 2] >= 180) & (hsv[:, :, 2] <= 240)
    white_ratio = float(white_mask.mean())
    edges = cv2.Canny(gray, 60, 150)
    edge_density = float(edges.mean()) / 255.0
    # text-like strokes: small connected components in semi-transparent white mask
    wm_mask = white_mask.astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)
    wm_mask = cv2.morphologyEx(wm_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(wm_mask, 8)
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 50 or area > 1000:
            continue
        if min(ww, hh) > 25:
            continue
        if max(ww, hh) < 8:
            continue
        aspect = ww / max(hh, 1)
        if aspect < 0.1 or aspect > 15:
            continue
        comps.append((cents[i][0], cents[i][1]))
    # cluster into horizontal text bands (tolerance 10px)
    bands = []
    for c in sorted(comps, key=lambda x: x[1]):
        placed = False
        for b in bands:
            if abs(b["y"] - c[1]) <= 15:
                b["items"].append(c)
                b["y"] = (b["y"] * (len(b["items"]) - 1) + c[1]) / len(b["items"])
                placed = True
                break
        if not placed:
            bands.append({"y": c[1], "items": [c]})
    text_lines = sum(1 for b in bands if len(b["items"]) >= 4)
    text_components = sum(len(b["items"]) for b in bands if len(b["items"]) >= 4)
    # drop shadow: gray pixels near white components
    gray_mask = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] >= 80) & (hsv[:, :, 2] <= 170)
    shadow_ratio = float(gray_mask.mean())
    gray_std = float(gray.std())
    score = 0.0
    if text_lines >= 1 and shadow_ratio > 0.01:
        score += 4.0
    if text_lines >= 2 and shadow_ratio > 0.01:
        score += 2.0
    if text_components >= 12 and shadow_ratio > 0.01:
        score += 2.0
    if white_ratio > 0.012:
        score += 1.0
    if shadow_ratio > 0.005:
        score += 1.5
    if edge_density > 0.05:
        score += 1.0
    if gray_std < 18 and text_lines == 0:
        score *= 0.5
    if text_lines == 0 or shadow_ratio <= 0.01:
        score = min(score, 2.0)
    details = {
        "white_ratio": round(white_ratio, 4),
        "edge_density": round(edge_density, 4),
        "text_lines": text_lines,
        "text_components": text_components,
        "shadow_ratio": round(shadow_ratio, 4),
        "gray_std": round(gray_std, 2),
    }
    return score, details


def detect_floating_overlay(bgr):
    """检测悬浮贴纸/合成图片浮层：位于画面顶部、漂浮在"大片平滑渐变背景"上的孤立小物体。
    只扫描顶部约22%区域，要求该区域背景高度平滑、且存在不连向下方主体的孤立小物体。"""
    if bgr is None or bgr.size == 0:
        return False
    try:
        h, w = bgr.shape[:2]
        band = bgr[0:max(1, int(h * 0.22)), :]
        bh, bw = band.shape[:2]
        scale = 640.0 / max(bh, bw)
        if scale < 1.0:
            band = cv2.resize(band, (max(1, int(bw * scale)), max(1, int(bh * scale))))
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)
        H, W = gray.shape
        if H < 20 or W < 20:
            return False
        m = cv2.blur(gray, (9, 9))
        m2 = cv2.blur(gray * gray, (9, 9))
        std = np.sqrt(np.maximum(m2 - m * m, 0))
        # 顶部背景需高度平滑（渐变/纯色/虚化）
        if float((std < 8.0).mean()) < 0.60:
            return False
        smooth = std < 8.0
        obj = (std > 20.0).astype(np.uint8)
        obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        num, lab, st, ce = cv2.connectedComponentsWithStats(obj, 8)
        total = H * W
        for i in range(1, num):
            area = st[i, cv2.CC_STAT_AREA]
            if area < total * 0.004 or area > total * 0.30:
                continue
            y = st[i, cv2.CC_STAT_TOP]
            hh = st[i, cv2.CC_STAT_HEIGHT]
            ww = st[i, cv2.CC_STAT_WIDTH]
            # 孤立小物体：不能连到该带底部（否则是主体人物的头/身延伸下来）
            if y + hh > H * 0.90:
                continue
            comp = (lab == i).astype(np.uint8)
            dil = cv2.dilate(comp, np.ones((11, 11), np.uint8))
            ring = (dil == 1) & (comp == 0)
            if int(ring.sum()) < 30:
                continue
            if float(smooth[ring].mean()) > 0.55:
                return True
        return False
    except Exception:
        return False


def detect_text_overlay(bgr):
    """检测图片中是否存在字幕/文字浮层/贴纸（任意位置），返回 True/False。优先使用 RapidOCR。"""
    if bgr is None or bgr.size == 0:
        return False
    # 1) 烧录字幕/水印文字：白色字体 + 明显深色描边，任意位置均可识别
    ocr = get_ocr()
    if ocr is not None:
        try:
            res, _ = ocr(bgr)
            if res:
                h, w = bgr.shape[:2]
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                V = hsv[:, :, 2]
                S = hsv[:, :, 1]
                for item in res:
                    box = item[0]
                    conf = float(item[2]) if len(item) > 2 else 1.0
                    if conf < 0.6:
                        continue
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    x1, x2 = int(min(xs)), int(max(xs))
                    y1, y2 = int(min(ys)), int(max(ys))
                    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        continue
                    bh = (y2 - y1) / h
                    if bh < 0.02 or bh > 0.30:
                        continue
                    roi_v = V[y1:y2, x1:x2]
                    roi_s = S[y1:y2, x1:x2]
                    if roi_v.size == 0:
                        continue
                    white = float(((roi_v > 195) & (roi_s < 60)).mean())
                    dark = float((roi_v < 75).mean())
                    # 烧录字幕特征：白色字 + 深色描边同时存在（包装/品牌彩色字无此特征）
                    if white > 0.06 and dark > 0.05:
                        return True
        except Exception:
            pass
    # 2) 悬浮贴纸/图片浮层：经典图像处理误报过高，默认关闭，待接入视觉大模型
    # if detect_floating_overlay(bgr):
    #     return True
    return False
    # fallback heuristic
    try:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = bgr.shape[:2]
        wm = (hsv[:, :, 1] < 70) & (hsv[:, :, 2] >= 170) & (hsv[:, :, 2] <= 245)
        wm = wm.astype(np.uint8) * 255
        kernel = np.ones((2, 2), np.uint8)
        wm = cv2.morphologyEx(wm, cv2.MORPH_OPEN, kernel, iterations=1)
        num, labels, stats, cents = cv2.connectedComponentsWithStats(wm, 8)
        comps = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            ww = int(stats[i, cv2.CC_STAT_WIDTH])
            hh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if area < 20 or area > 800:
                continue
            if min(ww, hh) > 15:
                continue
            if max(ww, hh) < 6:
                continue
            aspect = ww / max(hh, 1)
            if aspect < 0.08 or aspect > 20:
                continue
            comps.append((cents[i][0], cents[i][1], ww, hh))
        bands = []
        for c in sorted(comps, key=lambda x: x[1]):
            placed = False
            for b in bands:
                if abs(b["y"] - c[1]) <= 12:
                    b["items"].append(c)
                    b["y"] = (b["y"] * (len(b["items"]) - 1) + c[1]) / len(b["items"])
                    placed = True
                    break
            if not placed:
                bands.append({"y": c[1], "items": [c]})
        gray_mask = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] >= 80) & (hsv[:, :, 2] <= 170)
        shadow_ratio = float(gray_mask.mean())
        for b in bands:
            items = b["items"]
            if len(items) < 8:
                continue
            heights = [it[3] for it in items]
            h_std = float(np.std(heights)) if len(heights) > 1 else 0.0
            xs = [it[0] for it in items]
            x_span = (max(xs) - min(xs)) / max(w, 1)
            if h_std < 5 and x_span > 0.2 and shadow_ratio > 0.015:
                return True
        return False
    except Exception:
        return False


def detect_watermark(bgr, wm_threshold=5.0, suspect_threshold=3.0):
    # text overlay / subtitle detection (anywhere in image)
    has_overlay = detect_text_overlay(bgr)
    h, w = bgr.shape[:2]
    regions = OrderedDict()
    regions["bottom_left"] = bgr[int(h * 0.70):h, 0:int(w * 0.50)]
    regions["bottom_right"] = bgr[int(h * 0.70):h, int(w * 0.50):w]
    regions["bottom_center"] = bgr[int(h * 0.78):h, int(w * 0.15):int(w * 0.85)]
    regions["top_right"] = bgr[0:int(h * 0.20), int(w * 0.65):w]
    regions["top_left"] = bgr[0:int(h * 0.20), 0:int(w * 0.35)]
    best = (0.0, None, {})
    all_scores = {}
    for name, roi in regions.items():
        sc, det = region_score(roi)
        all_scores[name] = (sc, det)
        if sc > best[0]:
            best = (sc, name, det)
    decent = [n for n, (sc, _) in all_scores.items() if sc >= 3.0]
    score = best[0]
    if len(decent) >= 2:
        score += 1.0
    region = best[1]
    details = best[2]
    summary = {n: round(v[0], 2) for n, v in all_scores.items()}
    if _OCR_AVAILABLE:
        # 有 OCR 时，仅以 OCR 白字（字幕/水印）为准，不再使用不可靠的角落边缘密度启发式
        status = "watermarked" if has_overlay else "clean"
    else:
        if has_overlay:
            status = "watermarked"
        elif score >= wm_threshold:
            status = "watermarked"
        elif score >= suspect_threshold:
            status = "suspect"
        else:
            status = "clean"
    return {
        "status": status,
        "score": round(float(score if not has_overlay else max(score, wm_threshold)), 2),
        "region": region,
        "details": details,
        "region_scores": summary,
        "has_overlay": has_overlay,
    }


def download_image(url, dest, proxy=None):
    last_err = None
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Referer": "https://www.tiktok.com/",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    proxies = {"http": proxy, "https": proxy} if proxy else None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30, headers=headers, proxies=proxies)
            if r.status_code == 429 or r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
                time.sleep(1 + attempt)
                continue
            r.raise_for_status()
            data = r.content
            if not data or len(data) < 100:
                return False, None, "empty response"
            try:
                img = Image.open(io.BytesIO(data))
                img.verify()
            except Exception:
                return False, None, "not a valid image"
            try:
                img2 = Image.open(io.BytesIO(data))
                fmt = (img2.format or "JPEG").lower()
            except Exception:
                fmt = "jpeg"
            ext = "jpg" if fmt in ("jpeg", "jpg") else fmt
            dest = dest.with_suffix("." + ext)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return True, dest, None
        except Exception as e:
            last_err = str(e)
            time.sleep(1 + attempt)
    return False, None, last_err


def collect_images(input_path, work_dir, proxy, concurrency):
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        return [(p, str(p))]
    if p.is_dir():
        out = []
        for ext in IMG_EXTS:
            for f in p.rglob("*" + ext):
                out.append((f, str(f)))
        for ext in [e.upper() for e in IMG_EXTS]:
            for f in p.rglob("*" + ext):
                out.append((f, str(f)))
        seen = set()
        uniq = []
        for lp, src in out:
            if str(lp) in seen:
                continue
            seen.add(str(lp))
            uniq.append((lp, src))
        return uniq
    if p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".xls"):
        urls, _ = read_table(p)
        dl_dir = work_dir / "_downloads"
        dl_dir.mkdir(parents=True, exist_ok=True)
        results = []
        tasks = []
        for i, url in enumerate(urls):
            dest = dl_dir / f"cover_{i:05d}"
            tasks.append((i, url, dest))
        dl_map = {}
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            fut_map = {}
            for i, url, dest in tasks:
                fut = pool.submit(download_image, url, dest, proxy)
                fut_map[fut] = (i, url, dest)
            for fut in as_completed(fut_map):
                i, url, dest = fut_map[fut]
                ok, path, err = fut.result()
                if ok and path is not None:
                    dl_map[i] = (path, url)
                else:
                    dl_map[i] = (None, url + f"  [下载失败: {err}]")
        for i, url, dest in tasks:
            entry = dl_map.get(i)
            if entry is None:
                continue
            path, src = entry
            if path is not None:
                results.append((path, src))
        return results
    raise SystemExit(f"不支持的输入：{input_path}")


def main():
    parser = argparse.ArgumentParser(description="TikTok 视频封面水印识别与筛选（轻量版）")
    parser.add_argument("input", help="图片文件 / 目录 / CSV / XLSX")
    parser.add_argument("--output", default=None, help="输出根目录")
    parser.add_argument("--threshold", type=float, default=5.0, help="水印判定分数阈值（默认 5.0）")
    parser.add_argument("--suspect", type=float, default=3.0, help="疑似水印分数阈值（默认 3.0）")
    parser.add_argument("--proxy", default="http://127.0.0.1:7890", help="下载代理")
    parser.add_argument("--no-proxy", action="store_true", help="下载时不使用代理")
    parser.add_argument("--concurrency", type=int, default=4, help="下载并发数")
    parser.add_argument("--move", action="store_true", help="移动而非复制")
    parser.add_argument("--dry-run", action="store_true", help="仅报告，不移动/复制文件")
    parser.add_argument("--report", default=None, help="报告 CSV 路径")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"输入不存在：{inp}")

    if args.output:
        out_root = Path(args.output)
    else:
        base = inp.parent if inp.is_file() else inp
        out_root = base / "cover_filter_output"
    out_root.mkdir(parents=True, exist_ok=True)

    proxy = None if args.no_proxy else args.proxy
    work_dir = out_root
    items = collect_images(inp, work_dir, proxy, args.concurrency)
    if not items:
        print("未找到任何图片。")
        return

    print(f"共 {len(items)} 张封面待检测")

    for sub in ("clean", "suspect", "watermarked"):
        (out_root / sub).mkdir(exist_ok=True)

    do_move = args.move and not args.dry_run

    rows = []
    counts = {"clean": 0, "suspect": 0, "watermarked": 0}
    for idx, (local_path, src) in enumerate(items, 1):
        bgr = load_image(local_path)
        if bgr is None:
            print(f"[{idx}/{len(items)}] 读取失败：{local_path}")
            rows.append({
                "file": str(local_path), "source": src, "status": "error",
                "score": "", "region": "", "width": "", "height": "",
                "details": "读取失败", "region_scores": "", "new_path": "",
            })
            continue
        h, w = bgr.shape[:2]
        res = detect_watermark(bgr, wm_threshold=args.threshold, suspect_threshold=args.suspect)
        status = res["status"]
        counts[status] = counts.get(status, 0) + 1
        new_path = ""
        if not args.dry_run:
            dst_dir = out_root / status
            base_name = safe_name(Path(local_path).stem)
            ext = Path(local_path).suffix or ".jpg"
            candidate = dst_dir / f"{base_name}{ext}"
            i = 1
            while candidate.exists() and str(candidate.resolve()) != str(Path(local_path).resolve()):
                candidate = dst_dir / f"{base_name}_{i}{ext}"
                i += 1
            try:
                if do_move:
                    Path(local_path).replace(candidate)
                else:
                    shutil.copy2(local_path, candidate)
                new_path = str(candidate)
            except Exception as e:
                new_path = f"写入失败: {e}"
        rows.append({
            "file": str(local_path),
            "source": src,
            "status": status,
            "score": res["score"],
            "region": res["region"] or "",
            "width": w,
            "height": h,
            "details": json.dumps(res["details"], ensure_ascii=False),
            "region_scores": json.dumps(res["region_scores"], ensure_ascii=False),
            "new_path": new_path,
        })
        print(f"[{idx}/{len(items)}] {Path(local_path).name}: {status} (score={res['score']}, region={res['region']})")

    report_path = Path(args.report) if args.report else (out_root / "report.csv")
    fieldnames = ["file", "source", "status", "score", "region", "width", "height",
                  "details", "region_scores", "new_path"]
    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("\n===== 完成 =====")
    print(f"总计: {len(items)}")
    print(f"  clean (无水印):       {counts.get('clean', 0)}")
    print(f"  suspect (疑似水印):   {counts.get('suspect', 0)}")
    print(f"  watermarked (水印):   {counts.get('watermarked', 0)}")
    print(f"报告: {report_path}")
    print(f"输出目录: {out_root}")


if __name__ == "__main__":
    main()
