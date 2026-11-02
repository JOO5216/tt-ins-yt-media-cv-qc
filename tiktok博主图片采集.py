# -*- coding: utf-8 -*-
"""批量采集 TikTok 博主公开图文帖中的全部图片，并进行自动初筛。"""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_PATH = SCRIPT_DIR / "tiktok拍同款抓取.py"
OUTPUT_ROOT = SCRIPT_DIR / "output"
BROWSER_PROFILE = SCRIPT_DIR / ".tiktok_image_profile"
CDP_URL = "http://127.0.0.1:9222"
FACE_MIN = 128
BLUR_THRESHOLD = 35.0
FACE_BLUR_THRESHOLD = 18.0
MIN_IMAGE_SIDE = 256


def load_core():
    if not CORE_PATH.exists():
        raise SystemExit(f"缺少依赖脚本：{CORE_PATH}")
    spec = importlib.util.spec_from_file_location("tiktok_core", CORE_PATH)
    module = importlib.util.spec_from_file_location("tiktok_core", CORE_PATH)
    mod = importlib.util.module_from_spec(module)
    module.loader.exec_module(mod)
    return mod


core = load_core()
api_get = core.api_get
TIKWM = core.TIKWM
ApiUnavailable = core.ApiUnavailable

# ---------- InsightFace (可选) ----------
_face_app = None
_face_init_tried = False


def get_face_app():
    global _face_app, _face_init_tried
    if _face_init_tried:
        return _face_app
    _face_init_tried = True
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
    except Exception as e:
        print(f"[WARN] InsightFace 不可用：{e}")
        _face_app = None
    return _face_app


# ---------- 工具 ----------

def parse_username(url):
    url = (url or "").strip()
    if not url:
        return None
    try:
        parts = urlparse(url)
    except Exception:
        return None
    path = (parts.path or "").strip("/")
    # /@username 或 /@username/...
    m = re.search(r"@([^/?#]+)", path)
    if m:
        return m.group(1)
    # 也支持直接传 username
    if re.fullmatch(r"[A-Za-z0-9_.]+", url):
        return url
    return None


def find_url_column(headers):
    priority = ["url", "主页链接", "profile_url", "profile", "link", "链接"]
    lower = [str(h).strip().lower() if h is not None else "" for h in headers]
    for name in priority:
        if name.lower() in lower:
            return lower.index(name.lower())
    # fallback: first column containing http
    for i, h in enumerate(headers):
        if h is not None and "http" in str(h).lower():
            return i
    return 0


def read_input(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
        if not rows:
            return []
        header = list(rows[0])
        col = find_url_column(header)
        out = []
        for r in rows[1:]:
            if not r:
                continue
            val = r[col] if col < len(r) else None
            if val is not None:
                out.append(str(val).strip())
        return out
    # csv
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return []
    header = rows[0]
    col = find_url_column(header)
    out = []
    for r in rows[1:]:
        if not r:
            continue
        val = r[col] if col < len(r) else None
        if val is not None and str(val).strip():
            out.append(str(val).strip())
    return out


def safe_name(s):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(s))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def laplacian_variance(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def fetch_user_info(username):
    try:
        data = api_get(TIKWM + "/user/info", {"unique_id": username})["data"]
        return data or {}
    except Exception:
        return {}


def run_login():
    """打开有头浏览器，让用户手动登录/通过滑块验证，会话保存在持久化 profile。"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("需要 Playwright：python -m pip install playwright")
    print("将打开 Chrome，请在浏览器中完成 TikTok 登录/滑块验证。完成后回到此窗口按回车。")
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            str(BROWSER_PROFILE), channel="chrome", headless=False,
            proxy={"server": core.PROXY} if core.PROXY else None, args=["--no-sandbox"],
            viewport={"width": 1280, "height": 900})
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://www.tiktok.com/login", wait_until="domcontentloaded", timeout=60000)
        input("完成登录/验证后按回车保存会话……")
        ctx.close()
    print(f"会话已保存到 {BROWSER_PROFILE}")


def fetch_photo_urls_browser(username, scrolls=6, cdp_url=None):
    """用浏览器打开博主主页，拦截 /api/post/item_list 响应，提取图文帖图片。
    返回 [{post_id, image_urls:[...]}]。失败返回 None。"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [WARN] 未安装 Playwright，无法使用浏览器模式")
        return None

    photo_posts = OrderedDict()  # post_id -> list[url]
    seen_posts = set()
    target = username.lower()

    def harvest(items):
        for it in items or []:
            if len(covers) >= max_covers:
                return
            au = (it.get("author") or {})
            handle = str(au.get("uniqueId") or au.get("unique_id") or "").lower()
            # 没有作者信息时不收，避免把推荐流归到目标账号。
            if handle != target:
                continue
            pid = str(it.get("id") or it.get("video_id") or "")
            if pid:
                seen_posts.add(pid)
            ip = it.get("imagePost") or {}
            imgs = ip.get("images") or it.get("images") or []
            if not imgs:
                continue
            pid = str(it.get("id") or it.get("video_id") or "")
            urls = []
            for im in imgs:
                if isinstance(im, str):
                    urls.append(im)
                    continue
                if not isinstance(im, dict):
                    continue
                u = im.get("imageURL") or im.get("imageUrl") or im.get("image_url")
                cand = None
                if isinstance(u, dict):
                    lst = u.get("urlList") or u.get("url_list") or []
                    if lst:
                        cand = max(lst, key=lambda x: len(str(x)))
                elif isinstance(u, str):
                    cand = u
                if not cand:
                    cand = im.get("url")
                if cand:
                    urls.append(cand)
            if pid and urls:
                photo_posts.setdefault(pid, [])
                for u in urls:
                    if u not in photo_posts[pid]:
                        photo_posts[pid].append(u)

    try:
        with sync_playwright() as pw:
            owns_browser = not cdp_url
            if cdp_url:
                browser = pw.chromium.connect_over_cdp(cdp_url)
                ctx = browser.contexts[0]
            else:
                browser = None
                ctx = pw.chromium.launch_persistent_context(
                    str(BROWSER_PROFILE), channel="chrome", headless=True,
                    proxy={"server": core.PROXY} if core.PROXY else None,
                    args=["--no-sandbox"])
            page = ctx.new_page()
            page_owned = True

            def on_resp(resp):
                if "item_list" not in resp.url or "/api/post/item_list" not in resp.url:
                    return
                try:
                    body = resp.text()
                    if not body.strip().startswith("{"):
                        return
                    d = json.loads(body)
                    harvest(d.get("itemList"))
                except Exception:
                    pass

            page.on("response", on_resp)
            page.goto("https://www.tiktok.com/@" + username,
                      wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(6000)
            for _ in range(scrolls):
                if len(covers) >= max_covers:
                    break
                page.mouse.wheel(0, 2500)
                page.wait_for_timeout(2000)
            try:
                page.close()
            except Exception:
                pass
            if not cdp_url:
                ctx.close()
    except Exception as e:
        print(f"  [browser] 出错：{e}")
        return None

    return [{"post_id": pid, "image_urls": urls} for pid, urls in photo_posts.items()], len(seen_posts)


def fetch_video_covers_browser(username, cdp_url=None, scrolls=3, max_covers=5):
    """用浏览器打开博主主页，拦截 /api/post/item_list，提取每个视频的封面图 URL。
    返回 ([{post_id, image_urls:[cover_url]}], seen_count)。"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [WARN] 未安装 Playwright")
        return None

    covers = OrderedDict()
    seen_posts = set()
    target = username.lower()

    def harvest(items):
        for it in items or []:
            au = (it.get("author") or {})
            handle = str(au.get("uniqueId") or au.get("unique_id") or "").lower()
            if handle and handle != target:
                continue
            pid = str(it.get("id") or it.get("video_id") or "")
            if pid:
                seen_posts.add(pid)
            # cover url
            v = it.get("video") or {}
            cover = (v.get("cover") or v.get("originCover") or v.get("dynamicCover")
                     or it.get("cover") or it.get("originCover"))
            if not cover:
                # try nested
                for k in ("cover", "originCover", "dynamicCover"):
                    c = it.get(k)
                    if isinstance(c, str) and c.startswith("http"):
                        cover = c
                        break
            if pid and isinstance(cover, str) and cover.startswith("http"):
                covers.setdefault(pid, [])
                if cover not in covers[pid]:
                    covers[pid].append(cover)

    try:
        with sync_playwright() as pw:
            owns_browser = not cdp_url
            if cdp_url:
                browser = pw.chromium.connect_over_cdp(cdp_url)
                ctx = browser.contexts[0]
            else:
                browser = None
                ctx = pw.chromium.launch_persistent_context(
                    str(BROWSER_PROFILE), channel="chrome", headless=True,
                    proxy={"server": core.PROXY} if core.PROXY else None,
                    args=["--no-sandbox"])
            page = ctx.new_page()

            def on_resp(resp):
                if "/api/post/item_list" not in resp.url:
                    return
                try:
                    body = resp.text()
                    if not body.strip().startswith("{"):
                        return
                    d = json.loads(body)
                    harvest(d.get("itemList"))
                except Exception:
                    pass

            page.on("response", on_resp)
            page.goto("https://www.tiktok.com/@" + username,
                      wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(6000)
            for _ in range(scrolls):
                page.mouse.wheel(0, 2500)
                page.wait_for_timeout(2000)
            try:
                page.close()
            except Exception:
                pass
            if not cdp_url:
                ctx.close()
    except Exception as e:
        print(f"  [browser] 出错：{e}")
        return None

    result_items = list(covers.items())[:max_covers]
    return [{"post_id": pid, "image_urls": urls} for pid, urls in result_items], len(seen_posts)


def fetch_user_posts(username, sec_uid=None):
    """获取用户帖子（TikWM API）。优先 unique_id，失败/空时尝试 sec_uid。"""
    def try_fetch(params):
        posts = []
        cursor = 0
        seen = set()
        cur = dict(params)
        ok = False
        while True:
            cur["cursor"] = cursor
            cur["count"] = 30
            try:
                payload = api_get(TIKWM + "/user/posts", cur)
            except ApiUnavailable as e:
                print(f"  [API] user/posts 失败：{e}")
                return None
            if payload.get("code") != 0:
                print(f"  [API] user/posts 返回：{payload.get('msg')}")
                return None
            data = payload.get("data") or {}
            items = data.get("videos") or data.get("posts") or data.get("items") or data.get("itemList") or []
            ok = True
            for it in items:
                posts.append(it)
            next_cursor = data.get("cursor")
            has_more = data.get("hasMore")
            if has_more in (False, 0, "0", "false") or next_cursor is None or str(next_cursor) in seen or not items:
                break
            seen.add(str(cursor))
            cursor = next_cursor
            time.sleep(0.6)
        return posts if ok else None

    res = try_fetch({"unique_id": username})
    if res is None:
        return None
    if not res and sec_uid:
        res2 = try_fetch({"sec_uid": sec_uid})
        if res2:
            return res2
        if res2 is None:
            return None
    return res if res is not None else []


def extract_images(post):
    """从帖子中提取图片 URL 列表（仅图文帖）。返回 (is_photo, [url,...])。"""
    # detect photo post
    images = None
    for key in ("images", "imageList", "image_list", "photo", "photos", "imagePost", "image_post"):
        if isinstance(post.get(key), list) and post.get(key):
            images = post.get(key)
            break
    # sometimes nested
    if images is None:
        ip = post.get("imagePost") or post.get("image_post")
        if isinstance(ip, dict):
            for key in ("images", "imageList", "image_list"):
                if isinstance(ip.get(key), list) and ip.get(key):
                    images = ip.get(key)
                    break
    if not images:
        return False, []
    urls = []
    for img in images:
        if isinstance(img, str):
            urls.append(img)
            continue
        if not isinstance(img, dict):
            continue
        # pick highest res url
        # try common fields
        cand = None
        # nested urlList
        for k in ("urlList", "url_list", "urls"):
            v = img.get(k)
            if isinstance(v, list) and v:
                cand = v[0]
                # pick longest url (often highest res) - heuristic
                cand = max(v, key=lambda x: len(str(x)))
                break
        if cand is None:
            for k in ("imageURL", "imageUrl", "image_url", "url", "downloadAddr", "download_addr", "origin", "hd"):
                v = img.get(k)
                if isinstance(v, str) and v.startswith("http"):
                    cand = v
                    break
                if isinstance(v, dict):
                    for kk in ("urlList", "url_list", "url"):
                        vv = v.get(kk)
                        if isinstance(vv, list) and vv:
                            cand = max(vv, key=lambda x: len(str(x)))
                            break
                        if isinstance(vv, str) and vv.startswith("http"):
                            cand = vv
                            break
                    if cand:
                        break
        if cand:
            urls.append(str(cand))
    return True, urls


def post_id(post):
    for k in ("id", "aweme_id", "awemeId", "video_id", "postId"):
        v = post.get(k)
        if v is not None:
            return str(v)
    return None


def download_image(url, dest):
    """下载图片，返回 (success, path, error)."""
    last_err = None
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Referer": "https://www.tiktok.com/",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    proxies = {"http": core.PROXY, "https": core.PROXY} if core.PROXY else None
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=30, headers=headers, proxies=proxies, stream=True)
            if r.status_code == 429 or r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
                time.sleep(2 ** attempt + random.random())
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
            time.sleep(2 ** attempt + random.random())
    return False, None, last_err


def qc_image(img_path):
    """返回 dict of metrics and grade/reason."""
    result = {
        "image_width": None, "image_height": None,
        "face_count": 0, "face_bbox": None, "face_width": None,
        "face_height": None, "face_confidence": None,
        "blur_score": None, "watermark_status": "未检测",
        "logo_status": "未检测", "filter_status": "未检测",
        "quality_grade": "C", "review_reason": "",
    }
    try:
        img = Image.open(img_path)
        img.load()
        w, h = img.size
        result["image_width"] = w
        result["image_height"] = h
    except Exception as e:
        result["review_reason"] = f"图片损坏: {e}"
        result["quality_grade"] = "C"
        return result

    # sha handled outside
    # blur
    try:
        data = np.fromfile(str(img_path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        result["blur_score"] = round(laplacian_variance(gray), 2)
    except Exception:
        bgr = None
        gray = None

    # face detection
    app = get_face_app()
    faces = []
    if app is not None and bgr is not None:
        try:
            # resize for faster detection
            orig_h, orig_w = bgr.shape[:2]
            scale = 1.0
            max_side = 1024
            if max(orig_h, orig_w) > max_side:
                scale = max_side / max(orig_h, orig_w)
                small = cv2.resize(bgr, (int(orig_w * scale), int(orig_h * scale)))
            else:
                small = bgr
            small_faces = app.get(small)
            # scale back
            if scale != 1.0:
                for f in small_faces:
                    f.bbox = [v / scale for v in f.bbox]
                    if hasattr(f, "kps") and f.kps is not None:
                        f.kps = f.kps / scale
            faces = small_faces
        except Exception as e:
            print(f"  [FACE] 检测异常：{e}")
            faces = []
    result["face_count"] = len(faces)

    max_face = None
    if faces:
        max_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = [int(v) for v in max_face.bbox]
        result["face_bbox"] = [x1, y1, x2, y2]
        result["face_width"] = x2 - x1
        result["face_height"] = y2 - y1
        result["face_confidence"] = round(float(max_face.det_score), 4)
        # face blur
        try:
            crop = gray[y1:y2, x1:x2]
            if crop.size > 0:
                result["face_blur_score"] = round(laplacian_variance(crop), 2)
        except Exception:
            pass

    # watermark / subtitle detection (OCR-driven)
    watermark_level = "none"
    if bgr is not None:
        try:
            _mod = get_wm_module()
            if _mod is not None:
                _res = _mod.detect_watermark(bgr)
                if _res.get("status") == "watermarked":
                    watermark_level = "clear"
                elif _res.get("status") == "suspect":
                    watermark_level = "suspect"
                else:
                    watermark_level = "none"
            else:
                watermark_level = detect_watermark_heuristic(bgr)
        except Exception:
            watermark_level = detect_watermark_heuristic(bgr)
    if watermark_level == "clear":
        result["watermark_status"] = "检测到水印/Logo"
    elif watermark_level == "suspect":
        result["watermark_status"] = "疑似水印/Logo"
    else:
        result["watermark_status"] = "未检测到水印"

    # subtitle/overlay detection
    has_overlay = False
    if bgr is not None:
        try:
            _mod = get_wm_module()
            if _mod is not None:
                has_overlay = _mod.detect_text_overlay(bgr)
        except Exception:
            has_overlay = False

    # filter/beauty heuristic
    filter_suspect = False
    if bgr is not None:
        filter_suspect = detect_filter_heuristic(bgr, gray)

    if has_overlay:
        result["filter_status"] = "疑似字幕/浮层"
    elif filter_suspect:
        result["filter_status"] = "疑似滤镜/美颜"
    else:
        result["filter_status"] = "未检测到明显异常"

    # grading
    reasons = []
    fw = result["face_width"]; fh = result["face_height"]
    blur = result["blur_score"]
    if result["face_count"] == 0:
        result["quality_grade"] = "C"
        reasons.append("未检测到人脸")
    elif fw is None or fw < FACE_MIN or fh is None or fh < FACE_MIN:
        result["quality_grade"] = "C"
        reasons.append("人脸尺寸不足128px")
    elif blur is not None and blur < BLUR_THRESHOLD:
        result["quality_grade"] = "C"
        reasons.append("严重模糊")
    elif watermark_level == "clear":
        result["quality_grade"] = "C"
        reasons.append("检测到水印/Logo")
    elif has_overlay:
        result["quality_grade"] = "C"
        reasons.append("疑似字幕/浮层")
    else:
        # 到达这里说明人脸/清晰度/文字水印都已过关，再用本地视觉模型复核浮层/贴纸
        vlm_bad = False
        if bgr is not None and not has_overlay:
            try:
                import vlm_overlay
                vlm_bad, _ans = vlm_overlay.vlm_check(bgr)
            except Exception:
                vlm_bad = False
        if vlm_bad:
            result["quality_grade"] = "C"
            reasons.append("疑似贴纸/浮层(视觉模型)")
        else:
            # B conditions (仅保留较明确的软性问题)
            if watermark_level == "suspect":
                reasons.append("疑似水印/Logo")
            fblur = result.get("face_blur_score")
            if fblur is not None and fblur < FACE_BLUR_THRESHOLD:
                reasons.append("人脸模糊")
            if w < MIN_IMAGE_SIDE or h < MIN_IMAGE_SIDE:
                reasons.append("分辨率偏低")
            if reasons:
                result["quality_grade"] = "B"
            else:
                result["quality_grade"] = "A"
    result["review_reason"] = "；".join(reasons)
    return result


def detect_watermark_heuristic(bgr):
    """角落高对比边缘密度启发式，只输出疑似/明显，不替代人工复核。"""
    try:
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners = [
            gray[0:int(h*0.15), 0:int(w*0.25)],
            gray[0:int(h*0.15), int(w*0.75):w],
            gray[int(h*0.85):h, 0:int(w*0.25)],
            gray[int(h*0.85):h, int(w*0.75):w],
        ]
        max_density = 0.0
        for c in corners:
            if c.size == 0:
                continue
            edges = cv2.Canny(c, 80, 160)
            max_density = max(max_density, float(edges.mean()) / 255.0)
        if max_density >= 0.14:
            return "clear"
        if max_density >= 0.06:
            return "suspect"
        return "none"
    except Exception:
        return "none"


def detect_filter_heuristic(bgr, gray):
    """轻量滤镜/美颜启发式：过度磨皮(低高频方差)、饱和度过高。"""
    try:
        # saturation
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        if float(sat.mean()) > 170:
            return True
        # high-frequency variance (sharpness proxy) very low -> over-smoothed
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        if float(lap.var()) < 30:
            return True
        return False
    except Exception:
        return False


_wm_module_cache = {"mod": None}


def get_wm_module():
    if _wm_module_cache["mod"] is not None:
        return _wm_module_cache["mod"]
    try:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "tiktok_cover_wm", str(Path(__file__).resolve().parent / "tiktok封面水印筛选.py"))
        if _spec is None:
            return None
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _wm_module_cache["mod"] = _mod
        return _mod
    except Exception:
        return None


def load_checkpoint(out_dir):
    p = checkpoint_path(out_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_checkpoint(out_dir, state):
    p = checkpoint_path(out_dir)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def process_account(username, profile_url, out_root, ckpt, use_browser=False, cdp_url=None, concurrency=4, source_url="", use_covers=False, max_covers=5):
    """处理单个账号，返回 (stats dict, list of image records, list of errors)."""
    user_dir = out_root / safe_name(username)
    user_dir.mkdir(parents=True, exist_ok=True)
    for g in ("A_初筛通过", "B_待人工复核", "C_不合格"):
        (user_dir / g).mkdir(exist_ok=True)

    def _post_url(pid):
        if use_covers:
            return f"https://www.tiktok.com/@{username}/video/{pid}"
        if pid:
            return f"https://www.tiktok.com/@{username}/photo/{pid}"
        return f"https://www.tiktok.com/@{username}"

    errors = []
    records = []
    stats = {
        "username": username, "profile_url": profile_url, "source_url": source_url,
        "total_posts": 0, "image_posts": 0, "total_images": 0,
        "A_count": 0, "B_count": 0, "C_count": 0,
        "downloaded": 0, "status": "ok", "error": "",
    }

    # resume
    done_posts = set(ckpt.get(username, {}).get("done_posts", []))
    seen_hashes = set(ckpt.get(username, {}).get("seen_hashes", []))

    # 获取图文帖（浏览器拦截优先，失败回退 TikWM API）
    photo_posts = []  # (pid, [url,...])
    posts_count = 0
    source_used = "browser" if use_browser else "api"

    if use_browser:
        if use_covers:
            bres = fetch_video_covers_browser(username, cdp_url=cdp_url, max_covers=max_covers)
        else:
            bres = fetch_photo_urls_browser(username, cdp_url=cdp_url)
        if bres is not None:
            bposts, posts_count = bres
            for bp in bposts:
                photo_posts.append((bp["post_id"], bp["image_urls"]))
        else:
            stats["status"] = "failed"
            stats["error"] = "浏览器抓取失败"
            errors.append({"account": username, "type": "浏览器抓取失败", "detail": "Playwright 未取到数据"})
            return stats, records, errors
    else:
        info = fetch_user_info(username)
        sec_uid = info.get("secUid") or info.get("sec_uid")
        try:
            posts = fetch_user_posts(username, sec_uid)
        except Exception:
            posts = None
        if posts is None:
            stats["status"] = "failed"
            stats["error"] = "获取帖子失败: TikWM /user/posts 403或网络错误"
            errors.append({"account": username, "type": "API请求失败", "detail": "TikWM /user/posts 返回 403 (Cloudflare) 或网络错误"})
            return stats, records, errors
        posts_count = len(posts)
        for p in posts:
            is_photo, imgs = extract_images(p)
            if is_photo and imgs:
                photo_posts.append((post_id(p) or "unknown", imgs))

    stats["total_posts"] = posts_count
    stats["image_posts"] = len(photo_posts)
    if use_browser and posts_count == 0:
        stats["status"] = "empty"
        stats["error"] = "未获取到目标博主自己的作品列表；可能需要登录/验证"
        errors.append({"account": username, "type": "作品列表为空", "detail": "TikTok 返回空列表或触发验证，未将推荐流归属到目标账号"})

    # 构造下载任务
    tasks = []
    for pid, imgs in photo_posts:
        if pid in done_posts:
            continue
        for idx, url in enumerate(imgs, 1):
            tasks.append((pid, idx, url))
    stats["total_images"] = sum(len(imgs) for _, imgs in photo_posts)

    # 并发下载
    download_results = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as dl_pool:
        future_map = {}
        for pid, idx, url in tasks:
            dest_base = user_dir / "tmp" / f"{pid}_{idx}"
            dest_base.parent.mkdir(parents=True, exist_ok=True)
            fut = dl_pool.submit(download_image, url, dest_base)
            future_map[fut] = (pid, idx, url)
        for fut in as_completed(future_map):
            pid, idx, url = future_map[fut]
            ok, path, err = fut.result()
            download_results.append((pid, idx, url, ok, path, err))

    for pid, idx, url, ok, path, err in download_results:
        if not ok:
            errors.append({"account": username, "type": "下载失败", "detail": f"{pid}_{idx}: {err}"})
            records.append({
                "username": username, "profile_url": profile_url,
                "post_id": pid, "post_url": _post_url(pid),
                "image_index": idx, "image_url": url, "local_path": "",
                "image_width": None, "image_height": None,
                "face_count": None, "face_bbox": None, "face_width": None,
                "face_height": None, "face_confidence": None, "blur_score": None,
                "watermark_status": "", "logo_status": "", "filter_status": "",
                "quality_grade": "C", "review_reason": f"下载失败: {err}",
            })
            continue
        # 全部交给 qc_image 统一质检（人脸优先，再判断水印/字幕）
        stats["downloaded"] += 1
        try:
            h = sha256_file(path)
        except Exception:
            h = None
        if h and h in seen_hashes:
            try:
                path.unlink()
            except Exception:
                pass
            records.append({
                "username": username, "profile_url": profile_url,
                "post_id": pid, "post_url": _post_url(pid),
                "image_index": idx, "image_url": url, "local_path": str(path),
                "image_width": None, "image_height": None,
                "face_count": 0, "face_bbox": None, "face_width": None,
                "face_height": None, "face_confidence": None, "blur_score": None,
                "watermark_status": "重复文件", "logo_status": "", "filter_status": "",
                "quality_grade": "C", "review_reason": "SHA256重复",
            })
            continue
        if h:
            seen_hashes.add(h)

        qc = qc_image(path)
        grade = qc["quality_grade"]
        grade_dir = user_dir / {"A": "A_初筛通过", "B": "B_待人工复核", "C": "C_不合格"}[grade]
        new_path = grade_dir / path.name
        try:
            if path.exists():
                path.replace(new_path)
                path = new_path
        except Exception:
            pass

        if grade == "A":
            stats["A_count"] += 1
        elif grade == "B":
            stats["B_count"] += 1
        else:
            stats["C_count"] += 1

        records.append({
            "username": username, "profile_url": profile_url,
            "post_id": pid, "post_url": _post_url(pid),
            "image_index": idx, "image_url": url, "local_path": str(path),
            "image_width": qc["image_width"], "image_height": qc["image_height"],
            "face_count": qc["face_count"], "face_bbox": str(qc["face_bbox"]) if qc["face_bbox"] else None,
            "face_width": qc["face_width"], "face_height": qc["face_height"],
            "face_confidence": qc["face_confidence"], "blur_score": qc["blur_score"],
            "watermark_status": qc["watermark_status"],
            "logo_status": qc["logo_status"], "filter_status": qc["filter_status"],
            "quality_grade": grade, "review_reason": qc["review_reason"],
        })

        done_posts.add(pid)

    # checkpoint
    ckpt[username] = {"done_posts": sorted(done_posts), "seen_hashes": sorted(seen_hashes)}
    return stats, records, errors


def main():
    parser = argparse.ArgumentParser(description="TikTok 博主主页图片批量采集与自动质检")
    parser.add_argument("input", nargs="?", help="输入 CSV/XLSX 文件路径")
    parser.add_argument("--output", default=str(OUTPUT_ROOT), help="输出根目录")
    parser.add_argument("--concurrency", type=int, default=4, help="图片下载并发数")
    parser.add_argument("--limit", type=int, default=0, help="最多处理 N 个账号（0=全部）")
    parser.add_argument("--browser", action="store_true", help="用浏览器拦截方式抓取（推荐，绕过 API 403）")
    parser.add_argument("--cdp", default=None, nargs="?", const=CDP_URL, help="接管已登录的 Chrome（默认连 127.0.0.1:9222）")
    parser.add_argument("--login", action="store_true", help="打开有头浏览器手动登录/过验证码并保存会话")
    parser.add_argument("--covers", action="store_true", help="抓取视频封面图（而非图文帖），并筛除水印")
    parser.add_argument("--max-covers", type=int, default=5, help="每个博主最多抓取 N 个视频封面（默认 5）")
    args = parser.parse_args()

    if args.login:
        run_login()
        return

    if args.cdp:
        args.browser = True

    if not args.input:
        parser.error("必须提供输入 CSV/XLSX 文件，或使用 --login")
    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"输入文件不存在：{in_path}")

    urls = read_input(in_path)
    # dedup by username
    seen = set()
    accounts = []
    for u in urls:
        un = parse_username(u)
        if not un:
            continue
        if un.lower() in seen:
            continue
        seen.add(un.lower())
        # 统一 profile_url；原始链接作为 source_url 单独保留
        pu = f"https://www.tiktok.com/@{un}"
        accounts.append((un, pu, u))

    if args.limit and args.limit > 0:
        accounts = accounts[:args.limit]

    print(f"共 {len(accounts)} 个唯一账号待处理")

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_file = out_root / "checkpoints" / "checkpoint.json"
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {}
    if ckpt_file.exists():
        try:
            ckpt = json.loads(ckpt_file.read_text(encoding="utf-8"))
        except Exception:
            ckpt = {}

    all_stats = []
    all_records = []
    all_errors = []

    # 加载历史结果实现跨次运行增量合并
    hist_stats = {}
    hist_file = out_root / "checkpoints" / "stats.json"
    rec_file = out_root / "checkpoints" / "records.jsonl"
    if hist_file.exists():
        try:
            for s in json.loads(hist_file.read_text(encoding="utf-8")):
                hist_stats[s["username"]] = s
        except Exception:
            hist_stats = {}
    hist_records = {}
    if rec_file.exists():
        try:
            for line in rec_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                hist_records[(r.get("username"), str(r.get("post_id")), r.get("image_index"))] = r
        except Exception:
            hist_records = {}

    # invalid urls
    raw_urls = read_input(in_path)
    for u in raw_urls:
        if not parse_username(u):
            all_errors.append({"account": u, "type": "无效URL", "detail": "无法提取 username"})

    processed_users = set()
    for i, (username, profile_url, source_url) in enumerate(accounts, 1):
        print(f"\n[{i}/{len(accounts)}] 处理 @{username}")
        try:
            stats, records, errors = process_account(username, profile_url, out_root, ckpt, use_browser=args.browser, cdp_url=args.cdp, concurrency=args.concurrency, source_url=source_url, use_covers=args.covers, max_covers=args.max_covers)
        except Exception as e:
            stats = {"username": username, "profile_url": profile_url, "source_url": source_url,
                     "total_posts": 0, "image_posts": 0, "total_images": 0,
                     "A_count": 0, "B_count": 0, "C_count": 0,
                     "downloaded": 0, "status": "failed", "error": str(e)}
            records = []
            errors = [{"account": username, "type": "异常", "detail": str(e)}]
        hist_stats[username] = stats
        for r in records:
            hist_records[(r.get("username"), str(r.get("post_id")), r.get("image_index"))] = r
        # 依据合并后的明细重算该账号 A/B/C，避免断点续跑时统计归零
        ua = ub = uc = ud = 0
        for (ru, _pid, _idx), rec in hist_records.items():
            if ru != username:
                continue
            g = rec.get("quality_grade")
            if rec.get("local_path"):
                ud += 1
            if g == "A":
                ua += 1
            elif g == "B":
                ub += 1
            elif g == "C":
                uc += 1
        if ua or ub or uc:
            stats["A_count"], stats["B_count"], stats["C_count"] = ua, ub, uc
            stats["downloaded"] = ud
        processed_users.add(username)
        all_errors.extend(errors)
        # 持久化 checkpoint、聚合统计、增量明细
        try:
            ckpt_file.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
            hist_file.write_text(json.dumps(list(hist_stats.values()), ensure_ascii=False, indent=2), encoding="utf-8")
            with rec_file.open("w", encoding="utf-8") as rf:
                for rec in hist_records.values():
                    rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # 合并历史+本次生成 Excel
        merged_stats = list(hist_stats.values())
        merged_records = list(hist_records.values())
        if i % 5 == 0 or i == len(accounts):
            write_excel(out_root, merged_stats, merged_records, all_errors)
        print(f"  完成：posts={stats['total_posts']} img_posts={stats['image_posts']} "
              f"imgs={stats['total_images']} A={stats['A_count']} B={stats['B_count']} C={stats['C_count']} "
              f"dl={stats['downloaded']} status={stats['status']}")

    write_excel(out_root, list(hist_stats.values()), list(hist_records.values()), all_errors)
    # 导出筛选通过（A级）的博主主页 + 视频链接
    try:
        passed_path = out_root / "passed_videos.csv"
        with passed_path.open("w", newline="", encoding="utf-8-sig") as pf:
            pw = csv.writer(pf)
            pw.writerow(["username", "profile_url", "video_url", "local_path"])
            for rec in hist_records.values():
                if rec.get("quality_grade") == "A":
                    pw.writerow([rec.get("username"), rec.get("profile_url"),
                                 rec.get("post_url"), rec.get("local_path")])
        print(f"通过列表: {passed_path}")
    except Exception:
        pass
    print(f"\n全部完成。输出目录：{out_root}")


def write_excel(out_root, stats_list, records, errors):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "summary.xlsx"
    wb = Workbook()
    # Sheet 1: 博主统计
    ws = wb.active
    ws.title = "博主统计"
    headers1 = ["username", "profile_url", "source_url", "total_posts", "image_posts", "total_images",
                "A_count", "B_count", "C_count", "downloaded", "status", "error"]
    ws.append(headers1)
    for s in stats_list:
        ws.append([s.get(h) for h in headers1])
    # Sheet 2: 图片明细
    ws2 = wb.create_sheet("图片明细")
    headers2 = ["username", "profile_url", "post_id", "post_url", "image_index",
                "image_url", "local_path", "image_width", "image_height", "face_count",
                "face_bbox", "face_width", "face_height", "face_confidence", "blur_score",
                "watermark_status", "logo_status", "filter_status", "quality_grade", "review_reason"]
    ws2.append(headers2)
    for r in records:
        row = []
        for h in headers2:
            v = r.get(h)
            if isinstance(v, (list, tuple, dict)):
                v = json.dumps(v, ensure_ascii=False)
            row.append(v)
        ws2.append(row)
    # Sheet 3: 异常记录
    ws3 = wb.create_sheet("异常记录")
    headers3 = ["account", "type", "detail"]
    ws3.append(headers3)
    for e in errors:
        ws3.append([e.get("account"), e.get("type"), e.get("detail")])
    for sh in (ws, ws2, ws3):
        sh.freeze_panes = "A2"
    wb.save(path)


if __name__ == "__main__":
    main()
