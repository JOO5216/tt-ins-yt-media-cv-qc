# -*- coding: utf-8 -*-
"""API 优先抓取 TikTok 某个音乐下的视频链接和博主账号。"""

import csv
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit


TIKWM = "https://www.tikwm.com/api"
PROXY = "http://127.0.0.1:7890"  # FlClash HTTP 代理；配置代理时不自动回退直连
SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = SCRIPT_DIR / ".tiktok_browser_profile"
TARGET_CREATORS = 3000
VIDEO_RE = re.compile(r"^/@([^/]+)/video/(\d+)$", re.I)
MUSIC_ID_RE = re.compile(r"/music/[^/?#]*-(\d+)(?:[/?#]|$)", re.I)


class ApiUnavailable(Exception):
    pass


def parse_video_url(href):
    try:
        parts = urlsplit(href)
    except (TypeError, ValueError):
        return None
    if parts.netloc.lower() not in {"www.tiktok.com", "tiktok.com", "m.tiktok.com"}:
        return None
    match = VIDEO_RE.match(parts.path.rstrip("/"))
    if not match:
        return None
    handle, video_id = match.groups()
    return f"https://www.tiktok.com/@{handle}/video/{video_id}", f"@{handle}"


def unique_creators(rows, limit=TARGET_CREATORS):
    """每个博主只保留首次出现的一条视频。"""
    result = {}
    seen_handles = set()
    for video_url, handle in rows.items():
        key = handle.lower()
        if key in seen_handles:
            continue
        seen_handles.add(key)
        result[video_url] = handle
        if len(result) >= limit:
            break
    return result


_session_local = None

def _get_session():
    global _session_local
    try:
        import threading
    except Exception:
        threading = None
    # thread-local storage
    if not hasattr(_get_session, "_local"):
        _get_session._local = threading.local() if threading else None
    local = _get_session._local
    sess = getattr(local, "session", None) if local is not None else _session_local
    if sess is None:
        from curl_cffi import requests
        sess = requests.Session(trust_env=False)
        if local is not None:
            local.session = sess
        else:
            _session_local = sess
    return sess


def api_get(url, params):
    try:
        from curl_cffi import requests  # noqa: F401
    except ImportError:
        raise SystemExit("缺少 curl_cffi，请先运行：python -m pip install curl_cffi")

    proxy = {"http": PROXY, "https": PROXY} if PROXY else None
    routes = [proxy]
    last_error = None
    session = _get_session()
    for proxies in routes:
        attempts = 1 if proxies else 3
        for attempt in range(attempts):
            try:
                response = session.get(
                    url,
                    params=params,
                    impersonate="chrome131",
                    timeout=20,
                    proxy=PROXY or '',
                    headers={
                        "Accept": "application/json, text/plain, */*",
                        "Referer": "https://www.tikwm.com/",
                        "Origin": "https://www.tikwm.com",
                    },
                )
                if response.status_code == 403 or "Just a moment" in response.text[:100]:
                    last_error = f"TikWM HTTP {response.status_code}，代理={PROXY or '直连'}，server={response.headers.get('server')}"
                    break
                response.raise_for_status()
                return response.json()
            except Exception as error:
                last_error = error
                if attempt + 1 < attempts:
                    time.sleep(0.5 + attempt * 0.3 + random.uniform(0.05, 0.2))
    raise ApiUnavailable(f"TikWM 请求失败：{last_error}")


def resolve_music_id(source_url):
    match = MUSIC_ID_RE.search(source_url)
    if match:
        return match.group(1)

    payload = api_get(TIKWM, {"url": source_url, "hd": 1})
    if payload.get("code") != 0:
        message = payload.get("msg") or payload.get("message") or payload
        raise ApiUnavailable(f"TikWM 无法解析原视频：{message}")
    music_id = str(((payload.get("data") or {}).get("music_info") or {}).get("id") or "")
    if not music_id:
        raise ApiUnavailable("原视频响应中没有 music_id")
    return music_id


def fetch_all_by_api(music_id):
    rows = {}
    cursor = 0
    seen_cursors = set()
    page_no = 0

    while cursor not in seen_cursors:
        seen_cursors.add(cursor)
        payload = api_get(
            f"{TIKWM}/music/posts",
            {"music_id": music_id, "count": 30, "cursor": cursor},
        )
        if payload.get("code") != 0:
            message = payload.get("msg") or payload.get("message") or payload
            raise ApiUnavailable(f"音乐接口返回失败：{message}")

        data = payload.get("data") or {}
        videos = data.get("videos") or []
        page_no += 1
        before = len(rows)
        for video in videos:
            author = video.get("author") or {}
            handle = str(author.get("unique_id") or "").lstrip("@").strip()
            video_id = str(video.get("video_id") or video.get("id") or "").strip()
            if handle and video_id:
                rows[f"https://www.tiktok.com/@{handle}/video/{video_id}"] = f"@{handle}"
        creator_count = len({handle.lower() for handle in rows.values()})
        print(
            f"API 第 {page_no} 页：新增视频 {len(rows) - before}，"
            f"不同博主 {creator_count}/{TARGET_CREATORS}"
        )

        if creator_count >= TARGET_CREATORS:
            break

        if not data.get("hasMore") or not videos:
            break
        next_cursor = data.get("cursor")
        if next_cursor is None:
            break
        cursor = next_cursor
        time.sleep(random.uniform(0.4, 0.8))

    if not rows:
        raise ApiUnavailable("音乐接口没有返回视频")
    return rows


def find_music_url(page):
    if "/music/" in page.url:
        return page.url
    links = page.locator('a[href*="/music/"]')
    for index in range(links.count()):
        href = links.nth(index).get_attribute("href")
        if href and "/music/" in href:
            return href if href.startswith("http") else "https://www.tiktok.com" + href
    return None


def fetch_by_browser(source_url):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "API 不可用，浏览器兜底需要 Playwright。请运行：python -m pip install playwright"
        )

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(PROFILE_DIR), channel="chrome", headless=False, viewport=None
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(source_url, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(5000)

        music_url = find_music_url(page)
        if not music_url:
            input("请完成登录/验证并手动进入音乐页面，然后按回车：")
            music_url = find_music_url(page)
        if not music_url:
            context.close()
            raise SystemExit("没有识别到音乐页面。")
        if "/music/" not in page.url:
            page.goto(music_url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(5000)

        rows = {}
        no_new_rounds = 0
        try:
            for round_no in range(1, 501):
                hrefs = page.locator('a[href*="/video/"]').evaluate_all("els => els.map(a => a.href)")
                before = len(rows)
                for href in hrefs:
                    parsed = parse_video_url(href)
                    if parsed:
                        rows[parsed[0]] = parsed[1]
                added = len(rows) - before
                no_new_rounds = no_new_rounds + 1 if added == 0 else 0
                creator_count = len({handle.lower() for handle in rows.values()})
                print(
                    f"浏览器第 {round_no} 次滚动：新增视频 {added}，"
                    f"不同博主 {creator_count}/{TARGET_CREATORS}"
                )
                if creator_count >= TARGET_CREATORS:
                    break
                if no_new_rounds >= 10:
                    break
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1800)
        except KeyboardInterrupt:
            creator_count = len({handle.lower() for handle in rows.values()})
            print(f"\n已暂停，保留当前 {creator_count} 个不同博主并导出。")

        final_music_url = page.url
        try:
            context.close()
        except Exception:
            pass
        return rows, final_music_url


def export_csv(rows, source_url, mode):
    output = SCRIPT_DIR / f"TikTok拍同款_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["序号", "视频链接", "博主账号", "博主主页", "来源链接", "抓取方式"])
        for index, (video_url, handle) in enumerate(rows.items(), 1):
            writer.writerow([
                index,
                video_url,
                handle,
                f"https://www.tiktok.com/{handle}",
                source_url,
                mode,
            ])
    return output


def main():
    source_url = sys.argv[1] if len(sys.argv) > 1 else input("粘贴 TikTok 原视频或音乐页链接：").strip()
    if not source_url.startswith(("https://", "http://")):
        raise SystemExit("链接必须以 http:// 或 https:// 开头")

    try:
        music_id = resolve_music_id(source_url)
        print(f"music_id：{music_id}，开始使用 API 分页抓取")
        rows = fetch_all_by_api(music_id)
        mode = "TikWM API"
        final_source = source_url
    except ApiUnavailable as error:
        print(f"API 不可用：{error}\n切换到浏览器兜底……")
        rows, final_source = fetch_by_browser(source_url)
        mode = "Playwright 兜底"

    rows = unique_creators(rows)
    output = export_csv(rows, final_source, mode)
    print(f"完成：{len(rows)} 个不同博主，抓取方式：{mode}\n已保存到：{output}")


def self_test():
    assert parse_video_url("https://www.tiktok.com/@alice/video/123?x=1") == (
        "https://www.tiktok.com/@alice/video/123", "@alice"
    )
    assert parse_video_url("https://example.com/@alice/video/123") is None
    assert resolve_music_id("https://www.tiktok.com/music/Test-749123456789") == "749123456789"
    assert len(unique_creators({"v1": "@alice", "v2": "@Alice", "v3": "@bob"})) == 2
    print("解析测试通过")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
