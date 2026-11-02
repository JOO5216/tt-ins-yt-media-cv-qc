"""本地视觉模型 浮层/字幕 检测（SmolVLM-500M，CPU 优化版）。
用于识别经典图像处理难以判断的后处理痕迹：悬浮图片贴纸、彩色文字浮层等。
优化：多线程、输入缩放、限制生成 token，单轮判定。模型首次使用时加载并缓存。"""
import os
import threading
import numpy as np
import cv2
from PIL import Image

_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
_lock = threading.Lock()
_state = {"proc": None, "model": None, "tried": False, "torch": None}

# 判定：画面中出现"编辑叠加"痕迹（字幕/水印/logo/悬浮贴纸/装饰文字）即 yes；
# 真实产品包装、衣服上印刷的文字不算。保守避免误杀，拿不准时倾向 no，交给人工复核。
PROMPT = ("Does this photo have any editing overlays added on top of the scene: "
          "white subtitle/caption text with outline, a watermark or channel logo, decorated floating text, "
          "OR product images/stickers floating in the air (not held by a person)? "
          "Text printed on real product packaging or clothing does NOT count. Answer yes or no only.")

MAX_SIDE = 448      # 缩放长边：CPU 上生成 token 是瓶颈，预处理缩放影响有限
MAX_NEW = 4         # 只需 yes/no


def _load():
    if _state["tried"]:
        return _state["proc"], _state["model"]
    _state["tried"] = True
    try:
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 4)))
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        from transformers import AutoProcessor, AutoModelForImageTextToText as AutoVLM
        _state["torch"] = torch
        proc = AutoProcessor.from_pretrained(_MODEL)
        model = AutoVLM.from_pretrained(_MODEL, dtype=torch.float32)
        model.eval()
        _state["proc"] = proc
        _state["model"] = model
    except Exception as e:
        print(f"[WARN] 本地视觉模型加载失败：{e}")
        _state["proc"] = None
        _state["model"] = None
    return _state["proc"], _state["model"]


def vlm_check(bgr):
    """返回 (bool 是否含后处理浮层, str 原始回答)。bgr 为 BGR ndarray。"""
    if bgr is None or bgr.size == 0:
        return False, "empty"
    proc, model = _load()
    if model is None:
        return False, "model_unavailable"
    torch = _state["torch"]
    with _lock:
        try:
            h, w = bgr.shape[:2]
            sc = min(1.0, MAX_SIDE / max(h, w))
            if sc < 1.0:
                bgr = cv2.resize(bgr, (max(1, int(w * sc)), max(1, int(h * sc))))
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
            inputs = proc(text=prompt, images=[img], return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
            ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            return ("yes" in ans.lower()), ans
        except Exception as e:
            return False, f"err:{e}"
