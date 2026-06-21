import argparse
import importlib.util
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import cv2
import ffmpeg
import numpy as np
import openai
import json

from PIL import Image, ImageDraw, ImageFont

# python translate_workflow.py --input input/p1.mp4 --calibrate
# python translate_workflow.py --input input/p1.mp4 --load-segments


# 配置
DEEPSEEK_API_KEY = None  # 请在 .env 文件或环境变量中设置
SAMPLE_FRAME_INTERVAL = 15  # 默认每 15 帧检测一次对话变化
# MAD 像素差异检测配置
MAD_DIFF_THRESHOLD = 10.0  # 默认差异阈值，值越小越敏感
MAD_SAMPLE_SIZE = (160, 90)  # 用于比较的缩略图尺寸
FONT_PATH = None  # 可替换为本机 Noto Sans CJK 字体路径
FONT_SIZE = 42
TEXT_COLOR = (34, 34, 34)
RECT_FILL = (255, 255, 255, 230)
DEFAULT_MODEL = "deepseek-v4-flash"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
TESSDATA_DIR = os.path.join(CACHE_DIR, "tessdata")
INPUT_DIR = os.path.join(BASE_DIR, "input")
WORK_ROOT = os.path.join(BASE_DIR, "work")
WORK_DIR = os.path.join(WORK_ROOT, "default")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")

INPUT_VIDEO = os.path.join(INPUT_DIR, "input.mp4")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "final_cn.mp4")
TEMP_VIDEO = os.path.join(OUTPUT_DIR, "rendered_temp.mp4")
OUTPUT_SUBTITLES = os.path.join(OUTPUT_DIR, "subtitles.ass")
OUTPUT_JA_SUBTITLES = os.path.join(OUTPUT_DIR, "subtitles_ja.ass")
SEGMENTS_JSON = os.path.join(OUTPUT_DIR, "segments.json")

@dataclass
class Segment:
    start: float
    end: Optional[float]
    ja: str
    zh: Optional[str] = None


@dataclass
class RenderStyle:
    font_size: int
    box_color: Tuple[int, int, int]
    box_opacity: int
    text_color: Tuple[int, int, int]
    stroke_color: Tuple[int, int, int]
    stroke_width: int
    box_radius: int
    streaming: bool = True
    chars_per_sec: float = 15.0


def configure_environment() -> None:
    os.environ["HOME"] = os.path.join(CACHE_DIR, "home")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_DIR, "matplotlib"))
    os.environ.setdefault("TESSDATA_PREFIX", TESSDATA_DIR)

    # 修复 Windows 上 PyTorch DLL 加载问题
    try:
        os.add_dll_directory(r"D:\F\python\Lib\site-packages\torch\lib")
    except (OSError, AttributeError):
        pass


def safe_project_name(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return name or "video"


def configure_work_paths(video_path: str) -> None:
    global WORK_DIR, OUTPUT_DIR, OUTPUT_VIDEO, TEMP_VIDEO, OUTPUT_SUBTITLES, OUTPUT_JA_SUBTITLES, SEGMENTS_JSON

    WORK_DIR = os.path.join(WORK_ROOT, safe_project_name(video_path))
    OUTPUT_DIR = os.path.join(WORK_DIR, "output")
    OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "final_cn.mp4")
    TEMP_VIDEO = os.path.join(OUTPUT_DIR, "rendered_temp.mp4")
    OUTPUT_SUBTITLES = os.path.join(OUTPUT_DIR, "subtitles.ass")
    OUTPUT_JA_SUBTITLES = os.path.join(OUTPUT_DIR, "subtitles_ja.ass")
    SEGMENTS_JSON = os.path.join(OUTPUT_DIR, "segments.json")


def ensure_dirs() -> None:
    for path in [
        CACHE_DIR,
        os.environ["MPLCONFIGDIR"],
        TESSDATA_DIR,
        WORK_ROOT,
        WORK_DIR,
        INPUT_DIR,
        OUTPUT_DIR,
    ]:
        os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="视觉小说视频翻译工作流")
    parser.add_argument(
        "--input",
        default=os.path.join(INPUT_DIR, "input.mp4"),
        help="输入视频文件，默认 input/input.mp4",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="跳过 DeepSeek 翻译，用 OCR 原文生成字幕和预览视频",
    )
    parser.add_argument(
        "--load-segments",
        action="store_true",
        help="跳过 OCR 和翻译，从 segments.json 加载已校对文本直接渲染",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"DeepSeek 翻译模型，默认 {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--sample-frame-interval",
        type=int,
        default=SAMPLE_FRAME_INTERVAL,
        help="每隔多少帧检测一次对话变化，默认 15",
    )
    parser.add_argument(
        "--use-mad",
        action="store_true",
        help="使用 MAD 像素差异检测对话边界（替代默认的 OCR 文本相似度分组）",
    )
    parser.add_argument(
        "--mad-threshold",
        type=float,
        default=MAD_DIFF_THRESHOLD,
        help=f"MAD 差异阈值，值越小越敏感，默认 {MAD_DIFF_THRESHOLD}",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="交互式标定对话框区域（OCR 区域和字幕覆盖区域），坐标保存到 boxes.json",
    )
    parser.add_argument(
        "--box-color",
        default="#808080",
        help="对话框背景色，格式 #RRGGBB，默认 #808080",
    )
    parser.add_argument(
        "--box-opacity",
        type=int,
        default=230,
        help="对话框透明度，0-255，默认 230",
    )
    parser.add_argument(
        "--text-color",
        default="#87CEEB",
        help="文字颜色，格式 #RRGGBB，默认 #87CEEB",
    )
    parser.add_argument(
        "--stroke-color",
        default="#FFFFFF",
        help="文字描边颜色，格式 #RRGGBB，默认 #FFFFFF",
    )
    parser.add_argument(
        "--stroke-width",
        type=int,
        default=1,
        help="文字描边宽度，默认 1",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=FONT_SIZE,
        help=f"渲染字号，默认 {FONT_SIZE}",
    )
    parser.add_argument(
        "--box-radius",
        type=int,
        default=2,
        help="对话框圆角半径，默认 2",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="禁用字幕逐字流式显示（打字机效果）",
    )
    parser.add_argument(
        "--streaming-speed",
        type=float,
        default=15.0,
        help="流式字幕逐字速度（字/秒），默认 15。显示时长 = 文本长度 / 速度",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="ffmpeg H.264 压缩质量，数值越小质量越高文件越大，默认 23",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="ffmpeg x264 preset，越慢压缩率越好，默认 medium",
    )
    parser.add_argument(
        "--keep-temp-video",
        action="store_true",
        help="保留 OpenCV 生成的临时渲染视频 rendered_temp.mp4",
    )
    return parser.parse_args()


def parse_hex_color(value: str) -> Tuple[int, int, int]:
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6 or not re.fullmatch(r"[0-9A-Fa-f]{6}", text):
        raise argparse.ArgumentTypeError(f"颜色格式无效: {value}，请使用 #RRGGBB")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def clamp_opacity(value: int) -> int:
    if not 0 <= value <= 255:
        raise argparse.ArgumentTypeError("--box-opacity 必须在 0 到 255 之间")
    return value


def build_render_style(args: argparse.Namespace) -> RenderStyle:
    return RenderStyle(
        font_size=max(1, args.font_size),
        box_color=parse_hex_color(args.box_color),
        box_opacity=clamp_opacity(args.box_opacity),
        text_color=parse_hex_color(args.text_color),
        stroke_color=parse_hex_color(args.stroke_color),
        stroke_width=max(0, args.stroke_width),
        box_radius=max(0, args.box_radius),
        streaming=not args.no_streaming,
        chars_per_sec=float(args.streaming_speed),
    )


def parse_fps(value: str) -> float:
    try:
        return float(Fraction(value))
    except Exception:
        return 0.0


def analyze_video(path: str) -> Tuple[float, int, int, float]:
    ffprobe_path = shutil.which("ffprobe")
    try:
        if not ffprobe_path:
            raise FileNotFoundError("ffprobe")
        probe = ffmpeg.probe(path, cmd=ffprobe_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps_text = video_stream.get('r_frame_rate', '0/1')
        fps = parse_fps(fps_text) or parse_fps(video_stream.get('avg_frame_rate', '0/1'))
        duration = float(video_stream.get('duration', probe['format'].get('duration', 0)))
        return fps, width, height, duration
    except FileNotFoundError:
        print("警告: 未找到 ffprobe，可使用 Homebrew 安装：brew install ffmpeg 。将回退到 OpenCV 获取媒体信息，结果可能不如 ffprobe 精确。")
    except Exception as exc:
        # 如果 ffmpeg.probe 抛出其它异常，也尝试用 OpenCV 回退
        print(f"ffprobe 解析失败，回退到 OpenCV：{exc}")

    # 回退：使用 OpenCV 获取 fps/宽高/帧数以估算时长
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件进行分析: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    if fps == 0.0:
        # 作为最后手段，避免后续时间戳计算除零
        fps = 30.0
    return fps, width, height, duration


def estimate_boxes(width: int, height: int) -> Dict[str, Tuple[int, int, int, int]]:
    boxes_file = os.path.join(BASE_DIR, "boxes.json")
    if os.path.exists(boxes_file):
        with open(boxes_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("width") != width or saved.get("height") != height:
            print(
                f"警告: boxes.json 标定分辨率 {saved.get('width')}x{saved.get('height')} "
                f"与当前视频 {width}x{height} 不匹配，建议重新标定"
            )
        return {
            "ocr_box": tuple(saved["ocr_box"]),
            "overlay_box": tuple(saved["overlay_box"]),
        }

    # 回退：硬编码百分比
    ocr_box = (
        int(width * 0.135),
        int(height * 0.81),
        int(width * 0.73),
        int(height * 0.10),
    )
    overlay_box = (
        int(width * 0.135),
        int(height * 0.82),
        int(width * 0.73),
        int(height * 0.135),
    )
    return {"ocr_box": ocr_box, "overlay_box": overlay_box}


def crop_region(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return frame[y : y + h, x : x + w]


def sample_frames(video_path: str, frame_interval: int, fps: float, text_box: Tuple[int, int, int, int]) -> List[Dict]:
    cap = cv2.VideoCapture(video_path)
    step = max(1, frame_interval)
    samples = []
    index = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if index % step == 0:
            timestamp = index / fps
            crop = crop_region(frame, text_box)
            samples.append({"frame_index": index, "time": timestamp, "crop": crop})
            saved += 1
        index += 1
    cap.release()
    return samples


def normalize_japanese(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace('。 ', '。').replace('！ ', '！').replace('？ ', '？')
    text = re.sub(r'[ 　]+', ' ', text)
    text = clean_ocr_text(text)
    return text


def ocr_dialogue(crops: List[np.ndarray], ocr_engine: Any) -> List[str]:
    results = []
    for crop in crops:
        try:
            ocr_result = ocr_engine.ocr(crop, cls=False)  # det=True 检测多行文本
        except TypeError:
            ocr_result = ocr_engine.ocr(crop)
        text = extract_ocr_text(ocr_result)
        text = normalize_japanese(text)
        results.append(text)
    return results


def group_samples_by_text(
    samples: List[Dict],
    all_texts: List[str],
    duration: float,
) -> List[Segment]:
    """基于 OCR 文本相似度进行对话分组。

    遍历所有采样帧，将连续、文本相似的帧归为同一 segment。
    每组取最高分 OCR 结果作为代表文本。
    """
    if not samples:
        return []

    groups: List[List[int]] = []
    current_group = [0]
    for i in range(1, len(all_texts)):
        prev_text = all_texts[current_group[-1]]
        curr_text = all_texts[i]

        same_utterance = False
        if prev_text and curr_text:
            same_utterance = (
                is_progressive_text(prev_text, curr_text)
                or char_overlap_ratio(prev_text, curr_text) >= 0.5
                or prev_text == curr_text
            )
        elif not prev_text and not curr_text:
            same_utterance = True

        if same_utterance:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    if current_group:
        groups.append(current_group)

    segments: List[Segment] = []
    for group in groups:
        candidates = [(i, all_texts[i]) for i in group if all_texts[i] and not is_noise_text(all_texts[i])]
        if not candidates:
            continue
        _, best_text = max(candidates, key=lambda x: _score_ocr_text(x[1]))
        seg = Segment(
            start=samples[group[0]]['time'],
            end=None,
            ja=best_text,
        )
        segments.append(seg)

    for i in range(len(segments) - 1):
        segments[i].end = segments[i + 1].start
    if segments:
        segments[-1].end = duration

    return segments


def extract_ocr_text(ocr_result) -> str:
    """解析 PaddleOCR / pytesseract 返回结果，提取纯文本。

    PaddleOCR ocr() 返回格式（单图）：
      det=True:  [[ [[x1,y1],...], ('text', score) ], ...]   # 外层多包一层
      det=False: [[ ('text', score), ... ]]
    本函数先解包外层 [0]，再根据内层元素格式分别处理。
    """
    if not ocr_result:
        return ""

    # 第一步：解包 PaddleOCR 的图片级外层 list
    if isinstance(ocr_result, list) and len(ocr_result) == 1:
        inner = ocr_result[0]
        if isinstance(inner, list) and inner:
            first = inner[0]

            # === det=True 格式 ===
            # 每个元素: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ('text', score)]
            if (
                isinstance(first, list)
                and len(first) >= 2
                and isinstance(first[0], list)
                and len(first[0]) == 4
            ):
                lines = []
                for item in inner:
                    if isinstance(item, list) and len(item) >= 2:
                        bbox = item[0]
                        text_info = item[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = str(text_info[0])
                            conf = float(text_info[1])
                            y_center = sum(pt[1] for pt in bbox) / 4
                            lines.append((y_center, text, conf))
                if lines:
                    lines.sort(key=lambda x: x[0])
                    return "".join(text for _, text, _ in lines)

            # === det=False 格式 ===
            # 每个元素: ('text', score)
            if isinstance(first, (list, tuple)) and len(first) >= 1:
                if not isinstance(first[0], list):  # 排除 det=True 的 bbox
                    texts = []
                    for entry in inner:
                        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
                            texts.append(str(entry[0]))
                        elif isinstance(entry, str):
                            texts.append(entry)
                    return "".join(texts)

    # === 兼容 pytesseract / 旧版格式 ===
    texts = []
    for item in ocr_result:
        if isinstance(item, dict):
            texts.extend(str(t) for t in item.get("rec_texts", []) if t)
        elif isinstance(item, (list, tuple)) and len(item) > 1:
            value = item[1]
            if isinstance(value, (list, tuple)) and value:
                texts.append(str(value[0]))
            elif isinstance(value, str):
                texts.append(value)
    return "".join(texts)


def deduplicate_texts(texts: List[str]) -> List[str]:
    deduped = []
    for text in texts:
        if not text:
            continue
        if deduped and text == deduped[-1]:
            continue
        deduped.append(text)
    return deduped


def compact_for_compare(text: str) -> str:
    return re.sub(r"[\s、。！？!?…ー—\-・『』「」\"'（）()\[\]【】]+", "", text)


def is_noise_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return bool(re.fullmatch(r"[0-9A-Za-z_|/\\.,:;+\-=]+", stripped)) and len(stripped) <= 4


def clean_ocr_text(text: str) -> str:
    """去除 OCR 结果中的数字/符号前缀噪声（如 \"1000000あたし\" → \"あたし\"）。"""
    cleaned = re.sub(r"^[\d\s_|/\\.,:;+\-=\"']+", "", text.strip())
    return cleaned


def _score_ocr_text(text: str) -> float:
    """对 OCR 结果打分（越高越好）：长度 + 日语字符纯度。"""
    if not text:
        return 0.0
    jp_chars = sum(1 for c in text if (
        '぀' <= c <= 'ゟ' or
        '゠' <= c <= 'ヿ' or
        '一' <= c <= '鿿' or
        '　' <= c <= '〿'
    ))
    purity = jp_chars / len(text) if text else 0.0
    return min(len(text), 80) * 0.6 + purity * 40


def is_progressive_text(previous: str, current: str) -> bool:
    prev = compact_for_compare(previous)
    curr = compact_for_compare(current)
    if len(prev) < 3 or len(curr) < 3:
        return False
    shorter, longer = (prev, curr) if len(prev) <= len(curr) else (curr, prev)
    return longer.startswith(shorter) or shorter in longer


def char_overlap_ratio(a: str, b: str) -> float:
    """两个文本的字符重叠率（0~1），用于判断是否同一句对话的两次 OCR。"""
    if not a or not b:
        return 0.0
    set_a = set(compact_for_compare(a))
    set_b = set(compact_for_compare(b))
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / min(len(set_a), len(set_b))


def merge_progressive_segments(segments: List[Segment]) -> List[Segment]:
    merged: List[Segment] = []
    for segment in segments:
        if is_noise_text(segment.ja):
            if merged:
                merged[-1].end = segment.end
            continue

        if merged and (
            is_progressive_text(merged[-1].ja, segment.ja)
            or char_overlap_ratio(merged[-1].ja, segment.ja) >= 0.9
        ):
            # 同一句对话的两次 OCR → 保留更长/更完整的版本
            if len(compact_for_compare(segment.ja)) >= len(compact_for_compare(merged[-1].ja)):
                merged[-1].ja = segment.ja
            merged[-1].end = segment.end
            continue

        merged.append(segment)
    return merged


def load_dotenv(path: str = os.path.join(BASE_DIR, ".env")) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)


def translate_text(text: str, model: str) -> str:
    api_key = DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY。请在环境变量或项目 .env 中设置，或显式使用 --no-translate。")

    system_prompt = (
        "Translate Japanese game dialogue into natural Simplified Chinese.\n"
        "Rules:\n"
        "- Preserve character names.\n"
        "- Preserve honorific meaning.\n"
        "- Use natural game localization style.\n"
        "- Avoid machine-translation tone.\n"
        "- Output only the translated dialogue."
    )

    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    for attempt in range(3):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        result = response.choices[0].message.content
        if result:
            return result.strip()
        if attempt < 2:
            print(f"  翻译返回空，重试第 {attempt + 1} 次: {text[:40]}...")
    print(f"  警告: 翻译 3 次均返回空: {text[:60]}")
    return ""


def _patch_paddle_disable_onednn() -> None:
    """修补 PaddlePaddle 推理配置以禁用 OneDNN，避免 Windows 上
    "OneDnnContext does not have the input Filter" 推理错误。"""
    try:
        from paddle import inference

        _orig_config_init = inference.Config.__init__

        def _patched_config_init(self, *args, **kwargs):
            _orig_config_init(self, *args, **kwargs)
            # 必须在创建预测器之前禁用，否则 OneDNN 初始化后再禁用无效
            try:
                self.disable_onednn()
                self.disable_mkldnn()
            except Exception:
                pass

        inference.Config.__init__ = _patched_config_init
    except Exception:
        pass


def _setup_paddle_ocr_env() -> str:
    """设置 PaddleOCR 模型目录到纯 ASCII 路径（C++ 推理引擎不支持 Unicode 路径）。
    返回模型目录路径。
    """
    ascii_dir = r"C:\paddleocr_models"
    if os.path.isdir(ascii_dir):
        return ascii_dir

    # 尝试从默认 Unicode 路径迁移模型
    legacy_dir = os.path.normpath(os.path.join(os.path.expanduser("~"), ".paddleocr"))
    if os.path.isdir(legacy_dir):
        print(f"首次运行：迁移 PaddleOCR 模型到 ASCII 路径 {ascii_dir} ...")
        try:
            shutil.copytree(legacy_dir, ascii_dir)
            return ascii_dir
        except Exception as exc:
            print(f"模型迁移失败：{exc}")

    # 回退到 Unicode 路径（可能失败）
    return legacy_dir


def init_ocr_engine():
    if importlib.util.find_spec("paddle") is not None:
        # 必须在导入 paddleocr 之前设置，确保使用纯 ASCII 路径
        model_base = _setup_paddle_ocr_env()
        os.environ["PADDLE_OCR_BASE_DIR"] = model_base + os.sep

        try:
            # 必须先导入 torch 再导入 paddleocr，避免 Windows DLL 加载顺序问题
            try:
                import torch  # noqa: F401
            except ImportError:
                pass

            from paddleocr import PaddleOCR

            # 修补 PaddlePaddle 在 Windows 上 OneDNN 推理的 bug
            _patch_paddle_disable_onednn()

            try:
                return PaddleOCR(
                    lang="japan",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_rec_score_thresh=0.3,
                )
            except ValueError:
                return PaddleOCR(lang="japan", use_angle_cls=False)
        except Exception as exc:
            print(f"警告：PaddleOCR 初始化失败：{exc}")
    else:
        print("警告：未安装 paddlepaddle，跳过 PaddleOCR，尝试使用 pytesseract。")

    try:
        import pytesseract

        languages = set(pytesseract.get_languages(config=""))
        if "jpn" not in languages:
            raise RuntimeError(
                f"未找到 Tesseract 日语语言包 jpn。请将 jpn.traineddata 放入 {TESSDATA_DIR}"
            )

        class PyTessWrapper:
            def ocr(self, image, cls=False, det=False):
                processed = preprocess_for_tesseract(image)
                txt = pytesseract.image_to_string(processed, lang="jpn", config="--psm 7")
                lines = [line for line in txt.splitlines() if line.strip()]
                return [[None, (line, 1.0)] for line in lines]

        return PyTessWrapper()
    except Exception as exc:
        raise RuntimeError(
            "没有可用的 OCR 引擎。请安装 paddlepaddle 以使用 PaddleOCR，"
            f"或安装 pytesseract 并把 jpn.traineddata 放入 {TESSDATA_DIR}。原始错误：{exc}"
        ) from exc


def preprocess_for_tesseract(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 145), (180, 90, 255))
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    inverted = cv2.bitwise_not(mask)
    return cv2.resize(inverted, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def assign_translations(segments: List[Segment], raw_texts: List[str], translate_enabled: bool, model: str) -> int:
    translation_cache: Dict[str, str] = {}
    for segment, text in zip(segments, raw_texts):
        segment.ja = text
        if not text:
            segment.zh = ""
            continue
        if text not in translation_cache:
            translation_cache[text] = translate_text(text, model) if translate_enabled else text
        segment.zh = translation_cache[text]
    return len(translation_cache)


def save_segments(segments: List[Segment], path: str) -> None:
    """保存翻译片段到 JSON，方便校对后重新渲染。"""
    data = []
    for seg in segments:
        data.append({
            "start": seg.start,
            "end": seg.end,
            "ja": seg.ja,
            "zh": seg.zh or "",
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"segments": data}, f, indent=2, ensure_ascii=False)
    print(f"翻译片段已保存到 {path}，可编辑 zh 字段校对后使用 --load-segments 重新渲染")


def load_segments(path: str) -> List[Segment]:
    """从 JSON 文件加载校对后的翻译片段。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = []
    for item in data["segments"]:
        seg = Segment(
            start=item["start"],
            end=item["end"],
            ja=item.get("ja", ""),
            zh=item.get("zh", ""),
        )
        segments.append(seg)
    print(f"从 {path} 加载了 {len(segments)} 个翻译片段")
    return segments


def format_ass_time(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    return f"{hours:d}:{minutes:02d}:{secs:05.2f}"


def build_subtitles(segments: List[Segment], field: str = "zh") -> str:
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Source Han Sans,52,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,20,20,40,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for segment in segments:
        text = getattr(segment, field)
        if not text:
            continue
        start = format_ass_time(segment.start)
        end = format_ass_time(segment.end if segment.end is not None else segment.start + 2.0)
        events.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
        )
    return header + "\n".join(events)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    if FONT_PATH and os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, size)

    # Windows 系统字体路径（主显示中文翻译，优先雅黑）
    win_fonts = [
        "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑 — 中文渲染最佳
        "C:/Windows/Fonts/meiryo.ttc",     # Meiryo — 日文现代字体
        "C:/Windows/Fonts/msgothic.ttc",   # MS Gothic — 日文哥特体
        "C:/Windows/Fonts/yugothb.ttc",    # Yu Gothic Bold
    ]
    for path in win_fonts:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue

    # macOS 字体
    for name in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue

    # 项目目录下的本地字体
    for name in [
        "SourceHanSansCN-Regular.otf",
        "NotoSansCJK-Regular.otf",
        "msyh.ttf",
    ]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue

    print("警告：未找到任何中文字体，将使用默认字体（文字会非常小）")
    return ImageFont.load_default()


def text_size(font: ImageFont.ImageFont, text: str) -> Tuple[int, int]:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    if " " not in text:
        lines = []
        current_line = ""
        for char in text:
            candidate = current_line + char
            width, _ = text_size(font, candidate)
            if width <= max_width or not current_line:
                current_line = candidate
            else:
                lines.append(current_line)
                current_line = char
        if current_line:
            lines.append(current_line)
        return lines if lines else [text]

    lines = []
    current_line = ""
    for word in text.split(' '):
        candidate = word if not current_line else f"{current_line} {word}"
        width, _ = text_size(font, candidate)
        if width <= max_width:
            current_line = candidate
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines if lines else [text]


def streaming_text(
    full_text: str,
    elapsed: float,
    seg_duration: float,
    chars_per_sec: float = 15.0,
) -> str:
    """打字机效果：以恒定速度逐字显示。

    reveal_duration = len(text) / chars_per_sec（不超过 segment 时长）
    速度恒定，无论 segment 长短。
    """
    if not full_text:
        return ""
    if chars_per_sec <= 0:
        return full_text
    reveal_dur = min(len(full_text) / chars_per_sec, seg_duration)
    if reveal_dur <= 0 or elapsed >= reveal_dur:
        return full_text
    if elapsed <= 0:
        return full_text[0]
    progress = elapsed / reveal_dur
    char_count = max(1, int(len(full_text) * progress))
    return full_text[:char_count]


def render_frame(
    frame: np.ndarray,
    segment: Optional[Segment],
    overlay_box: Tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont,
    style: RenderStyle,
    timestamp: float = 0.0,
) -> Image.Image:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image).convert("RGBA")
    draw = ImageDraw.Draw(pil, "RGBA")
    x, y, w, h = overlay_box

    # 计算流式文本
    display_text = ""
    if segment and segment.zh:
        if style.streaming and segment.end and segment.end > segment.start:
            elapsed = timestamp - segment.start
            display_text = streaming_text(
                segment.zh, elapsed,
                segment.end - segment.start,
                style.chars_per_sec,
            )
        else:
            display_text = segment.zh

    if segment:
        draw.rounded_rectangle(
            [x, y, x + w, y + h],
            radius=style.box_radius,
            fill=(*style.box_color, style.box_opacity),
        )
        if display_text:
            lines = wrap_text(display_text, font, w - 32)
            total_height = sum(text_size(font, line)[1] for line in lines) + 12 * (len(lines) - 1)
            current_y = y + (h - total_height) // 3
            for line in lines:
                line_width, line_height = text_size(font, line)
                line_x = x + 24
                draw.text(
                    (line_x, current_y),
                    line,
                    font=font,
                    fill=style.text_color,
                    stroke_width=style.stroke_width,
                    stroke_fill=style.stroke_color,
                )
                current_y += line_height + 12
    return pil.convert("RGB")


def find_active_segment(segments: List[Segment], timestamp: float) -> Optional[Segment]:
    for segment in reversed(segments):
        if segment.start <= timestamp and (segment.end is None or timestamp < segment.end):
            return segment
    return None


def render_video_with_opencv(
    video_path: str,
    output_path: str,
    segments: List[Segment],
    overlay_box: Tuple[int, int, int, int],
    fps: float,
    style: RenderStyle,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件进行渲染: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("无法获取视频尺寸，不能创建输出视频")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("OpenCV 无法创建输出 mp4。请安装 ffmpeg：brew install ffmpeg")

    idx = 0
    font = load_font(style.font_size)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = idx / fps
        segment = find_active_segment(segments, timestamp)
        rendered = render_frame(frame, segment, overlay_box, font, style, timestamp)
        writer.write(cv2.cvtColor(np.array(rendered), cv2.COLOR_RGB2BGR))
        idx += 1

    writer.release()
    cap.release()


def compress_with_ffmpeg(source_video: str, original_video: str, output_video: str, crf: int, preset: str) -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        source_video,
        "-i",
        original_video,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-shortest",
        "-movflags",
        "+faststart",
        output_video,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return True


def prepare_crop_for_diff(crop: np.ndarray, target_size: Tuple[int, int] = MAD_SAMPLE_SIZE) -> np.ndarray:
    """预处理裁剪区域用于 MAD 比较：缩小 + 灰度化。"""
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray


def mean_abs_diff(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两幅图像的 Mean Absolute Difference。"""
    diff = cv2.absdiff(img1.astype(np.float32), img2.astype(np.float32))
    return float(np.mean(diff))


def detect_dialogue_segments_mad(
    samples: List[Dict],
    duration: float,
    diff_threshold: float = MAD_DIFF_THRESHOLD,
) -> Tuple[List[Segment], List[int]]:
    """使用 MAD 像素差异检测对话变化边界。

    比较相邻采样帧的 OCR 区域像素差异，差异超过阈值时标记为新 segment 的起点。

    Returns:
        (segments, boundary_indices): segments 的 ja 字段为空，由调用方 OCR 填充；
        boundary_indices 是每个 segment 的起始采样帧在 samples 中的索引（即变化帧）。
    """
    if not samples:
        return [], []

    if len(samples) == 1:
        seg = Segment(start=samples[0]['time'], end=duration, ja="")
        return [seg], [0]

    preprocessed = [prepare_crop_for_diff(s['crop']) for s in samples]

    # 计算相邻帧差异，检测变化边界
    diffs = []
    boundaries = [0]
    for i in range(1, len(preprocessed)):
        diff = mean_abs_diff(preprocessed[i - 1], preprocessed[i])
        diffs.append(diff)
        if diff > diff_threshold:
            boundaries.append(i)

    avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
    max_diff = max(diffs) if diffs else 0.0
    print(
        f"MAD 分析: {len(samples)} 个采样帧, "
        f"平均差异={avg_diff:.2f}, 最大差异={max_diff:.2f}, "
        f"检测到 {len(boundaries)} 个边界 (阈值={diff_threshold:.1f})"
    )

    segments = []
    for j in range(len(boundaries)):
        start_idx = boundaries[j]
        segments.append(Segment(
            start=samples[start_idx]['time'],
            end=None,
            ja="",
        ))

    # 修正 segment 结束时间为下一 segment 的开始时间
    for i in range(len(segments) - 1):
        segments[i].end = segments[i + 1].start
    if segments:
        segments[-1].end = duration

    return segments, boundaries


def calibrate_boxes(video_path: str, duration: float) -> Dict[str, Any]:
    """抽取视频参考帧，让用户用鼠标框选 OCR 区域和字幕覆盖区域。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    target_time = duration * 0.3
    cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
    ret, frame = cap.read()
    if not ret:
        # 回退到第一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("无法从视频中抽取参考帧用于标定")

    height, width = frame.shape[:2]

    cv2.namedWindow("calibrate-ocr", cv2.WINDOW_NORMAL)
    cv2.setWindowTitle("calibrate-ocr", "框选 OCR 文字区域后按 ENTER")
    ocr_roi = cv2.selectROI("calibrate-ocr", frame, False)
    cv2.destroyWindow("calibrate-ocr")

    cv2.namedWindow("calibrate-overlay", cv2.WINDOW_NORMAL)
    cv2.setWindowTitle("calibrate-overlay", "框选字幕覆盖区域后按 ENTER")
    overlay_roi = cv2.selectROI("calibrate-overlay", frame, False)
    cv2.destroyAllWindows()

    ocr_box = [int(v) for v in ocr_roi]
    overlay_box = [int(v) for v in overlay_roi]

    if any(v <= 0 for v in ocr_box):
        raise RuntimeError("OCR 区域标定无效，所有值必须 > 0")
    if any(v <= 0 for v in overlay_box):
        raise RuntimeError("字幕覆盖区域标定无效，所有值必须 > 0")

    return {
        "ocr_box": ocr_box,
        "overlay_box": overlay_box,
        "width": width,
        "height": height,
    }


def main() -> None:
    configure_environment()
    load_dotenv()
    args = parse_args()
    style = build_render_style(args)
    input_video = os.path.abspath(args.input)
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"请先将原始视频放入 {input_video}")

    configure_work_paths(input_video)
    ensure_dirs()
    if not args.calibrate and not args.load_segments and not args.no_translate and not (DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY")):
        raise RuntimeError("未设置 DEEPSEEK_API_KEY，不能调用 DeepSeek API 翻译。请设置环境变量/项目 .env，或显式加 --no-translate。")

    print(f"项目目录: {WORK_DIR}")

    fps, width, height, duration = analyze_video(input_video)

    if args.calibrate:
        boxes = calibrate_boxes(input_video, duration)
        boxes_path = os.path.join(BASE_DIR, "boxes.json")
        with open(boxes_path, "w", encoding="utf-8") as f:
            json.dump(boxes, f, indent=2, ensure_ascii=False)
        print(f"标定完成，坐标已保存到 {boxes_path}")
        print(f"  OCR 区域: {boxes['ocr_box']}")
        print(f"  覆盖区域: {boxes['overlay_box']}")
        return

    boxes = estimate_boxes(width, height)
    print(f"视频分析: fps={fps:.2f}, 分辨率={width}x{height}, 时长={duration:.2f}s")
    print(f"OCR 区域: {boxes['ocr_box']}")
    print(f"覆盖区域: {boxes['overlay_box']}")

    if args.load_segments:
        if not os.path.exists(SEGMENTS_JSON):
            raise FileNotFoundError(
                f"未找到 {SEGMENTS_JSON}，请先正常运行一次生成翻译片段。"
            )
        segments = load_segments(SEGMENTS_JSON)
    else:
        samples = sample_frames(input_video, args.sample_frame_interval, fps, boxes['ocr_box'])
        print(f"已按每 {args.sample_frame_interval} 帧采样，得到 {len(samples)} 帧")

        ocr_engine = init_ocr_engine()

        if args.use_mad:
            # === Step 4: MAD 像素差异检测 ===
            segments, boundary_indices = detect_dialogue_segments_mad(
                samples, duration, args.mad_threshold
            )
            print(f"MAD 边界检测后 {len(segments)} 个对话片段")

            # === Step 5: 初始 OCR（仅变化帧 / 边界帧）===
            boundary_crops = [samples[idx]['crop'] for idx in boundary_indices]
            boundary_texts = ocr_dialogue(boundary_crops, ocr_engine)
            for seg, text in zip(segments, boundary_texts):
                seg.ja = text
            print(f"初始 OCR（{len(boundary_indices)} 个边界帧）完成")

            # === Step 6: 多帧 OCR 择优（每个 segment 取首/中/尾帧，选最高分）===
            for j, seg in enumerate(segments):
                start_idx = boundary_indices[j]
                end_idx = (
                    boundary_indices[j + 1]
                    if j + 1 < len(boundary_indices)
                    else len(samples)
                )
                seg_sample_indices = list(range(start_idx, end_idx))
                if len(seg_sample_indices) <= 1:
                    continue  # 仅一帧，已 OCR

                # 中间帧 + 末尾帧
                mid_offset = len(seg_sample_indices) // 2
                mid_idx = seg_sample_indices[mid_offset]
                last_idx = seg_sample_indices[-1]

                extra_crops = [samples[mid_idx]['crop'], samples[last_idx]['crop']]
                extra_texts = ocr_dialogue(extra_crops, ocr_engine)

                # 三元候选：起始帧(已OCR) / 中间帧 / 末尾帧
                candidates = [
                    (seg.ja, start_idx),
                    (extra_texts[0], mid_idx),
                    (extra_texts[1], last_idx),
                ]
                valid = [(t, i) for t, i in candidates if t and not is_noise_text(t)]
                if valid:
                    best_text, _ = max(valid, key=lambda x: _score_ocr_text(x[0]))
                    seg.ja = best_text
            print("多帧 OCR 择优完成")

            # === Step 7: 长 segment 拆分（> 5s 的 segment 检查 1/3 与 2/3 处是否不同）===
            new_segments: List[Segment] = []
            split_count = 0
            for j, seg in enumerate(segments):
                seg_duration = (seg.end if seg.end is not None else duration) - seg.start
                if seg_duration <= 5.0:
                    new_segments.append(seg)
                    continue

                start_idx = boundary_indices[j]
                end_idx = (
                    boundary_indices[j + 1]
                    if j + 1 < len(boundary_indices)
                    else len(samples)
                )
                seg_sample_indices = list(range(start_idx, end_idx))
                if len(seg_sample_indices) < 3:
                    new_segments.append(seg)
                    continue

                # 在 1/3 和 2/3 处 OCR
                third_1_offset = len(seg_sample_indices) // 3
                third_2_offset = 2 * len(seg_sample_indices) // 3
                idx_1 = seg_sample_indices[third_1_offset]
                idx_2 = seg_sample_indices[third_2_offset]

                split_crops = [samples[idx_1]['crop'], samples[idx_2]['crop']]
                split_texts = ocr_dialogue(split_crops, ocr_engine)
                text_1, text_2 = split_texts[0], split_texts[1]

                # 判断是否拆分：不是渐进、重叠率低、不相同
                if (
                    text_1 and text_2
                    and not is_progressive_text(text_1, text_2)
                    and char_overlap_ratio(text_1, text_2) <= 0.5
                    and text_1 != text_2
                ):
                    seg1 = Segment(
                        start=seg.start,
                        end=samples[idx_2]['time'],
                        ja=text_1,
                    )
                    seg2 = Segment(
                        start=samples[idx_2]['time'],
                        end=seg.end,
                        ja=text_2,
                    )
                    new_segments.append(seg1)
                    new_segments.append(seg2)
                    split_count += 1
                else:
                    # 保持合并，用更长文本
                    if text_1 and text_2:
                        seg.ja = (
                            text_1
                            if len(compact_for_compare(text_1)) >= len(compact_for_compare(text_2))
                            else text_2
                        )
                    new_segments.append(seg)

            segments = new_segments
            print(f"长 segment 拆分后 {len(segments)} 个对话片段 (拆分 {split_count} 个)")

            # 过滤空文本 / 噪声 segment
            segments = [s for s in segments if s.ja and not is_noise_text(s.ja)]
            print(f"过滤后 {len(segments)} 个对话片段")
        else:
            # === 默认 OCR 文本相似度分组管线 ===
            all_crops = [s['crop'] for s in samples]
            all_texts = ocr_dialogue(all_crops, ocr_engine)
            print(f"OCR 完成: {len(all_texts)} 帧")

            segments = group_samples_by_text(samples, all_texts, duration)
            print(f"文本分组后 {len(segments)} 个对话片段")

            # === 长 segment 拆分（> 5s）===
            new_segments: List[Segment] = []
            split_count = 0
            for seg in segments:
                seg_duration = (seg.end if seg.end is not None else duration) - seg.start
                if seg_duration <= 5.0:
                    new_segments.append(seg)
                    continue

                # 找到该 segment 时间范围内的采样帧索引
                seg_end = seg.end if seg.end is not None else (duration + 1)
                seg_sample_indices = [
                    i for i, s in enumerate(samples)
                    if seg.start <= s['time'] < seg_end
                ]
                if len(seg_sample_indices) < 3:
                    new_segments.append(seg)
                    continue

                # 在 1/3 和 2/3 处取已 OCR 的文本
                third_1_offset = len(seg_sample_indices) // 3
                third_2_offset = 2 * len(seg_sample_indices) // 3
                text_1 = all_texts[seg_sample_indices[third_1_offset]]
                text_2 = all_texts[seg_sample_indices[third_2_offset]]

                # 判断是否拆分
                if (
                    text_1 and text_2
                    and not is_progressive_text(text_1, text_2)
                    and char_overlap_ratio(text_1, text_2) <= 0.5
                    and text_1 != text_2
                ):
                    seg1 = Segment(
                        start=seg.start,
                        end=samples[seg_sample_indices[third_2_offset]]['time'],
                        ja=text_1,
                    )
                    seg2 = Segment(
                        start=samples[seg_sample_indices[third_2_offset]]['time'],
                        end=seg.end,
                        ja=text_2,
                    )
                    new_segments.append(seg1)
                    new_segments.append(seg2)
                    split_count += 1
                else:
                    # 保持合并，用更长文本
                    if text_1 and text_2:
                        seg.ja = (
                            text_1
                            if len(compact_for_compare(text_1)) >= len(compact_for_compare(text_2))
                            else text_2
                        )
                    new_segments.append(seg)

            segments = new_segments
            print(f"长 segment 拆分后 {len(segments)} 个对话片段 (拆分 {split_count} 个)")

        translate_enabled = bool(DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY")) and not args.no_translate
        if args.no_translate:
            print("警告: 指定了 --no-translate，将使用 OCR 原文生成字幕和视频。")

        before_merge_count = len(segments)
        segments = merge_progressive_segments(segments)
        print(f"递进文本合并后 {len(segments)} 个字幕片段，过滤/合并 {before_merge_count - len(segments)} 个片段")

        translate_count = assign_translations(segments, [segment.ja for segment in segments], translate_enabled, args.model)
        print(f"翻译调用次数: {translate_count}")

        save_segments(segments, SEGMENTS_JSON)

    with open(OUTPUT_SUBTITLES, 'w', encoding='utf-8') as f:
        f.write(build_subtitles(segments, "zh"))

    with open(OUTPUT_JA_SUBTITLES, 'w', encoding='utf-8') as f:
        f.write(build_subtitles(segments, "ja"))

    ffmpeg_available = shutil.which("ffmpeg") is not None
    render_target = TEMP_VIDEO if ffmpeg_available else OUTPUT_VIDEO
    render_video_with_opencv(input_video, render_target, segments, boxes['overlay_box'], fps, style)
    if ffmpeg_available:
        compress_with_ffmpeg(TEMP_VIDEO, input_video, OUTPUT_VIDEO, args.crf, args.preset)
        if not args.keep_temp_video and os.path.exists(TEMP_VIDEO):
            os.remove(TEMP_VIDEO)
        print(f"已使用 ffmpeg libx264 压缩输出: crf={args.crf}, preset={args.preset}")
    else:
        print("警告: 未找到 ffmpeg，已使用 OpenCV mp4v 流式编码；输出视频不包含原始音频，文件可能较大。")

    print(f"输出已生成: {OUTPUT_VIDEO}")
    print(f"字幕 ASS: {OUTPUT_SUBTITLES}")
    print(f"日语字幕 ASS: {OUTPUT_JA_SUBTITLES}")


if __name__ == '__main__':
    main()
