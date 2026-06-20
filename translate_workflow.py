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
from PIL import Image, ImageDraw, ImageFont

# python translate_workflow.py --input input/test.mp4


# 配置
SAMPLE_FRAME_INTERVAL = 30  # 默认每 30 帧检测一次对话变化
DIFF_THRESHOLD = 10.0
SAMPLE_SIZE = (160, 90)
FONT_PATH = None  # 可替换为本机 Noto Sans CJK 字体路径
FONT_SIZE = 44
TEXT_COLOR = (34, 34, 34)
RECT_FILL = (255, 255, 255, 230)
DEFAULT_OPENAI_MODEL = "gpt-5"

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


def configure_environment() -> None:
    os.environ["HOME"] = os.path.join(CACHE_DIR, "home")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_DIR, "matplotlib"))
    os.environ.setdefault("TESSDATA_PREFIX", TESSDATA_DIR)


def safe_project_name(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return name or "video"


def configure_work_paths(video_path: str) -> None:
    global WORK_DIR, OUTPUT_DIR, OUTPUT_VIDEO, TEMP_VIDEO, OUTPUT_SUBTITLES, OUTPUT_JA_SUBTITLES

    WORK_DIR = os.path.join(WORK_ROOT, safe_project_name(video_path))
    OUTPUT_DIR = os.path.join(WORK_DIR, "output")
    OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "final_cn.mp4")
    TEMP_VIDEO = os.path.join(OUTPUT_DIR, "rendered_temp.mp4")
    OUTPUT_SUBTITLES = os.path.join(OUTPUT_DIR, "subtitles.ass")
    OUTPUT_JA_SUBTITLES = os.path.join(OUTPUT_DIR, "subtitles_ja.ass")


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
        help="跳过 OpenAI 翻译，用 OCR 原文生成字幕和预览视频",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI 翻译模型，默认 {DEFAULT_OPENAI_MODEL}",
    )
    parser.add_argument(
        "--sample-frame-interval",
        type=int,
        default=SAMPLE_FRAME_INTERVAL,
        help="每隔多少帧检测一次对话变化，默认 30",
    )
    parser.add_argument(
        "--box-color",
        default="#FFFFFF",
        help="对话框背景色，格式 #RRGGBB，默认 #FFFFFF",
    )
    parser.add_argument(
        "--box-opacity",
        type=int,
        default=230,
        help="对话框透明度，0-255，默认 230",
    )
    parser.add_argument(
        "--text-color",
        default="#222222",
        help="文字颜色，格式 #RRGGBB，默认 #222222",
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
    name_box = (
        int(width * 0.10),
        int(height * 0.70),
        int(width * 0.30),
        int(height * 0.08),
    )
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
    return {"name_box": name_box, "ocr_box": ocr_box, "overlay_box": overlay_box}


def crop_region(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return frame[y : y + h, x : x + w]


def prepare_crop_for_diff(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, SAMPLE_SIZE)
    return small


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


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


def detect_dialogue_segments(samples: List[Dict]) -> Tuple[List[Segment], List[Dict]]:
    segments: List[Segment] = []
    changed_samples: List[Dict] = []
    prev_crop = None
    for sample in samples:
        current_crop = prepare_crop_for_diff(sample['crop'])
        new_segment = False
        if prev_crop is None:
            new_segment = True
        else:
            diff = mean_abs_diff(current_crop, prev_crop)
            new_segment = diff > DIFF_THRESHOLD
        if new_segment:
            segments.append(Segment(start=sample['time'], end=None, ja=""))
            changed_samples.append(sample)
        prev_crop = current_crop
    for i in range(len(segments) - 1):
        segments[i].end = segments[i + 1].start
    return segments, changed_samples


def normalize_japanese(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace('。 ', '。').replace('！ ', '！').replace('？ ', '？')
    text = re.sub(r'[ 　]+', ' ', text)
    return text


def ocr_dialogue(crops: List[np.ndarray], ocr_engine: Any) -> List[str]:
    results = []
    for crop in crops:
        try:
            ocr_result = ocr_engine.ocr(crop, cls=False, det=False)
        except TypeError:
            ocr_result = ocr_engine.ocr(crop)
        text = extract_ocr_text(ocr_result)
        text = normalize_japanese(text)
        results.append(text)
    return results


def extract_ocr_text(ocr_result) -> str:
    if not ocr_result:
        return ""
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


def is_progressive_text(previous: str, current: str) -> bool:
    prev = compact_for_compare(previous)
    curr = compact_for_compare(current)
    if len(prev) < 3 or len(curr) < 3:
        return False
    shorter, longer = (prev, curr) if len(prev) <= len(curr) else (curr, prev)
    return longer.startswith(shorter) or shorter in longer


def merge_progressive_segments(segments: List[Segment]) -> List[Segment]:
    merged: List[Segment] = []
    for segment in segments:
        if is_noise_text(segment.ja):
            if merged:
                merged[-1].end = segment.end
            continue

        if merged and is_progressive_text(merged[-1].ja, segment.ja):
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
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY。请在环境变量或项目 .env 中设置，或显式使用 --no-translate。")

    system_prompt = (
        "Translate Japanese game dialogue into natural Simplified Chinese.\n"
        "Rules:\n"
        "- Preserve character names.\n"
        "- Preserve honorific meaning.\n"
        "- Use natural game localization style.\n"
        "- Avoid machine-translation tone.\n"
        "- Output only the translated dialogue."
    )
    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=api_key)
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=text,
            )
            return response.output_text.strip()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return response['choices'][0]['message']['content'].strip()


def init_ocr_engine():
    if importlib.util.find_spec("paddle") is not None:
        try:
            from paddleocr import PaddleOCR

            try:
                return PaddleOCR(
                    lang="japan",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_rec_score_thresh=0.0,
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
    for name in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "SourceHanSansCN-Regular.otf",
        "NotoSansCJK-Regular.otf",
        "msyh.ttf",
    ]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
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


def render_frame(
    frame: np.ndarray,
    segment: Optional[Segment],
    overlay_box: Tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    style: RenderStyle,
) -> Image.Image:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image).convert("RGBA")
    draw = ImageDraw.Draw(pil, "RGBA")
    x, y, w, h = overlay_box
    if segment and segment.zh:
        draw.rounded_rectangle(
            [x, y, x + w, y + h],
            radius=style.box_radius,
            fill=(*style.box_color, style.box_opacity),
        )
        lines = wrap_text(segment.zh, font, w - 32)
        total_height = sum(text_size(font, line)[1] for line in lines) + 12 * (len(lines) - 1)
        current_y = y + (h - total_height) // 2
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
        rendered = render_frame(frame, segment, overlay_box, font, style)
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
    if not args.no_translate and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("未设置 OPENAI_API_KEY，不能调用 OpenAI API 翻译。请设置环境变量/项目 .env，或显式加 --no-translate。")

    print(f"项目目录: {WORK_DIR}")

    fps, width, height, duration = analyze_video(input_video)
    boxes = estimate_boxes(width, height)
    print(f"视频分析: fps={fps:.2f}, 分辨率={width}x{height}, 时长={duration:.2f}s")
    print(f"OCR 区域: {boxes['ocr_box']}")
    print(f"覆盖区域: {boxes['overlay_box']}")

    samples = sample_frames(input_video, args.sample_frame_interval, fps, boxes['ocr_box'])
    print(f"已按每 {args.sample_frame_interval} 帧采样，得到 {len(samples)} 帧用于对话变化检测")

    segments, changed_samples = detect_dialogue_segments(samples)
    if segments:
        segments[-1].end = duration
    print(f"检测到 {len(segments)} 个对话片段")

    ocr_engine = init_ocr_engine()

    crops = [sample["crop"] for sample in changed_samples]
    raw_texts = ocr_dialogue(crops, ocr_engine)
    unique_text_count = len(deduplicate_texts(raw_texts))
    print(f"OCR 调用 {len(raw_texts)} 次，连续去重后 {unique_text_count} 条文本")

    translate_enabled = bool(os.environ.get("OPENAI_API_KEY")) and not args.no_translate
    if args.no_translate:
        print("警告: 指定了 --no-translate，将使用 OCR 原文生成字幕和视频。")
    for segment, text in zip(segments, raw_texts):
        segment.ja = text
    before_merge_count = len(segments)
    segments = merge_progressive_segments(segments)
    print(f"递进文本合并后 {len(segments)} 个字幕片段，过滤/合并 {before_merge_count - len(segments)} 个片段")

    translate_count = assign_translations(segments, [segment.ja for segment in segments], translate_enabled, args.openai_model)
    print(f"翻译调用次数: {translate_count}")

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
