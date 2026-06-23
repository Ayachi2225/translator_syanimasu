"""
decorate_workflow.py —— 多区域装饰叠加脚本

将用户自定义的文本框 / 图片叠加到视频上。
与 translate_workflow.py 独立，但复用其工具函数。

两种模式：
  --calibrate  交互式标定多个命名区域，保存到 decor_boxes.json
  （默认）      读取 decor_boxes.json + decorations.json 合成视频

用法：
  # 标定（用参考图）
  python decorate_workflow.py --calibrate --image ref.png --input input/p1.mp4

  # 标定（从视频抽帧）
  python decorate_workflow.py --calibrate --input input/p1.mp4

  # 合成
  python decorate_workflow.py --input input/p1.mp4
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 复用 translate_workflow 中的工具函数
from translate_workflow import (
    analyze_video,
    clamp_opacity,
    compress_with_ffmpeg,
    configure_environment,
    load_dotenv,
    load_font,
    parse_hex_color,
    safe_project_name,
    streaming_text,
    text_size,
    wrap_text,
)

# ---------------------------------------------------------------------------
# 全局路径常量
# ---------------------------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
INPUT_DIR = os.path.join(BASE_DIR, "input")
WORK_ROOT = os.path.join(BASE_DIR, "work")

# 默认值（运行时由 configure_work_paths 覆盖）
WORK_DIR = os.path.join(WORK_ROOT, "default")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")

# 默认文件路径
DECOR_BOXES_JSON = os.path.join(WORK_DIR, "decor_boxes.json")
DECORATIONS_JSON = os.path.join(OUTPUT_DIR, "decorations.json")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "final_decorated.mp4")
TEMP_VIDEO = os.path.join(OUTPUT_DIR, "rendered_temp.mp4")

# 默认样式
DEFAULT_FONT_SIZE = 42
DEFAULT_BOX_COLOR = (128, 128, 128)    # #808080
DEFAULT_BOX_OPACITY = 230
DEFAULT_TEXT_COLOR = (135, 206, 235)   # #87CEEB
DEFAULT_STROKE_COLOR = (255, 255, 255)  # #FFFFFF
DEFAULT_STROKE_WIDTH = 1
DEFAULT_BOX_RADIUS = 2
DEFAULT_CHARS_PER_SEC = 0  # 默认禁用流式


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

def configure_work_paths(video_path: str) -> None:
    """根据输入视频路径设置工作目录及所有输出路径。"""
    global WORK_DIR, OUTPUT_DIR, DECOR_BOXES_JSON, DECORATIONS_JSON
    global OUTPUT_VIDEO, TEMP_VIDEO

    WORK_DIR = os.path.join(WORK_ROOT, safe_project_name(video_path))
    OUTPUT_DIR = os.path.join(WORK_DIR, "output")
    DECOR_BOXES_JSON = os.path.join(WORK_DIR, "decor_boxes.json")
    DECORATIONS_JSON = os.path.join(OUTPUT_DIR, "decorations.json")
    OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "final_decorated.mp4")
    TEMP_VIDEO = os.path.join(OUTPUT_DIR, "rendered_temp.mp4")


def ensure_dirs() -> None:
    """创建所有必要的目录。"""
    for path in [CACHE_DIR, WORK_ROOT, WORK_DIR, INPUT_DIR, OUTPUT_DIR]:
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多区域装饰叠加 —— 标定命名区域或合成视频"
    )
    parser.add_argument("--input", default=None, help="输入视频路径")
    parser.add_argument(
        "--calibrate", action="store_true",
        help="进入交互式标定模式",
    )
    parser.add_argument(
        "--image", default=None,
        help="标定用的参考图片（不传则从视频 30%% 位置抽帧）",
    )
    parser.add_argument(
        "--boxes", default=None,
        help="decor_boxes.json 路径（默认 work/<项目>/decor_boxes.json）",
    )
    parser.add_argument(
        "--decorations", default=None,
        help="decorations.json 路径（默认 work/<项目>/output/decorations.json）",
    )
    parser.add_argument(
        "--output", default=None,
        help="输出视频路径（默认 work/<项目>/output/final_decorated.mp4）",
    )

    # 全局样式
    parser.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE)
    parser.add_argument("--box-color", default="#808080")
    parser.add_argument("--box-opacity", type=int, default=DEFAULT_BOX_OPACITY)
    parser.add_argument("--text-color", default="#87CEEB")
    parser.add_argument("--stroke-color", default="#FFFFFF")
    parser.add_argument("--stroke-width", type=int, default=DEFAULT_STROKE_WIDTH)
    parser.add_argument("--box-radius", type=int, default=DEFAULT_BOX_RADIUS)
    parser.add_argument(
        "--streaming-speed", type=float, default=DEFAULT_CHARS_PER_SEC,
        help="流式速度（字/秒），0 = 禁用",
    )
    parser.add_argument("--no-streaming", action="store_true", help="强制禁用流式")

    # 编码
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--keep-temp-video", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 全局样式构建
# ---------------------------------------------------------------------------

def build_global_style(args: argparse.Namespace) -> Dict[str, Any]:
    """将 CLI 参数组装为全局默认样式字典。"""
    chars_per_sec = 0 if args.no_streaming else args.streaming_speed
    return {
        "font_size": max(1, args.font_size),
        "box_color": parse_hex_color(args.box_color),
        "box_opacity": clamp_opacity(args.box_opacity),
        "text_color": parse_hex_color(args.text_color),
        "stroke_color": parse_hex_color(args.stroke_color),
        "stroke_width": max(0, args.stroke_width),
        "box_radius": max(0, args.box_radius),
        "chars_per_sec": chars_per_sec,
    }


def resolve_box_style(box_def: Dict, global_style: Dict) -> Dict:
    """将逐 box 的 style 覆盖合并到全局默认样式上。"""
    style = dict(global_style)
    overrides = box_def.get("style", {}) or {}
    for k, v in overrides.items():
        if v is not None:
            style[k] = v
    # 对颜色字段特殊处理：支持 hex 字符串
    for color_key in ("box_color", "text_color", "stroke_color"):
        if isinstance(style.get(color_key), str):
            style[color_key] = parse_hex_color(style[color_key])
    return style


# ---------------------------------------------------------------------------
# JSON 加载 / 保存
# ---------------------------------------------------------------------------

def load_boxes(path: str) -> Dict[str, Any]:
    """加载 decor_boxes.json。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到标定文件: {path}\n"
            f"请先运行: python decorate_workflow.py --calibrate --input <视频>"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "boxes" not in data:
        raise ValueError(f"{path} 格式错误：缺少 'boxes' 字段")
    if not isinstance(data["boxes"], dict):
        raise ValueError(f"{path} 格式错误：'boxes' 必须是 dict")
    return data


def save_boxes(boxes_data: Dict[str, Any], path: str) -> None:
    """保存 decor_boxes.json。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(boxes_data, f, indent=2, ensure_ascii=False)
    print(f"标定已保存到 {path}")


def load_decorations(path: str, boxes_def: Dict[str, Dict]) -> List[Dict]:
    """加载 decorations.json 并校验引用的 box 名。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到装饰数据: {path}\n"
            f"请创建一个 decorations.json，格式参考 README。"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"{path} 格式错误：'entries' 必须是数组")

    # 收集所有引用的 box 名
    box_names = set(boxes_def.keys())
    unknown = set()
    for entry in entries:
        name = entry.get("box", "")
        if name and name not in box_names:
            unknown.add(name)

    if unknown:
        print(f"  警告: decorations.json 引用了未定义的 box: {unknown}")

    print(f"从 {path} 加载了 {len(entries)} 条装饰数据")
    return entries


# ---------------------------------------------------------------------------
# 标定模式
# ---------------------------------------------------------------------------

def extract_reference_frame(video_path: str) -> np.ndarray:
    """从视频 30% 位置抽取一帧作为标定参考图。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0
    target_time = duration * 0.3

    cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("无法从视频中抽取参考帧")
    return frame


def calibrate_interactive(image: np.ndarray) -> Dict[str, Any]:
    """交互式多区域标定。

    用户在图片上逐次框选区域并命名，按 ENTER 确认选区后在终端输入名称，
    按 ESC 结束标定循环。
    """
    height, width = image.shape[:2]
    boxes: Dict[str, Dict] = {}
    display = image.copy()
    box_count = 0

    print("\n=== 交互式标定 ===")
    print("  操作说明:")
    print("    1. 用鼠标框选一个矩形区域")
    print("    2. 按 ENTER 确认选区")
    print("    3. 在终端输入该区域的名称（如 'dialogue', 'speaker_tag'）")
    print("    4. 重复以上步骤添加更多区域")
    print("    5. 按 ESC 或在名称提示中输入 'done' 结束标定")
    print()

    while True:
        win_name = "decorate-calibrate"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        title = (
            f"Box #{box_count + 1} | ENTER=confirm | ESC=finish"
            if box_count == 0
            else f"Box #{box_count + 1} | ENTER=confirm | ESC=finish | 已标定: {', '.join(boxes.keys())}"
        )
        cv2.setWindowTitle(win_name, title)
        roi = cv2.selectROI(win_name, display, False)
        cv2.destroyAllWindows()

        x, y, w, h = roi
        if w == 0 and h == 0:
            print("标定结束（ESC）。")
            break

        if w <= 0 or h <= 0:
            print("  选区无效（宽高必须 > 0），请重试。")
            continue

        # 终端输入名称
        name = input(
            f"  为此区域命名 (Box #{box_count + 1})，"
            f"回车跳过 / 输入 'done' 结束: "
        ).strip()

        if not name:
            print("  跳过该区域（未命名）。")
            continue
        if name.lower() in ("done", "quit", "exit"):
            print("标定完成。")
            break

        if name in boxes:
            print(f"  名称 '{name}' 已存在，覆盖之前的定义。")

        boxes[name] = {
            "type": "text",
            "box": [int(x), int(y), int(w), int(h)],
            "style": {},
        }
        box_count += 1
        print(f"  已记录: '{name}' -> {boxes[name]['box']}")

        # 在展示图上绘制已确认的矩形
        cv2.rectangle(
            display, (int(x), int(y)), (int(x + w), int(y + h)),
            (0, 255, 0), 2,
        )
        cv2.putText(
            display, name, (int(x), int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

    return {"width": width, "height": height, "boxes": boxes}


def run_calibrate(args: argparse.Namespace) -> None:
    """执行标定模式：加载参考图 → 交互标定 → 保存。"""
    # 获取参考图像
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"参考图片不存在: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            raise RuntimeError(f"无法读取图片: {args.image}")
        print(f"从图片加载参考帧: {args.image}")
    elif args.input:
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"视频不存在: {args.input}")
        image = extract_reference_frame(args.input)
        print(f"从视频抽取参考帧: {args.input}")
    else:
        raise RuntimeError("标定模式需要 --input <视频> 或 --image <图片>")

    result = calibrate_interactive(image)

    if not result["boxes"]:
        print("未标定任何区域，退出。")
        return

    # 确定保存路径
    if args.input:
        configure_work_paths(args.input)
        ensure_dirs()
        boxes_path = args.boxes or DECOR_BOXES_JSON
    else:
        boxes_path = args.boxes or os.path.join(
            os.path.dirname(os.path.abspath(args.image)), "decor_boxes.json"
        )

    save_boxes(result, boxes_path)
    print(f"  Box 总数: {len(result['boxes'])}")
    for name, bdef in result["boxes"].items():
        print(f"    {name}: type={bdef['type']}, box={bdef['box']}")


# ---------------------------------------------------------------------------
# 合成模式
# ---------------------------------------------------------------------------

def find_active_entries(
    entries: List[Dict],
    timestamp: float,
) -> List[Dict]:
    """返回当前时间点所有活跃的装饰条目，附带 elapsed / seg_duration。"""
    active = []
    for entry in entries:
        start = entry.get("start", 0.0)
        end = entry.get("end")
        if end is None:
            end = float("inf")
        if start <= timestamp < end:
            item = dict(entry)
            item["elapsed"] = timestamp - start
            item["seg_duration"] = max(end - start, 0.001)
            active.append(item)
    return active


def render_frame_multi(
    frame: np.ndarray,
    active_entries: List[Dict],
    boxes_def: Dict[str, Dict],
    fonts: Dict[int, ImageFont.FreeTypeFont],
    global_style: Dict,
) -> Image.Image:
    """对单帧绘制所有活跃的装饰条目。

    参数
    ----
    frame: BGR numpy 数组
    active_entries: find_active_entries() 的返回
    boxes_def: decor_boxes.json 的 "boxes" 部分
    fonts: {font_size: ImageFont} 缓存
    global_style: 全局默认样式
    """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image).convert("RGBA")
    draw = ImageDraw.Draw(pil, "RGBA")

    for entry in active_entries:
        box_name = entry.get("box", "")
        box_def = boxes_def.get(box_name)
        if not box_def:
            continue

        style = resolve_box_style(box_def, global_style)
        bx, by, bw, bh = box_def["box"]
        content_type = entry.get("type", box_def.get("type", "text"))

        if content_type == "text":
            content = entry.get("content", "")
            if not content:
                continue

            font_size = style["font_size"]
            if font_size not in fonts:
                fonts[font_size] = load_font(font_size)
            font = fonts[font_size]

            # 流式效果
            display_text = content
            chars_per_sec = style.get("chars_per_sec", 0)
            if chars_per_sec > 0:
                display_text = streaming_text(
                    content,
                    entry.get("elapsed", 0),
                    entry.get("seg_duration", 5.0),
                    chars_per_sec,
                )

            # 背景框
            draw.rounded_rectangle(
                [bx, by, bx + bw, by + bh],
                radius=style["box_radius"],
                fill=(*style["box_color"], style["box_opacity"]),
            )

            # 文字
            lines = wrap_text(display_text, font, bw - 32)
            if not lines:
                continue
            total_h = sum(text_size(font, line)[1] for line in lines) + 12 * (len(lines) - 1)
            cy = by + max(0, (bh - total_h)) // 3
            for line in lines:
                lw, lh = text_size(font, line)
                lx = bx + 24
                draw.text(
                    (lx, cy), line,
                    font=font,
                    fill=style["text_color"],
                    stroke_width=style["stroke_width"],
                    stroke_fill=style["stroke_color"],
                )
                cy += lh + 12

        elif content_type == "image":
            img_path = entry.get("content", "")
            if not img_path or not os.path.exists(img_path):
                if img_path:
                    print(f"  警告: 图片不存在: {img_path}")
                continue
            overlay_img = Image.open(img_path).convert("RGBA")
            overlay_img = overlay_img.resize((bw, bh), Image.LANCZOS)
            pil.paste(overlay_img, (bx, by), overlay_img)

    return pil.convert("RGB")


def render_video_with_opencv_multi(
    video_path: str,
    output_path: str,
    entries: List[Dict],
    boxes_def: Dict[str, Dict],
    fps: float,
    global_style: Dict,
) -> None:
    """逐帧渲染，将装饰叠加到视频上。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("无法获取视频尺寸")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "OpenCV 无法创建输出 mp4，请安装 ffmpeg: brew install ffmpeg"
        )

    fonts: Dict[int, ImageFont.FreeTypeFont] = {}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = idx / fps
        active = find_active_entries(entries, timestamp)
        rendered = render_frame_multi(frame, active, boxes_def, fonts, global_style)
        writer.write(cv2.cvtColor(np.array(rendered), cv2.COLOR_RGB2BGR))
        idx += 1

        if idx % 300 == 0:
            print(f"  渲染进度: {idx} 帧 ({timestamp:.1f}s)")

    writer.release()
    cap.release()
    print(f"  渲染完成: {idx} 帧")


def run_compose(args: argparse.Namespace) -> None:
    """执行合成模式：加载配置 → 渲染 → 编码。"""
    # 路径
    boxes_path = args.boxes or DECOR_BOXES_JSON
    decorations_path = args.decorations or DECORATIONS_JSON
    output_path = args.output or OUTPUT_VIDEO

    # 加载
    boxes_data = load_boxes(boxes_path)
    boxes_def = boxes_data["boxes"]
    print(f"加载 {len(boxes_def)} 个 box: {list(boxes_def.keys())}")

    entries = load_decorations(decorations_path, boxes_def)

    # 视频分析
    fps, width, height, duration = analyze_video(args.input)
    print(f"视频: {width}x{height}, {fps:.2f}fps, {duration:.2f}s")

    # 分辨率校验
    saved_w = boxes_data.get("width")
    saved_h = boxes_data.get("height")
    if saved_w and saved_h and (saved_w != width or saved_h != height):
        print(
            f"  警告: decor_boxes.json 标定分辨率 {saved_w}x{saved_h} "
            f"与视频 {width}x{height} 不匹配"
        )

    # 全局样式
    global_style = build_global_style(args)
    print(
        f"  全局样式: font_size={global_style['font_size']}, "
        f"box_color={global_style['box_color']}, "
        f"text_color={global_style['text_color']}"
    )

    # 渲染 + 编码
    has_ffmpeg = shutil.which("ffmpeg") is not None
    temp_path = TEMP_VIDEO if has_ffmpeg else output_path

    print("开始渲染...")
    render_video_with_opencv_multi(
        args.input, temp_path, entries, boxes_def, fps, global_style,
    )

    if has_ffmpeg:
        print("正在用 ffmpeg 压缩 + 合并音频...")
        ok = compress_with_ffmpeg(temp_path, args.input, output_path, args.crf, args.preset)
        if ok:
            print(f"输出: {output_path}")
            if not args.keep_temp_video and os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            print(f"ffmpeg 失败，使用未压缩视频: {temp_path}")
    else:
        print(f"未检测到 ffmpeg，使用 OpenCV 编码: {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_environment()
    load_dotenv()
    args = parse_args()

    if args.calibrate:
        run_calibrate(args)
        return

    # 合成模式需要 input
    if not args.input:
        print("错误: 合成模式需要 --input <视频>")
        print("或使用 --calibrate 进入标定模式")
        sys.exit(1)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"视频不存在: {args.input}")

    configure_work_paths(args.input)
    ensure_dirs()
    print(f"项目目录: {WORK_DIR}")

    run_compose(args)


if __name__ == "__main__":
    main()
