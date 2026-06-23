# 日语视频翻译工作流

主要是将 youtube 上闪彩手游剧情视频自动翻译为简体中文，生成时间轴和字幕。

## 前置依赖

- Python 3.8+
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（日语 OCR）
- [FFmpeg](https://ffmpeg.org/)（视频分析 + H.264 压缩，可选但强烈推荐）
- DeepSeek API Key（翻译）

```bash
pip install -r requirements.txt
```

### 安装 FFmpeg

| 平台                 | 命令                         |
| -------------------- | ---------------------------- |
| macOS                | `brew install ffmpeg`        |
| Windows (winget)     | `winget install Gyan.FFmpeg` |
| Windows (Scoop)      | `scoop install ffmpeg`       |
| Windows (Chocolatey) | `choco install ffmpeg`       |

安装后确认：`ffmpeg -version && ffprobe -version`

未安装 FFmpeg 时，脚本回退到 OpenCV 编码。

## 快速开始

```bash
# 1. 放入视频
cp your_video.mp4 input/p1.mp4

# 2. 设置 API Key（二选一）
#    方式 A: 环境变量
     Windows: set DEEPSEEK_API_KEY=sk-xxx
     macOS/Linux: export DEEPSEEK_API_KEY=sk-xxx

#    方式 B: 项目根目录创建 .env
echo DEEPSEEK_API_KEY=sk-xxx > .env

# 3. 运行（首次运行会自动弹出标定窗口）
python translate_workflow.py --input input/p1.mp4
```

## 完整工作流（两步法）

```
首次运行                      校对                        重新渲染
───────────                ───────────                 ──────────────
OCR + 翻译                  编辑 zh 字段               只渲染不改动文本
  │                            │                          │
  ▼                            ▼                          ▼
segments.json ──────▶ 人工校对 segments.json ──────▶ final_cn.mp4
subtitles.ass                                   subtitles.ass
final_cn.mp4                                    subtitles_ja.ass
```

```bash
# 第一步：生成 segments.json
python translate_workflow.py --input input/p1.mp4

# 第二步：打开 work/p1/output/segments.json，编辑 zh 字段校对译文

# 第三步：跳过 OCR + 翻译，直接用校对后的文本重新渲染
python translate_workflow.py --input input/p1.mp4 --load-segments
```

`--load-segments` 从 `segments.json` 读取 `ja` / `zh`，跳过采样、OCR、翻译，直接生成字幕和视频。

## 命令行参数

### 核心流程

| 参数                      | 默认值              | 说明                                           |
| ------------------------- | ------------------- | ---------------------------------------------- |
| `--input`                 | `input/input.mp4`   | 输入视频路径                                   |
| `--no-translate`          | 关                  | 跳过翻译，OCR 原文作为字幕（快速预览）         |
| `--load-segments`         | 关                  | 从 segments.json 加载校对后文本，跳过 OCR+翻译 |
| `--model`                 | `deepseek-v4-flash` | DeepSeek 翻译模型                              |
| `--sample-frame-interval` | `15`                | 每隔多少帧采样一次                             |

### 对话检测

| 参数              | 默认值 | 说明                                           |
| ----------------- | ------ | ---------------------------------------------- |
| `--use-mad`       | 关     | 启用 MAD 像素差异检测（替代默认 OCR 文本分组） |
| `--mad-threshold` | `10.0` | MAD 差异阈值，越小越敏感                       |

### 渲染样式

| 参数                | 默认值    | 说明                                  |
| ------------------- | --------- | ------------------------------------- |
| `--box-color`       | `#808080` | 对话框背景色 (#RRGGBB)                |
| `--box-opacity`     | `230`     | 对话框透明度 (0–255)                  |
| `--text-color`      | `#87CEEB` | 文字颜色 (#RRGGBB)                    |
| `--stroke-color`    | `#FFFFFF` | 文字描边色 (#RRGGBB)                  |
| `--stroke-width`    | `1`       | 描边宽度                              |
| `--font-size`       | `42`      | 渲染字号                              |
| `--box-radius`      | `2`       | 对话框圆角半径                        |
| `--no-streaming`    | 关        | 禁用逐字流式显示（打字机效果）        |
| `--streaming-speed` | `15`      | 流式速度（字/秒），时长 = 字数 ÷ 速度 |

### 视频输出

| 参数                | 默认值   | 说明                                                 |
| ------------------- | -------- | ---------------------------------------------------- |
| `--crf`             | `23`     | H.264 压缩质量；`18` 高质量大文件，`28` 低质量小文件 |
| `--preset`          | `medium` | x264 编码预设；`slow` 换更小体积                     |
| `--keep-temp-video` | 关       | 保留 OpenCV 临时渲染视频                             |

## 两条检测管线

### 默认：OCR 文本相似度分组

```
采样(每 N 帧) → 全量 OCR → 字符重叠率分组 → 长段拆分(>5s) → 递进合并 → 翻译
```

- 对所有采样帧做 OCR，相邻帧文本相似则归为同一对话
- 分组后用日语字符纯度打分，选最佳 OCR 结果
- 优点：分组精确，OCR 噪声容错好
- 缺点：OCR 调用量大

### `--use-mad`：MAD 像素差异检测

```
采样 → MAD 灰度比较 → 差异 > 阈值则新段 → OCR 边界帧 → 多帧择优 → 长段拆分 → 递进合并 → 翻译
```

- 比较相邻采样帧的 OCR 区域像素差异
- 差异超过阈值则认为对话切换
- 每个段 OCR 首/中/尾三帧，取最高分
- 优点：OCR 调用少，适合长视频
- 缺点：阈值需调校，依赖背景稳定

两条管线共享相同的后处理：**递进文本合并 → DeepSeek 翻译(空返回重试 3 次) → 渲染**。

## 配置

### `.env` — API Key

```
DEEPSEEK_API_KEY=sk-你的密钥
```

### `boxes.json` — 对话框区域

首次运行时会自动弹出标定窗口，无需手动指定。坐标保存在 `work/<视频名>/` 目录下，每个视频独立管理：

```json
{
  "ocr_box": [192, 874, 1051, 108],
  "overlay_box": [192, 885, 1051, 146],
  "width": 1440,
  "height": 1080
}
```

- 存在 → 自动加载，分辨率不匹配时警告
- 不存在 → 自动弹出交互式标定窗口
- 删除后重新运行即可重新标定

### `segments.json` — 翻译片段

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "ja": "ふあ",
      "zh": "呼啊——"
    }
  ]
}
```

编辑 `zh` 字段即可校对译文，然后 `--load-segments` 重新渲染。

## 目录结构

```
translator_syanimasu/
├── translate_workflow.py
├── decorate_workflow.py      # 装饰叠加脚本
├── .env                       # API Key
├── README.md
├── requirements.txt
├── input/
│   └── p1.mp4                 # 输入视频
└── work/
    └── p1/
        ├── boxes.json             # 翻译标定坐标
        ├── decor_boxes.json       # 装饰标定坐标（多区域命名）
        └── output/
            ├── segments.json      # 翻译片段（校对编辑此文件）
            ├── decorations.json   # 装饰数据（用户手写）
            ├── subtitles.ass      # 中文字幕
            ├── subtitles_ja.ass   # 日语字幕
            ├── final_cn.mp4       # 翻译成品视频
            └── final_decorated.mp4 # 装饰成品视频
```

每个输入视频使用独立的 `work/<视频名>/` 目录，互不干扰。

## 常用场景

```bash
# 快速测试 OCR 效果（不调 API）
python translate_workflow.py --input input/p1.mp4 --no-translate

# 校对后重新渲染
python translate_workflow.py --input input/p1.mp4 --load-segments

# MAD 管线 + 调低阈值
python translate_workflow.py --input input/p1.mp4 --use-mad --mad-threshold 5

# 自定义样式
python translate_workflow.py --input input/p1.mp4 \
  --box-color "#1E293B" --box-opacity 210 \
  --text-color "#F8FAFC" --stroke-color "#0F172A" \
  --font-size 46 --box-radius 8

# 高质量输出
python translate_workflow.py --input input/p1.mp4 --crf 18 --preset slow


# 密集采样（适合快节奏对话）
python translate_workflow.py --input input/p1.mp4 --sample-frame-interval 5
```

## 装饰叠加（decorate_workflow.py）

独立脚本，将用户自定义的文本框/图片叠加到视频上。与翻译管线互不干扰。

### 两步法

```
标定                              合成
───────────                    ─────────────────────
参考图 → 框选区域并命名           视频 + decor_boxes.json
  │                                + decorations.json
  ▼                                  │
decor_boxes.json ──────────▶  final_decorated.mp4
```

### 快速开始

```bash
# 第一步：标定（从视频抽参考帧）
python decorate_workflow.py --calibrate --input input/p1.mp4

# 或指定参考图片
python decorate_workflow.py --calibrate --input input/p1.mp4 --time 1:30

# 第二步：手写 work/<项目>/output/decorations.json

# 第三步：合成
python decorate_workflow.py --input input/p1.mp4
```

### decor_boxes.json（标定生成）

```json
{
  "width": 1920,
  "height": 1080,
  "boxes": {
    "speaker_tag": {
      "type": "text",
      "box": [80, 40, 320, 64],
      "style": {}
    },
    "dialogue": {
      "type": "text",
      "box": [120, 820, 1680, 180],
      "style": {"font_size": 46}
    }
  }
}
```

每个 box 可指定独立的 `style`，缺省继承 CLI 全局参数。

### decorations.json（用户手写）

```json
{
  "entries": [
    {
      "box": "speaker_tag",
      "start": 0.0, "end": 5.0,
      "type": "text",
      "content": "羽那"
    },
    {
      "box": "dialogue",
      "start": 0.0, "end": 5.0,
      "type": "text",
      "content": "呼啊——"
    }
  ]
}
```

同一时间戳可有多条 entry 指向不同 box，同时渲染。`type` 支持 `"text"` 和 `"image"`。

### 命令行参数

| 参数              | 默认值                              | 说明                           |
| ----------------- | ----------------------------------- | ------------------------------ |
| `--input`         | 无（合成模式必需）                  | 输入视频                       |
| `--calibrate`     | 关                                  | 进入交互式标定模式             |
| `--image`         | 无                                  | 标定参考图（也可用 --time 从视频抽帧） |
| `--time`          | 30% 位置                            | 从视频指定时间点抽帧（`1:30` 或 `90.5`） |
| `--boxes`         | `work/<项目>/decor_boxes.json`      | boxes 文件路径                 |
| `--decorations`   | `work/<项目>/output/decorations.json` | decorations 文件路径         |
| `--output`        | `work/<项目>/output/final_decorated.mp4` | 输出视频                 |

样式参数与 `translate_workflow.py` 共用同一套：`--font-size`、`--box-color`、`--box-opacity`、`--text-color`、`--stroke-color`、`--stroke-width`、`--box-radius`、`--streaming-speed`、`--no-streaming`。

## 字体

自动按以下优先级查找：

| 优先级 | 来源     | 字体                                                         |
| ------ | -------- | ------------------------------------------------------------ |
| 1      | Windows  | 微软雅黑 → Meiryo → MS Gothic → Yu Gothic                    |
| 2      | macOS    | Hiragino Sans → ヒラギノ角ゴシック → STHeiti → Arial Unicode |
| 3      | 项目目录 | SourceHanSansCN / NotoSansCJK / msyh                         |

未找到时可能回退 PIL 默认字体（文字极小）。
