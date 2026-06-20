# Visual Novel Video Translation Workflow

这是一个用于将本地日语视觉小说 / 手游剧情视频翻译为简体中文的工作流。

## 目标

- 保留原始 UI、角色立绘和名牌区域
- 默认不翻译名牌区域
- 仅识别并替换对话文本
- 自动检测对话变化并生成字幕时间轴
- 输出简体中文字幕、日语 OCR 字幕和覆盖渲染后的视频
- 不覆盖 `input/` 中的原始视频

## 项目结构

```text
Translator/
  README.md
  requirements.txt
  translate_workflow.py
  input/                  # 手动放入本地视频文件
  work/
    <video_name>/
      output/
        final_cn.mp4      # 覆盖渲染后的视频
        subtitles.ass     # 中文字幕
        subtitles_ja.ass  # 日语 OCR 字幕
```

每个输入视频都会使用自己的 `work/<video_name>/output/` 目录，避免不同视频的结果混在一起。

## 安装依赖

```bash
pip install -r requirements.txt
```

建议另外安装系统 `ffmpeg`，用于更准确地读取媒体信息和更高质量的视频编码。

macOS:

```bash
brew install ffmpeg
```

Windows 推荐用 `winget`：

```powershell
winget install Gyan.FFmpeg
```

如果你使用 Chocolatey：

```powershell
choco install ffmpeg
```

如果你使用 Scoop：

```powershell
scoop install ffmpeg
```

安装后重新打开终端，确认命令可用：

```bash
ffmpeg -version
ffprobe -version
```

如果 Windows 提示找不到命令，需要把 `ffmpeg.exe` 和 `ffprobe.exe` 所在的 `bin` 目录加入系统 `Path` 环境变量，然后重新打开终端。

如果没有安装 `ffmpeg/ffprobe`，脚本会回退到 OpenCV；回退编码的视频不包含原始音频。

安装 `ffmpeg` 后，脚本会先渲染临时视频，再用 `libx264` 压缩并复制原视频音频。可以用 `--crf` 控制文件大小：

- `--crf 18`: 质量高，文件较大
- `--crf 23`: 默认平衡
- `--crf 26` 到 `--crf 28`: 文件更小，画质略降

## 放入视频

只支持本地文件输入。请把视频文件直接放入 `input/`，例如：

```text
input/input.mp4
```

或保留任意文件名，然后运行时通过 `--input` 指定。

## 运行

真实翻译需要设置 OpenAI API key：

```bash
export OPENAI_API_KEY="你的 key"
python translate_workflow.py --input input/input.mp4
```

也可以在项目根目录创建 `.env`：

```env
OPENAI_API_KEY=你的 key
```

如果只想测试 OCR、字幕和覆盖渲染，不调用 OpenAI API：

```bash
python translate_workflow.py --input input/input.mp4 --no-translate
```

## 常用参数

```bash
python translate_workflow.py \
  --input input/input.mp4 \
  --box-color "#1E293B" \
  --box-opacity 210 \
  --text-color "#F8FAFC" \
  --stroke-color "#0F172A" \
  --stroke-width 1 \
  --font-size 46 \
  --box-radius 8 \
  --crf 23 \
  --preset medium
```

参数说明：

- `--input`: 输入视频路径，默认 `input/input.mp4`
- `--no-translate`: 跳过 OpenAI 翻译，用 OCR 原文生成预览
- `--openai-model`: OpenAI 翻译模型，默认 `gpt-5`
- `--sample-frame-interval`: 每隔多少帧检测一次对话变化，默认 `30`
- `--box-color`: 覆盖对话框背景色，格式 `#RRGGBB`
- `--box-opacity`: 覆盖对话框透明度，范围 `0-255`
- `--text-color`: 渲染文字颜色
- `--stroke-color`: 文字描边颜色
- `--stroke-width`: 文字描边宽度
- `--font-size`: 渲染字号
- `--box-radius`: 对话框圆角半径
- `--crf`: ffmpeg H.264 压缩质量，数值越小质量越高、文件越大，默认 `23`
- `--preset`: ffmpeg x264 压缩速度/压缩率，默认 `medium`；可用 `slow` 换更小体积

## 输出

运行完成后，结果会出现在对应视频的输出目录：

```text
work/<video_name>/output/final_cn.mp4
work/<video_name>/output/subtitles.ass
work/<video_name>/output/subtitles_ja.ass
```

其中 `subtitles_ja.ass` 是 OCR 识别出的日语字幕，建议保留，便于检查 OCR 质量和后续修正翻译。
