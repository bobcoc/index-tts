# LipSync CLI 使用文档

独立的口型同步视频生成命令行工具，支持 LatentSync 和 Wav2Lip 两种引擎。

## 功能特点

- 直接使用已有的音频和视频生成口型同步视频
- 无需加载 TTS 模型，节省显存
- 支持视频/音频时长自动对齐
- 支持低显存模式 (256px)
- 支持 LatentSync 和 Wav2Lip 两种引擎

## 环境要求

### 依赖工具
- **ffmpeg**: 用于视频/音频处理
- **ffprobe**: 用于获取媒体时长信息

### LatentSync 引擎（推荐）
- 需要单独安装 LatentSync 项目
- 建议使用独立的 conda 环境
- 显存需求：
  - 标准模式 (512px): ~20GB
  - 低显存模式 (256px): ~8GB

### Wav2Lip 引擎
- 需要单独安装 Wav2Lip 项目
- 显存需求较低

## 参数说明

### 必需参数

| 参数 | 简写 | 说明 |
|------|------|------|
| `--video` | `-v` | 输入视频路径 |
| `--audio` | `-a` | 输入音频路径 |
| `--output` | `-o` | 输出视频路径 |

### 引擎选择

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--engine` | `latentsync` | 口型同步引擎，可选 `latentsync` 或 `wav2lip` |

### LatentSync 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--latentsync_dir` | - | LatentSync 仓库路径（必需） |
| `--latentsync_python` | 系统 python | LatentSync Python 解释器路径 |
| `--low_vram` | False | 低显存模式，使用 256px 分辨率 |
| `--inference_steps` | 20 | 推理步数，越高质量越好但速度越慢 |
| `--guidance_scale` | 1.5 | 引导比例 |

### Wav2Lip 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--wav2lip_dir` | - | Wav2Lip 仓库路径（必需） |
| `--wav2lip_python` | 系统 python | Wav2Lip Python 解释器路径 |

### 视频/音频对齐

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--align_mode` | `none` | 时长不匹配时的对齐模式 |
| `--quiet` / `-q` | False | 静默模式，减少输出 |

## 对齐模式详解

当视频和音频时长不一致时，可使用 `--align_mode` 参数自动对齐：

### `speed` - 调整播放速度

| 情况 | 处理方式 |
|------|---------|
| 视频短于音频 | 加快音频播放速度以适配视频长度 |
| 视频长于音频 | 加快视频播放速度以适配音频长度 |

**适用场景**：时长差异较小（<30%），希望保持原始内容完整

### `extend_tail` - 从后部处理

| 情况 | 处理方式 |
|------|---------|
| 视频短于音频 | 从视频后半部分截取片段拼接到末尾 |
| 视频长于音频 | 从视频末尾截断多余部分 |

**适用场景**：视频开头有重要内容，希望保留开头

### `extend_head` - 从前部处理

| 情况 | 处理方式 |
|------|---------|
| 视频短于音频 | 从视频前半部分截取片段拼接到末尾 |
| 视频长于音频 | 从视频开头截断多余部分 |

**适用场景**：视频结尾有重要内容，希望保留结尾

### `none` - 不处理（默认）

不进行任何对齐处理，仅给出警告提示。

## 使用示例

### 基本用法

```bash
python tools/lipsync_cli.py \
    --video input.mp4 \
    --audio speech.wav \
    --output output.mp4 \
    --latentsync_dir ~/LatentSync \
    --latentsync_python ~/miniconda3/envs/latentsync/bin/python
```

### 低显存模式

适用于显存不足 20GB 的情况：

```bash
python tools/lipsync_cli.py \
    --video input.mp4 \
    --audio speech.wav \
    --output output.mp4 \
    --latentsync_dir ~/LatentSync \
    --latentsync_python ~/miniconda3/envs/latentsync/bin/python \
    --low_vram
```

### 视频短于音频 - 从后部补充

```bash
python tools/lipsync_cli.py \
    --video short_video.mp4 \
    --audio long_speech.wav \
    --output output.mp4 \
    --latentsync_dir ~/LatentSync \
    --latentsync_python ~/miniconda3/envs/latentsync/bin/python \
    --align_mode extend_tail
```

### 视频短于音频 - 加快音频速度

```bash
python tools/lipsync_cli.py \
    --video short_video.mp4 \
    --audio long_speech.wav \
    --output output.mp4 \
    --latentsync_dir ~/LatentSync \
    --latentsync_python ~/miniconda3/envs/latentsync/bin/python \
    --align_mode speed
```

### 视频长于音频 - 从后部截断

```bash
python tools/lipsync_cli.py \
    --video long_video.mp4 \
    --audio short_speech.wav \
    --output output.mp4 \
    --latentsync_dir ~/LatentSync \
    --latentsync_python ~/miniconda3/envs/latentsync/bin/python \
    --align_mode extend_tail
```

### 使用 Wav2Lip 引擎

```bash
python tools/lipsync_cli.py \
    --video input.mp4 \
    --audio speech.wav \
    --output output.mp4 \
    --engine wav2lip \
    --wav2lip_dir ~/Wav2Lip \
    --wav2lip_python ~/miniconda3/envs/wav2lip/bin/python
```

### 静默模式

减少输出信息：

```bash
python tools/lipsync_cli.py \
    --video input.mp4 \
    --audio speech.wav \
    --output output.mp4 \
    --latentsync_dir ~/LatentSync \
    --quiet
```

## 常见问题

### Q: 显存不足怎么办？

**A**: 使用 `--low_vram` 参数启用低显存模式，将分辨率从 512px 降到 256px，显存需求从 ~20GB 降到 ~8GB。

### Q: 视频比音频短怎么办？

**A**: 使用 `--align_mode` 参数：
- `extend_tail`: 从视频后部截取片段补充（推荐）
- `extend_head`: 从视频前部截取片段补充
- `speed`: 加快音频播放速度

### Q: 视频比音频长怎么办？

**A**: 使用 `--align_mode` 参数：
- `extend_tail`: 从视频后部截断（推荐）
- `extend_head`: 从视频前部截断
- `speed`: 加快视频播放速度

### Q: LatentSync 和 Wav2Lip 有什么区别？

| 特性 | LatentSync | Wav2Lip |
|------|-----------|---------|
| 质量 | 较高 | 中等 |
| 显存需求 | 高 (~20GB) | 低 (~4GB) |
| 速度 | 较慢 | 较快 |
| 分辨率 | 256px / 512px | 96px |

### Q: ONNX Runtime CUDA 警告怎么办？

如果看到类似以下警告：
```
Failed to load library libonnxruntime_providers_cuda.so
Applied providers: ['CPUExecutionProvider']
```

这只是人脸检测回退到 CPU 运行，**不影响最终结果**，可以忽略。主要的扩散模型仍然使用 GPU。

### Q: 如何提高生成质量？

- 使用标准模式（不加 `--low_vram`）
- 增加推理步数：`--inference_steps 30`
- 确保输入视频质量较高
- 确保输入视频中人脸清晰可见

## 输出示例

```
>> 音频时长: 285.37s
>> 视频时长: 164.67s
>> 视频较短: 164.67s < 音频 285.37s (差 120.70s)
>> 从视频后部 43.97s 截取 120.70s 补充
>> 拼接后视频时长: 285.41s
>> 运行 LatentSync...
   配置: configs/unet/stage2_512.yaml
   视频: /home/user/outputs/video.extended.mp4
   音频: /home/user/outputs/speech.wav
   输出: /home/user/outputs/output.mp4
>> 口型同步视频已保存: /home/user/outputs/output.mp4

✅ 完成! 输出: outputs/output.mp4
```
