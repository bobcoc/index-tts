#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LipSync Command Line Tool - 独立的口型同步视频生成工具

直接使用已生成的音频和视频进行口型同步，无需加载 TTS 模型。

用法:
    # 基本用法
    python tools/lipsync_cli.py --video input.mp4 --audio speech.wav --output output.mp4 \
        --latentsync_dir ~/LatentSync --latentsync_python ~/miniconda3/envs/latentsync/bin/python

    # 低显存模式 (256px)
    python tools/lipsync_cli.py --video input.mp4 --audio speech.wav --output output.mp4 \
        --latentsync_dir ~/LatentSync --latentsync_python ~/miniconda3/envs/latentsync/bin/python \
        --low_vram

    # 使用 Wav2Lip
    python tools/lipsync_cli.py --video input.mp4 --audio speech.wav --output output.mp4 \
        --wav2lip_dir ~/Wav2Lip --engine wav2lip
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get duration of {video_path}")
    return float(result.stdout.strip())


def get_audio_duration(audio_path: str) -> float:
    """获取音频时长（秒）"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get duration of {audio_path}")
    return float(result.stdout.strip())


def extend_video_if_needed(video_path: str, target_duration: float, output_path: str, verbose: bool = True) -> str:
    """
    如果视频时长不足，从视频后半部分截取片段来补充。
    
    Args:
        video_path: 原视频路径
        target_duration: 目标时长（秒）
        output_path: 输出视频路径
        verbose: 是否打印详细信息
    
    Returns:
        处理后的视频路径
    """
    video_duration = get_video_duration(video_path)
    
    if video_duration >= target_duration:
        # 视频足够长，直接截取需要的部分
        if abs(video_duration - target_duration) < 0.1:
            return video_path
        
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-t", str(target_duration),
            "-c:v", "libx264", "-c:a", "aac",
            output_path
        ]
        if verbose:
            print(f">> 视频时长足够，截取前 {target_duration:.2f}s")
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    
    # 视频不够长，需要从后面截取补充
    shortage = target_duration - video_duration
    
    if verbose:
        print(f">> 视频时长不足: {video_duration:.2f}s < {target_duration:.2f}s")
        print(f">> 需要补充 {shortage:.2f}s")
    
    # 计算从视频后面截取的起始位置
    # 策略：从视频的后半部分开始截取，避免重复开头
    if video_duration > shortage:
        # 从视频后面截取
        start_time = max(0, video_duration - shortage)
    else:
        # 视频太短，循环使用
        start_time = 0
    
    temp_dir = tempfile.mkdtemp(prefix="lipsync_extend_")
    
    try:
        # 截取补充片段
        extra_clip = os.path.join(temp_dir, "extra.mp4")
        cmd_extract = [
            "ffmpeg", "-y", "-i", video_path,
            "-ss", str(start_time),
            "-t", str(shortage),
            "-c:v", "libx264", "-c:a", "aac",
            extra_clip
        ]
        subprocess.run(cmd_extract, capture_output=True, check=True)
        
        if verbose:
            print(f">> 从 {start_time:.2f}s 截取 {shortage:.2f}s 补充片段")
        
        # 创建拼接文件列表
        concat_list = os.path.join(temp_dir, "concat.txt")
        with open(concat_list, "w") as f:
            f.write(f"file '{os.path.abspath(video_path)}'\n")
            f.write(f"file '{os.path.abspath(extra_clip)}'\n")
        
        # 拼接视频
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:v", "libx264", "-c:a", "aac",
            output_path
        ]
        subprocess.run(cmd_concat, capture_output=True, check=True)
        
        if verbose:
            final_duration = get_video_duration(output_path)
            print(f">> 拼接后视频时长: {final_duration:.2f}s")
        
        return output_path
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_latentsync(
    video_path: str,
    audio_path: str,
    output_path: str,
    latentsync_dir: str,
    python_executable: str = None,
    low_vram: bool = False,
    inference_steps: int = 20,
    guidance_scale: float = 1.5,
    verbose: bool = True
) -> bool:
    """
    运行 LatentSync 推理
    """
    latentsync_dir = Path(latentsync_dir)
    python_executable = python_executable or sys.executable
    
    # 选择配置
    if low_vram:
        config_efficient = latentsync_dir / "configs/unet/stage2_efficient.yaml"
        config_256 = latentsync_dir / "configs/unet/stage2.yaml"
        if config_efficient.exists():
            config = "configs/unet/stage2_efficient.yaml"
        elif config_256.exists():
            config = "configs/unet/stage2.yaml"
        else:
            config = "configs/unet/stage2_512.yaml"
        if verbose:
            print(f">> 使用低显存模式 (256px)")
    else:
        config_512 = latentsync_dir / "configs/unet/stage2_512.yaml"
        if config_512.exists():
            config = "configs/unet/stage2_512.yaml"
        else:
            config = "configs/unet/stage2.yaml"
    
    checkpoint = "checkpoints/latentsync_unet.pt"
    
    cmd = [
        python_executable,
        "-m", "scripts.inference",
        "--unet_config_path", str(latentsync_dir / config),
        "--inference_ckpt_path", str(latentsync_dir / checkpoint),
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--video_out_path", output_path,
        "--enable_deepcache",
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(latentsync_dir)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if verbose:
        print(f">> 运行 LatentSync...")
        print(f"   配置: {config}")
        print(f"   视频: {video_path}")
        print(f"   音频: {audio_path}")
        print(f"   输出: {output_path}")
    
    result = subprocess.run(
        cmd,
        cwd=str(latentsync_dir),
        env=env,
        capture_output=not verbose,
        text=True
    )
    
    if result.returncode != 0:
        if not verbose:
            print(f"ERROR: LatentSync 失败!")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        return False
    
    if verbose:
        print(f">> 口型同步视频已保存: {output_path}")
    
    return True


def run_wav2lip(
    video_path: str,
    audio_path: str,
    output_path: str,
    wav2lip_dir: str,
    python_executable: str = None,
    verbose: bool = True
) -> bool:
    """
    运行 Wav2Lip 推理
    """
    wav2lip_dir = Path(wav2lip_dir)
    python_executable = python_executable or sys.executable
    
    cmd = [
        python_executable,
        str(wav2lip_dir / "inference.py"),
        "--checkpoint_path", str(wav2lip_dir / "checkpoints/wav2lip_gan.pth"),
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_path,
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(wav2lip_dir)
    
    if verbose:
        print(f">> 运行 Wav2Lip...")
        print(f"   视频: {video_path}")
        print(f"   音频: {audio_path}")
        print(f"   输出: {output_path}")
    
    result = subprocess.run(
        cmd,
        cwd=str(wav2lip_dir),
        env=env,
        capture_output=not verbose,
        text=True
    )
    
    if result.returncode != 0:
        if not verbose:
            print(f"ERROR: Wav2Lip 失败!")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        return False
    
    if verbose:
        print(f">> 口型同步视频已保存: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="LipSync 命令行工具 - 使用已有音频生成口型同步视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 LatentSync
  python tools/lipsync_cli.py \\
      --video input.mp4 --audio speech.wav --output output.mp4 \\
      --latentsync_dir ~/LatentSync \\
      --latentsync_python ~/miniconda3/envs/latentsync/bin/python

  # 低显存模式
  python tools/lipsync_cli.py \\
      --video input.mp4 --audio speech.wav --output output.mp4 \\
      --latentsync_dir ~/LatentSync --low_vram

  # 如果视频比音频短，自动从视频后面截取补充
  python tools/lipsync_cli.py \\
      --video short.mp4 --audio long_speech.wav --output output.mp4 \\
      --latentsync_dir ~/LatentSync --extend_video
        """
    )
    
    # 必需参数
    parser.add_argument("--video", "-v", type=str, required=True, help="输入视频路径")
    parser.add_argument("--audio", "-a", type=str, required=True, help="输入音频路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出视频路径")
    
    # 引擎选择
    parser.add_argument("--engine", type=str, default="latentsync", 
                       choices=["latentsync", "wav2lip"],
                       help="口型同步引擎 (默认: latentsync)")
    
    # LatentSync 参数
    parser.add_argument("--latentsync_dir", type=str, help="LatentSync 仓库路径")
    parser.add_argument("--latentsync_python", type=str, help="LatentSync Python 解释器路径")
    parser.add_argument("--low_vram", action="store_true", help="低显存模式 (256px)")
    parser.add_argument("--inference_steps", type=int, default=20, help="推理步数 (默认: 20)")
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="引导比例 (默认: 1.5)")
    
    # Wav2Lip 参数
    parser.add_argument("--wav2lip_dir", type=str, help="Wav2Lip 仓库路径")
    parser.add_argument("--wav2lip_python", type=str, help="Wav2Lip Python 解释器路径")
    
    # 视频处理
    parser.add_argument("--extend_video", action="store_true", 
                       help="如果视频比音频短，从视频后面截取片段补充")
    
    # 其他
    parser.add_argument("--quiet", "-q", action="store_true", help="静默模式")
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # 验证输入文件
    if not os.path.exists(args.video):
        print(f"ERROR: 视频文件不存在: {args.video}")
        sys.exit(1)
    
    if not os.path.exists(args.audio):
        print(f"ERROR: 音频文件不存在: {args.audio}")
        sys.exit(1)
    
    # 获取时长
    audio_duration = get_audio_duration(args.audio)
    video_duration = get_video_duration(args.video)
    
    if verbose:
        print(f">> 音频时长: {audio_duration:.2f}s")
        print(f">> 视频时长: {video_duration:.2f}s")
    
    # 处理视频长度不足
    video_to_use = args.video
    temp_video = None
    
    if video_duration < audio_duration:
        if args.extend_video:
            temp_video = args.output + ".extended.mp4"
            video_to_use = extend_video_if_needed(
                args.video, audio_duration, temp_video, verbose
            )
        else:
            print(f"警告: 视频 ({video_duration:.2f}s) 比音频 ({audio_duration:.2f}s) 短")
            print("使用 --extend_video 参数可自动补充视频")
    
    try:
        # 运行口型同步
        if args.engine == "latentsync":
            if not args.latentsync_dir:
                print("ERROR: 使用 LatentSync 需要指定 --latentsync_dir")
                sys.exit(1)
            
            success = run_latentsync(
                video_path=video_to_use,
                audio_path=args.audio,
                output_path=args.output,
                latentsync_dir=args.latentsync_dir,
                python_executable=args.latentsync_python,
                low_vram=args.low_vram,
                inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                verbose=verbose
            )
        else:  # wav2lip
            if not args.wav2lip_dir:
                print("ERROR: 使用 Wav2Lip 需要指定 --wav2lip_dir")
                sys.exit(1)
            
            success = run_wav2lip(
                video_path=video_to_use,
                audio_path=args.audio,
                output_path=args.output,
                wav2lip_dir=args.wav2lip_dir,
                python_executable=args.wav2lip_python,
                verbose=verbose
            )
        
        if not success:
            sys.exit(1)
            
    finally:
        # 清理临时文件
        if temp_video and os.path.exists(temp_video):
            os.remove(temp_video)
    
    if verbose:
        print(f"\n✅ 完成! 输出: {args.output}")


if __name__ == "__main__":
    main()
