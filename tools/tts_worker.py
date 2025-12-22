#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Subprocess Worker - 独立进程运行 TTS 推理

这个脚本在独立进程中运行 TTS，完成后进程退出，显存完全释放。

用法:
    python tools/tts_worker.py \
        --prompt_audio ref.wav \
        --text "要合成的文本" \
        --output output.wav \
        --model_dir checkpoints \
        --fp16
"""

import os
import sys
import json
import argparse

# Add project root to path
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="TTS Worker - 独立进程运行 TTS 推理")
    
    # 必需参数
    parser.add_argument("--prompt_audio", "-p", type=str, required=True, help="参考音频路径")
    parser.add_argument("--text", "-t", type=str, required=True, help="要合成的文本")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出音频路径")
    
    # 模型参数
    parser.add_argument("--model_dir", type=str, default="./checkpoints", help="模型目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径 (默认: model_dir/config.yaml)")
    parser.add_argument("--fp16", action="store_true", help="使用 FP16 推理")
    
    # 情感参数
    parser.add_argument("--emo_alpha", type=float, default=1.0, help="情感强度")
    parser.add_argument("--use_emo_text", action="store_true", help="从文本推断情感")
    
    # 其他参数
    parser.add_argument("--max_tokens", type=int, default=120, help="每段最大 token 数")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 确定配置文件路径
    config_path = args.config or os.path.join(args.model_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"ERROR: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    if not os.path.exists(args.prompt_audio):
        print(f"ERROR: 参考音频不存在: {args.prompt_audio}")
        sys.exit(1)
    
    if args.verbose:
        print(f">> TTS Worker 启动")
        print(f"   模型目录: {args.model_dir}")
        print(f"   配置文件: {config_path}")
        print(f"   FP16: {args.fp16}")
    
    try:
        # 导入并初始化 TTS
        from indextts.infer_v2 import IndexTTS2
        
        if args.verbose:
            print(f">> 加载 TTS 模型...")
        
        tts = IndexTTS2(
            cfg_path=config_path,
            model_dir=args.model_dir,
            use_fp16=args.fp16,
            use_cuda_kernel=False,
            use_deepspeed=False
        )
        
        if args.verbose:
            print(f">> TTS 模型加载完成")
            print(f">> 开始生成音频...")
            print(f"   文本: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
        
        # 生成音频
        tts.infer(
            spk_audio_prompt=args.prompt_audio,
            text=args.text,
            output_path=args.output,
            emo_alpha=args.emo_alpha,
            use_emo_text=args.use_emo_text,
            max_text_tokens_per_segment=args.max_tokens,
            verbose=args.verbose
        )
        
        if args.verbose:
            print(f">> 音频已保存: {args.output}")
        
        # 验证输出文件
        if not os.path.exists(args.output):
            print(f"ERROR: 输出文件未生成: {args.output}")
            sys.exit(1)
        
        print(f"SUCCESS: {args.output}")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: TTS 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
