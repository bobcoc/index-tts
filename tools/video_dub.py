#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Dubbing Tool - IndexTTS2 + Lip Sync Integration

This script provides an end-to-end pipeline for video dubbing with lip sync:
1. Extract reference audio from input video (for voice cloning)
2. Generate new audio using IndexTTS2 (text-to-speech with voice cloning)
3. Generate lip-synced video using Wav2Lip (or LatentSync in the future)

Usage:
    uv run tools/video_dub.py \\
        --video input.mp4 \\
        --script "Your text script here" \\
        --output output.mp4

Requirements:
    - ffmpeg installed and available in PATH
    - IndexTTS2 models downloaded to checkpoints/
    - Wav2Lip repository cloned and models downloaded (see --wav2lip_dir)
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Utility Functions
# =============================================================================

def run_command(cmd: list, desc: str = "", verbose: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and handle errors."""
    if verbose:
        print(f">> {desc}")
        print(f"   Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {desc} failed!")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    
    return result


def get_audio_duration(audio_path: str) -> float:
    """Get duration of an audio file in seconds using ffprobe."""
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


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "ffmpeg/ffprobe not found. Please install ffmpeg:\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  macOS: brew install ffmpeg"
        )


# =============================================================================
# Step 1: Extract Audio from Video
# =============================================================================

def extract_audio_from_video(
    video_path: str,
    output_audio_path: str,
    sample_rate: int = 24000,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
    verbose: bool = True
) -> str:
    """
    Extract audio from video file using ffmpeg.
    
    Args:
        video_path: Path to input video
        output_audio_path: Path to output audio file
        sample_rate: Target sample rate (24000 for IndexTTS2)
        start_time: Optional start time in seconds (for extracting a segment)
        duration: Optional duration in seconds (for extracting a segment)
        verbose: Print progress info
    
    Returns:
        Path to extracted audio file
    """
    cmd = ["ffmpeg", "-y", "-i", video_path]
    
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    
    cmd.extend([
        "-vn",                    # No video
        "-ac", "1",               # Mono
        "-ar", str(sample_rate),  # Sample rate
        "-acodec", "pcm_s16le",   # WAV format
        output_audio_path
    ])
    
    run_command(cmd, f"Extracting audio from {video_path}", verbose)
    return output_audio_path


# =============================================================================
# Step 2: Generate Audio with IndexTTS2
# =============================================================================

def generate_tts_audio(
    text: str,
    spk_audio_prompt: str,
    output_path: str,
    config_path: str = "checkpoints/config.yaml",
    model_dir: str = "checkpoints",
    use_fp16: bool = True,
    emo_audio_prompt: Optional[str] = None,
    emo_vector: Optional[list] = None,
    emo_text: Optional[str] = None,
    emo_alpha: float = 1.0,
    use_emo_text: bool = False,
    verbose: bool = True,
    **kwargs
) -> str:
    """
    Generate speech audio using IndexTTS2.
    
    Args:
        text: Text script to synthesize
        spk_audio_prompt: Path to speaker reference audio (for voice cloning)
        output_path: Path to output audio file
        config_path: Path to IndexTTS2 config file
        model_dir: Path to model directory
        use_fp16: Use FP16 inference (faster, less VRAM)
        emo_audio_prompt: Optional emotional reference audio
        emo_vector: Optional emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        emo_text: Optional emotion description text
        emo_alpha: Emotion intensity (0.0-1.0)
        use_emo_text: Use text content for emotion inference
        verbose: Print progress info
    
    Returns:
        Path to generated audio file
    """
    if verbose:
        print(f">> Generating TTS audio with IndexTTS2...")
        print(f"   Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"   Speaker prompt: {spk_audio_prompt}")
    
    # Import here to avoid loading models when not needed
    from indextts.infer_v2 import IndexTTS2
    
    # Initialize TTS model (singleton pattern would be better for batch processing)
    tts = IndexTTS2(
        cfg_path=config_path,
        model_dir=model_dir,
        use_fp16=use_fp16,
        use_cuda_kernel=False,
        use_deepspeed=False
    )
    
    # Generate audio
    tts.infer(
        spk_audio_prompt=spk_audio_prompt,
        text=text,
        output_path=output_path,
        emo_audio_prompt=emo_audio_prompt,
        emo_vector=emo_vector,
        emo_text=emo_text,
        emo_alpha=emo_alpha,
        use_emo_text=use_emo_text,
        verbose=verbose,
        **kwargs
    )
    
    if verbose:
        duration = get_audio_duration(output_path)
        print(f"   Generated audio duration: {duration:.2f}s")
    
    return output_path


# =============================================================================
# Step 3: Lip Sync Engine (Abstraction for easy replacement)
# =============================================================================

class LipSyncEngine(ABC):
    """Abstract base class for lip sync engines (Wav2Lip, LatentSync, etc.)"""
    
    @abstractmethod
    def generate(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        verbose: bool = True,
        **kwargs
    ) -> str:
        """Generate lip-synced video."""
        pass
    
    @abstractmethod
    def check_requirements(self) -> bool:
        """Check if all requirements are met."""
        pass


class Wav2LipEngine(LipSyncEngine):
    """Wav2Lip lip sync engine."""
    
    def __init__(
        self,
        wav2lip_dir: str,
        checkpoint: str = "checkpoints/wav2lip_gan.pth",
        face_det_checkpoint: str = "checkpoints/s3fd.pth",
        python_executable: Optional[str] = None
    ):
        """
        Initialize Wav2Lip engine.
        
        Args:
            wav2lip_dir: Path to Wav2Lip repository
            checkpoint: Path to Wav2Lip model checkpoint (relative to wav2lip_dir)
            face_det_checkpoint: Path to face detection checkpoint (relative to wav2lip_dir)
            python_executable: Python executable to use (default: sys.executable)
        """
        self.wav2lip_dir = Path(wav2lip_dir)
        self.checkpoint = checkpoint
        self.face_det_checkpoint = face_det_checkpoint
        self.python_executable = python_executable or sys.executable
    
    def check_requirements(self) -> bool:
        """Check if Wav2Lip is properly set up."""
        inference_script = self.wav2lip_dir / "inference.py"
        checkpoint_path = self.wav2lip_dir / self.checkpoint
        
        if not inference_script.exists():
            print(f"ERROR: Wav2Lip inference.py not found at {inference_script}")
            print("Please clone Wav2Lip repository:")
            print("  git clone https://github.com/Rudrabha/Wav2Lip.git")
            return False
        
        if not checkpoint_path.exists():
            print(f"ERROR: Wav2Lip checkpoint not found at {checkpoint_path}")
            print("Please download the checkpoint from:")
            print("  https://github.com/Rudrabha/Wav2Lip#getting-the-weights")
            return False
        
        return True
    
    def generate(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        verbose: bool = True,
        resize_factor: int = 1,
        pad_top: int = 0,
        pad_bottom: int = 10,
        pad_left: int = 0,
        pad_right: int = 0,
        **kwargs
    ) -> str:
        """
        Generate lip-synced video using Wav2Lip.
        
        Args:
            video_path: Path to input video
            audio_path: Path to audio file
            output_path: Path to output video
            verbose: Print progress info
            resize_factor: Reduce resolution by this factor (for faster processing)
            pad_*: Padding for face detection
        
        Returns:
            Path to output video
        """
        if not self.check_requirements():
            raise RuntimeError("Wav2Lip requirements not met")
        
        cmd = [
            self.python_executable,
            str(self.wav2lip_dir / "inference.py"),
            "--checkpoint_path", str(self.wav2lip_dir / self.checkpoint),
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--resize_factor", str(resize_factor),
            "--pads", str(pad_top), str(pad_bottom), str(pad_left), str(pad_right),
        ]
        
        # Run Wav2Lip inference
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.wav2lip_dir)
        
        if verbose:
            print(f">> Generating lip-synced video with Wav2Lip...")
            print(f"   Video: {video_path}")
            print(f"   Audio: {audio_path}")
            print(f"   Output: {output_path}")
        
        result = subprocess.run(
            cmd,
            cwd=str(self.wav2lip_dir),
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"ERROR: Wav2Lip failed!")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError("Wav2Lip inference failed")
        
        if verbose:
            print(f"   Lip-synced video saved to: {output_path}")
        
        return output_path


class LatentSyncEngine(LipSyncEngine):
    """
    LatentSync lip sync engine - High quality lip sync using latent diffusion.
    
    Based on ByteDance's LatentSync:
    https://github.com/bytedance/LatentSync
    
    Requirements:
    - Minimum 8GB VRAM (LatentSync 1.5) or 18GB VRAM (LatentSync 1.6)
    - checkpoints/latentsync_unet.pt
    - checkpoints/whisper/tiny.pt
    """
    
    def __init__(
        self,
        latentsync_dir: str,
        checkpoint: str = "checkpoints/latentsync_unet.pt",
        config: str = "auto",  # auto, stage2, stage2_512, stage2_efficient
        python_executable: Optional[str] = None,
        low_vram_mode: bool = False
    ):
        """
        Initialize LatentSync engine.
        
        Args:
            latentsync_dir: Path to LatentSync repository
            checkpoint: Path to UNet checkpoint (relative to latentsync_dir)
            config: Config to use: 'auto' (detect from model), 'stage2' (256px), 
                   'stage2_512' (512px), 'stage2_efficient' (256px, lower VRAM)
            python_executable: Python executable to use (default: sys.executable)
            low_vram_mode: Use 256px resolution to reduce VRAM usage
        """
        self.latentsync_dir = Path(latentsync_dir)
        self.checkpoint = checkpoint
        self.python_executable = python_executable or sys.executable
        self.low_vram_mode = low_vram_mode
        
        # Auto-detect config based on checkpoint or use default
        if low_vram_mode:
            # Force use efficient/256px config for low VRAM
            config_efficient = self.latentsync_dir / "configs/unet/stage2_efficient.yaml"
            config_256 = self.latentsync_dir / "configs/unet/stage2.yaml"
            if config_efficient.exists():
                self.config = "configs/unet/stage2_efficient.yaml"
                print(f">> Using LatentSync efficient config (256px, low VRAM mode)")
            elif config_256.exists():
                self.config = "configs/unet/stage2.yaml"
                print(f">> Using LatentSync 256px config (low VRAM mode)")
            else:
                self.config = "configs/unet/stage2_512.yaml"
                print(f">> Warning: Low VRAM mode requested but only 512px config available")
        elif config == "auto":
            # Default to stage2_512 for LatentSync 1.6 (512 resolution model)
            # TTS model will be offloaded to CPU before running LatentSync
            config_512 = self.latentsync_dir / "configs/unet/stage2_512.yaml"
            if config_512.exists():
                self.config = "configs/unet/stage2_512.yaml"
                print(f">> Using LatentSync 512x512 config (1.6)")
            else:
                self.config = "configs/unet/stage2.yaml"
        elif config == "efficient":
            self.config = "configs/unet/stage2_efficient.yaml"
            print(f">> Using LatentSync efficient config (256px, lower VRAM)")
        else:
            self.config = f"configs/unet/{config}.yaml"
    
    def check_requirements(self) -> bool:
        """Check if LatentSync is properly set up."""
        scripts_dir = self.latentsync_dir / "scripts"
        checkpoint_path = self.latentsync_dir / self.checkpoint
        config_path = self.latentsync_dir / self.config
        
        if not scripts_dir.exists():
            print(f"ERROR: LatentSync scripts not found at {scripts_dir}")
            print("Please clone LatentSync repository:")
            print("  git clone https://github.com/bytedance/LatentSync.git")
            return False
        
        if not checkpoint_path.exists():
            print(f"ERROR: LatentSync checkpoint not found at {checkpoint_path}")
            print("Please download the checkpoint:")
            print("  cd LatentSync && source setup_env.sh")
            print("  Or download manually from: https://huggingface.co/ByteDance/LatentSync-1.6")
            return False
        
        if not config_path.exists():
            print(f"ERROR: LatentSync config not found at {config_path}")
            return False
        
        return True
    
    def generate(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        verbose: bool = True,
        inference_steps: int = 20,
        guidance_scale: float = 1.5,
        seed: int = 0,
        enable_deepcache: bool = True,
        **kwargs
    ) -> str:
        """
        Generate lip-synced video using LatentSync.
        
        Args:
            video_path: Path to input video
            audio_path: Path to audio file
            output_path: Path to output video
            verbose: Print progress info
            inference_steps: Number of diffusion steps (20-50, higher = better quality but slower)
            guidance_scale: CFG scale (1.0-3.0, higher = better lip sync but may cause distortion)
            seed: Random seed (0 for random)
            enable_deepcache: Use DeepCache for faster inference
        
        Returns:
            Path to output video
        """
        if not self.check_requirements():
            raise RuntimeError("LatentSync requirements not met")
        
        cmd = [
            self.python_executable,
            "-m", "scripts.inference",
            "--unet_config_path", str(self.latentsync_dir / self.config),
            "--inference_ckpt_path", str(self.latentsync_dir / self.checkpoint),
            "--inference_steps", str(inference_steps),
            "--guidance_scale", str(guidance_scale),
            "--video_path", video_path,
            "--audio_path", audio_path,
            "--video_out_path", output_path,
        ]
        
        if seed > 0:
            cmd.extend(["--seed", str(seed)])
        
        if enable_deepcache:
            cmd.append("--enable_deepcache")
        
        # Run LatentSync inference
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.latentsync_dir)
        # Optimize CUDA memory allocation to reduce fragmentation
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        if verbose:
            print(f">> Generating lip-synced video with LatentSync...")
            print(f"   Video: {video_path}")
            print(f"   Audio: {audio_path}")
            print(f"   Output: {output_path}")
            print(f"   Steps: {inference_steps}, Guidance: {guidance_scale}")
        
        result = subprocess.run(
            cmd,
            cwd=str(self.latentsync_dir),
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"ERROR: LatentSync failed!")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError("LatentSync inference failed")
        
        if verbose:
            print(f"   Lip-synced video saved to: {output_path}")
        
        return output_path


# =============================================================================
# Step 4 (Optional): Audio Duration Alignment
# =============================================================================

def align_audio_duration(
    audio_path: str,
    target_duration: float,
    output_path: str,
    orig_audio_path: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Adjust audio duration to match target duration.
    
    - If audio is shorter: pad with original audio (if provided) or silence
    - If audio is much longer: truncate to target duration
    - If difference is small: use ffmpeg atempo to adjust speed
    
    Args:
        audio_path: Path to input audio (TTS generated)
        target_duration: Target duration in seconds (original video duration)
        output_path: Path to output audio
        orig_audio_path: Path to original audio (for padding shorter audio)
        verbose: Print progress info
    
    Returns:
        Path to adjusted audio
    """
    current_duration = get_audio_duration(audio_path)
    
    if abs(current_duration - target_duration) < 0.1:
        # Duration already close enough, just copy
        if audio_path != output_path:
            shutil.copy(audio_path, output_path)
        return output_path
    
    # Calculate tempo factor
    tempo = current_duration / target_duration
    
    # Case 1: TTS audio is much shorter than video - pad with original audio
    if tempo < 0.5:
        if verbose:
            print(f">> TTS audio too short ({current_duration:.2f}s vs {target_duration:.2f}s), padding with original audio")
        
        if orig_audio_path and os.path.exists(orig_audio_path):
            # Mix: TTS audio at beginning + original audio for the rest
            # Extract the remaining part from original audio
            remaining_start = current_duration
            remaining_duration = target_duration - current_duration
            
            # Create temp file for the remaining part
            temp_remaining = output_path + ".remaining.wav"
            cmd_extract = [
                "ffmpeg", "-y",
                "-i", orig_audio_path,
                "-ss", str(remaining_start),
                "-t", str(remaining_duration),
                "-acodec", "pcm_s16le",
                temp_remaining
            ]
            run_command(cmd_extract, "Extracting remaining original audio", verbose=False)
            
            # Concatenate TTS audio + remaining original audio
            concat_list = output_path + ".concat.txt"
            with open(concat_list, "w") as f:
                f.write(f"file '{os.path.abspath(audio_path)}'\n")
                f.write(f"file '{os.path.abspath(temp_remaining)}'\n")
            
            cmd_concat = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list,
                "-acodec", "pcm_s16le",
                output_path
            ]
            run_command(cmd_concat, "Concatenating audio", verbose=False)
            
            # Cleanup temp files
            os.remove(temp_remaining)
            os.remove(concat_list)
            
            if verbose:
                print(f"   Combined audio: TTS ({current_duration:.2f}s) + Original tail ({remaining_duration:.2f}s)")
        else:
            # No original audio, pad with silence
            silence_duration = target_duration - current_duration
            cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-af", f"apad=whole_dur={target_duration}",
                "-acodec", "pcm_s16le",
                output_path
            ]
            run_command(cmd, "Padding audio with silence", verbose)
            if verbose:
                print(f"   Padded with {silence_duration:.2f}s silence")
        
        return output_path
    
    # Case 2: TTS audio is much longer than video - truncate
    if tempo > 2.0:
        if verbose:
            print(f">> TTS audio too long ({current_duration:.2f}s vs {target_duration:.2f}s), truncating")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-t", str(target_duration),
            "-acodec", "pcm_s16le",
            output_path
        ]
        run_command(cmd, "Truncating audio", verbose=False)
        
        if verbose:
            print(f"   Truncated to {target_duration:.2f}s")
        
        return output_path
    
    # Case 3: Moderate difference - use atempo to adjust speed
    if verbose:
        print(f">> Aligning audio duration: {current_duration:.2f}s -> {target_duration:.2f}s (tempo={tempo:.2f})")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-filter:a", f"atempo={tempo}",
        "-acodec", "pcm_s16le",
        output_path
    ]
    
    run_command(cmd, "Adjusting audio duration", verbose)
    return output_path


# =============================================================================
# Main Pipeline
# =============================================================================

def dub_video(
    video_path: str,
    script_text: str,
    output_path: str,
    spk_audio_prompt: Optional[str] = None,
    spk_extract_start: float = 0.0,
    spk_extract_duration: float = 10.0,
    align_duration: bool = True,
    lip_sync_engine: Optional[LipSyncEngine] = None,
    wav2lip_dir: Optional[str] = None,
    tts_config: str = "checkpoints/config.yaml",
    tts_model_dir: str = "checkpoints",
    use_fp16: bool = True,
    emo_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    keep_temp_files: bool = False,
) -> str:
    """
    Main video dubbing pipeline.
    
    Args:
        video_path: Path to input video
        script_text: Text script for dubbing
        output_path: Path to output video
        spk_audio_prompt: Optional path to speaker reference audio
                         (if None, will extract from video)
        spk_extract_start: Start time for extracting speaker reference (seconds)
        spk_extract_duration: Duration for extracting speaker reference (seconds)
        align_duration: Whether to align TTS audio duration to original
        lip_sync_engine: LipSyncEngine instance (if None, will create Wav2LipEngine)
        wav2lip_dir: Path to Wav2Lip repository (required if lip_sync_engine is None)
        tts_config: Path to IndexTTS2 config file
        tts_model_dir: Path to IndexTTS2 model directory
        use_fp16: Use FP16 inference for TTS
        emo_params: Optional emotion parameters for IndexTTS2
        verbose: Print progress info
        keep_temp_files: Keep temporary files for debugging
    
    Returns:
        Path to output video
    """
    check_ffmpeg()
    
    # Create temp directory for intermediate files
    temp_dir = tempfile.mkdtemp(prefix="video_dub_")
    if verbose:
        print(f">> Temporary directory: {temp_dir}")
    
    try:
        # =====================================================================
        # Step 1: Extract reference audio from video
        # =====================================================================
        if spk_audio_prompt is None:
            spk_audio_prompt = os.path.join(temp_dir, "spk_prompt.wav")
            extract_audio_from_video(
                video_path,
                spk_audio_prompt,
                sample_rate=24000,
                start_time=spk_extract_start,
                duration=spk_extract_duration,
                verbose=verbose
            )
        
        # Also extract full audio for duration reference
        orig_audio_path = os.path.join(temp_dir, "orig_audio.wav")
        extract_audio_from_video(video_path, orig_audio_path, verbose=verbose)
        orig_duration = get_audio_duration(orig_audio_path)
        if verbose:
            print(f"   Original audio duration: {orig_duration:.2f}s")
        
        # =====================================================================
        # Step 2: Generate TTS audio
        # =====================================================================
        tts_audio_raw = os.path.join(temp_dir, "tts_audio_raw.wav")
        
        emo_kwargs = emo_params or {}
        generate_tts_audio(
            text=script_text,
            spk_audio_prompt=spk_audio_prompt,
            output_path=tts_audio_raw,
            config_path=tts_config,
            model_dir=tts_model_dir,
            use_fp16=use_fp16,
            verbose=verbose,
            **emo_kwargs
        )
        
        # Optional: Align duration
        if align_duration:
            tts_audio = os.path.join(temp_dir, "tts_audio.wav")
            align_audio_duration(tts_audio_raw, orig_duration, tts_audio, 
                               orig_audio_path=orig_audio_path, verbose=verbose)
        else:
            tts_audio = tts_audio_raw
        
        # =====================================================================
        # Step 3: Generate lip-synced video
        # =====================================================================
        if lip_sync_engine is None:
            if wav2lip_dir is None:
                raise ValueError(
                    "Either lip_sync_engine or wav2lip_dir must be provided. "
                    "Please specify --wav2lip_dir pointing to your Wav2Lip repository."
                )
            lip_sync_engine = Wav2LipEngine(wav2lip_dir)
        
        lip_sync_engine.generate(
            video_path=video_path,
            audio_path=tts_audio,
            output_path=output_path,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n>> Video dubbing complete!")
            print(f"   Output: {output_path}")
        
        return output_path
    
    finally:
        # Cleanup temp files
        if not keep_temp_files:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f">> Temporary files kept at: {temp_dir}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Video Dubbing Tool - IndexTTS2 + Lip Sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Wav2Lip
  uv run tools/video_dub.py \\
      --video input.mp4 \\
      --script "Hello, this is the new dubbing text." \\
      --output output.mp4 \\
      --wav2lip_dir /path/to/Wav2Lip

  # With custom speaker reference audio
  uv run tools/video_dub.py \\
      --video input.mp4 \\
      --script "Hello, this is the new dubbing text." \\
      --output output.mp4 \\
      --spk_audio reference_voice.wav \\
      --wav2lip_dir /path/to/Wav2Lip

  # With emotion control
  uv run tools/video_dub.py \\
      --video input.mp4 \\
      --script "I am so happy to see you!" \\
      --output output.mp4 \\
      --wav2lip_dir /path/to/Wav2Lip \\
      --emo_alpha 0.8 \\
      --use_emo_text
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--video", "-v",
        type=str, required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--script", "-s",
        type=str, required=True,
        help="Text script for dubbing (or path to .txt file)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str, required=True,
        help="Path to output video file"
    )
    
    # Lip sync engine
    parser.add_argument(
        "--wav2lip_dir",
        type=str, default=None,
        help="Path to Wav2Lip repository"
    )
    parser.add_argument(
        "--lip_sync_engine",
        type=str, default="wav2lip",
        choices=["wav2lip", "latentsync"],
        help="Lip sync engine to use (default: wav2lip)"
    )
    
    # Speaker/voice options
    parser.add_argument(
        "--spk_audio",
        type=str, default=None,
        help="Path to speaker reference audio (if not provided, will extract from video)"
    )
    parser.add_argument(
        "--spk_extract_start",
        type=float, default=0.0,
        help="Start time for extracting speaker reference from video (seconds)"
    )
    parser.add_argument(
        "--spk_extract_duration",
        type=float, default=10.0,
        help="Duration for extracting speaker reference from video (seconds)"
    )
    
    # TTS options
    parser.add_argument(
        "--tts_config",
        type=str, default="checkpoints/config.yaml",
        help="Path to IndexTTS2 config file"
    )
    parser.add_argument(
        "--tts_model_dir",
        type=str, default="checkpoints",
        help="Path to IndexTTS2 model directory"
    )
    parser.add_argument(
        "--fp16",
        action="store_true", default=True,
        help="Use FP16 inference for TTS (default: True)"
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable FP16 inference"
    )
    
    # Emotion options
    parser.add_argument(
        "--emo_audio",
        type=str, default=None,
        help="Path to emotional reference audio"
    )
    parser.add_argument(
        "--emo_alpha",
        type=float, default=1.0,
        help="Emotion intensity (0.0-1.0, default: 1.0)"
    )
    parser.add_argument(
        "--use_emo_text",
        action="store_true",
        help="Use script text for emotion inference"
    )
    parser.add_argument(
        "--emo_text",
        type=str, default=None,
        help="Emotion description text (separate from script)"
    )
    
    # Other options
    parser.add_argument(
        "--no_align",
        action="store_true",
        help="Don't align TTS audio duration to original"
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary files for debugging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        sys.exit(1)
    
    # Read script from file if it's a path
    script_text = args.script
    if os.path.exists(args.script) and args.script.endswith('.txt'):
        with open(args.script, 'r', encoding='utf-8') as f:
            script_text = f.read().strip()
    
    if not script_text:
        print("ERROR: Script text is empty")
        sys.exit(1)
    
    # Prepare emotion parameters
    emo_params = {}
    if args.emo_audio:
        emo_params["emo_audio_prompt"] = args.emo_audio
    if args.emo_alpha != 1.0:
        emo_params["emo_alpha"] = args.emo_alpha
    if args.use_emo_text:
        emo_params["use_emo_text"] = True
    if args.emo_text:
        emo_params["emo_text"] = args.emo_text
    
    # Determine FP16 usage
    use_fp16 = args.fp16 and not args.no_fp16
    
    # Create lip sync engine
    lip_sync_engine = None
    if args.lip_sync_engine == "latentsync":
        print("ERROR: LatentSync is not yet implemented. Please use wav2lip for now.")
        sys.exit(1)
    
    # Run the pipeline
    try:
        dub_video(
            video_path=args.video,
            script_text=script_text,
            output_path=args.output,
            spk_audio_prompt=args.spk_audio,
            spk_extract_start=args.spk_extract_start,
            spk_extract_duration=args.spk_extract_duration,
            align_duration=not args.no_align,
            lip_sync_engine=lip_sync_engine,
            wav2lip_dir=args.wav2lip_dir,
            tts_config=args.tts_config,
            tts_model_dir=args.tts_model_dir,
            use_fp16=use_fp16,
            emo_params=emo_params if emo_params else None,
            verbose=not args.quiet,
            keep_temp_files=args.keep_temp,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
