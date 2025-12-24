import html
import json
import os
import sys
import threading
import time
import tempfile
import shutil
import subprocess

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
parser.add_argument("--wav2lip_dir", type=str, default=None, help="Path to Wav2Lip repository for video dubbing")
parser.add_argument("--wav2lip_python", type=str, default=None, help="Python executable for Wav2Lip environment")
parser.add_argument("--latentsync_dir", type=str, default=None, help="Path to LatentSync repository for video dubbing (higher quality)")
parser.add_argument("--latentsync_python", type=str, default=None, help="Python executable for LatentSync environment")
parser.add_argument("--latentsync_low_vram", action="store_true", default=False, help="Use 256px resolution for LatentSync to reduce VRAM usage (~8GB instead of ~18GB)")
parser.add_argument("--subprocess_dub", action="store_true", default=False, help="Run video dubbing in subprocess mode to completely isolate VRAM usage")
parser.add_argument("--extend_video", action="store_true", default=False, help="Extend video from tail if audio is longer (instead of truncating audio)")
parser.add_argument("--lip_sync_engine", type=str, default="auto", choices=["auto", "wav2lip", "latentsync"], help="Lip sync engine to use")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16,
                use_deepspeed=cmd_args.deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                )

# Video dubbing support
VIDEO_DUB_ENABLED = False
lip_sync_engine = None
LIP_SYNC_ENGINE_NAME = None

# Initialize lip sync engine based on settings
def init_lip_sync_engine():
    global VIDEO_DUB_ENABLED, lip_sync_engine, LIP_SYNC_ENGINE_NAME
    
    engine_choice = cmd_args.lip_sync_engine
    
    # Try LatentSync first if specified or auto
    if engine_choice in ["auto", "latentsync"] and cmd_args.latentsync_dir and os.path.exists(cmd_args.latentsync_dir):
        try:
            from tools.video_dub import extract_audio_from_video, align_audio_duration, get_audio_duration, LatentSyncEngine, check_ffmpeg
            check_ffmpeg()
            latentsync_python = cmd_args.latentsync_python
            if latentsync_python is None:
                possible_paths = [
                    os.path.expanduser("~/miniconda3/envs/latentsync/bin/python"),
                    os.path.expanduser("~/anaconda3/envs/latentsync/bin/python"),
                    os.path.join(cmd_args.latentsync_dir, "venv/bin/python"),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        latentsync_python = path
                        print(f">> Auto-detected LatentSync Python: {latentsync_python}")
                        break
            lip_sync_engine = LatentSyncEngine(
                cmd_args.latentsync_dir, 
                python_executable=latentsync_python,
                low_vram_mode=cmd_args.latentsync_low_vram
            )
            if lip_sync_engine.check_requirements():
                VIDEO_DUB_ENABLED = True
                LIP_SYNC_ENGINE_NAME = "LatentSync"
                print(f">> Video dubbing enabled with LatentSync at: {cmd_args.latentsync_dir}")
                if latentsync_python:
                    print(f">> Using LatentSync Python: {latentsync_python}")
                return
            else:
                print(f"WARNING: LatentSync requirements not met.")
        except Exception as e:
            print(f"WARNING: Failed to initialize LatentSync: {e}")
    
    # Try Wav2Lip if specified or auto (fallback)
    if engine_choice in ["auto", "wav2lip"] and cmd_args.wav2lip_dir and os.path.exists(cmd_args.wav2lip_dir):
        try:
            from tools.video_dub import extract_audio_from_video, align_audio_duration, get_audio_duration, Wav2LipEngine, check_ffmpeg
            check_ffmpeg()
            wav2lip_python = cmd_args.wav2lip_python
            if wav2lip_python is None:
                possible_paths = [
                    os.path.expanduser("~/miniconda3/envs/wav2lip/bin/python"),
                    os.path.expanduser("~/anaconda3/envs/wav2lip/bin/python"),
                    os.path.join(cmd_args.wav2lip_dir, "venv/bin/python"),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        wav2lip_python = path
                        print(f">> Auto-detected Wav2Lip Python: {wav2lip_python}")
                        break
            lip_sync_engine = Wav2LipEngine(cmd_args.wav2lip_dir, python_executable=wav2lip_python)
            if lip_sync_engine.check_requirements():
                VIDEO_DUB_ENABLED = True
                LIP_SYNC_ENGINE_NAME = "Wav2Lip"
                print(f">> Video dubbing enabled with Wav2Lip at: {cmd_args.wav2lip_dir}")
                if wav2lip_python:
                    print(f">> Using Wav2Lip Python: {wav2lip_python}")
                return
            else:
                print(f"WARNING: Wav2Lip requirements not met.")
        except Exception as e:
            print(f"WARNING: Failed to initialize Wav2Lip: {e}")

init_lip_sync_engine()

# Import video dubbing utilities if enabled
if VIDEO_DUB_ENABLED:
    from tools.video_dub import extract_audio_from_video, align_audio_duration, get_audio_duration

# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]  # skip experimental features

os.makedirs(os.path.join(current_dir, "outputs/tasks"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "prompts"), exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
example_cases = []
examples_file = os.path.join(current_dir, "examples/cases.jsonl")
if os.path.exists(examples_file):
    with open(examples_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            if example.get("emo_audio",None):
                emo_audio_path = os.path.join(current_dir, "examples", example["emo_audio"])
            else:
                emo_audio_path = None

            example_cases.append([os.path.join(current_dir, "examples", example.get("prompt_audio", "sample_prompt.wav")),
                                  EMO_CHOICES_ALL[example.get("emo_mode",0)],
                                  example.get("text"),
                                 emo_audio_path,
                                 example.get("emo_weight",1.0),
                                 example.get("emo_text",""),
                                 example.get("emo_vec_1",0),
                                 example.get("emo_vec_2",0),
                                 example.get("emo_vec_3",0),
                                 example.get("emo_vec_4",0),
                                 example.get("emo_vec_5",0),
                                 example.get("emo_vec_6",0),
                                 example.get("emo_vec_7",0),
                                 example.get("emo_vec_8",0),
                                 ])

def get_example_cases(include_experimental = False):
    if include_experimental:
        return example_cases  # show every example

    # exclude emotion control mode 3 (emotion from text description)
    return [x for x in example_cases if x[1] != EMO_CHOICES_ALL[3]]

def format_glossary_markdown():
    """将词汇表转换为Markdown表格格式"""
    if not tts.normalizer.term_glossary:
        return i18n("暂无术语")

    lines = [f"| {i18n('术语')} | {i18n('中文读法')} | {i18n('英文读法')} |"]
    lines.append("|---|---|---|")

    for term, reading in tts.normalizer.term_glossary.items():
        zh = reading.get("zh", "") if isinstance(reading, dict) else reading
        en = reading.get("en", "") if isinstance(reading, dict) else reading
        lines.append(f"| {term} | {zh} | {en} |")

    return "\n".join(lines)

def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment=120,
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join(current_dir, "outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    if emo_control_method == 1:  # emotion from reference audio
        pass
    if emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    else:
        # don't use the emotion vector inputs for the other modes
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                       **kwargs)
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message(i18n('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。'))

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')

    with gr.Tab(i18n("音频生成")):
        with gr.Row():
            os.makedirs(os.path.join(current_dir, "prompts"), exist_ok=True)
            prompt_audio = gr.Audio(label=i18n("音色参考音频"),key="prompt_audio",
                                    sources=["upload","microphone"],type="filepath")
            prompt_list = os.listdir(os.path.join(current_dir, "prompts"))
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label=i18n("文本"),key="input_text_single", placeholder=i18n("请输入目标文本"), info=f"{i18n('当前模型版本')}{tts.model_version or '1.0'}")
                gen_button = gr.Button(i18n("生成语音"), key="gen_button",interactive=True)
            output_audio = gr.Audio(label=i18n("生成结果"), visible=True,key="output_audio")

        with gr.Row():
            experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)
            glossary_checkbox = gr.Checkbox(label=i18n("开启术语词汇读音"), value=tts.normalizer.enable_glossary)
        with gr.Accordion(i18n("功能设置")):
            # 情感控制选项部分
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_OFFICIAL,
                    type="index",
                    value=EMO_CHOICES_OFFICIAL[0],label=i18n("情感控制方式"))
                # we MUST have an extra, INVISIBLE list of *all* emotion control
                # methods so that gr.Dataset() can fetch ALL control mode labels!
                # otherwise, the gr.Dataset()'s experimental labels would be empty!
                emo_control_method_all = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0], label=i18n("情感控制方式"),
                    visible=False)  # do not render
        # 情感参考音频部分
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")

        # 情感随机采样
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            create_experimental_warning_message()
            with gr.Row():
                emo_text = gr.Textbox(label=i18n("情感描述文本"),
                                      placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                                      value="",
                                      info=i18n("例如：委屈巴巴、危险在悄悄逼近"))

        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)

        # 术语词汇表管理
        with gr.Accordion(i18n("自定义术语词汇读音"), open=False, visible=tts.normalizer.enable_glossary) as glossary_accordion:
            gr.Markdown(i18n("自定义个别专业术语的读音"))
            with gr.Row():
                with gr.Column(scale=1):
                    glossary_term = gr.Textbox(
                        label=i18n("术语"),
                        placeholder="IndexTTS2",
                    )
                    glossary_reading_zh = gr.Textbox(
                        label=i18n("中文读法"),
                        placeholder="Index T-T-S 二",
                    )
                    glossary_reading_en = gr.Textbox(
                        label=i18n("英文读法"),
                        placeholder="Index T-T-S two",
                    )
                    btn_add_term = gr.Button(i18n("添加术语"), scale=1)
                with gr.Column(scale=2):
                    glossary_table = gr.Markdown(
                        value=format_glossary_markdown()
                    )

        with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                        )
                    with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                # typical_sampling, typical_mass,
            ]

        # we must use `gr.Dataset` to support dynamic UI rewrites, since `gr.Examples`
        # binds tightly to UI and always restores the initial state of all components,
        # such as the list of available choices in emo_control_method.
        example_table = gr.Dataset(label="Examples",
            samples_per_page=20,
            samples=get_example_cases(include_experimental=False),
            type="values",
            # these components are NOT "connected". it just reads the column labels/available
            # states from them, so we MUST link to the "all options" versions of all components,
            # such as `emo_control_method_all` (to be able to see EXPERIMENTAL text labels)!
            components=[prompt_audio,
                        emo_control_method_all,  # important: support all mode labels!
                        input_text_single,
                        emo_upload,
                        emo_weight,
                        emo_text,
                        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        )

    def on_example_click(example):
        print(f"Example clicked: ({len(example)} values) = {example!r}")
        return (
            gr.update(value=example[0]),
            gr.update(value=example[1]),
            gr.update(value=example[2]),
            gr.update(value=example[3]),
            gr.update(value=example[4]),
            gr.update(value=example[5]),
            gr.update(value=example[6]),
            gr.update(value=example[7]),
            gr.update(value=example[8]),
            gr.update(value=example[9]),
            gr.update(value=example[10]),
            gr.update(value=example[11]),
            gr.update(value=example[12]),
            gr.update(value=example[13]),
        )

    # click() event works on both desktop and mobile UI
    example_table.click(on_example_click,
                        inputs=[example_table],
                        outputs=[prompt_audio,
                                 emo_control_method,
                                 input_text_single,
                                 emo_upload,
                                 emo_weight,
                                 emo_text,
                                 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
    )

    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = []
            for i, s in enumerate(segments):
                segment_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {
                segments_preview: gr.update(value=df),
            }

    # 术语词汇表事件处理函数
    def on_add_glossary_term(term, reading_zh, reading_en):
        """添加术语到词汇表并自动保存"""
        term = term.rstrip()
        reading_zh = reading_zh.rstrip()
        reading_en = reading_en.rstrip()

        if not term:
            gr.Warning(i18n("请输入术语"))
            return gr.update()
            
        if not reading_zh and not reading_en:
            gr.Warning(i18n("请至少输入一种读法"))
            return gr.update()
        

        # 构建读法数据
        if reading_zh and reading_en:
            reading = {"zh": reading_zh, "en": reading_en}
        elif reading_zh:
            reading = {"zh": reading_zh}
        elif reading_en:
            reading = {"en": reading_en}
        else:
            reading = reading_zh or reading_en

        # 添加到词汇表
        tts.normalizer.term_glossary[term] = reading

        # 自动保存到文件
        try:
            tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
            gr.Info(i18n("词汇表已更新"), duration=1)
        except Exception as e:
            gr.Error(i18n("保存词汇表时出错"))
            print(f"Error details: {e}")
            return gr.update()

        # 更新Markdown表格
        return gr.update(value=format_glossary_markdown())
        

    def on_method_change(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )

    def on_experimental_change(is_experimental, current_mode_index):
        # 切换情感控制选项
        new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
        # if their current mode selection doesn't exist in new choices, reset to 0.
        # we don't verify that OLD index means the same in NEW list, since we KNOW it does.
        new_index = current_mode_index if current_mode_index < len(new_choices) else 0

        return (
            gr.update(choices=new_choices, value=new_choices[new_index]),
            gr.update(samples=get_example_cases(include_experimental=is_experimental)),
        )

    experimental_checkbox.change(
        on_experimental_change,
        inputs=[experimental_checkbox, emo_control_method],
        outputs=[emo_control_method, example_table]
    )

    def on_glossary_checkbox_change(is_enabled):
        """控制术语词汇表的可见性"""
        tts.normalizer.enable_glossary = is_enabled
        return gr.update(visible=is_enabled)

    glossary_checkbox.change(
        on_glossary_checkbox_change,
        inputs=[glossary_checkbox],
        outputs=[glossary_accordion]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    def on_demo_load():
        """页面加载时重新加载glossary数据"""
        try:
            tts.normalizer.load_glossary_from_yaml(tts.glossary_path)
        except Exception as e:
            gr.Error(i18n("加载词汇表时出错"))
            print(f"Failed to reload glossary on page load: {e}")
        return gr.update(value=format_glossary_markdown())

    # 术语词汇表事件绑定
    btn_add_term.click(
        on_add_glossary_term,
        inputs=[glossary_term, glossary_reading_zh, glossary_reading_en],
        outputs=[glossary_table]
    )

    # 页面加载时重新加载glossary
    demo.load(
        on_demo_load,
        inputs=[],
        outputs=[glossary_table]
    )

    gen_button.click(gen_single,
                     inputs=[emo_control_method,prompt_audio, input_text_single, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text,emo_random,
                             max_text_tokens_per_segment,
                             *advanced_params,
                     ],
                     outputs=[output_audio])

    # =========================================================================
    # Video Dubbing Tab
    # =========================================================================
    with gr.Tab(i18n("视频配音") + (f" ({LIP_SYNC_ENGINE_NAME})" if LIP_SYNC_ENGINE_NAME else ""), visible=VIDEO_DUB_ENABLED):
        if not VIDEO_DUB_ENABLED:
            gr.Markdown(f"""
            ### {i18n("视频配音功能未启用")}
            
            {i18n("请使用以下参数启动 WebUI 以启用视频配音功能")}：
            
            **Wav2Lip** ({i18n("较快，显存要求低")}):
            ```
            python webui.py --wav2lip_dir /path/to/Wav2Lip
            ```
            
            **LatentSync** ({i18n("质量更高，需要8GB+显存")}):
            ```
            python webui.py --latentsync_dir /path/to/LatentSync
            ```
            """)
        else:
            engine_info = f"**{i18n('当前引擎')}:** {LIP_SYNC_ENGINE_NAME}"
            gr.Markdown(f"""
            ### {i18n("视频配音 + 口型同步")}
            {i18n("上传视频和配音文本，自动生成口型同步的新视频")} | {engine_info}
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(
                        label=i18n("输入视频"),
                        sources=["upload"],
                    )
                    video_spk_audio = gr.Audio(
                        label=i18n("音色参考音频（可选）"),
                        sources=["upload", "microphone"],
                        type="filepath",
                    )
                    gr.Markdown(i18n("不提供则自动从视频中提取"))
                
                with gr.Column(scale=1):
                    video_script = gr.TextArea(
                        label=i18n("配音文本"),
                        placeholder=i18n("请输入配音脚本"),
                        lines=6
                    )
                    video_gen_button = gr.Button(i18n("生成视频"), variant="primary")
            
            with gr.Accordion(i18n("高级设置"), open=False):
                with gr.Row():
                    video_align_duration = gr.Checkbox(
                        label=i18n("对齐原视频时长"),
                        value=True,
                    )
                    video_spk_extract_start = gr.Number(
                        label=i18n("参考音频提取起始时间(秒)"),
                        value=0.0,
                        minimum=0.0,
                    )
                    video_spk_extract_duration = gr.Number(
                        label=i18n("参考音频提取时长(秒)"),
                        value=10.0,
                        minimum=1.0,
                        maximum=30.0,
                    )
                
                with gr.Row():
                    video_emo_control = gr.Radio(
                        choices=[i18n("与音色参考音频相同"), i18n("根据文本推断情感")],
                        value=i18n("与音色参考音频相同"),
                        label=i18n("情感控制")
                    )
                    video_emo_alpha = gr.Slider(
                        label=i18n("情感强度"),
                        minimum=0.0,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                        visible=False
                    )
            
            with gr.Row():
                video_output = gr.Video(label=i18n("生成结果"))
            
            video_status = gr.Markdown("")
            
            def on_video_emo_control_change(emo_control):
                if emo_control == i18n("根据文本推断情感"):
                    return gr.update(visible=True)
                return gr.update(visible=False)
            
            video_emo_control.change(
                on_video_emo_control_change,
                inputs=[video_emo_control],
                outputs=[video_emo_alpha]
            )
            
            def generate_dubbed_video(
                video_path,
                script_text,
                spk_audio,
                align_duration,
                spk_extract_start,
                spk_extract_duration,
                emo_control,
                emo_alpha,
                progress=gr.Progress()
            ):
                if not video_path:
                    gr.Warning(i18n("请上传视频文件"))
                    return None, ""
                
                if not script_text or not script_text.strip():
                    gr.Warning(i18n("请输入配音文本"))
                    return None, ""
                
                # Use persistent temp dir for debugging (not /tmp)
                temp_base = os.path.join(current_dir, "outputs", "video_dub_temp")
                os.makedirs(temp_base, exist_ok=True)
                temp_dir = os.path.join(temp_base, f"session_{int(time.time())}")
                os.makedirs(temp_dir, exist_ok=True)
                print(f">> Temp dir: {temp_dir}")
                
                # 子进程模式：TTS 和 LipSync 都在独立进程中运行，完全隔离显存
                if cmd_args.subprocess_dub:
                    return generate_dubbed_video_subprocess(
                        video_path, script_text, spk_audio, align_duration,
                        spk_extract_start, spk_extract_duration, emo_control, emo_alpha,
                        temp_dir, progress
                    )
                
                # 常规模式（保持原有逻辑）
                return generate_dubbed_video_inprocess(
                    video_path, script_text, spk_audio, align_duration,
                    spk_extract_start, spk_extract_duration, emo_control, emo_alpha,
                    temp_dir, progress
                )
            
            def generate_dubbed_video_subprocess(
                video_path, script_text, spk_audio, align_duration,
                spk_extract_start, spk_extract_duration, emo_control, emo_alpha,
                temp_dir, progress
            ):
                """
                子进程模式：TTS 和 LipSync 都在独立进程中运行，完成后显存完全释放。
                """
                import subprocess
                import shutil
                
                try:
                    progress(0.1, desc=i18n("提取参考音频..."))
                    
                    # 复制原始视频到session目录，便于溯源
                    video_ext = os.path.splitext(video_path)[1] or '.mp4'
                    local_video_path = os.path.join(temp_dir, f"input_video{video_ext}")
                    shutil.copy(video_path, local_video_path)
                    print(f">> [DEBUG] 原始视频已复制到: {local_video_path}")
                    video_path = local_video_path  # 后续使用本地副本
                    
                    # Step 1: Extract reference audio
                    if spk_audio is None:
                        spk_audio_path = os.path.join(temp_dir, "spk_prompt.wav")
                        extract_audio_from_video(
                            video_path,
                            spk_audio_path,
                            sample_rate=24000,
                            start_time=float(spk_extract_start),
                            duration=float(spk_extract_duration),
                            verbose=cmd_args.verbose
                        )
                    else:
                        spk_audio_path = spk_audio
                    
                    progress(0.2, desc=i18n("生成配音 (子进程)..."))
                    
                    # Step 2: Generate TTS audio using subprocess
                    tts_audio_raw = os.path.join(temp_dir, "tts_audio_raw.wav")
                    use_emo_text = (emo_control == i18n("根据文本推断情感"))
                    
                    tts_cmd = [
                        sys.executable,
                        os.path.join(current_dir, "tools", "tts_worker.py"),
                        "--prompt_audio", spk_audio_path,
                        "--text", script_text.strip(),
                        "--output", tts_audio_raw,
                        "--model_dir", cmd_args.model_dir,
                        "--max_tokens", str(cmd_args.gui_seg_tokens),
                    ]
                    
                    if cmd_args.fp16:
                        tts_cmd.append("--fp16")
                    if use_emo_text:
                        tts_cmd.append("--use_emo_text")
                        tts_cmd.extend(["--emo_alpha", str(float(emo_alpha))])
                    if cmd_args.verbose:
                        tts_cmd.append("--verbose")
                    
                    print(f">> Running TTS subprocess...")
                    tts_result = subprocess.run(tts_cmd, capture_output=True, text=True)
                    
                    if tts_result.returncode != 0:
                        print(f"TTS subprocess failed!")
                        print(f"STDOUT: {tts_result.stdout}")
                        print(f"STDERR: {tts_result.stderr}")
                        raise RuntimeError(f"TTS 生成失败: {tts_result.stderr}")
                    
                    if not os.path.exists(tts_audio_raw):
                        raise RuntimeError("TTS 输出文件未生成")
                    
                    print(f">> TTS audio generated: {tts_audio_raw}")
                    
                    # Get audio duration for video extension
                    tts_duration = get_audio_duration(tts_audio_raw)
                    
                    progress(0.5, desc=i18n("处理音视频时长..."))
                    
                    # Handle video/audio duration mismatch
                    from tools.video_dub import get_video_duration, extend_video_from_tail
                    video_duration = get_video_duration(video_path)
                    
                    # 调试信息：显示音视频时长对比
                    print(f">> [DEBUG] 音视频时长对比:")
                    print(f"   - 视频时长: {video_duration:.3f}s")
                    print(f"   - TTS音频时长: {tts_duration:.3f}s")
                    print(f"   - 差异: {tts_duration - video_duration:.3f}s (正数=音频长于视频, 负数=音频短于视频)")
                    print(f"   - align_duration={align_duration}, extend_video={cmd_args.extend_video}")
                    
                    video_to_use = video_path
                    tts_audio = tts_audio_raw
                    
                    if tts_duration > video_duration and cmd_args.extend_video:
                        # 扩展视频而不是截断音频
                        print(f">> [DEBUG] 选择扩展视频模式 (TTS音频较长且extend_video=True)")
                        extended_video = os.path.join(temp_dir, "extended_video.mp4")
                        video_to_use = extend_video_from_tail(
                            video_path, tts_duration, extended_video, 
                            verbose=cmd_args.verbose
                        )
                        print(f">> Extended video: {video_to_use}")
                    elif align_duration and abs(tts_duration - video_duration) > 0.1:
                        # 对齐音频时长到视频
                        print(f">> [DEBUG] 选择对齐音频时长模式")
                        tts_audio = os.path.join(temp_dir, "tts_audio.wav")
                        align_audio_duration(
                            tts_audio_raw, video_duration, tts_audio,
                            no_speed_change=cmd_args.extend_video,
                            verbose=True  # 强制开启详细输出
                        )
                        # 验证对齐后的音频时长
                        aligned_duration = get_audio_duration(tts_audio)
                        print(f">> [DEBUG] 对齐后音频时长: {aligned_duration:.3f}s")
                    else:
                        print(f">> [DEBUG] 无需对齐 (差异<0.1s 或 align_duration=False)")
                    
                    progress(0.6, desc=i18n("生成口型同步视频 (子进程)..."))
                    
                    # Step 3: Generate lip-synced video using subprocess
                    output_path = os.path.join(current_dir, "outputs", f"video_dub_{int(time.time())}.mp4")
                    
                    lipsync_cmd = [
                        sys.executable,
                        os.path.join(current_dir, "tools", "lipsync_cli.py"),
                        "--video", video_to_use,
                        "--audio", tts_audio,
                        "--output", output_path,
                        "--engine", "latentsync",
                        "--latentsync_dir", cmd_args.latentsync_dir,
                        "--align_mode", "none",  # 禁止lipsync_cli调整音频速度，音频对齐已在上游处理
                    ]
                    
                    if cmd_args.latentsync_python:
                        lipsync_cmd.extend(["--latentsync_python", cmd_args.latentsync_python])
                    if cmd_args.latentsync_low_vram:
                        lipsync_cmd.append("--low_vram")
                    if not cmd_args.verbose:
                        lipsync_cmd.append("--quiet")
                    
                    # 调试信息：显示传递给lipsync_cli的参数
                    print(f">> [DEBUG] LipSync命令参数:")
                    print(f"   - video: {video_to_use}")
                    print(f"   - audio: {tts_audio}")
                    print(f"   - output: {output_path}")
                    print(f"   - align_mode: none")
                    
                    # 检查传递给lipsync的音频时长
                    final_audio_duration = get_audio_duration(tts_audio)
                    print(f"   - 传递给lipsync的音频时长: {final_audio_duration:.3f}s")
                    
                    print(f">> Running LipSync subprocess...")
                    lipsync_result = subprocess.run(lipsync_cmd, capture_output=not cmd_args.verbose, text=True)
                    
                    if lipsync_result.returncode != 0:
                        print(f"LipSync subprocess failed!")
                        if not cmd_args.verbose:
                            print(f"STDOUT: {lipsync_result.stdout}")
                            print(f"STDERR: {lipsync_result.stderr}")
                        raise RuntimeError(f"口型同步失败")
                    
                    if not os.path.exists(output_path):
                        raise RuntimeError("输出视频文件未生成")
                    
                    # 调试信息：检查最终输出视频的时长
                    from tools.video_dub import get_video_duration as get_vid_dur
                    output_video_duration = get_vid_dur(output_path)
                    print(f">> [DEBUG] 最终输出视频时长: {output_video_duration:.3f}s")
                    print(f">> [DEBUG] session目录: {temp_dir}")
                    
                    progress(1.0, desc=i18n("完成"))
                    
                    return output_path, f"✅ {i18n('视频生成成功')} ({LIP_SYNC_ENGINE_NAME} 子进程模式): {output_path}"
                    
                except Exception as e:
                    import traceback
                    error_msg = f"❌ {i18n('生成失败')}: {str(e)}"
                    print(f"Video dubbing error: {e}")
                    traceback.print_exc()
                    return None, error_msg
            
            def generate_dubbed_video_inprocess(
                video_path, script_text, spk_audio, align_duration,
                spk_extract_start, spk_extract_duration, emo_control, emo_alpha,
                temp_dir, progress
            ):
                """
                常规模式：TTS 在主进程运行，LipSync 在子进程运行。
                """
                import shutil
                
                try:
                    progress(0.1, desc=i18n("提取参考音频..."))
                    
                    # 复制原始视频到session目录，便于溯源
                    video_ext = os.path.splitext(video_path)[1] or '.mp4'
                    local_video_path = os.path.join(temp_dir, f"input_video{video_ext}")
                    shutil.copy(video_path, local_video_path)
                    print(f">> [DEBUG] 原始视频已复制到: {local_video_path}")
                    video_path = local_video_path  # 后续使用本地副本
                    
                    # Step 1: Extract reference audio
                    if spk_audio is None:
                        spk_audio_path = os.path.join(temp_dir, "spk_prompt.wav")
                        extract_audio_from_video(
                            video_path,
                            spk_audio_path,
                            sample_rate=24000,
                            start_time=float(spk_extract_start),
                            duration=float(spk_extract_duration),
                            verbose=cmd_args.verbose
                        )
                    else:
                        spk_audio_path = spk_audio
                    
                    # Get original audio duration for alignment
                    orig_audio_path = os.path.join(temp_dir, "orig_audio.wav")
                    extract_audio_from_video(video_path, orig_audio_path, verbose=False)
                    orig_duration = get_audio_duration(orig_audio_path)
                    
                    progress(0.3, desc=i18n("生成配音..."))
                    
                    # Step 2: Generate TTS audio
                    tts_audio_raw = os.path.join(temp_dir, "tts_audio_raw.wav")
                    
                    use_emo_text = (emo_control == i18n("根据文本推断情感"))
                    
                    tts.infer(
                        spk_audio_prompt=spk_audio_path,
                        text=script_text.strip(),
                        output_path=tts_audio_raw,
                        emo_alpha=float(emo_alpha) if use_emo_text else 1.0,
                        use_emo_text=use_emo_text,
                        verbose=cmd_args.verbose
                    )
                    
                    # 调试信息：显示音视频时长对比
                    tts_duration = get_audio_duration(tts_audio_raw)
                    print(f">> [DEBUG] 音视频时长对比:")
                    print(f"   - 视频时长(原始音频): {orig_duration:.3f}s")
                    print(f"   - TTS音频时长: {tts_duration:.3f}s")
                    print(f"   - 差异: {tts_duration - orig_duration:.3f}s")
                    print(f"   - align_duration={align_duration}")
                    
                    # Optional: Align duration
                    if align_duration:
                        progress(0.5, desc=i18n("对齐音频时长..."))
                        print(f">> [DEBUG] 选择对齐音频时长模式")
                        tts_audio = os.path.join(temp_dir, "tts_audio.wav")
                        align_audio_duration(tts_audio_raw, orig_duration, tts_audio, 
                                           orig_audio_path=orig_audio_path, verbose=True)
                        # 验证对齐后的音频时长
                        aligned_duration = get_audio_duration(tts_audio)
                        print(f">> [DEBUG] 对齐后音频时长: {aligned_duration:.3f}s")
                    else:
                        print(f">> [DEBUG] 未启用对齐")
                        tts_audio = tts_audio_raw
                    
                    progress(0.6, desc=i18n("生成口型同步视频..."))
                    
                    # Step 3: Generate lip-synced video
                    # Free up GPU memory before calling LatentSync
                    import torch
                    import gc
                    
                    # Move TTS models to CPU temporarily to free GPU memory
                    tts_models_to_offload = [
                        'gpt', 'semantic_model', 'semantic_codec', 's2mel',
                        'campplus_model', 'bigvgan'
                    ]
                    tts_tensors_to_offload = [
                        'semantic_mean', 'semantic_std', 'emo_matrix', 'spk_matrix',
                        # Cache tensors that may hold GPU memory
                        'cache_spk_cond', 'cache_s2mel_style', 'cache_s2mel_prompt',
                        'cache_emo_cond', 'cache_mel'
                    ]
                    
                    # Also offload QwenEmotion model if it exists
                    if hasattr(tts, 'qwen_emo') and hasattr(tts.qwen_emo, 'model'):
                        try:
                            tts.qwen_emo.model.cpu()
                            if cmd_args.verbose:
                                print(f">> Moved qwen_emo.model to CPU")
                        except Exception as e:
                            if cmd_args.verbose:
                                print(f">> Failed to move qwen_emo.model to CPU: {e}")
                    
                    for attr_name in tts_models_to_offload:
                        if hasattr(tts, attr_name):
                            model = getattr(tts, attr_name)
                            if model is not None and hasattr(model, 'cpu'):
                                try:
                                    model.cpu()
                                except Exception as e:
                                    if cmd_args.verbose:
                                        print(f">> Failed to move {attr_name} to CPU: {e}")
                    
                    for attr_name in tts_tensors_to_offload:
                        if hasattr(tts, attr_name):
                            tensor = getattr(tts, attr_name)
                            if tensor is not None:
                                try:
                                    if isinstance(tensor, (list, tuple)):
                                        # Handle split tensors like emo_matrix and spk_matrix
                                        setattr(tts, attr_name, tuple(t.cpu() if hasattr(t, 'cpu') else t for t in tensor))
                                    elif hasattr(tensor, 'cpu'):
                                        setattr(tts, attr_name, tensor.cpu())
                                except Exception as e:
                                    if cmd_args.verbose:
                                        print(f">> Failed to move {attr_name} to CPU: {e}")
                    
                    # Clear CUDA cache aggressively
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Wait for all CUDA operations to complete
                    
                    if cmd_args.verbose:
                        # Print current GPU memory usage
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / 1024**3
                            reserved = torch.cuda.memory_reserved() / 1024**3
                            print(f">> GPU memory after offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                    
                    output_path = os.path.join(current_dir, "outputs", f"video_dub_{int(time.time())}.mp4")
                    
                    # 调试信息：显示传递给lip_sync_engine的参数
                    final_audio_duration = get_audio_duration(tts_audio)
                    print(f">> [DEBUG] LipSync参数:")
                    print(f"   - video: {video_path}")
                    print(f"   - audio: {tts_audio}")
                    print(f"   - output: {output_path}")
                    print(f"   - 传递给lipsync的音频时长: {final_audio_duration:.3f}s")
                    
                    try:
                        lip_sync_engine.generate(
                            video_path=video_path,
                            audio_path=tts_audio,
                            output_path=output_path,
                            verbose=cmd_args.verbose
                        )
                    finally:
                        # Reload TTS models to GPU after lip sync
                        device = torch.device(tts.device if hasattr(tts, 'device') else 'cuda')
                        
                        # Reload QwenEmotion model if it exists
                        if hasattr(tts, 'qwen_emo') and hasattr(tts.qwen_emo, 'model'):
                            try:
                                tts.qwen_emo.model.to(device)
                                if cmd_args.verbose:
                                    print(f">> Moved qwen_emo.model to GPU")
                            except Exception as e:
                                if cmd_args.verbose:
                                    print(f">> Failed to move qwen_emo.model to GPU: {e}")
                        
                        for attr_name in tts_models_to_offload:
                            if hasattr(tts, attr_name):
                                model = getattr(tts, attr_name)
                                if model is not None and hasattr(model, 'to'):
                                    try:
                                        model.to(device)
                                    except Exception as e:
                                        if cmd_args.verbose:
                                            print(f">> Failed to move {attr_name} to GPU: {e}")
                        
                        for attr_name in tts_tensors_to_offload:
                            if hasattr(tts, attr_name):
                                tensor = getattr(tts, attr_name)
                                if tensor is not None:
                                    try:
                                        if isinstance(tensor, (list, tuple)):
                                            setattr(tts, attr_name, tuple(t.to(device) if hasattr(t, 'to') else t for t in tensor))
                                        elif hasattr(tensor, 'to'):
                                            setattr(tts, attr_name, tensor.to(device))
                                    except Exception as e:
                                        if cmd_args.verbose:
                                            print(f">> Failed to move {attr_name} to GPU: {e}")
                        
                        if cmd_args.verbose:
                            print(f">> TTS models reloaded to GPU")
                    
                    # 调试信息：检查最终输出视频的时长
                    from tools.video_dub import get_video_duration as get_vid_dur
                    output_video_duration = get_vid_dur(output_path)
                    print(f">> [DEBUG] 最终输出视频时长: {output_video_duration:.3f}s")
                    print(f">> [DEBUG] session目录: {temp_dir}")
                    
                    progress(1.0, desc=i18n("完成"))
                    
                    return output_path, f"✅ {i18n('视频生成成功')} ({LIP_SYNC_ENGINE_NAME}): {output_path}"
                    
                except Exception as e:
                    import traceback
                    error_str = str(e)
                    if "Face not detected" in error_str:
                        error_msg = f"❌ {i18n('生成失败')}: 视频中未检测到人脸，请确保视频中有清晰可见的正脸"
                    else:
                        error_msg = f"❌ {i18n('生成失败')}: {error_str}"
                    print(f"Video dubbing error: {e}")
                    traceback.print_exc()
                    return None, error_msg
                    
                finally:
                    # Keep temp files for debugging on error
                    # To clean up manually: rm -rf outputs/video_dub_temp/
                    pass
            
            video_gen_button.click(
                generate_dubbed_video,
                inputs=[
                    video_input,
                    video_script,
                    video_spk_audio,
                    video_align_duration,
                    video_spk_extract_start,
                    video_spk_extract_duration,
                    video_emo_control,
                    video_emo_alpha
                ],
                outputs=[video_output, video_status]
            )


if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
