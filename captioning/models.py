#!/usr/bin/env python3
"""
Unified model loader and runner for all ALM backends.

Supported backends:
  - qwen_omni         (Qwen2.5-Omni-7B, local HF)
  - qwen3_omni        (Qwen3-Omni-30B-A3B, local HF)
  - audio_flamingo    (NVIDIA Audio Flamingo 3, local HF)
  - audio_flamingo_next (NVIDIA Audio Flamingo Next, local HF)
  - kimi_audio        (Moonshot Kimi-Audio-7B, local)
  - music_flamingo    (NVIDIA Music Flamingo, local HF)

Usage:
    model_bundle = load_model(backend, model_id=...)
    text = run_model(model_bundle, audio_array, prompt, sample_rate, max_new_tokens)
"""

import os
import tempfile
import numpy as np
import soundfile as sf

DEFAULT_MODEL_IDS = {
    "qwen_omni":           "Qwen/Qwen2.5-Omni-7B",
    "qwen3_omni":          "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "qwen3_captioner":     "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    "audio_flamingo":      "nvidia/audio-flamingo-3-hf",
    "audio_flamingo_next": "nvidia/audio-flamingo-next-hf",
    "kimi_audio":          "moonshotai/Kimi-Audio-7B-Instruct",
    "music_flamingo":      "nvidia/music-flamingo-hf",
}


#  Qwen2.5-Omni (local HF)
def _load_qwen_omni(model_id):
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    return {"backend": "qwen_omni", "model": model, "processor": processor}


def _run_qwen_omni(bundle, audio_array, prompt, sample_rate, max_new_tokens):
    import torch

    model = bundle["model"]
    processor = bundle["processor"]

    # Write audio to temp file (processor expects a path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_array, sample_rate)
        tmp_path = tmp.name

    try:
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful audio analysis assistant. Provide extremely brief captions. Do not elaborate or provide long descriptions."}],        
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": tmp_path},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, return_audio=False)

        gen_ids = gen[0] if isinstance(gen, tuple) else gen
        prompt_len = inputs["input_ids"].shape[1]
        gen_only = gen_ids[:, prompt_len:]

        text_out = processor.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return text_out
    finally:
        os.unlink(tmp_path)


#  Qwen3-Omni (local HF)
def _load_qwen3_omni(model_id):
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    model.eval()
    return {"backend": "qwen3_omni", "model": model, "processor": processor}


def _run_qwen3_omni(bundle, audio_array, prompt, sample_rate, max_new_tokens):
    import torch
    from qwen_omni_utils import process_mm_info

    model = bundle["model"]
    processor = bundle["processor"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_array, sample_rate)
        tmp_path = tmp.name

    try:
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful audio analysis assistant. Provide extremely brief captions. Do not elaborate or provide long descriptions."}],    
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": tmp_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        USE_AUDIO_IN_VIDEO = False

        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )

        inputs = {
            k: (v.to(model.device).to(model.dtype) if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
                 else v.to(model.device) if isinstance(v, torch.Tensor)
                 else v)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            text_ids, _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_audio=False,
                thinker_return_dict_in_generate=True,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
            )

        text_out = processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return text_out
    finally:
        os.unlink(tmp_path)

#  Audio Flamingo 3 (local HF)
def _load_audio_flamingo(model_id):
    import torch
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return {"backend": "audio_flamingo", "model": model, "processor": processor}


def _run_audio_flamingo(bundle, audio_array, prompt, sample_rate, max_new_tokens):
    import torch

    model = bundle["model"]
    processor = bundle["processor"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_array, sample_rate)
        tmp_path = tmp.name

    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": tmp_path},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(model.device)

        with torch.inference_mode():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        prompt_len = inputs["input_ids"].shape[1]
        gen_only = gen_ids[:, prompt_len:]

        text_out = processor.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return text_out
    finally:
        os.unlink(tmp_path)


#  Audio Flamingo Next (local HF)
def _load_audio_flamingo_next(model_id):
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return {"backend": "audio_flamingo_next", "model": model, "processor": processor}


def _run_audio_flamingo_next(bundle, audio_array, prompt, sample_rate, max_new_tokens):
    import torch

    model = bundle["model"]
    processor = bundle["processor"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_array, sample_rate)
        tmp_path = tmp.name

    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": tmp_path},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(model.device)

        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(model.dtype)

        with torch.inference_mode():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        prompt_len = inputs["input_ids"].shape[1]
        gen_only = gen_ids[:, prompt_len:]

        text_out = processor.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return text_out
    finally:
        os.unlink(tmp_path)

#  Kimi Audio (local, custom API)
def _load_kimi_audio(model_id):
    from kimia_infer.api.kimia import KimiAudio

    model = KimiAudio(model_path=model_id, load_detokenizer=False)
    return {"backend": "kimi_audio", "model": model}


def _run_kimi_audio(bundle, audio_array, prompt, sample_rate, max_new_tokens):
    import torch

    model = bundle["model"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_array, sample_rate)
        tmp_path = tmp.name

    try:
        messages = [
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "user", "message_type": "audio", "content": tmp_path},
        ]

        sampling_params = {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }

        with torch.inference_mode():
            _, text_out = model.generate(
                messages,
                **sampling_params,
                output_type="text",
                max_new_tokens=max_new_tokens,
            )

        return text_out.strip()
    finally:
        os.unlink(tmp_path)


#  Music Flamingo (local HF)
def _load_music_flamingo(model_id):
    import torch
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return {"backend": "music_flamingo", "model": model, "processor": processor}


def _run_music_flamingo(bundle, audio_array, prompt, sample_rate, max_new_tokens):
    import torch

    model = bundle["model"]
    processor = bundle["processor"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_array, sample_rate)
        tmp_path = tmp.name

    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": tmp_path},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(model.device)

        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)

        with torch.inference_mode():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        prompt_len = inputs["input_ids"].shape[1]
        gen_only = gen_ids[:, prompt_len:]

        text_out = processor.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return text_out
    finally:
        os.unlink(tmp_path)

#  Qwen3-Omni Captioner (local HF, audio-only, NO text prompt)
def _load_qwen3_captioner(model_id):
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
    return {"backend": "qwen3_captioner", "model": model, "processor": processor}


def _run_qwen3_captioner(bundle, audio_array, prompt, sample_rate, max_new_tokens):
    """Run Qwen3-Omni-Captioner.

    NOTE: This model does NOT accept text prompts.  The *prompt* argument
    is **ignored** — the conversation contains only the audio input.
    """
    import torch
    from qwen_omni_utils import process_mm_info

    if prompt:
        print("[qwen3_captioner] WARNING: prompt ignored — model is audio-only.")

    model = bundle["model"]
    processor = bundle["processor"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_array, sample_rate)
        tmp_path = tmp.name

    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": tmp_path},
                ],
            },
        ]

        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        with torch.inference_mode():
            text_ids, _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_audio=False,
                thinker_return_dict_in_generate=True,
            )

        text_out = processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return text_out
    finally:
        os.unlink(tmp_path)

#  Dispatcher tables
_LOADERS = {
    "qwen_omni":           _load_qwen_omni,
    "qwen3_omni":          _load_qwen3_omni,
    "audio_flamingo":      _load_audio_flamingo,
    "audio_flamingo_next": _load_audio_flamingo_next,
    "kimi_audio":          _load_kimi_audio,
    "music_flamingo":      _load_music_flamingo,
    "qwen3_captioner":     _load_qwen3_captioner
}

_RUNNERS = {
    "qwen_omni":           _run_qwen_omni,
    "qwen3_omni":          _run_qwen3_omni,
    "audio_flamingo":      _run_audio_flamingo,
    "audio_flamingo_next": _run_audio_flamingo_next,
    "kimi_audio":          _run_kimi_audio,
    "music_flamingo":      _run_music_flamingo,
    "qwen3_captioner":     _run_qwen3_captioner
}


#  Public API
def load_model(backend, model_id=None):
    """
    Load a model bundle for the given backend.

    Parameters
    ----------
    backend : str
        One of the supported backend names.
    model_id : str, optional
        HuggingFace model ID or API model name. Falls back to DEFAULT_MODEL_IDS.

    Returns
    -------
    dict
        A model bundle dict consumed by run_model().
    """
    if backend not in _LOADERS:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: {list(_LOADERS.keys())}"
        )
    if model_id is None:
        model_id = DEFAULT_MODEL_IDS[backend]

    print(f"[models] Loading backend='{backend}' model_id='{model_id}' ...")
    bundle = _LOADERS[backend](model_id)
    print(f"[models] Backend '{backend}' loaded successfully.")
    return bundle


def run_model(model_bundle, audio_array, prompt, sample_rate=16000, max_new_tokens=2048):
    """
    Run inference on an audio array with the given prompt.

    Parameters
    ----------
    model_bundle : dict
        The bundle returned by load_model().
    audio_array : np.ndarray
        Mono audio waveform.
    prompt : str
        Text prompt for the model.
    sample_rate : int
        Sample rate of the audio array.
    max_new_tokens : int
        Maximum number of tokens to generate.

    Returns
    -------
    str
        Generated text response.
    """
    backend = model_bundle["backend"]
    if backend not in _RUNNERS:
        raise ValueError(f"No runner for backend '{backend}'.")
    return _RUNNERS[backend](model_bundle, audio_array, prompt, sample_rate, max_new_tokens)




