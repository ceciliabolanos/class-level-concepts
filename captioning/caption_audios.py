#!/usr/bin/env python3
"""
Captioning script for all ALM backends.

Iterates over explanation directories and generates audio descriptions
using any of the supported ALM backends.

Usage:
    python caption_audios.py --backend qwen_omni
    ...

Supported backends:
    qwen_omni, qwen3_omni, audio_flamingo,
    audio_flamingo_next, kimi_audio, music_flamingo,
    qwen3_omni_captioner
"""

import json
import os
import time
import numpy as np
from pathlib import Path
import librosa
import argparse
import soundfile as sf
import tempfile
from models import load_model, run_model, DEFAULT_MODEL_IDS
import ast
import pandas as pd
from utils import mute_audio_hamming, mute_audio_except_timestamps

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run audio captioning with any ALM backend."
)
parser.add_argument(
    "--backend",
    choices=list(DEFAULT_MODEL_IDS.keys()),
    required=True,
    help="ALM backend to use for captioning.",
)
parser.add_argument(
    "--model_id",
    default=None,
    help="Override the default model ID for the backend.",
)
parser.add_argument(
    "--base_dir",
    type=str,
    help="Base directory with explanation sub-folders.",
)
parser.add_argument(
    "--methods",
    nargs="+",
    default=["RF"],
    help="Explanation methods to process (e.g. RF TRUE_TS).",
)
parser.add_argument(
    "--method",
    nargs="+",
    default="complete",
    help="Explanation methods to process (e.g. RF TRUE_TS).",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=2048,
    help="Maximum tokens for generation.",
)
parser.add_argument(
    "--percentile",
    type=int,
    default=90,
    help="Percentile threshold for selecting important time windows.",
)
parser.add_argument(
    "--metadata_csv",
    type=str,
    help="CSV con la columna 'path' (tupla sqf_path, rel_path).",
)
args = parser.parse_args()

BACKEND = args.backend
base_dir = Path(args.base_dir)
methods = args.methods
SAMPLE_RATE = 16_000
df_test = pd.read_csv(args.metadata_csv)

MODEL = load_model(BACKEND, model_id=args.model_id)
    
possible_labels = ["Applause", "Dog", "Bird", "Car", "Liquid"]
for label in possible_labels:
    for subdir in sorted(base_dir.iterdir(), reverse=True):
        if not subdir.is_dir():
            continue
        json_path = subdir / f"ft_7000_{label}.json"
        out_path  = subdir / f"captions_{BACKEND}_{label}_{args.method}_{args.percentile}.json"
    
        print(  f"\nProcesando carpeta: {subdir.name} con archivo: {json_path.name}")

        if not json_path.exists():
            continue
        if out_path.exists():
            continue
        explanations = {}

        print(f"\nProcesando carpeta: {subdir.name}")

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        idx = data.get('metadata', {}).get('filename', None)  
        
        if idx is None or idx not in df_test['segment_id'].values:
            print(f"  -> Índice no encontrado en CSV: {idx}. Omitiendo.")
            continue

        row = df_test.loc[df_test['segment_id'] == idx].iloc[0]
        _sqf_path, rel_path = ast.literal_eval(row['path'])
        audio_path = str(Path('/') / rel_path)

        if not os.path.exists(audio_path):
            print(f"  -> Archivo de audio no encontrado: {audio_path}. Omitiendo.")
            continue

        print(f"  -> Cargando audio con librosa...")
        audio_array, _ = librosa.core.load(audio_path, sr=SAMPLE_RATE)
        audio_array = audio_array.astype(np.float32)
        print(f"  -> Audio cargado: {len(audio_array)/SAMPLE_RATE:.2f}s")

        for method in methods:
            values = np.array(data["importance_scores"][method]["values"])
            threshold = np.percentile(values, args.percentile)
            important_indices = np.where(values >= threshold)[0]
            timestamp_strings = [
                f"[{i * 0.1:.1f}s to {(i + 1) * 0.1:.1f}s]"
                for i in important_indices
            ]

            formatted_timestamps = ", ".join(timestamp_strings)
            prompt = """Listen to the audio and describe what happens during the moments when there is sound. Completely ignore the silent parts. 

            Give me a simple, chronological description of the events, voices, or noises you hear."""

            try:
                if args.method == "hamming":
                    silence_audio_array = mute_audio_hamming(audio_array, timestamp_strings, SAMPLE_RATE)
                elif args.method == "complete":
                    silence_audio_array = audio_array
                else:    
                    silence_audio_array = mute_audio_except_timestamps(audio_array, timestamp_strings, SAMPLE_RATE)

                t0 = time.perf_counter()
                full_description = run_model(
                    model_bundle=MODEL,
                    audio_array=silence_audio_array,
                    prompt=prompt,
                    sample_rate=SAMPLE_RATE,
                    max_new_tokens=args.max_new_tokens,
                )
                elapsed = time.perf_counter() - t0

                explanations[method] = {
                    "alm_backend":         BACKEND,
                    "alm_model_id":        args.model_id or DEFAULT_MODEL_IDS[BACKEND],
                    "alm_prompt":          prompt,
                    "description":         full_description,
                    "timestamps_used":     timestamp_strings,
                    "inference_time_sec":  round(elapsed, 2),
                }
                print(f"  -> [{method}] OK ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  -> Error procesando {method}: {e}")

        with open(out_path, 'w') as f:
            json.dump(explanations, f, indent=4)

