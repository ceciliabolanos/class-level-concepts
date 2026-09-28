import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from explanations.explainers import explanation_config
from explanations.explainers.explanation_config import get_explanation_config
from explanations.explainers.explanation_generator import ExplanationGenerator
from explanations.perturbations.temporal import TemporalPerturbations
from explanations.perturbations.config import get_temporal_config
from explanations.utils import get_segments
from model.model import AudiosetModel


DICT_EXPLAINERS = {'Applause': 0, 'Dog': 1, 'Bird': 2, 'Car': 3, 'Liquid': 4}


def parse_args():
    parser = argparse.ArgumentParser(description="Generación de explicaciones para AudioSet")

    parser.add_argument('--data-root', type=str, default=f'/home/{os.environ.get("USER", "")}',
                         help="Raíz donde viven repos/ y mnt-beta/ (default: /mnt/data/$USER)")
    parser.add_argument('--labels', type=str, nargs='+', required=True,
                         help="Uno o dos labels. Ej: --labels Bird  |  --labels Bird Wind (con --correlacion)")
    parser.add_argument('--correlacion', action='store_true',)
    parser.add_argument('--labels-csv', type=str, default=None,
                         help="Override manual del CSV de metadata (si no, se infiere de --data-root/--labels/--correlacion)")
    parser.add_argument('--model-checkpoint', type=str, default=None,
                         help="Override manual del checkpoint del modelo (si no, se infiere igual que labels-csv)")

    parser.add_argument('--experiments', type=str, nargs='+', default=['audioset_dog_noise'])
    parser.add_argument('--explainer-configs', type=str, nargs='+', default=['paper_default'])
    parser.add_argument('--num-samples', type=int, default=7000)
    parser.add_argument('--test-batch-indices', type=int, nargs='+', default=[14])

    return parser.parse_args()


def main():
    args = parse_args()

    labels = args.labels
    correlacion = args.correlacion
    if correlacion and len(labels) < 2:
        raise ValueError("--correlacion requiere pasar dos labels: --labels <label1> <label2>")

    model_number = DICT_EXPLAINERS[labels[0]]

    if args.labels_csv is not None:
        LABELS_PATH = args.labels_csv
    elif correlacion:
        LABELS_PATH = f'/repos/audioset/datasets/model{model_number}_spu.csv'
    else:
        LABELS_PATH = f'/repos/audioset/datasets/model{model_number}_cln.csv'

    if args.model_checkpoint is not None:
        MODEL_PATH = args.model_checkpoint

    name = f'{labels[0]}_{labels[1]}' if correlacion else f'{labels[0]}'

    print('LABELS_PATH:', LABELS_PATH)
    print('MODEL_PATH:', MODEL_PATH)
    print('name:', name)

    df = pd.read_csv(LABELS_PATH)
    df = df.drop_duplicates(subset=['mapped_name', 'segment_id']).reset_index(drop=True)
    df_test = df.loc[df['split'] == 'test'].reset_index(drop=True)
    print(len(df_test), 'test samples','in model', model_number, 'correlacion:', correlacion)

    for explainer_config_name in args.explainer_configs:
        for exp_name in args.experiments:
            DICT_AMOUNT = {'Applause': 0, 'Dog': 0, 'Bird': 0, 'Car': 0, 'Liquid': 0}
            config = get_temporal_config(exp_name)
            num_batches = 8000 // config.num_samples

            config_dir = f'/scratch/cb6117/explanations-timelocalized/subset_audioset/perturbations/{name}/configs'
            config.save(config_dir)
            print(f"\n✓ Config saved to: {config_dir}/{config.get_experiment_name()}_config.json")

            model = AudiosetModel(None, 1, MODEL_PATH)
            for i in tqdm(range(len(df_test)), desc=f"Processing {exp_name}"):
                filename = df_test.loc[i, 'segment_id']
                try:
                    model.audio_path = df_test.loc[i, 'path']
                    label = df_test.loc[i, 'mapped_name']
                    input_audio, outputs = model.process_input()

                    predicted_class = np.argmax(outputs)
                    if (predicted_class != DICT_EXPLAINERS[label]) or (DICT_AMOUNT[label] >= 40):
                        print(f"  Skipping {filename} as predicted class {predicted_class} does not match label {label}.")
                        continue
                    DICT_AMOUNT[label] += 1
                    real_score_id = outputs[predicted_class]
                    predict_fn = model.get_predict_fn()

                    output_dir = f'/scratch/cb6117/explanations-timelocalized/subset_audioset/perturbations/{name}/{config.get_experiment_name()}'
                    audio_dir = f'{output_dir}/{os.path.basename(filename)}'

                    for batch_idx in range(num_batches):
                        output_filepath = f'{audio_dir}/batch{batch_idx}.npz'

                        if os.path.exists(output_filepath):
                            if batch_idx == num_batches - 1:
                                print(f"  Perturbations exist for {os.path.basename(filename)}, skipping generation...")
                            continue

                        data_generator = TemporalPerturbations(
                            audio=input_audio,
                            sample_rate=16000,
                            config=config,
                            predict_fn=predict_fn,
                            path=output_filepath
                        )
                        data_generator.generate()
                    print(DICT_AMOUNT)
                    # Si hay correlacion, calcular AUC con labels[0] y labels[1] para ver qué usa el modelo para predecir
                    true_markers = get_segments(filename, label, LABELS_PATH)
                    print(f"  True markers for {label} in {filename}: {true_markers}")
                    explanation_config_obj = get_explanation_config(explainer_config_name)
                    output_path = f'/scratch/cb6117/explanations-timelocalized/subset_audioset/{name}/{exp_name}/{os.path.basename(filename)}'
                    result_file = f'{output_path}/ft_{args.num_samples}_{label}.json'
                    os.makedirs(output_path, exist_ok=True)
                    print(f"This are the true markers: {true_markers}")
                    if os.path.exists(result_file):
                        continue

                    generator = ExplanationGenerator()
                    generator.generate_surrogate_explanations(
                        filename=filename,
                        id_to_explain=predicted_class,
                        path=audio_dir,
                        true_markers=true_markers,
                        num_samples=args.num_samples,
                        num_batch=config.num_samples,
                        output_path=result_file,
                        explanation_config=explanation_config_obj,
                        test_batch_indices=args.test_batch_indices,
                        audio=input_audio,
                        sample_rate=16000
                    )

                    if correlacion & (predicted_class == DICT_EXPLAINERS[labels[0]]):
                        correlation_markers = get_segments(filename, labels[1], LABELS_PATH)
                        print(f"  Correlation markers for {labels[1]} in {filename}: {correlation_markers}")
                        result_file = f'{output_path}/ft_{args.num_samples}_correlacion_{label}.json'

                        if os.path.exists(result_file):
                            continue

                        generator = ExplanationGenerator()
                        generator.generate_surrogate_explanations(
                            filename=filename,
                            id_to_explain=predicted_class,
                            path=audio_dir,
                            true_markers=correlation_markers,
                            num_samples=args.num_samples,
                            num_batch=config.num_samples,
                            output_path=result_file,
                            explanation_config=explanation_config_obj,
                            test_batch_indices=args.test_batch_indices,
                            audio=input_audio,
                            sample_rate=16000
                        )
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"  Error processing {filename}: {e}")

            del model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()