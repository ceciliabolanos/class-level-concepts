import numpy as np

def mute_audio_except_timestamps(audio_array, timestamp_strings, sample_rate):
    """
    Devuelve una copia de audio_array donde todo el audio fuera de los
    intervalos indicados en timestamp_strings se reemplaza por ceros.
    Los intervalos SÍ conservan el audio original.
    
    timestamp_strings: lista de strings tipo "[0.3s to 0.4s]"
    """
    muted = np.zeros_like(audio_array)

    for ts in timestamp_strings:
        # extrae los dos números del string "[0.3s to 0.4s]"
        start_str, end_str = ts.strip("[]").split(" to ")
        start_sec = float(start_str.replace("s", ""))
        end_sec = float(end_str.replace("s", ""))

        start_idx = int(round(start_sec * sample_rate))
        end_idx = int(round(end_sec * sample_rate))

        # clamp por si algún índice se pasa del largo del audio
        start_idx = max(0, min(start_idx, len(audio_array)))
        end_idx = max(0, min(end_idx, len(audio_array)))

        muted[start_idx:end_idx] = audio_array[start_idx:end_idx]

    return muted

def mute_audio_hamming(audio_array, timestamp_strings, sample_rate, fade_duration=0.05):
    mask = np.zeros(len(audio_array), dtype=np.float32)

    for ts in timestamp_strings:
        start_str, end_str = ts.strip("[]").split(" to ")
        start_sec = float(start_str.replace("s", ""))
        end_sec = float(end_str.replace("s", ""))

        start_idx = int(round(start_sec * sample_rate))
        end_idx = int(round(end_sec * sample_rate))

        start_idx = max(0, min(start_idx, len(audio_array)))
        end_idx = max(0, min(end_idx, len(audio_array)))

        mask[start_idx:end_idx] = 1.0

    window_size = int(fade_duration * sample_rate)
    if window_size % 2 == 0:
        window_size += 1
        
    hamming_kernel = np.hamming(window_size)
    hamming_kernel /= hamming_kernel.sum() 

    smoothed_mask = np.convolve(mask, hamming_kernel, mode='same')
    return audio_array * smoothed_mask
