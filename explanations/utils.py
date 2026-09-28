import pandas as pd

def get_segments(base_segment_id, label, path):
    df = pd.read_csv(path)
    filtered_df = df[
        (df['segment_id'] == base_segment_id) &
        (df['mapped_name'] == label)
    ]
    if filtered_df.empty:
        return []

    return filtered_df[['start_time_seconds', 'end_time_seconds']].values.tolist()