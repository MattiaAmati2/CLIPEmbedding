"""
Process interpolation_experiment *_complete.csv files.

    Actions performed for each matching file under the `results` tree:
    - detect delimiter and read file
    - ensure the second column is named 'point'
    - sort rows by 'shot_number' then by numeric 'point' ascending
    - after sorting, replace each numeric point value with (value - 1) / 127 rounded to 4 decimals
    - write the file back using the same delimiter

This script only modifies the 'point' values (and column name if necessary).
Other numeric values are preserved as-read; the point column is written with exactly two decimals.
"""
from pathlib import Path
import pandas as pd
import csv
import sys


def detect_separator(path: Path):
    # Try to sniff the delimiter using csv.Sniffer on a sample
    text = path.read_text()
    sample = text[:1024]
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        # fallback: if there are tabs, use '\t', else comma
        if '\t' in sample:
            return '\t'
        return ','


def process_file(path: Path):
    sep = detect_separator(path)
    # Read with inferred separator. Use engine='python' to accept regex separators from sniff.
    try:
        df = pd.read_csv(path, sep=sep, engine='python')
    except Exception as e:
        print(f"ERROR reading {path}: {e}")
        return

    if df.shape[1] < 2:
        print(f"SKIP {path}: less than 2 columns")
        return

    # Ensure second column is named 'point'
    cols = list(df.columns)
    if cols[1] != 'point':
        cols[1] = 'point'
        df.columns = cols

    # Prepare numeric shot column
    if 'shot_number' not in df.columns:
        # try common alternative names
        possible = [c for c in df.columns if 'shot' in c.lower()]
        if possible:
            df = df.rename(columns={possible[0]: 'shot_number'})
        else:
            print(f"SKIP {path}: no shot_number column found")
            return

    # Create numeric sort key for point (coerce non-numeric to NaN)
    sort_point = pd.to_numeric(df['point'], errors='coerce')
    # For rows where point is NaN (non-numeric) place them after numeric points
    sort_point_filled = pd.Series(sort_point).fillna(1e9)

    # Sort by shot_number (ascending numeric if possible) then by point
    shot_sort_series = pd.Series(pd.to_numeric(df['shot_number'], errors='coerce')).fillna(pd.Series(df['shot_number']))
    df['_sort_shot'] = shot_sort_series
    df['_sort_point'] = sort_point_filled
    df = df.sort_values(by=['_sort_shot', '_sort_point'], kind='mergesort')
    df = df.drop(columns=['_sort_shot', '_sort_point'])

    # Now transform the point values: numeric -> (value - 1) / 127 rounded to 4 decimals
    def transform_point(v):
        try:
            num = float(v)
        except Exception:
            return v  # leave non-numeric as-is
        # perform transform and round to 4 decimals
        new = round((num - 1.0) / 127.0, 4)
        # format to four decimals exactly
        return f"{new:.4f}"

    df['point'] = df['point'].apply(transform_point)

    # Write back using the original separator
    try:
        # When using tab, ensure sep='\t'
        df.to_csv(path, sep=sep, index=False)
        print(f"Updated: {path}")
    except Exception as e:
        print(f"ERROR writing {path}: {e}")


def main():
    results_dir = Path('results')
    if not results_dir.exists():
        print('results directory not found')
        sys.exit(1)

    # Walk recursively and process files with 'interpolation_experiment' and 'complete' in name
    files = list(results_dir.rglob('*interpolation*complete*.csv'))
    # also catch possible typo 'completel'
    files += list(results_dir.rglob('*interpolation*completel*.csv'))

    if not files:
        print('No interpolation complete files found under results')
        return

    for f in sorted(set(files)):
        process_file(f)


if __name__ == '__main__':
    main()


