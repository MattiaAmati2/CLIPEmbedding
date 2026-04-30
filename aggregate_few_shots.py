import os
import pandas as pd
from pathlib import Path

#ai-generated script to help with the csv setup required for plotting

# Define directories
previous_results_dir = Path('/mnt/c/Users/Admin/PycharmProjects/CLIPEmbedding/previous_results')
results_dir = Path('/mnt/c/Users/Admin/PycharmProjects/CLIPEmbedding/results')

# Get all subdirectories in previous_results
subdirs = sorted([d for d in previous_results_dir.iterdir() if d.is_dir()])

print(f'Processing {len(subdirs)} directories...\n')

for subdir in subdirs:
    dir_name = subdir.name
    print(f'{'=' * 70}')
    print(f'Processing: {dir_name}')
    print(f'{'=' * 70}')

    # Extract model and dataset from directory name
    if 'B16' in dir_name:
        patch = 'patch16'
    elif 'B32' in dir_name:
        patch = 'patch32'
    else:
        print(f'  ERROR: Could not determine patch from {dir_name}')
        continue

    # Extract dataset name - handle special cases
    if 'FGVC' in dir_name:
        dataset_name = 'FGVC-Aircraft'
    elif 'food101' in dir_name:
        dataset_name = 'food101'
    elif 'Oxford' in dir_name:
        dataset_name = 'oxford-pets'
    elif 'Stanford' in dir_name:
        dataset_name = 'stanford-dogs'
    else:
        print(f'  ERROR: Could not determine dataset from {dir_name}')
        continue

    # Construct model_dataset string for file matching
    model_dataset = f'openai-clip-vit-base-{patch}_{dataset_name}'

    print(f'  Model-Dataset: {model_dataset}')

    # Find mahalanobis file in results directory
    mahal_pattern = f'mahalanobis_ncm_{model_dataset}_val_results.csv'
    mahal_path = results_dir / mahal_pattern

    if not mahal_path.exists():
        print(f'  SKIP: No mahalanobis file found for this directory')
        print(f'  (Would expect: {mahal_pattern})')
        print()
        continue

    # Find ncm_few_shot file in previous_results/subdir
    few_shot_pattern = f'ncm_few_shot_{model_dataset}_val_results.csv'
    few_shot_path = subdir / few_shot_pattern

    if not few_shot_path.exists():
        print(f'  ERROR: Could not find {few_shot_pattern}')
        print(f'  Looking for: {few_shot_path}')
        continue

    print(f'  ✓ Found mahalanobis file: {mahal_pattern}')
    print(f'  ✓ Found ncm_few_shot file: {few_shot_pattern}')

    # Read the mahalanobis file
    mahal_df = pd.read_csv(mahal_path)

    # Read the ncm_few_shot file
    few_shot_df = pd.read_csv(few_shot_path)

    # For each shot_number in mahalanobis, find the row with max mu_f1
    selected_rows = []
    for shot_num in sorted(mahal_df['shot_number'].unique()):
        shot_data = mahal_df[mahal_df['shot_number'] == shot_num]
        max_f1_idx = shot_data['mu_f1'].idxmax()
        selected_rows.append(mahal_df.loc[max_f1_idx].copy())

    selected_df = pd.DataFrame(selected_rows)

    print(f'  Selected {len(selected_df)} best rows from mahalanobis (one per shot_number)')

    # Rename the second column to 'distance' in both dataframes
    # For few_shot_df (rows E), set second column to 'Euclidean'
    few_shot_df_copy = few_shot_df.copy()
    few_shot_cols = list(few_shot_df_copy.columns)
    few_shot_cols[1] = 'distance'
    few_shot_df_copy.columns = few_shot_cols
    few_shot_df_copy['distance'] = 'Euclidean'

    # For selected_df (rows M), set second column to 'Mahalanobis, V'
    selected_df_copy = selected_df.copy()
    # Create the distance value first (before dropping regularization_factor)
    selected_df_copy['distance'] = 'Mahalanobis, ' + selected_df_copy['regularization_factor'].astype(str)
    # Remove the old regularization_factor column
    selected_df_copy = selected_df_copy.drop('regularization_factor', axis=1)
    # Reorder columns so distance is in the second position
    cols = list(selected_df_copy.columns)
    cols.remove('distance')
    cols.insert(1, 'distance')
    selected_df_copy = selected_df_copy[cols]

    # Combine the two dataframes
    combined_df = pd.concat([few_shot_df_copy, selected_df_copy], ignore_index=True)

    # Create output filename
    output_filename = f'few_shots_aggregate_{model_dataset}_results.csv'
    output_path = subdir / output_filename

    # Save the combined file
    combined_df.to_csv(output_path, index=False)
    print(f'  ✓ Created: {output_filename}')
    print(f'  Total rows in output: {len(combined_df)}')
    print()

print(f'{'=' * 70}')
print('✓ ALL PROCESSING COMPLETE!')
print(f'{'=' * 70}')

