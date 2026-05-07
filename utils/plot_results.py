import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True, help="Path to your CSV file")
    parser.add_argument("--x", required=True, help="Exact column name for the X-axis")
    parser.add_argument("--y", required=True, help="Exact column name for the Y-axis")
    parser.add_argument("--group", required=True, help="Column name to group by (each unique value gets a line)")
    parser.add_argument("--zero_shot_file", required=False, help="Zero shot classification report on the same dataset")

    parser.add_argument("--xlabel", required=False, help="Clean label for the X-axis (LaTeX syntax supported)")
    parser.add_argument("--ylabel", required=False, help="Clean label for the Y-axis (LaTeX syntax supported)")
    parser.add_argument("--output", default="learning_curve.pdf", help="Name of the saved image file (use .pdf)")

    parser.add_argument("--legend_title", required=False, help="Adds a title to the top of the legend box")
    parser.add_argument("--group_prefix", required=False, default="",
                        help="Text to prepend to each line's label (e.g., 'Size: ')")

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: Could not find '{args.csv}'")
        return

    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)

    required_cols = [args.x, args.y, args.group]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found. Available columns are: {list(df.columns)}")
            return

    plt.figure(figsize=(10, 6))

    grouped_data = df.groupby(args.group)

    for group_name, group_df in grouped_data:
        group_df = group_df.sort_values(by=args.x)
        label_text = f"{args.group_prefix}{group_name}" if args.group_prefix else str(group_name)
        plt.plot(group_df[args.x], group_df[args.y], marker='', linewidth=2, label=label_text)

    if args.zero_shot_file and os.path.exists(args.zero_shot_file):
        print(f"Loading Zero-Shot baseline from {args.zero_shot_file}...")
        # Read the CSV, treating the first column (class names / summaries) as the Index
        zs_df = pd.read_csv(args.zero_shot_file, index_col=0)

        try:
            zs_val = None
            zs_label = "Zero-Shot Baseline"

            if args.y == "mu_f1":
                zs_val = zs_df.loc['macro avg', 'f1-score']
                zs_label = f"Zero-Shot"

            elif args.y == "mu_acc":
                acc_row = zs_df.loc['accuracy']
                zs_val = acc_row['f1-score'] if 'f1-score' in acc_row else acc_row.dropna().iloc[0]
                zs_label = f"Zero-Shot"

            if zs_val is not None:
                plt.axhline(y=zs_val, color='black', linestyle='--', linewidth=2, alpha=0.8, label=zs_label)

        except KeyError as e:
            print(f"Warning: Could not parse zero-shot file correctly. Missing expected structure: {e}")

    final_xlabel = args.xlabel if args.xlabel else args.x
    final_ylabel = args.ylabel if args.ylabel else args.y

    plt.xlabel(final_xlabel, fontsize=14)
    plt.ylabel(final_ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)

    if len(grouped_data) > 1 or args.zero_shot_file:
        plt.legend(title=args.legend_title, title_fontsize=13, fontsize=12)

    plt.tight_layout()

    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{args.output}'")


if __name__ == "__main__":
    main()