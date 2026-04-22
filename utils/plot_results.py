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

    parser.add_argument("--output", default="learning_curve.png", help="Name of the saved image file")

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

        plt.plot(group_df[args.x], group_df[args.y], marker='', linewidth=2, label=f"{args.group}: {group_name}")

    plt.xlabel(args.x, fontsize=12, fontweight='bold')
    plt.ylabel(args.y, fontsize=12, fontweight='bold')
    plt.title(f"{args.y} vs {args.x} (Grouped by {args.group})", fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title= args.group)
    plt.tight_layout()

    plt.savefig(args.output, dpi=300)
    print(f"Plot saved as '{args.output}'")


if __name__ == "__main__":
    main()