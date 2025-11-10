#!/usr/bin/env python3
"""
Append cells from one notebook to another (non-destructive).

Usage:
  python append_cells_to_notebook.py \
      --base notebooks/01_train_decision_tree.ipynb \
      --add  /path/to/01_additions_for_original_notebook.ipynb \
      --out  notebooks/01_train_decision_tree_with_additions.ipynb
"""
import argparse, nbformat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to the original .ipynb")
    ap.add_argument("--add", required=True, help="Path to the addendum .ipynb")
    ap.add_argument("--out", required=True, help="Output path for merged notebook")
    args = ap.parse_args()

    base_nb = nbformat.read(args.base, as_version=4)
    add_nb = nbformat.read(args.add, as_version=4)

    # Append all cells from add_nb to base_nb
    base_nb.cells.extend(add_nb.cells)

    nbformat.write(base_nb, args.out)
    print("Merged ->", args.out)

if __name__ == "__main__":
    main()
