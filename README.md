# GIM Mess Food-Waste Forecasting

Predict **daily food_waste_kg** to reduce leftover while meeting demand at the Goa Institute of Management (Sanquelim, Goa).

## Repo Layout
```
gim-food-waste/
├─ data/
│  ├─ mess_waste_GIM_500.csv
│  └─ mess_waste_GIM_daily_exams.csv
├─ notebooks/
│  └─ 01_train_decision_tree.ipynb
├─ scripts/
│  └─ train_tree.py
├─ artifacts/           # saved model/metrics after running the notebook
├─ reports/
│  ├─ figures/          # exported plots
│  ├─ GIM_Food_Waste_Project_Report.docx
│  └─ overleaf/
│     └─ GIM_MLBA_IEEE_Overleaf.zip
├─ requirements.txt
├─ .gitignore
└─ LICENSE
```

## Quickstart (Colab)
1. Open `notebooks/01_train_decision_tree.ipynb` in Google Colab (File → Open Notebook → Upload).
2. Run all cells. It will:
   - Load `data/mess_waste_GIM_500.csv` (or the exams file)
   - Train a Decision Tree baseline (with tiny tuning)
   - Save artifacts to `artifacts/` and plots to `reports/figures/`
3. Use `reports/overleaf/GIM_MLBA_IEEE_Overleaf.zip` in Overleaf, and replace the placeholder figure/table with your outputs.
4. Update `reports/GIM_Food_Waste_Project_Report.docx` with metrics and figures for submission.

## Notes
- Replace the synthetic CSVs with real daily logs when available.
- Keep the simple feature set for interpretability; scale up only if needed.
- License: MIT
