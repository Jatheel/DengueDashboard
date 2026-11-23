"""
export_results.py - reproduce the results in results/ and models/ without the dashboard.

Usage (from the project folder):
    python export_results.py              # split year = same default as the dashboard
    python export_results.py 2022         # train on years up to 2022 instead
Writes:
    results/model_comparison.csv          all 8 models (Train & Compare page)
    results/hybrid_arima_comparison.csv   ARIMA vs RF vs RF + ARIMA hybrids (Hybrid page)
    results/next_week_predictions.csv     next-week risk per district from the best model
    models/dengue_<best model>.pkl        best model, can be uploaded on the Save / Load page
Takes about 2 minutes (Stacking is the slowest model).
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

import ml_core as core

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

clean = core.load_raw(pd.read_csv(ROOT / "data" / "sl_dengue_weekly.csv"))
feats = core.build_features(clean)
years = sorted(clean["year"].unique())
valid = [y for y in years if years[0] + 3 <= y <= years[-1] - 1]
split = int(sys.argv[1]) if len(sys.argv) > 1 else valid[-2]      # dashboard default
print(f"Data: {len(clean):,} rows, {clean['district'].nunique()} districts, "
      f"{years[0]}-{years[-1]} | train <= {split}, test > {split}")

# 1. All models (same as the Train & Compare page)
Xtr, Xte, ytr, yte, th, _ = core.make_dataset(feats, split)
rows, models = {}, {}
for name in core.available_models():
    print(f"  training {name} ...")
    model, res = core.train_and_evaluate(name, Xtr, ytr, Xte, yte)
    models[name] = (model, res)
    rows[name] = {k: round(v, 4) for k, v in res.items() if not k.startswith("_")}
table = pd.DataFrame(rows).T
table.index.name = "Model"
table.to_csv(ROOT / "results" / "model_comparison.csv")
print(table.sort_values("Macro F1", ascending=False).to_string())

# 2. Best model -> .pkl + next-week predictions
best = table["Macro F1"].idxmax()
model, res = models[best]
data = core.make_bundle(model, best, th, res, split, core.district_map_of(feats))
fname = "dengue_" + best.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
(ROOT / "models" / fname).write_bytes(data)
print(f"Best model: {best} -> models/{fname} ({len(data) / 1024:.0f} KB)")
pred = core.predict_next_week(core.load_bundle(data), clean)
pred.to_csv(ROOT / "results" / "next_week_predictions.csv", index=False)

# 3. RF + ARIMA hybrids (same as the Hybrid page, ARIMA(2,1,1))
out = core.run_arima_hybrids(feats, split, (2, 1, 1))
hyb = pd.DataFrame({n: {k: round(v, 4) for k, v in r.items() if not k.startswith("_")}
                    for n, r in out["results"].items()}).T
hyb.index.name = "Approach"
hyb.to_csv(ROOT / "results" / "hybrid_arima_comparison.csv")
print(hyb.to_string())
print("Done - see results/ and models/")
