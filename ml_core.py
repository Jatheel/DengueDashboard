"""
ml_core.py - all the machine-learning logic for the Dengue Risk Dashboard.

Kept separate from app.py so it can be used and tested without the UI:
load -> features -> target -> split -> train -> evaluate.
"""
import io
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, recall_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Optional boosting libraries - the app still runs if one is missing.
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None

RANDOM_STATE = 42
CLASSES = ["Low", "Medium", "High"]          # encoded as 0, 1, 2

FEATURE_COLS = [
    "district_code",
    "week", "week_sin", "week_cos", "monsoon_code",
    "cases",                                   # this week's cases
    "lag_1", "lag_2", "lag_3", "lag_4",        # previous weeks
    "roll_mean_4", "roll_mean_8", "roll_max_4",
    "roll_mean_52",                            # district's "normal" level in the past year
    "diff_1",                                  # change from last week
    "ratio_to_normal",                         # this week vs the past year
]

MONSOONS = {0: "Inter-monsoon", 1: "South-West (May-Sep)", 2: "North-East (Dec-Feb)"}

# The Epidemiology Unit reports by health area (RDHS), not strictly by district.
# Kalmunai is a separate RDHS but lies inside Ampara district, so its cases are
# added to Ampara -> 25 administrative districts.
DISTRICT_MERGE = {"Kalmune": "Ampara", "Kalmunai": "Ampara"}
# Official spellings for names that are written differently in the source data.
DISTRICT_NAMES = {"Hambanthota": "Hambantota", "Nuwaraeliya": "Nuwara Eliya"}


# --------------------------------------------------------------------------
# 1. Load + clean
# --------------------------------------------------------------------------
def load_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and types. Needs: year, week, district, cases."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(".", "_").replace(" ", "_") for c in df.columns]
    needed = {"year", "week", "district", "cases"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}. "
                         f"Found: {list(df.columns)}")
    df["district"] = df["district"].astype(str).str.strip().str.title()
    df["district"] = df["district"].replace(DISTRICT_MERGE)   # Kalmunai -> Ampara
    df["district"] = df["district"].replace(DISTRICT_NAMES)   # official spellings
    for c in ["year", "week", "cases"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["year", "week", "cases"])
    df[["year", "week"]] = df[["year", "week"]].astype(int)
    df["cases"] = df["cases"].clip(lower=0)
    # If a district/week appears twice (e.g. Ampara + Kalmunai), add the counts together.
    df = (df.groupby(["district", "year", "week"], as_index=False)["cases"].sum()
            .sort_values(["district", "year", "week"]).reset_index(drop=True))
    return df


# --------------------------------------------------------------------------
# 2. Feature engineering (only uses PAST data -> no leakage)
# --------------------------------------------------------------------------
def monsoon_of_week(week: int) -> int:
    if 18 <= week <= 39:                 # roughly May - September
        return 1
    if week >= 49 or week <= 9:          # roughly December - February
        return 2
    return 0


def build_features(df: pd.DataFrame, district_map: dict | None = None) -> pd.DataFrame:
    """district_map: pass the saved mapping when predicting with a loaded model,
    so 'Colombo' always gets the same code it had during training."""
    df = df.sort_values(["district", "year", "week"]).copy()
    g = df.groupby("district")["cases"]

    for k in (1, 2, 3, 4):
        df[f"lag_{k}"] = g.shift(k)
    # Rolling stats over the previous weeks (shift(1) = exclude this week)
    df["roll_mean_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    df["roll_mean_8"] = g.transform(lambda s: s.shift(1).rolling(8).mean())
    df["roll_max_4"] = g.transform(lambda s: s.shift(1).rolling(4).max())
    df["roll_mean_52"] = g.transform(lambda s: s.shift(1).rolling(52, min_periods=26).mean())
    df["diff_1"] = df["cases"] - df["lag_1"]
    df["ratio_to_normal"] = df["cases"] / (df["roll_mean_52"] + 1)

    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
    df["monsoon_code"] = df["week"].apply(monsoon_of_week)

    if district_map is None:
        district_map = {d: i for i, d in enumerate(sorted(df["district"].unique()))}
    df["district_code"] = df["district"].map(district_map)  # unknown district -> NaN (skipped)

    # What we want to predict: NEXT week's cases in the same district
    df["next_cases"] = g.shift(-1)
    return df


# --------------------------------------------------------------------------
# 3. Target: next-week risk level compared with the district's OWN history
#    Thresholds are learned from the TRAINING years only (no leakage).
# --------------------------------------------------------------------------
def compute_thresholds(df: pd.DataFrame, train_end_year: int) -> dict:
    train = df[df["year"] <= train_end_year]
    q = train.groupby("district")["cases"].quantile([0.50, 0.85]).unstack()
    return {d: (float(r[0.50]), float(r[0.85])) for d, r in q.iterrows()}


def label_risk(values: pd.Series, districts: pd.Series, thresholds: dict) -> pd.Series:
    lo = districts.map(lambda d: thresholds.get(d, (np.nan, np.nan))[0])
    hi = districts.map(lambda d: thresholds.get(d, (np.nan, np.nan))[1])
    out = pd.Series(np.nan, index=values.index)
    out[values <= lo] = 0
    out[(values > lo) & (values <= hi)] = 1
    out[values > hi] = 2
    return out


def make_dataset(df_feat: pd.DataFrame, train_end_year: int, feature_cols=None):
    """Return X_train, X_test, y_train, y_test, thresholds, test_frame.
    feature_cols: defaults to FEATURE_COLS (the hybrid page adds 'arima_next')."""
    feature_cols = feature_cols or FEATURE_COLS
    thresholds = compute_thresholds(df_feat, train_end_year)
    df = df_feat.copy()
    df["target"] = label_risk(df["next_cases"], df["district"], thresholds)
    df = df.dropna(subset=feature_cols + ["target"])
    df["target"] = df["target"].astype(int)

    # Time-based split: train on the past, test on the future.
    train = df[df["year"] <= train_end_year]
    test = df[df["year"] > train_end_year]
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Train or test set is empty - pick a different split year.")
    return (train[feature_cols], test[feature_cols],
            train["target"], test["target"], thresholds, test)


# --------------------------------------------------------------------------
# 4. Models
# --------------------------------------------------------------------------
def available_models() -> list:
    names = ["Random Forest", "Decision Tree"]
    if XGBClassifier is not None:
        names.append("XGBoost")
    if CatBoostClassifier is not None:
        names.append("CatBoost")
    if LGBMClassifier is not None:
        names.append("LightGBM")
    names += ["Stacking", "ANN (MLP)", "RF + ANN (hybrid)"]
    return names


def make_model(name: str, rf_params: dict | None = None):
    rf_params = rf_params or {}
    if name == "Random Forest":
        params = dict(n_estimators=300, max_depth=12, min_samples_leaf=2,
                      class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
        params.update(rf_params)
        return RandomForestClassifier(**params)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=5,
                                      class_weight="balanced", random_state=RANDOM_STATE)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9,
                             eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              class_weight="balanced", random_state=RANDOM_STATE,
                              verbose=-1)
    if name == "CatBoost":
        return CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08,
                                  auto_class_weights="Balanced",
                                  random_seed=RANDOM_STATE, verbose=0,
                                  allow_writing_files=False)   # no catboost_info/ folder
    if name == "Stacking":
        base = [("rf", make_model("Random Forest", rf_params)),
                ("dt", make_model("Decision Tree"))]
        if XGBClassifier is not None:
            base.append(("xgb", make_model("XGBoost")))
        if LGBMClassifier is not None:
            base.append(("lgbm", make_model("LightGBM")))
        return StackingClassifier(estimators=base,
                                  final_estimator=LogisticRegression(max_iter=2000),
                                  cv=5, n_jobs=-1)
    if name == "ANN (MLP)":
        # Neural networks need scaled inputs (trees don't), so scale first.
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                          max_iter=500, random_state=RANDOM_STATE))
    if name == "RF + ANN (hybrid)":
        # Soft voting: average the class probabilities of the forest and the network.
        return VotingClassifier(estimators=[("rf", make_model("Random Forest", rf_params)),
                                            ("ann", make_model("ANN (MLP)"))],
                                voting="soft")
    raise ValueError(f"Unknown model: {name}")


def evaluate(model, X_test, y_test) -> dict:
    pred = np.asarray(model.predict(X_test)).ravel().astype(int)
    return score_predictions(y_test, pred)


def score_predictions(y_test, pred) -> dict:
    """Metrics for any list of predicted classes (also used for ARIMA, which is not
    a scikit-learn model)."""
    pred = np.asarray(pred).ravel().astype(int)
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Macro F1": f1_score(y_test, pred, average="macro", zero_division=0),
        "Precision (macro)": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, pred, average="macro", zero_division=0),
        "High-risk recall": recall_score(y_test, pred, labels=[2], average="macro",
                                         zero_division=0),
        "_pred": pred,
        "_cm": confusion_matrix(y_test, pred, labels=[0, 1, 2]),
    }


def train_and_evaluate(name, X_train, y_train, X_test, y_test, rf_params=None):
    model = make_model(name, rf_params)
    t0 = time.time()
    model.fit(X_train, y_train)
    secs = time.time() - t0
    res = evaluate(model, X_test, y_test)
    res["Train time (s)"] = secs
    return model, res


def feature_importance(model) -> pd.Series | None:
    if hasattr(model, "feature_importances_"):
        names = getattr(model, "feature_names_in_", FEATURE_COLS)
        return (pd.Series(model.feature_importances_, index=list(names))
                  .sort_values(ascending=False))
    return None


# --------------------------------------------------------------------------
# 4b. Hybrid RF + ARIMA (time-series)
#     ARIMA is fitted per district on the TRAINING years only, then used with
#     those fixed parameters to make one-step-ahead forecasts for every week.
# --------------------------------------------------------------------------
def arima_next_week(df_feat: pd.DataFrame, train_end_year: int,
                    order=(2, 1, 1)) -> pd.Series:
    """For each row (district, week t) return ARIMA's forecast of NEXT week's cases,
    made only from weeks up to t."""
    if ARIMA is None:
        raise ImportError("statsmodels is not installed - run: pip install statsmodels")
    out = pd.Series(np.nan, index=df_feat.index, name="arima_next")
    for _, grp in df_feat.sort_values(["year", "week"]).groupby("district"):
        y = np.log1p(grp["cases"].to_numpy(dtype=float))     # log scale: tames outbreak spikes
        n_train = int((grp["year"] <= train_end_year).sum())
        if n_train < 60:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                  # convergence chatter
            res = ARIMA(y[:n_train], order=order).fit()      # learn parameters on the past
            full = res.apply(y)                              # same parameters, whole series
            one_step = np.asarray(full.predict())            # one_step[t] = forecast of y[t]
            last = float(np.asarray(full.forecast(1))[0])    # forecast after the final week
        nxt = np.append(one_step[1:], last)                  # nxt[t] = forecast of y[t+1]
        out.loc[grp.index] = np.clip(np.expm1(nxt), 0, None)
    return out


def run_arima_hybrids(df_feat: pd.DataFrame, train_end_year: int,
                      order=(2, 1, 1)) -> dict:
    """Compare ARIMA, Random Forest and two RF + ARIMA hybrids on the same test weeks."""
    df = df_feat.copy()
    t0 = time.time()
    df["arima_next"] = arima_next_week(df, train_end_year, order)
    arima_secs = time.time() - t0

    hybrid_cols = FEATURE_COLS + ["arima_next"]
    Xtr, Xte, ytr, yte, th, test = make_dataset(df, train_end_year, hybrid_cols)
    results = {}

    # 1. ARIMA only: forecast next week's cases -> turn into Low/Medium/High
    pred = label_risk(test["arima_next"], test["district"], th)
    res = score_predictions(yte, pred)
    res["Train time (s)"] = arima_secs
    res["MAE (cases)"] = mean_absolute_error(test["next_cases"], test["arima_next"])
    results["ARIMA only"] = res

    # 2. Random Forest only (the normal features, same rows)
    _, res = train_and_evaluate("Random Forest", Xtr[FEATURE_COLS], ytr,
                                Xte[FEATURE_COLS], yte)
    res["MAE (cases)"] = np.nan
    results["Random Forest only"] = res

    # 3. Hybrid 1: ARIMA's forecast is given to the RF as one extra feature
    rf_h1, res = train_and_evaluate("Random Forest", Xtr, ytr, Xte, yte)
    res["Train time (s)"] += arima_secs
    res["MAE (cases)"] = np.nan
    results["Hybrid 1: RF + ARIMA feature"] = res

    # 4. Hybrid 2 - Zhang's (2003) residual-hybrid method, with an RF in place of his ANN:
    #    ARIMA forecasts, RF learns ARIMA's mistakes (residuals)
    train_rows = df.loc[Xtr.index]
    resid = np.log1p(train_rows["next_cases"]) - np.log1p(train_rows["arima_next"])
    reg = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=2,
                                n_jobs=-1, random_state=RANDOM_STATE)
    t0 = time.time()
    reg.fit(Xtr, resid)
    secs = time.time() - t0
    hybrid_fc = np.clip(np.expm1(np.log1p(test["arima_next"]) + reg.predict(Xte)), 0, None)
    pred = label_risk(hybrid_fc, test["district"], th)
    res = score_predictions(yte, pred)
    res["Train time (s)"] = arima_secs + secs
    res["MAE (cases)"] = mean_absolute_error(test["next_cases"], hybrid_fc)
    results["Hybrid 2: ARIMA + RF residuals"] = res

    forecasts = test[["district", "year", "week", "next_cases", "arima_next"]].copy()
    forecasts["hybrid_next"] = hybrid_fc
    return {"results": results, "forecasts": forecasts,
            "importance": feature_importance(rf_h1),
            "n_train": len(Xtr), "n_test": len(Xte)}


# --------------------------------------------------------------------------
# 5. Pickle save / load and "next week" prediction
# --------------------------------------------------------------------------
def district_map_of(df_feat: pd.DataFrame) -> dict:
    return (df_feat.dropna(subset=["district_code"])
                   .drop_duplicates("district").set_index("district")["district_code"]
                   .astype(int).to_dict())


def bundle_dict(model, model_name, thresholds, metrics, train_end_year,
                district_map) -> dict:
    """Everything needed to predict with a model, as a plain dict (no pickling)."""
    return {
        "model": model,
        "model_name": model_name,
        "feature_cols": FEATURE_COLS,
        "classes": CLASSES,
        "thresholds": thresholds,
        "train_end_year": train_end_year,
        "district_map": district_map,
        "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
    }


def make_bundle(model, model_name, thresholds, metrics, train_end_year,
                district_map) -> bytes:
    bundle = bundle_dict(model, model_name, thresholds, metrics, train_end_year,
                         district_map)
    buf = io.BytesIO()
    pickle.dump(bundle, buf)
    return buf.getvalue()


def load_bundle(file_bytes: bytes) -> dict:
    # NOTE: only load .pkl files you created yourself - pickle can run code.
    bundle = pickle.loads(file_bytes)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("This .pkl was not saved by this dashboard.")
    return bundle


def predict_next_week(bundle: dict, df_clean: pd.DataFrame) -> pd.DataFrame:
    """Use each district's most recent week to predict next week's risk."""
    district_map = bundle.get("district_map")
    if district_map:
        # Models saved before the district clean-up used the raw names
        # (Kalmune, Hambanthota, Nuwaraeliya) - translate them so no district is lost.
        # Kalmunai is now part of Ampara, so Ampara keeps its own code.
        district_map = {DISTRICT_NAMES.get(d, d): code for d, code in district_map.items()
                        if d not in DISTRICT_MERGE}
    df_feat = build_features(df_clean, district_map)
    cols = bundle["feature_cols"]
    latest = (df_feat.dropna(subset=cols)
                     .sort_values(["year", "week"])
                     .groupby("district").tail(1))
    if latest.empty:
        raise ValueError("Not enough recent weeks to build features (need ~26+ weeks per district).")
    X = latest[cols]
    model = bundle["model"]
    pred = np.asarray(model.predict(X)).ravel().astype(int)
    out = latest[["district", "year", "week", "cases"]].copy()
    out["predicted_risk"] = [bundle["classes"][p] for p in pred]
    if hasattr(model, "predict_proba"):
        out["p_high"] = model.predict_proba(X)[:, 2].round(3)
    out["_order"] = out["predicted_risk"].map({"High": 0, "Medium": 1, "Low": 2})
    return (out.sort_values(["_order", "cases"], ascending=[True, False])
               .drop(columns="_order").reset_index(drop=True))
