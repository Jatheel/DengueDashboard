"""
Sri Lanka Dengue Outbreak Risk - Random Forest Dashboard
Run with:  streamlit run app.py
"""
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import ml_core as core

APP_DIR = Path(__file__).resolve().parent          # works from any working directory
DEFAULT_DATA = APP_DIR / "data" / "sl_dengue_weekly.csv"
BUNDLED_MODEL = APP_DIR / "models" / "dengue_lightgbm.pkl"
PRECOMPUTED = APP_DIR / "results" / "model_comparison.csv"
# Uploading a .pkl lets the uploader run code on the server (pickle), so it is OFF unless
# ALLOW_PKL_UPLOAD=1 is set. run_dashboard.bat sets it for local use; leave it unset
# on a public deployment.
ALLOW_PKL_UPLOAD = os.environ.get("ALLOW_PKL_UPLOAD", "").strip().lower() in ("1", "true", "yes")
# Everything that belongs to trained models - cleared when the data or split changes.
MODEL_STATE = ["results", "thresholds", "split", "y_test", "X_train", "y_train", "X_test",
               "district_map", "n_train", "n_test", "tuned", "hybrid", "hybrid_key",
               "pkl_cache"]

st.set_page_config(page_title="Dengue Risk Dashboard - Sri Lanka", page_icon="🦟",
                   layout="wide")

BLUE = "#2a78d6"
RISK_COLORS = {"High": "#e34948", "Medium": "#eda100", "Low": "#1baf7a"}
TEMPLATE = "plotly_white"
METRICS = ["Accuracy", "Macro F1", "Precision (macro)", "Recall (macro)",
           "High-risk recall", "Train time (s)"]
# Black bold text on the light highlight so it is readable in dark mode too.
BEST_CELL = "background-color: #d6e6fa; color: #000000; font-weight: bold"


# max_entries keeps memory bounded when many visitors upload their own CSVs.
@st.cache_data(show_spinner=False, max_entries=4)
def prepare(raw: pd.DataFrame):
    clean = core.load_raw(raw)
    feats = core.build_features(clean)
    return clean, feats


@st.cache_data(show_spinner=False)
def read_default_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False, max_entries=8)
def run_hybrids(feats: pd.DataFrame, split: int, order: tuple):
    return core.run_arima_hybrids(feats, split, order)


@st.cache_resource(show_spinner=False)
def load_bundled_model(path: str) -> dict:
    # Trusted file shipped with the repo (re-created by export_results.py).
    return core.load_bundle(Path(path).read_bytes())


def clear_model_state(keep=()):
    for k in MODEL_STATE:
        if k not in keep:
            st.session_state.pop(k, None)


def week_to_date(df):
    return pd.to_datetime(df["year"].astype(str) + "-01-01") + \
        pd.to_timedelta((df["week"] - 1) * 7, unit="D")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🦟 Dengue Risk Dashboard")
st.sidebar.caption("Sri Lanka · weekly cases by district · Epidemiology Unit data")

uploaded = st.sidebar.file_uploader("Upload dengue CSV (optional)", type=["csv"],
                                    help="Columns needed: year, week, district, cases. "
                                         "If nothing is uploaded, the built-in "
                                         "data/sl_dengue_weekly.csv is used.")
try:
    source = None
    if uploaded is not None:                       # an upload always wins
        raw, source = pd.read_csv(uploaded), f"{uploaded.name} (uploaded)"
    elif DEFAULT_DATA.exists():
        raw, source = read_default_csv(str(DEFAULT_DATA)), "data/sl_dengue_weekly.csv"
    if source is not None:
        clean_df, feats_df = prepare(raw)
        if st.session_state.get("file_name") != source:
            clear_model_state()                    # models trained on other data no longer apply
        st.session_state["clean"], st.session_state["feats"] = clean_df, feats_df
        st.session_state["file_name"] = source
        st.sidebar.caption(f"Data in use: **{source}**")
except Exception as e:
    st.sidebar.error(f"Could not read file: {e}")

page = st.sidebar.radio("Go to", ["📊 Data Explorer", "⚙️ Train & Compare Models",
                                  "🌲 Random Forest Lab", "🔗 Hybrid: RF + ARIMA",
                                  "💾 Save / Load Model"])

if "clean" not in st.session_state:
    st.title("Sri Lanka Dengue Outbreak Risk Dashboard")
    st.info("⬅️ No data found. Upload **sl_dengue_weekly.csv** in the sidebar, or create "
            "`data/sl_dengue_weekly.csv` with `python convert_rda.py` (see README).")
    st.markdown("""
**What this dashboard does**
1. Explores weekly dengue cases for every district in Sri Lanka.
2. Predicts whether a district's dengue risk **next week** will be *Low, Medium or High*,
   compared with that district's own history.
3. Compares Random Forest, Decision Tree, XGBoost, CatBoost, LightGBM, Stacking,
   an ANN and an RF + ANN hybrid.
4. Tests RF + ARIMA hybrids for time-series forecasting.
5. Saves the trained model as a `.pkl` file and loads it again to make predictions.
""")
    st.stop()

clean = st.session_state["clean"]
feats = st.session_state["feats"]
years = sorted(clean["year"].unique())

# ================================================================ PAGE 1
if page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    st.caption(f"File: {st.session_state.get('file_name', '')} · "
               f"{years[0]}–{years[-1]}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(clean):,}")
    c2.metric("Districts", clean["district"].nunique())
    c3.metric("Years", len(years), help=f"{years[0]} to {years[-1]}")
    c4.metric("Total cases", f"{int(clean['cases'].sum()):,}")

    nat = clean.groupby(["year", "week"], as_index=False)["cases"].sum()
    nat["date"] = week_to_date(nat)
    fig = px.line(nat, x="date", y="cases", template=TEMPLATE,
                  title="Weekly dengue cases - all of Sri Lanka",
                  labels={"date": "", "cases": "Cases per week"})
    fig.update_traces(line=dict(color=BLUE, width=2))
    st.plotly_chart(fig)

    left, right = st.columns(2)
    with left:
        by_d = clean.groupby("district", as_index=False)["cases"].sum() \
                    .sort_values("cases")
        fig = px.bar(by_d, x="cases", y="district", orientation="h", template=TEMPLATE,
                     title="Total cases by district", height=650,
                     labels={"cases": "Total cases", "district": ""})
        fig.update_traces(marker_color=BLUE)
        st.plotly_chart(fig)
    with right:
        season = clean.groupby("week", as_index=False)["cases"].mean()
        fig = px.line(season, x="week", y="cases", template=TEMPLATE,
                      title="Seasonal pattern - average cases per district by week of year",
                      labels={"week": "Week of year", "cases": "Avg cases"})
        fig.update_traces(line=dict(color=BLUE, width=2))
        fig.add_vrect(x0=18, x1=39, fillcolor="#2a78d6", opacity=0.08, line_width=0,
                      annotation_text="SW monsoon", annotation_position="top left")
        st.plotly_chart(fig)

        heat = clean.pivot_table(index="district", columns="year", values="cases",
                                 aggfunc="sum")
        fig = px.imshow(heat, color_continuous_scale="Blues", aspect="auto",
                        template=TEMPLATE, title="Cases by district and year",
                        labels={"color": "Cases"})
        st.plotly_chart(fig)

    dist = st.selectbox("Look at one district", sorted(clean["district"].unique()))
    one = clean[clean["district"] == dist].copy()
    one["date"] = week_to_date(one)
    fig = px.line(one, x="date", y="cases", template=TEMPLATE,
                  title=f"Weekly cases - {dist}", labels={"date": "", "cases": "Cases"})
    fig.update_traces(line=dict(color=BLUE, width=2))
    st.plotly_chart(fig)

    with st.expander("Show raw data"):
        st.dataframe(clean)

# ================================================================ PAGE 2
elif page == "⚙️ Train & Compare Models":
    st.title("⚙️ Train & Compare Models")
    st.markdown("""
**Prediction task:** using this week's and previous weeks' cases, predict a district's
dengue risk **next week**:
- **Low** = at or below the district's usual (median) level
- **Medium** = above usual, up to its 85th percentile
- **High** = above its 85th percentile (outbreak-level for that district)

Thresholds come from the **training years only**, and the split is **by time**
(train on the past, test on the future), so the model never sees the future.
""")
    valid_years = [y for y in years if years[0] + 3 <= y <= years[-1] - 1]
    if not valid_years:
        st.error("Need at least 5 years of data to train.")
        st.stop()
    split = st.select_slider("Train on years up to (test on the years after):",
                             options=valid_years,
                             value=valid_years[-2] if len(valid_years) > 1 else valid_years[-1])
    choices = st.multiselect("Models to train", core.available_models(),
                             default=[m for m in core.available_models() if m != "Stacking"],
                             help="Stacking is left out by default: it takes about 1 minute "
                                  "on its own. Add it to train all 8 models.")
    missing = {"XGBoost", "CatBoost", "LightGBM"} - set(core.available_models())
    if missing:
        st.warning(f"Not installed: {', '.join(sorted(missing))}. "
                   "Run `pip install -r requirements.txt` to enable them.")

    if st.button("🚀 Train selected models", type="primary", disabled=not choices):
        clear_model_state()        # old tuned RF / hybrid / saved files belong to the old run
        Xtr, Xte, ytr, yte, th, test_df = core.make_dataset(feats, split)
        results = {}
        bar = st.progress(0.0, text="Training...")
        for i, name in enumerate(choices):
            bar.progress(i / len(choices), text=f"Training {name}...")
            model, res = core.train_and_evaluate(name, Xtr, ytr, Xte, yte)
            results[name] = {"model": model, "res": res}
        bar.progress(1.0, text="Done ✅")
        st.session_state.update(results=results, thresholds=th, split=split,
                                y_test=yte, X_train=Xtr, y_train=ytr, X_test=Xte,
                                district_map=core.district_map_of(feats),
                                n_train=len(Xtr), n_test=len(Xte))

    if "results" not in st.session_state and PRECOMPUTED.exists():
        st.info("Nothing trained in this session yet. Below are **pre-computed results** "
                "for all 8 models (train on 2006–2023, test on 2024–2025, made by "
                "`export_results.py`). Click **Train selected models** to run them live.")
        pre = pd.read_csv(PRECOMPUTED, index_col="Model")
        pre = pre[[c for c in METRICS if c in pre.columns]].sort_values("Macro F1",
                                                                       ascending=False)
        st.dataframe(pre.style.format("{:.3f}"))

    if "results" in st.session_state:
        results = st.session_state["results"]
        st.caption(f"Train rows: {st.session_state['n_train']:,} · "
                   f"Test rows: {st.session_state['n_test']:,} · "
                   f"Split year: {st.session_state['split']}")

        table = pd.DataFrame({n: {k: v for k, v in r["res"].items()
                                  if not k.startswith("_")}
                              for n, r in results.items()}).T[METRICS]
        best = table["Macro F1"].idxmax()
        st.subheader("📋 Performance comparison")
        st.dataframe(table.style.format("{:.3f}")
                     .highlight_max(subset=METRICS[:-1], props=BEST_CELL)
                     .highlight_min(subset=["Train time (s)"], props=BEST_CELL))
        st.success(f"🏆 Best model by Macro F1: **{best}** "
                   f"({table.loc[best, 'Macro F1']:.3f})")

        metric = st.selectbox("Compare models by", METRICS, index=1)
        chart = table[[metric]].reset_index().rename(columns={"index": "Model"}) \
                               .sort_values(metric, ascending=metric == "Train time (s)")
        fig = px.bar(chart, x=metric, y="Model", orientation="h", template=TEMPLATE,
                     text=chart[metric].round(3), title=f"{metric} by model")
        fig.update_traces(marker_color=BLUE, textposition="outside")
        fig.update_layout(yaxis_title="", xaxis_title=metric)
        fig.update_xaxes(range=[0, chart[metric].max() * 1.18])   # room for the labels
        st.plotly_chart(fig)

        st.subheader("🔍 Inspect one model")
        pick = st.selectbox("Model", list(results.keys()),
                            index=list(results.keys()).index(best))
        c1, c2 = st.columns(2)
        with c1:
            cm = results[pick]["res"]["_cm"]
            fig = px.imshow(cm, x=core.CLASSES, y=core.CLASSES, text_auto=True,
                            color_continuous_scale="Blues", template=TEMPLATE,
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            title=f"Confusion matrix - {pick}")
            st.plotly_chart(fig)
        with c2:
            fi = core.feature_importance(results[pick]["model"])
            if fi is not None:
                fi = fi.head(12).sort_values().reset_index()
                fi.columns = ["Feature", "Importance"]
                fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                             template=TEMPLATE, title=f"Top features - {pick}")
                fig.update_traces(marker_color=BLUE)
                fig.update_layout(yaxis_title="")
                st.plotly_chart(fig)
            else:
                st.info(f"{pick} has no single feature-importance chart (it combines "
                        "several models, or is a neural network). Check Random Forest instead.")

# ================================================================ PAGE 3
elif page == "🌲 Random Forest Lab":
    st.title("🌲 Random Forest Lab")
    st.write("Tune the Random Forest and see how each setting changes the result.")
    if "X_train" not in st.session_state:
        st.info("Train the models on the **Train & Compare** page first "
                "(this sets the train/test split).")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        n_est = st.slider("Number of trees (n_estimators)", 50, 800, 300, 50)
        depth = st.slider("Max depth (0 = no limit)", 0, 40, 12)
    with c2:
        leaf = st.slider("Min samples per leaf", 1, 20, 2)
        feat = st.selectbox("Max features per split", ["sqrt", "log2", "all"])
    balanced = st.checkbox("Balance classes (helps catch High-risk weeks)", True)

    if st.button("🌲 Train tuned Random Forest", type="primary"):
        params = dict(n_estimators=n_est, max_depth=None if depth == 0 else depth,
                      min_samples_leaf=leaf,
                      max_features=None if feat == "all" else feat,
                      class_weight="balanced" if balanced else None)
        model, res = core.train_and_evaluate(
            "Random Forest", st.session_state["X_train"], st.session_state["y_train"],
            st.session_state["X_test"], st.session_state["y_test"], rf_params=params)
        st.session_state["results"]["Random Forest (tuned)"] = {"model": model, "res": res}
        st.session_state["tuned"] = res
        st.session_state.get("pkl_cache", {}).pop("Random Forest (tuned)", None)

    if "tuned" in st.session_state:
        res = st.session_state["tuned"]
        base = st.session_state["results"].get("Random Forest", {}).get("res")
        cols = st.columns(5)
        for col, m in zip(cols, METRICS[:5]):
            delta = f"{res[m] - base[m]:+.3f} vs default" if base else None
            col.metric(m, f"{res[m]:.3f}", delta)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.imshow(res["_cm"], x=core.CLASSES, y=core.CLASSES, text_auto=True,
                            color_continuous_scale="Blues", template=TEMPLATE,
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            title="Confusion matrix - tuned RF")
            st.plotly_chart(fig)
        with c2:
            fi = core.feature_importance(
                st.session_state["results"]["Random Forest (tuned)"]["model"])
            fi = fi.sort_values().reset_index()
            fi.columns = ["Feature", "Importance"]
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                         template=TEMPLATE, title="Feature importance - tuned RF")
            fig.update_traces(marker_color=BLUE)
            fig.update_layout(yaxis_title="")
            st.plotly_chart(fig)
        st.caption("The tuned model is also added to the comparison table and the "
                   "Save page as 'Random Forest (tuned)'.")

# ================================================================ PAGE 4
elif page == "🔗 Hybrid: RF + ARIMA":
    st.title("🔗 Hybrid: RF + ARIMA")
    st.markdown("""
**ARIMA** is a classic time-series model: it forecasts next week's cases in a district from that
district's own recent weeks (**p** = past weeks used, **d** = differencing to remove trend,
**q** = past forecast errors used). It is fitted separately for each district on the
**training years only**, then forecasts one week ahead at a time.

This page compares four approaches on the **same test weeks**:
1. **ARIMA only** - forecast next week's cases, then convert to Low / Medium / High.
2. **Random Forest only** - the normal RF classifier.
3. **Hybrid 1 - RF + ARIMA feature** - ARIMA's forecast is given to the RF as an extra input.
4. **Hybrid 2 - ARIMA + RF residuals** - Zhang's (2003) residual-hybrid method, with a
   Random Forest in place of his neural network: ARIMA makes the forecast and an RF learns to
   correct ARIMA's mistakes (ARIMA captures the linear part, RF the non-linear part).
""")
    if "split" not in st.session_state:
        st.info("Train the models on the **Train & Compare** page first "
                "(this sets the train/test split).")
        st.stop()
    split = st.session_state["split"]
    st.caption(f"Same split as Train & Compare: train on years up to {split}, "
               f"test on the years after.")

    c1, c2, c3 = st.columns(3)
    p = c1.selectbox("p - past weeks (AR)", [0, 1, 2, 3], index=2)
    d = c2.selectbox("d - differencing", [0, 1], index=1)
    q = c3.selectbox("q - past errors (MA)", [0, 1, 2], index=1)
    key = (split, (p, d, q))

    if st.button("🔗 Run ARIMA + hybrids", type="primary"):
        with st.spinner("Fitting ARIMA for each district and training the hybrids "
                        "(about 15 seconds)..."):
            try:
                st.session_state["hybrid"] = run_hybrids(feats, split, (p, d, q))
                st.session_state["hybrid_key"] = key
            except Exception as e:
                st.error(f"Hybrid run failed: {e}")

    if "hybrid" in st.session_state and st.session_state.get("hybrid_key") != key:
        st.info("The split or ARIMA order changed - click **Run ARIMA + hybrids** again.")
    elif "hybrid" in st.session_state:
        out = st.session_state["hybrid"]
        st.caption(f"ARIMA{(p, d, q)} · Train rows: {out['n_train']:,} · "
                   f"Test rows: {out['n_test']:,}")
        cols = METRICS[:5] + ["MAE (cases)", "Train time (s)"]
        table = pd.DataFrame({n: {k: v for k, v in r.items() if not k.startswith("_")}
                              for n, r in out["results"].items()}).T[cols]
        best = table["Macro F1"].idxmax()
        st.subheader("📋 ARIMA vs Random Forest vs hybrids")
        # MAE only exists for models that forecast case numbers: show "–" for the others.
        mae = table["MAE (cases)"]
        shown = table.copy()
        shown["MAE (cases)"] = mae.map(lambda v: "–" if pd.isna(v) else f"{v:.3f}")
        st.dataframe(shown.style
                     .format("{:.3f}", subset=[c for c in cols if c != "MAE (cases)"])
                     .highlight_max(subset=METRICS[:5], props=BEST_CELL)
                     .highlight_min(subset=["Train time (s)"], props=BEST_CELL)
                     .apply(lambda s: [BEST_CELL if v == mae.min() else "" for v in mae],
                            subset=["MAE (cases)"]))
        st.caption("MAE (cases) = average error of the forecast number of cases next week. "
                   "Only models that forecast a number of cases have it.")
        st.success(f"🏆 Best by Macro F1: **{best}** ({table.loc[best, 'Macro F1']:.3f})")

        left, right = st.columns(2)
        with left:
            chart = table[["Macro F1"]].reset_index().rename(columns={"index": "Model"}) \
                                       .sort_values("Macro F1")
            fig = px.bar(chart, x="Macro F1", y="Model", orientation="h", template=TEMPLATE,
                         text=chart["Macro F1"].round(3), title="Macro F1 by approach")
            fig.update_traces(marker_color=BLUE, textposition="outside")
            fig.update_layout(yaxis_title="")
            fig.update_xaxes(range=[0, chart["Macro F1"].max() * 1.18])   # room for labels
            st.plotly_chart(fig)
        with right:
            fi = out["importance"]
            if fi is not None:
                fi = fi.head(10).sort_values().reset_index()
                fi.columns = ["Feature", "Importance"]
                fi["is_arima"] = fi["Feature"].eq("arima_next").map(
                    {True: "ARIMA forecast", False: "Other features"})
                fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                             color="is_arima", template=TEMPLATE,
                             color_discrete_map={"ARIMA forecast": "#eda100",
                                                 "Other features": BLUE},
                             title="Hybrid 1 - where does the ARIMA forecast rank?",
                             labels={"is_arima": ""})
                fig.update_layout(yaxis_title="")
                fig.update_yaxes(categoryorder="total ascending")   # keep sorted with 2 colours
                st.plotly_chart(fig)

        st.subheader("📈 Forecasts for one district (test years)")
        fc = out["forecasts"]
        dist = st.selectbox("District", sorted(fc["district"].unique()))
        one = fc[fc["district"] == dist].copy()
        one["date"] = week_to_date(one) + pd.Timedelta(days=7)   # the week being forecast
        long = one.melt(id_vars="date", value_vars=["next_cases", "arima_next", "hybrid_next"],
                        var_name="series", value_name="cases")
        long["series"] = long["series"].map({"next_cases": "Actual",
                                             "arima_next": "ARIMA forecast",
                                             "hybrid_next": "Hybrid 2 forecast"})
        fig = px.line(long, x="date", y="cases", color="series", template=TEMPLATE,
                      color_discrete_map={"Actual": "#555555", "ARIMA forecast": "#eda100",
                                          "Hybrid 2 forecast": BLUE},
                      title=f"Next-week cases - {dist}", labels={"date": "", "series": ""})
        st.plotly_chart(fig)
        st.caption("These hybrids are for analysis and comparison. To save a .pkl and "
                   "predict, use the models from the Train & Compare page.")

# ================================================================ PAGE 5
else:
    st.title("💾 Save / Load Model")
    left, right = st.columns(2)

    with left:
        st.subheader("1. Save a trained model")
        if "results" not in st.session_state:
            st.info("Train models first on the **Train & Compare** page.")
        else:
            results = st.session_state["results"]
            names = list(results.keys())
            best = max(names, key=lambda n: results[n]["res"]["Macro F1"])
            name = st.selectbox("Choose model to save", names, index=names.index(best),
                                help="Pre-selected: the best model by Macro F1.")
            # Pickling a big forest takes a while - do it once per model, not on every click.
            cache = st.session_state.setdefault("pkl_cache", {})
            if name not in cache:
                cache[name] = core.make_bundle(results[name]["model"], name,
                                               st.session_state["thresholds"],
                                               results[name]["res"], st.session_state["split"],
                                               st.session_state["district_map"])
            data = cache[name]
            fname = "dengue_" + name.lower().replace(" ", "_").replace("(", "") \
                                    .replace(")", "") + ".pkl"
            st.download_button("⬇️ Download .pkl", data, file_name=fname,
                               mime="application/octet-stream", type="primary")
            st.caption(f"Size: {len(data) / 1024:.0f} KB. The file holds the model plus "
                       "its feature list, risk thresholds and test scores.")

    with right:
        st.subheader("2. Choose a model and predict next week")
        sources = ["Bundled model (models/dengue_lightgbm.pkl)"]
        if "results" in st.session_state:
            sources.append("A model trained in this session")
        if ALLOW_PKL_UPLOAD:
            sources.append("Upload a .pkl file")
        source = st.radio("Model source", sources)
        bundle = None
        try:
            if source.startswith("Bundled"):
                if BUNDLED_MODEL.exists():
                    bundle = load_bundled_model(str(BUNDLED_MODEL))
                else:
                    st.info("No bundled model found. Run `python export_results.py` to "
                            "create it.")
            elif source.startswith("A model trained"):
                results = st.session_state["results"]
                pick = st.selectbox("Trained model", list(results.keys()), key="predict_pick")
                bundle = core.bundle_dict(results[pick]["model"], pick,
                                          st.session_state["thresholds"],
                                          results[pick]["res"], st.session_state["split"],
                                          st.session_state["district_map"])
            else:
                pkl = st.file_uploader("Upload model file", type=["pkl"])
                st.caption("⚠️ Only upload .pkl files you created yourself - pickle files "
                           "can run code when opened.")
                if pkl is not None:
                    bundle = core.load_bundle(pkl.getvalue())
        except Exception as e:
            st.error(f"Could not load model: {e}")
            bundle = None
        if bundle is not None:
            st.success(f"Using **{bundle['model_name']}** "
                       f"(trained up to {bundle['train_end_year']})")
            st.json({k: round(v, 3) for k, v in bundle["metrics"].items()})
        if not ALLOW_PKL_UPLOAD:
            st.caption("Uploading .pkl files is turned off on this server, because pickle "
                       "files can run code. Run the app locally to use it (see README).")

    if bundle is not None:
        st.divider()
        st.subheader("🔮 Next-week dengue risk by district")
        try:
            pred = core.predict_next_week(bundle, clean)
            last = pred.iloc[0]
            st.caption(f"Based on the latest week in the data "
                       f"(week {int(last['week'])}, {int(last['year'])}).")
            c1, c2, c3 = st.columns(3)
            for col, lvl in zip((c1, c2, c3), ("High", "Medium", "Low")):
                col.metric(f"{lvl} risk districts", int((pred["predicted_risk"] == lvl).sum()))
            if "p_high" in pred:
                chart = pred.sort_values("p_high")
                fig = px.bar(chart, x="p_high", y="district", orientation="h",
                             color="predicted_risk", color_discrete_map=RISK_COLORS,
                             template=TEMPLATE, height=650,
                             title="Probability of High risk next week",
                             labels={"p_high": "P(High)", "district": "",
                                     "predicted_risk": "Predicted risk"})
                st.plotly_chart(fig)
            st.dataframe(pred)
            st.download_button("⬇️ Download predictions (CSV)",
                               pred.to_csv(index=False), "next_week_predictions.csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
