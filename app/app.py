import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "credit_risk_pipeline.pkl"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "credit-risk-portfolio-demo")

ARTIFACT = None
LOAD_ERROR = None


def load_artifact():
    global ARTIFACT, LOAD_ERROR
    try:
        ARTIFACT = joblib.load(MODEL_PATH)
        LOAD_ERROR = None
    except Exception:
        ARTIFACT = None
        LOAD_ERROR = "The model file could not be loaded. Train the notebook so models/credit_risk_pipeline.pkl exists."


load_artifact()


def human_label(name: str) -> str:
    return name.replace("_", " ").strip()


def field_groups(artifact):
    engineered = set(artifact.get("engineered_features", []))
    cats = set(artifact["categorical_features"])
    groups = {
        "Applicant": [],
        "Bureau accounts": [],
        "Delinquency & payments": [],
        "Enquiries": [],
        "Other numeric": [],
        "Categorical": [],
    }
    applicant_keys = {"AGE", "GENDER", "MARITALSTATUS", "EDUCATION", "NETMONTHLYINCOME", "Time_With_Curr_Empr"}
    for col in artifact["input_columns"]:
        if col in engineered:
            continue
        if col in cats or col in artifact["categorical_features"]:
            groups["Categorical"].append(col)
        elif col in applicant_keys:
            groups["Applicant"].append(col)
        elif any(k in col.lower() for k in ["deliq", "delinq", "dpd", "missed", "std", "sub", "dbt", "lss", "payment"]):
            groups["Delinquency & payments"].append(col)
        elif "enq" in col.lower():
            groups["Enquiries"].append(col)
        elif any(k in col.lower() for k in ["tl", "flag", "util", "bal", "exposure"]):
            groups["Bureau accounts"].append(col)
        else:
            groups["Other numeric"].append(col)
    return {k: v for k, v in groups.items() if v}


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    tl = out["Total_TL"].fillna(0)
    out["missed_pmnt_per_tl"] = out["Tot_Missed_Pmnt"] / (tl + 1)
    out["delinquent_per_tl"] = out["num_times_delinquent"] / (tl + 1)
    out["recent_enq_share"] = out["enq_L3m"] / out["enq_L12m"].replace(0, np.nan)
    return out


def parse_form(artifact, form):
    data = {}
    errors = []
    for col in artifact["numeric_features"]:
        if col in artifact.get("engineered_features", []):
            continue
        raw = form.get(col, "")
        if raw is None or str(raw).strip() == "":
            errors.append(f"{human_label(col)} is required.")
            continue
        try:
            data[col] = float(raw)
        except ValueError:
            errors.append(f"{human_label(col)} must be a number.")
    for col in artifact["categorical_features"]:
        raw = form.get(col, "")
        allowed = artifact["category_values"][col]
        if raw not in allowed:
            errors.append(f"{human_label(col)} has an invalid value.")
        else:
            data[col] = raw
    if errors:
        return None, errors
    row = pd.DataFrame([data])
    row = add_engineered_features(row)
    for col in artifact["input_columns"]:
        if col not in row.columns:
            row[col] = np.nan
    row = row[artifact["input_columns"]]
    return row, []


def top_factors(artifact, row, pred_label, n=5):
    pipeline = artifact["pipeline"]
    estimator = pipeline.named_steps["model"]
    prep = pipeline.named_steps["prep"]
    ohe = prep.named_transformers_["cat"].named_steps["onehot"]
    names = artifact["numeric_features"] + list(ohe.get_feature_names_out(artifact["categorical_features"]))
    transformed = prep.transform(row)
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
        contrib = pd.Series(importances, index=names)
        ranked = contrib.sort_values(ascending=False).head(n)
        return [human_label(i) for i in ranked.index]
    if hasattr(estimator, "coef_"):
        classes = list(estimator.classes_)
        label_map = artifact["label_map"]
        key = label_map.get(pred_label, pred_label)
        cls_index = classes.index(key)
        contrib = pd.Series(np.abs(estimator.coef_[cls_index]), index=names)
        ranked = contrib.sort_values(ascending=False).head(n)
        return [human_label(i) for i in ranked.index]
    return []


def form_context(values=None, errors=None):
    if ARTIFACT is None:
        return {"load_error": LOAD_ERROR, "artifact": None}
    bounds = ARTIFACT.get("numeric_bounds", {})
    current = {col: bounds.get(col, {}).get("median", "") for col in ARTIFACT["numeric_features"]}
    current.update({col: ARTIFACT["category_values"][col][0] for col in ARTIFACT["categorical_features"]})
    if values:
        current.update(values)
    return {
        "load_error": LOAD_ERROR,
        "artifact": ARTIFACT,
        "groups": field_groups(ARTIFACT),
        "values": current,
        "bounds": bounds,
        "human_label": human_label,
        "errors": errors,
    }


@app.route("/")
def home():
    return render_template("index.html", **form_context())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if ARTIFACT is None:
        return render_template("index.html", **form_context()), 500
    if request.method == "GET":
        return redirect(url_for("home") + "#application")

    values = {key: request.form.get(key) for key in request.form}
    row, errors = parse_form(ARTIFACT, request.form)
    if errors:
        return render_template("index.html", **form_context(values=values, errors=errors)), 400
    try:
        pipeline = ARTIFACT["pipeline"]
        proba = pipeline.predict_proba(row)[0]
        pred_raw = pipeline.predict(row)[0]
        if ARTIFACT["is_xgboost"]:
            inv = ARTIFACT["inv_label_map"]
            pred_label = inv.get(pred_raw, inv.get(int(pred_raw), str(pred_raw)))
        else:
            pred_label = pred_raw
        classes = ARTIFACT["class_order"]
        proba_map = {str(c): float(p) for c, p in zip(classes, proba)}
        confidence = float(max(proba))
        factors = top_factors(ARTIFACT, row, pred_label)
        return render_template(
            "result.html",
            pred_label=pred_label,
            confidence=confidence,
            proba_map=proba_map,
            factors=factors,
        )
    except Exception:
        return render_template(
            "index.html",
            **form_context(values=values, errors=["Prediction failed. Check that every field is filled with a valid value."]),
        ), 400


@app.errorhandler(404)
def not_found(_e):
    return render_template("index.html", **form_context(errors=["Page not found."])), 404


@app.errorhandler(500)
def server_error(_e):
    return render_template("index.html", **form_context(errors=["An unexpected server error occurred."])), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
