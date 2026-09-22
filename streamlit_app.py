import streamlit as st

from app.app import ARTIFACT, LOAD_ERROR, human_label, parse_form, top_factors


def _field_groups(artifact):
    engineered = set(artifact.get("engineered_features", []))
    categorical = set(artifact["categorical_features"])
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
        if col in categorical:
            groups["Categorical"].append(col)
        elif col in applicant_keys:
            groups["Applicant"].append(col)
        elif any(key in col.lower() for key in ["deliq", "delinq", "dpd", "missed", "std", "sub", "dbt", "lss", "payment"]):
            groups["Delinquency & payments"].append(col)
        elif "enq" in col.lower():
            groups["Enquiries"].append(col)
        elif any(key in col.lower() for key in ["tl", "flag", "util", "bal", "exposure"]):
            groups["Bureau accounts"].append(col)
        else:
            groups["Other numeric"].append(col)
    return {name: columns for name, columns in groups.items() if columns}


st.set_page_config(page_title="Credit Risk Prediction", page_icon="CRP", layout="wide")

st.title("Credit Risk Prediction")
st.caption("Portfolio machine learning system for applicant priority classification")

if ARTIFACT is None:
    st.error(LOAD_ERROR or "The model could not be loaded.")
    st.stop()

st.info(
    "This educational model predicts priority levels from application and bureau features. "
    "It should not be used as the sole basis for real-world lending decisions."
)

bounds = ARTIFACT.get("numeric_bounds", {})
defaults = {
    col: bounds.get(col, {}).get("median", 0.0)
    for col in ARTIFACT["numeric_features"]
}
defaults.update(
    {col: ARTIFACT["category_values"][col][0] for col in ARTIFACT["categorical_features"]}
)

with st.form("risk_form"):
    st.subheader("Applicant information")
    st.caption("Numeric fields start at their training-set medians. Engineered ratios are calculated automatically.")
    values = {}

    for group_name, columns in _field_groups(ARTIFACT).items():
        st.markdown(f"**{group_name}**")
        widgets = st.columns(3)
        for index, col in enumerate(columns):
            with widgets[index % len(widgets)]:
                if col in ARTIFACT["categorical_features"]:
                    values[col] = st.selectbox(
                        human_label(col),
                        ARTIFACT["category_values"][col],
                        index=ARTIFACT["category_values"][col].index(defaults[col]),
                        key=f"category_{col}",
                    )
                else:
                    values[col] = st.number_input(
                        human_label(col),
                        value=float(defaults[col]),
                        step=1.0,
                        key=f"numeric_{col}",
                    )

    submitted = st.form_submit_button("Run prediction", type="primary", use_container_width=True)

if submitted:
    row, errors = parse_form(ARTIFACT, values)
    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    try:
        pipeline = ARTIFACT["pipeline"]
        probabilities = pipeline.predict_proba(row)[0]
        prediction = pipeline.predict(row)[0]
        if ARTIFACT["is_xgboost"]:
            inverse_labels = ARTIFACT["inv_label_map"]
            prediction = inverse_labels.get(prediction, inverse_labels.get(int(prediction), str(prediction)))
        prediction = str(prediction)
        probability_map = {
            str(label): float(probability)
            for label, probability in zip(ARTIFACT["class_order"], probabilities)
        }
        confidence = max(probabilities)
        factors = top_factors(ARTIFACT, row, prediction)
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    st.divider()
    st.subheader("Prediction result")
    result_column, probability_column = st.columns([1, 2])
    with result_column:
        st.metric("Predicted priority", prediction)
        st.metric("Prediction probability", f"{confidence:.1%}")
    with probability_column:
        st.markdown("**Class probabilities**")
        for label, probability in probability_map.items():
            st.write(f"{label}: {probability:.1%}")
            st.progress(probability)

    if factors:
        st.markdown("**Key associated features**")
        st.write(", ".join(factors))
        st.caption("These are model associations, not proof of causation.")
