import streamlit as st
import pandas as pd
import ollama
import os
import re
import json
from datetime import datetime

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
client = ollama.Client(host=ollama_host)

st.set_page_config(page_title="Data Quality Agent", layout="wide")
st.title("🤖 Data Quality Agent")
st.write("Upload any CSV and the AI will validate it and explain the results.")

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
NULL_STRINGS = {"", "none", "n/a", "na", "not provided", "nan", "null", "unknown"}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d"]

def is_null_like(val):
    return str(val).strip().lower() in NULL_STRINGS

def check_email(val):
    if is_null_like(val): return False
    return not re.match(r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$", str(val).strip())

def check_range(val, min_val, max_val):
    if is_null_like(val): return False
    try:
        v = float(str(val).strip())
        return v < min_val or v > max_val
    except:
        return True

def check_numeric_positive(val):
    if is_null_like(val): return False
    try:
        return float(str(val).strip()) <= 0
    except:
        return True

def check_date(val):
    if is_null_like(val): return False
    for fmt in DATE_FORMATS:
        try:
            datetime.strptime(str(val).strip(), fmt)
            return False
        except:
            continue
    return True

def check_categorical(val, allowed):
    if is_null_like(val): return False
    return str(val).strip().lower() not in [a.lower() for a in allowed]

def check_duplicate(series):
    return series.duplicated(keep=False)

def apply_rule(df, col, rule):
    if rule == "email_format":
        return df[col].apply(check_email)
    elif rule == "numeric_positive":
        return df[col].apply(check_numeric_positive)
    elif rule == "date_format":
        return df[col].apply(check_date)
    elif rule == "not_null":
        return df[col].apply(is_null_like)
    elif rule == "unique":
        return check_duplicate(df[col])
    elif rule.startswith("range:"):
        _, mn, mx = rule.split(":")
        return df[col].apply(lambda v: check_range(v, float(mn), float(mx)))
    elif rule.startswith("categorical:"):
        allowed = rule.split(":")[1].split(",")
        return df[col].apply(lambda v: check_categorical(v, allowed))
    return pd.Series([False] * len(df))

# ─────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    st.success(f"File loaded — {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10))

    # ─────────────────────────────────────────
    # STEP 2 — DATA PROFILE
    # ─────────────────────────────────────────
    st.subheader("📊 Data Profile")

    profile = []
    for col in df.columns:
        null_count = df[col].apply(is_null_like).sum()
        unique_count = df[col].nunique()
        sample = df[col][~df[col].apply(is_null_like)].iloc[0] if null_count < len(df) else "ALL NULL"
        profile.append({
            "column": col,
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 1),
            "unique_values": unique_count,
            "sample_value": sample
        })

    profile_df = pd.DataFrame(profile)
    st.dataframe(profile_df)

    # ─────────────────────────────────────────
    # STEP 3 — AI DECIDES RULES
    # ─────────────────────────────────────────
    st.subheader("🤖 AI Rule Detection")

    if st.button("Analyse & Validate with AI"):
        with st.spinner("AI is deciding validation rules..."):

            profile_text = profile_df.to_string(index=False)
            columns = list(df.columns)
            samples = {col: df[col][~df[col].apply(is_null_like)].iloc[0] if df[col].apply(is_null_like).sum() < len(df) else "ALL NULL" for col in columns}

            prompt = f"""
You are a data quality expert. Based on the column names and sample values below, return a JSON object mapping each column to the best validation rule.

Only return valid JSON. No explanation, no markdown, no extra text.

Available rules:
- "not_null" — column must not be empty
- "email_format" — must be valid email
- "numeric_positive" — must be a positive number
- "date_format" — must be a valid date
- "unique" — values must be unique
- "range:MIN:MAX" — value must be between MIN and MAX (e.g. "range:18:100")
- "categorical:val1,val2,val3" — value must be one of these

Columns and sample values:
{json.dumps(samples, indent=2)}

Return only a JSON object like:
{{
  "column_name": "rule",
  "column_name2": "range:0:100"
}}
"""

            raw = ""
            for chunk in client.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            ):
                raw += chunk["message"]["content"]

            st.write("**AI detected these rules:**")

            # parse JSON from AI response
            try:
                json_match = re.search(r"\{.*\}", raw, re.DOTALL)
                rules = json.loads(json_match.group()) if json_match else {}
            except:
                rules = {}

            if rules:
                rules_df = pd.DataFrame(list(rules.items()), columns=["column", "rule"])
                st.dataframe(rules_df)
            else:
                st.warning("AI could not detect rules. Using not_null for all columns.")
                rules = {col: "not_null" for col in df.columns}

        # ─────────────────────────────────────────
        # STEP 4 — RUN VALIDATION
        # ─────────────────────────────────────────
        st.subheader("✅ Validation Results")

        with st.spinner("Running validation..."):
            validation_results = []
            failed_mask = pd.Series([False] * len(df))

            for col, rule in rules.items():
                if col not in df.columns:
                    continue
                col_failed = apply_rule(df, col, rule)
                failed_count = col_failed.sum()
                failed_mask = failed_mask | col_failed
                validation_results.append({
                    "column": col,
                    "rule": rule,
                    "failed_rows": int(failed_count),
                    "failed_pct": round(failed_count / len(df) * 100, 1),
                    "status": "✅ PASSED" if failed_count == 0 else "❌ FAILED"
                })

            results_df = pd.DataFrame(validation_results)
            st.dataframe(results_df)

            df_passed = df[~failed_mask]
            df_failed = df[failed_mask]

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", len(df))
            col2.metric("Passed Rows", len(df_passed))
            col3.metric("Failed Rows", len(df_failed))

            st.dataframe(df_passed.head(10))

            csv = df_passed.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Passed Rows as CSV",
                data=csv,
                file_name="passed_rows.csv",
                mime="text/csv"
            )