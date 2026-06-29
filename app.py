import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime

st.set_page_config(page_title="Data Quality Agent", layout="wide")
st.title("🤖 Data Quality Agent")
st.write("Upload any CSV and it will automatically validate your data.")

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

def check_range(val, min_val, max_val):
    if is_null_like(val): return False
    try:
        v = float(str(val).strip())
        return v < min_val or v > max_val
    except:
        return True

def check_categorical(val, allowed):
    if is_null_like(val): return False
    return str(val).strip().lower() not in [a.lower() for a in allowed]

def check_phone(val):
    if is_null_like(val): return False
    v = str(val).strip().replace("-", "").replace(" ", "")
    if v in {"0000000000", "9999999999"}: return True
    return not re.match(r"^(\+91)?[6-9]\d{9}$", v)

def detect_rule(col, sample_val):
    col_lower = col.lower()
    if "email" in col_lower:
        return "email_format"
    elif "phone" in col_lower or "mobile" in col_lower or "contact" in col_lower:
        return "phone_format"
    elif "age" in col_lower:
        return "range:18:100"
    elif "price" in col_lower or "amount" in col_lower or "spent" in col_lower or "cost" in col_lower:
        return "numeric_positive"
    elif "quantity" in col_lower or "qty" in col_lower:
        return "range:1:999"
    elif "discount" in col_lower:
        return "range:0:100"
    elif "date" in col_lower:
        return "date_format"
    elif "status" in col_lower:
        return "categorical:Delivered,Shipped,Pending,Cancelled,Returned"
    elif "payment" in col_lower or "method" in col_lower:
        return "categorical:UPI,Credit Card,Debit Card,Net Banking,Cash on Delivery,Wallet,EMI"
    elif "gender" in col_lower:
        return "categorical:Male,Female,Other"
    elif "tier" in col_lower:
        return "categorical:Gold,Silver,Bronze,Platinum"
    elif "country" in col_lower:
        return "categorical:India"
    elif "id" in col_lower:
        return "unique"
    else:
        return "not_null"

def apply_rule(df, col, rule):
    if rule == "email_format":
        return df[col].apply(check_email)
    elif rule == "phone_format":
        return df[col].apply(check_phone)
    elif rule == "numeric_positive":
        return df[col].apply(check_numeric_positive)
    elif rule == "date_format":
        return df[col].apply(check_date)
    elif rule == "not_null":
        return df[col].apply(is_null_like)
    elif rule == "unique":
        return df[col].duplicated(keep=False)
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
    # STEP 3 — AUTO RULE DETECTION
    # ─────────────────────────────────────────
    st.subheader("🔍 Auto Detected Rules")

    rules = {}
    for col in df.columns:
        sample = df[col][~df[col].apply(is_null_like)].iloc[0] if df[col].apply(is_null_like).sum() < len(df) else ""
        rules[col] = detect_rule(col, sample)

    rules_df = pd.DataFrame(list(rules.items()), columns=["column", "rule"])
    st.dataframe(rules_df)

    # ─────────────────────────────────────────
    # STEP 4 — VALIDATION
    # ─────────────────────────────────────────
    st.subheader("✅ Validation Results")

    validation_results = []
    failed_mask = pd.Series([False] * len(df))

    for col, rule in rules.items():
        col_failed = apply_rule(df, col, rule)
        failed_count = int(col_failed.sum())
        failed_mask = failed_mask | col_failed
        validation_results.append({
            "column": col,
            "rule": rule,
            "failed_rows": failed_count,
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