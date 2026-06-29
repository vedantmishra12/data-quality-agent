import streamlit as st
import pandas as pd
import ollama

st.set_page_config(page_title="Data Quality Agent", layout="wide")
st.title("🤖 Data Quality Agent")
st.write("Upload any CSV and the AI will validate it and explain the results.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    st.success(f"File loaded — {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10))

    # ─────────────────────────────────────────
    # STEP 2 — DATA PROFILE
    # ─────────────────────────────────────────
    st.subheader("📊 Data Profile")

    NULL_STRINGS = {"", "none", "n/a", "na", "not provided", "nan"}

    def is_null_like(val):
        return str(val).strip().lower() in NULL_STRINGS

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
    # STEP 3 — AI ANALYSIS
    # ─────────────────────────────────────────
    st.subheader("🤖 AI Analysis")

    if st.button("Analyse with AI"):
        with st.spinner("AI is analysing your data..."):

            profile_text = profile_df.to_string(index=False)

            prompt = f"""
            You are a data quality expert. Analyse this data profile and:
            1. Identify the most critical data quality issues
            2. Explain what each issue means in simple terms
            3. Suggest how to fix each issue

            Data Profile:
            {profile_text}

            Total rows: {len(df)}
            Columns: {list(df.columns)}
            """

            response = ollama.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}]
            )

            ai_response = response["message"]["content"]
            st.markdown(ai_response)

    # ─────────────────────────────────────────
    # STEP 4 — VALIDATION & DOWNLOAD
    # ─────────────────────────────────────────
    st.subheader("✅ Validation & Clean Data")

    NULL_LIKE = {"", "none", "n/a", "na", "not provided", "nan", "null", "unknown"}

    def not_null(val):
        return str(val).strip().lower() not in NULL_LIKE

    def is_numeric_positive(val):
        try:
            return float(str(val).strip()) > 0
        except:
            return False

    def is_between(val, min_val, max_val):
        try:
            v = float(str(val).strip())
            return min_val <= v <= max_val
        except:
            return False

    mask = pd.Series([True] * len(df))

    for col in df.columns:
        mask = mask & df[col].apply(not_null)

    df_passed = df[mask]
    df_failed = df[~mask]

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