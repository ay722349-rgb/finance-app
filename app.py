import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_fscore_support


# ----------------------------
# Page setup + small styling
# ----------------------------
st.set_page_config(
    page_title="Personal Finance Assistant",
    page_icon="💰",
    layout="wide",
)

st.markdown(
    """
    <style>
    .small-note {opacity: 0.75; font-size: 0.9rem;}
    .section-title {font-size: 1.25rem; font-weight: 700; margin: 0.2rem 0 0.6rem 0;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💰 AI-Driven Personal Finance Assistant")
st.caption("Upload a transaction CSV → categorize → visualize → forecast → get recommendations → evaluate results.")


# ----------------------------
# Helpers
# ----------------------------
REQUIRED_COLS = {"date", "description", "amount"}
CATEGORIES = ["Food", "Transport", "Rent", "Entertainment", "Utilities", "Miscellaneous"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def parse_amount(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def categorize_description(desc: str) -> str:
    d = str(desc).lower()

    food = ["grocery", "supermarket", "restaurant", "lunch", "dinner", "food", "pizza", "cafe", "coffee"]
    transport = ["uber", "ola", "taxi", "bus", "train", "metro", "fuel", "petrol", "diesel", "ticket"]
    entertainment = ["netflix", "movie", "cinema", "spotify", "concert", "entertainment"]
    utilities = ["electric", "electricity", "water", "gas", "bill", "recharge", "broadband", "internet", "mobile"]
    rent = ["rent"]

    if any(k in d for k in food):
        return "Food"
    if any(k in d for k in transport):
        return "Transport"
    if any(k in d for k in entertainment):
        return "Entertainment"
    if any(k in d for k in utilities):
        return "Utilities"
    if any(k in d for k in rent):
        return "Rent"
    return "Miscellaneous"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["date"] = parse_date(df["date"])
    df["amount"] = parse_amount(df["amount"])
    df["description"] = df["description"].astype(str)

    df = df.dropna(subset=["date"]).copy()
    df["category"] = df["description"].apply(categorize_description)

    # Helpful columns
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["type"] = np.where(df["amount"] < 0, "Income/Credit", "Expense/Debit")
    return df


def kpis(df: pd.DataFrame):
    expenses = df.loc[df["amount"] > 0, "amount"].sum()
    income = -df.loc[df["amount"] < 0, "amount"].sum()  # positive income
    net = income - expenses

    tx_count = len(df)
    avg_expense = df.loc[df["amount"] > 0, "amount"].mean()
    avg_expense = 0 if np.isnan(avg_expense) else avg_expense

    return {
        "expenses": float(expenses),
        "income": float(income),
        "net": float(net),
        "tx_count": int(tx_count),
        "avg_expense": float(avg_expense),
    }


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    s = (
        df[df["amount"] > 0]
        .groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )
    return s


def monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    m = (
        df[df["amount"] > 0]
        .groupby("month", as_index=False)["amount"]
        .sum()
    )
    m["_m"] = pd.to_datetime(m["month"] + "-01", errors="coerce")
    m = m.sort_values("_m").drop(columns="_m")
    m["t"] = np.arange(len(m))
    return m


def compute_category_baseline(df: pd.DataFrame, lookback_months: int = 3) -> pd.DataFrame:
    """Compute baseline spending per category.

    Baseline = average category spend over the previous `lookback_months` months,
    excluding the latest month. Also returns last month's spend and delta.
    """
    exp = df[df["amount"] > 0].copy()
    if exp.empty:
        return pd.DataFrame(columns=["category", "last_month", "last_spend", "baseline", "delta"])

    g = exp.groupby(["month", "category"], as_index=False)["amount"].sum()
    g["_m"] = pd.to_datetime(g["month"] + "-01", errors="coerce")
    g = g.sort_values("_m").dropna(subset=["_m"])

    months = g["month"].unique().tolist()
    if len(months) < 2:
        last_month = months[-1] if months else ""
        last = g[g["month"] == last_month][["category", "amount"]].rename(columns={"amount": "last_spend"})
        last["last_month"] = last_month
        last["baseline"] = np.nan
        last["delta"] = np.nan
        return last[["category", "last_month", "last_spend", "baseline", "delta"]].sort_values("last_spend", ascending=False)

    last_month = months[-1]
    prev_months = months[:-1][-lookback_months:]  # previous N months (excluding last)

    last = g[g["month"] == last_month][["category", "amount"]].rename(columns={"amount": "last_spend"})
    base = (
        g[g["month"].isin(prev_months)]
        .groupby("category", as_index=False)["amount"]
        .mean()
        .rename(columns={"amount": "baseline"})
    )

    out = last.merge(base, on="category", how="left")
    out["last_month"] = last_month
    out["delta"] = out["last_spend"] - out["baseline"]
    out = out[["category", "last_month", "last_spend", "baseline", "delta"]].sort_values("last_spend", ascending=False)
    return out



def forecast_next_months(monthly_df: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
    if len(monthly_df) < 3:
        return pd.DataFrame(columns=["month", "forecast"])

    X = monthly_df[["t"]].values
    y = monthly_df["amount"].values

    model = LinearRegression()
    model.fit(X, y)

    start_t = monthly_df["t"].max() + 1
    future_t = np.arange(start_t, start_t + horizon)

    last_month = pd.to_datetime(monthly_df["month"].iloc[-1] + "-01")
    future_months = pd.date_range(last_month, periods=horizon + 1, freq="MS")[1:]
    future_labels = future_months.strftime("%Y-%m").tolist()

    preds = model.predict(future_t.reshape(-1, 1))
    preds = np.maximum(preds, 0)

    return pd.DataFrame({"month": future_labels, "forecast": preds.round(2)})


def baseline_forecast(monthly_df: pd.DataFrame, horizon: int = 6, method: str = "Last month (naive)") -> pd.DataFrame:
    """
    Fallback forecast for short histories (1–2 months).
    method:
      - "Last month (naive)"  -> forecast = last actual month value
      - "Average of available months" -> forecast = mean of available months
    """
    if monthly_df.empty:
        return pd.DataFrame(columns=["month", "forecast"])

    last_month = pd.to_datetime(monthly_df["month"].iloc[-1] + "-01")
    future_months = pd.date_range(last_month, periods=horizon + 1, freq="MS")[1:]
    future_labels = future_months.strftime("%Y-%m")

    last_val = float(monthly_df["amount"].iloc[-1])
    avg_val = float(monthly_df["amount"].mean())

    base = last_val if method == "Last month (naive)" else avg_val
    base = max(base, 0)

    return pd.DataFrame({"month": list(future_labels), "forecast": [round(base, 2)] * horizon})


def backtest_forecast(monthly_df: pd.DataFrame):
    """Rolling one-step-ahead backtest on monthly total expenses.

    Fixes:
    - Always passes a 2D array to sklearn.predict (prevents 'Expected 2D container' error)
    - Uses the correct feature value (t) for previous prediction in direction accuracy
    """
    if monthly_df is None or monthly_df.empty or len(monthly_df) < 6:
        return None

    abs_err, sq_err, ape, dir_hits = [], [], [], []

    for i in range(3, len(monthly_df)):
        train = monthly_df.iloc[:i].copy()
        test_row = monthly_df.iloc[i]  # Series row

        model = LinearRegression()
        model.fit(train[["t"]], train["amount"])

        # ✅ Force 2D input for sklearn
        x_test = np.array([[float(test_row["t"])]], dtype=float)
        pred = float(model.predict(x_test)[0])
        pred = max(pred, 0.0)

        actual = float(test_row["amount"])
        e = actual - pred

        abs_err.append(abs(e))
        sq_err.append(e * e)
        if actual != 0:
            ape.append(abs(e) / actual)

        # Direction accuracy: compare change from previous month
        prev_actual = float(monthly_df.iloc[i - 1]["amount"])
        prev_t = float(monthly_df.iloc[i - 1]["t"])
        prev_pred = float(model.predict(np.array([[prev_t]], dtype=float))[0])

        actual_dir = actual - prev_actual
        pred_dir = pred - prev_pred
        dir_hits.append(np.sign(actual_dir) == np.sign(pred_dir))

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(sq_err)))
    mape = float(np.mean(ape) * 100) if ape else 0.0
    direction_acc = float(np.mean(dir_hits)) if dir_hits else 0.0

    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "Direction Accuracy": direction_acc}


def generate_recommendations(df: pd.DataFrame):
    m = (
        df[df["amount"] > 0]
        .groupby(["month", "category"], as_index=False)["amount"]
        .sum()
    )
    if m.empty:
        return []

    m["_m"] = pd.to_datetime(m["month"] + "-01", errors="coerce")
    m = m.sort_values("_m").drop(columns="_m")

    last_month = m["month"].iloc[-1]
    last = m[m["month"] == last_month].copy()

    prev = m[m["month"] != last_month].copy()
    prev_tail = prev.groupby("category").tail(3)

    baseline = (
        prev_tail.groupby("category", as_index=False)["amount"]
        .mean()
        .rename(columns={"amount": "baseline"})
    )

    out = last.merge(baseline, on="category", how="left").fillna({"baseline": 0})
    out["delta"] = out["amount"] - out["baseline"]

    recs = []
    overs = out.sort_values("delta", ascending=False)

    tips = {
        "Food": ["Plan meals weekly", "Avoid impulse snacks", "Cook in batches"],
        "Transport": ["Combine trips", "Use public transport when possible", "Track fuel spending"],
        "Rent": ["Ensure rent fits 30–35% of income", "Negotiate annually where possible"],
        "Entertainment": ["Set a monthly cap", "Pause unused subscriptions", "Try free alternatives"],
        "Utilities": ["Reduce standby power", "Switch to LED", "Monitor high-consumption devices"],
        "Miscellaneous": ["Review unknown spends weekly", "Create sub-categories", "Set a weekly limit"],
    }

    for _, r in overs.iterrows():
        cat = r["category"]
        delta = float(r["delta"])
        if delta <= 0:
            continue

        # Priority is assigned by overspend severity relative to baseline:
        #   Priority 1: overspend >= 50% of baseline
        #   Priority 2: overspend 20%–49% of baseline
        #   Priority 3: overspend < 20% of baseline
        baseline_val = float(r.get("baseline", 0.0))
        last_spend = float(r.get("amount", 0.0))

        if baseline_val <= 0:
            overspend_ratio = float("inf") if last_spend > 0 else 0.0
        else:
            overspend_ratio = (last_spend - baseline_val) / baseline_val

        if overspend_ratio >= 0.50:
            priority = 1
        elif overspend_ratio >= 0.20:
            priority = 2
        else:
            priority = 3
        recs.append({
            "priority": priority,
            "title": f"Reduce {cat} overspend vs baseline",
            "why": f"Last month {cat} spend is higher than baseline by about ₹{delta:,.0f}.",
            "actions": tips.get(cat, ["Track spends weekly", "Set a budget cap", "Review large transactions"]),
        })

    if not recs:
        recs.append({
            "priority": 3,
            "title": "Maintain your current spending discipline",
            "why": "No major overspend detected compared to recent baseline.",
            "actions": ["Keep tracking monthly", "Set small savings goals", "Review subscriptions quarterly"]
        })

    recs.sort(key=lambda x: x["priority"])
    return recs


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    forecast_horizon = st.number_input("Forecast horizon (months)", min_value=1, max_value=36, value=6, step=1)
    income_input = st.text_input("Monthly income (optional)", value="")
    ef_months = st.slider("Emergency fund months", min_value=1, max_value=12, value=6, step=1)
    st.markdown('<div class="small-note">Tip: Income is used only for savings & emergency fund guidance.</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("📤 Upload your transactions CSV (must contain: date, description, amount)", type=["csv"])


# ----------------------------
# Main UI
# ----------------------------
if not uploaded:
    st.info("Upload a CSV to begin. Example columns: **date, description, amount**.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded)
    df = preprocess(raw_df)
except Exception as e:
    st.error(f"Could not process file: {e}")
    st.stop()

tabs = st.tabs(["📄 Data", "📊 Insights", "📈 Forecast", "🧠 Recommendations", "✅ Evaluation", "🗣️ Feedback"])


# --- Tab: Data
with tabs[0]:
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown('<div class="section-title">Raw preview</div>', unsafe_allow_html=True)
        st.dataframe(raw_df.head(30), use_container_width=True, height=360)

    with c2:
        st.markdown('<div class="section-title">Processed data</div>', unsafe_allow_html=True)
        st.dataframe(df.head(30), use_container_width=True, height=360)

    st.download_button(
        "⬇️ Download processed_transactions.csv",
        df.to_csv(index=False).encode("utf-8"),
        file_name="processed_transactions.csv",
        mime="text/csv"
    )


# --- Tab: Insights
with tabs[1]:
    metrics = kpis(df)
    exp = metrics["expenses"]
    inc = metrics["income"]
    net = metrics["net"]

    m1, m2, m3, m4 = st.columns(4, gap="large")
    m1.metric("Total Expenses", f"₹{exp:,.0f}")
    m2.metric("Total Income", f"₹{inc:,.0f}")
    m3.metric("Net (Income − Expense)", f"₹{net:,.0f}")
    m4.metric("Transactions", f"{metrics['tx_count']}")

    s = category_summary(df)
    if s.empty:
        st.warning("No expense (debit) transactions detected.")
    else:
        c1, c2 = st.columns([1.2, 1], gap="large")
        with c1:
            st.markdown('<div class="section-title">Category-wise expense (₹)</div>', unsafe_allow_html=True)
            fig = px.bar(s, x="category", y="amount", text="amount")
            fig.update_traces(texttemplate="₹%{text:.0f}", textposition="outside")
            fig.update_layout(yaxis_title="Amount (₹)", xaxis_title="", height=420)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-title">Expense share</div>', unsafe_allow_html=True)
            fig2 = px.pie(s, names="category", values="amount", hole=0.45)
            fig2.update_layout(height=420)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">Category summary table</div>', unsafe_allow_html=True)
        s_show = s.copy()
        s_show["amount"] = s_show["amount"].map(lambda x: f"₹{x:,.2f}")
        st.dataframe(s_show, use_container_width=True, height=260)
        # --- Baseline display (previous months average vs last month)
        show_baseline = st.toggle("Show baseline comparison (advanced)", value=False)
        if show_baseline:
            st.markdown('<div class="section-title">Baseline (previous months average) vs Last Month</div>', unsafe_allow_html=True)
            baseline_df = compute_category_baseline(df, lookback_months=3)

            if baseline_df.empty:
                st.info("No expense data available to compute baseline.")
            elif baseline_df["baseline"].isna().all():
                st.warning("Baseline needs at least 2 unique months of expense history. Add more months to see baseline.")
                st.dataframe(baseline_df, use_container_width=True)
            else:
                show_df = baseline_df.copy()
                show_df["last_spend"] = show_df["last_spend"].map(lambda x: f"₹{x:,.2f}")
                show_df["baseline"] = show_df["baseline"].map(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "")
                show_df["delta"] = baseline_df["delta"].map(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "")
                st.dataframe(show_df, use_container_width=True, height=260)

                # Optional visual comparison
                chart_df = baseline_df.dropna(subset=["baseline"]).copy()
                chart_df = chart_df.melt(
                    id_vars=["category"],
                    value_vars=["last_spend", "baseline"],
                    var_name="metric",
                    value_name="amount"
                )
                figb = px.bar(chart_df, x="category", y="amount", color="metric", barmode="group")
                figb.update_layout(yaxis_title="Amount (₹)", xaxis_title="", height=380)
                st.plotly_chart(figb, use_container_width=True)



# --- Tab: Forecast (FIXED for User 2)
with tabs[2]:
    st.markdown('<div class="section-title">Monthly totals (expenses only)</div>', unsafe_allow_html=True)
    m = monthly_totals(df)

    if m.empty:
        st.warning("No expense (debit) transactions found, so forecasting cannot be performed.")
        st.stop()

    st.info(f"Months available: **{len(m)}** → " + ", ".join(m["month"].tolist()))

    fig = px.line(m, x="month", y="amount", markers=True)
    fig.update_layout(yaxis_title="Amount (₹)", xaxis_title="", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Forecast (next months)</div>', unsafe_allow_html=True)

    if len(m) >= 3:
        fc = forecast_next_months(m, horizon=int(forecast_horizon))
        st.success("Forecast method: **Linear Trend (Regression)**")
    else:
        st.warning("Limited history (< 3 months). Using fallback forecast (baseline).")
        method = st.selectbox(
            "Fallback forecast method",
            ["Last month (naive)", "Average of available months"],
            index=0
        )
        fc = baseline_forecast(m, horizon=int(forecast_horizon), method=method)
        st.success(f"Forecast method: **Baseline – {method}**")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.dataframe(fc, use_container_width=True, height=260)
    with c2:
        figf = px.line(fc, x="month", y="forecast", markers=True)
        figf.update_layout(yaxis_title="Forecast (₹)", xaxis_title="", height=260)
        st.plotly_chart(figf, use_container_width=True)

    st.download_button(
        "⬇️ Download forecast_summary.csv",
        fc.to_csv(index=False).encode("utf-8"),
        file_name="forecast_summary.csv",
        mime="text/csv"
    )

    # Optional savings guidance
    try:
        income_val = float(str(income_input).replace(",", "").strip()) if income_input else None
    except:
        income_val = None

    if income_val:
        next_month = float(fc["forecast"].iloc[0]) if len(fc) else 0.0
        est_savings = income_val - next_month
        target_ef = income_val * ef_months

        st.markdown('<div class="section-title">Savings guidance</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3, gap="large")
        a1.metric("Monthly Income (entered)", f"₹{income_val:,.0f}")
        a2.metric("Next-month forecast expense", f"₹{next_month:,.0f}")
        a3.metric("Estimated savings", f"₹{est_savings:,.0f}")

        st.info(f"Suggested emergency fund target: **₹{target_ef:,.0f}** (≈ {ef_months} months of income)")


# --- Tab: Recommendations
with tabs[3]:
    st.markdown('<div class="section-title">Personalized recommendations</div>', unsafe_allow_html=True)

    recs = generate_recommendations(df)
    for r in recs:
        with st.container(border=True):
            st.write(f"**Priority {r['priority']} — {r['title']}**")
            st.caption(r["why"])
            st.write("**Action steps:**")
            st.write("\n".join([f"- {a}" for a in r["actions"]]))

    rec_df = pd.DataFrame([{
        "priority": r["priority"],
        "title": r["title"],
        "why": r["why"],
        "actions": " | ".join(r["actions"])
    } for r in recs])

    st.download_button(
        "⬇️ Download recommendations.csv",
        rec_df.to_csv(index=False).encode("utf-8"),
        file_name="recommendations.csv",
        mime="text/csv"
    )


# --- Tab: Evaluation
with tabs[4]:
    st.markdown('<div class="section-title">Evaluate categorization</div>', unsafe_allow_html=True)
    st.caption("Upload a labeled dataset with columns: description, true_category. The app will compute Precision/Recall/F1.")

    eval_file = st.file_uploader("Upload labeled evaluation CSV", type=["csv"], key="eval_labeled")
    if eval_file:
        try:
            eval_df = pd.read_csv(eval_file)
            eval_df.columns = [c.strip().lower() for c in eval_df.columns]
            if not {"description", "true_category"}.issubset(set(eval_df.columns)):
                st.error("Labeled file must contain: description, true_category")
            else:
                eval_df["predicted_category"] = eval_df["description"].apply(categorize_description)

                labels = CATEGORIES
                p, r, f1, _ = precision_recall_fscore_support(
                    eval_df["true_category"],
                    eval_df["predicted_category"],
                    labels=labels,
                    zero_division=0
                )

                metrics_df = pd.DataFrame({
                    "Category": labels,
                    "Precision": np.round(p, 2),
                    "Recall": np.round(r, 2),
                    "F1-Score": np.round(f1, 2),
                })
                avg = pd.DataFrame([{
                    "Category": "Average",
                    "Precision": round(float(np.mean(p)), 2),
                    "Recall": round(float(np.mean(r)), 2),
                    "F1-Score": round(float(np.mean(f1)), 2),
                }])
                metrics_df = pd.concat([metrics_df, avg], ignore_index=True)

                st.dataframe(metrics_df, use_container_width=True)

                st.download_button(
                    "⬇️ Download categorization_metrics.csv",
                    metrics_df.to_csv(index=False).encode("utf-8"),
                    file_name="categorization_metrics.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Could not evaluate categorization: {e}")

    st.divider()
    st.markdown('<div class="section-title">Evaluate forecasting (backtest)</div>', unsafe_allow_html=True)
    bt = backtest_forecast(monthly_totals(df))
    if not bt:
        st.warning("Not enough monthly history for backtesting. Need at least ~6 months.")
    else:
        bt_df = pd.DataFrame({"Metric": list(bt.keys()), "Value": [bt[k] for k in bt.keys()]})
        bt_df["Value"] = bt_df.apply(
            lambda x: f"{x['Value']:.2f}" if x["Metric"] != "Direction Accuracy" else f"{x['Value']:.2f} ({x['Value']*100:.0f}%)",
            axis=1
        )
        st.dataframe(bt_df, use_container_width=True)


# --- Tab: Feedback
with tabs[5]:
    st.markdown('<div class="section-title">Recommendation feedback</div>', unsafe_allow_html=True)
    st.caption("Collect user feedback on clarity, relevance, and actionability.")

    if "feedback_log" not in st.session_state:
        st.session_state["feedback_log"] = []

    c1, c2, c3, c4 = st.columns(4, gap="large")
    relevance = c1.slider("Relevance", 1, 5, 4)
    clarity = c2.slider("Clarity", 1, 5, 4)
    actionability = c3.slider("Actionability", 1, 5, 4)
    overall = c4.slider("Overall satisfaction", 1, 10, 8)

    comment = st.text_area("Optional comment", placeholder="What would you improve?")

    if st.button("Submit feedback ✅"):
        st.session_state["feedback_log"].append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "relevance": relevance,
            "clarity": clarity,
            "actionability": actionability,
            "overall": overall,
            "comment": comment,
        })
        st.success("Thanks! Feedback recorded.")

    if st.session_state["feedback_log"]:
        fb = pd.DataFrame(st.session_state["feedback_log"])
        st.markdown('<div class="section-title">Feedback dashboard</div>', unsafe_allow_html=True)
        st.dataframe(fb, use_container_width=True, height=240)

        st.download_button(
            "⬇️ Download feedback.csv",
            fb.to_csv(index=False).encode("utf-8"),
            file_name="feedback.csv",
            mime="text/csv"
        )
    else:
        st.info("No feedback submitted yet.")
