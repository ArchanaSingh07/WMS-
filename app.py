
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

st.set_page_config(page_title="AI Warehouse Operations Control Tower", page_icon="🏭", layout="wide")

BASE = Path(__file__).parent
DATA = BASE / "data"

@st.cache_data
def load_csv(name):
    return pd.read_csv(DATA / name)

def load_data():
    names = ["warehouses.csv","leases.csv","assets.csv","maintenance.csv","logistics.csv"]
    return {n.replace(".csv",""): load_csv(n) for n in names}

def apply_uploads(data):
    st.sidebar.markdown("### Optional: Load your data")
    st.sidebar.caption("Upload CSVs using the matching template. Uploaded data is used for this browser session.")
    for key, filename in [
        ("warehouses","warehouses.csv"),("leases","leases.csv"),("assets","assets.csv"),
        ("maintenance","maintenance.csv"),("logistics","logistics.csv")]:
        f = st.sidebar.file_uploader(filename, type=["csv"], key=f"upload_{key}")
        if f is not None:
            data[key] = pd.read_csv(f)
    return data

def risk_engine(data):
    w = data["warehouses"].copy()
    a = data["assets"].copy()
    m = data["maintenance"].copy()
    l = data["leases"].copy()
    g = data["logistics"].copy()

    today = pd.Timestamp.today().normalize()
    l["expiry_date"] = pd.to_datetime(l["expiry_date"], errors="coerce")
    l["notice_date"] = l["expiry_date"] - pd.to_timedelta(pd.to_numeric(l["notice_days"], errors="coerce"), unit="D")
    l["days_to_notice"] = (l["notice_date"] - today).dt.days

    asset_risk = a.groupby("warehouse_id").apply(
        lambda x: max(0, (1-x["health_score"].mean())) * 100 + x["failures_90d"].sum()*2
    ).reset_index(name="asset_risk")
    maint_risk = m[m["status"].isin(["Open","Overdue"])].groupby("warehouse_id").size().reset_index(name="open_maintenance")
    g2 = g.copy()
    g2["logistics_risk"] = (
        (100-g2["dispatch_sla_pct"]).clip(lower=0)*0.7 +
        (100-g2["delivery_sla_pct"]).clip(lower=0)*0.4 +
        g2["vehicles_unavailable"]*2 +
        g2["external_disruption_risk"]*25
    )
    out = w.merge(asset_risk,on="warehouse_id",how="left").merge(maint_risk,on="warehouse_id",how="left").merge(
        g2[["warehouse_id","backlog","dispatch_sla_pct","delivery_sla_pct","logistics_risk"]],on="warehouse_id",how="left"
    )
    out["asset_risk"] = out["asset_risk"].fillna(0)
    out["open_maintenance"] = out["open_maintenance"].fillna(0)
    out["lease_risk"] = 0.0
    for _, r in l.iterrows():
        if r["warehouse_id"] in set(out["warehouse_id"]):
            if pd.notna(r["days_to_notice"]):
                if r["days_to_notice"] < 0: score=100
                elif r["days_to_notice"] <= 60: score=85
                elif r["days_to_notice"] <= 180: score=65
                elif r["days_to_notice"] <= 365: score=30
                else: score=10
                out.loc[out.warehouse_id==r.warehouse_id,"lease_risk"] = score
    out["capacity_risk"] = ((out["utilisation"]-0.80).clip(lower=0)/0.20*100).clip(0,100)
    out["maintenance_risk"] = (out["asset_risk"]*0.65 + out["open_maintenance"].clip(0,5)*7).clip(0,100)
    out["overall_risk"] = (
        out["capacity_risk"]*0.22 + out["maintenance_risk"]*0.28 +
        out["lease_risk"]*0.20 + out["logistics_risk"].fillna(0).clip(0,100)*0.30
    ).clip(0,100)
    out["severity"] = pd.cut(out["overall_risk"], bins=[-1,35,60,75,101], labels=["Low","Medium","High","Critical"])
    return out, l

def metric_card(label, value, delta=None):
    st.metric(label, value, delta)

data = apply_uploads(load_data())
risk, lease = risk_engine(data)

st.title("🏭 AI Warehouse Operations Control Tower")
st.caption("MVP v0.1 — decision-support dashboard. Risk scores are illustrative and configurable; connect authoritative systems before production use.")

with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("Go to", ["Executive Control Tower","Warehouse 360","Lease Intelligence","Maintenance Intelligence","Logistics Risk","Data & Configuration"])
    st.divider()
    st.markdown("### MVP status")
    st.success("Dashboard: Live")
    st.info("AI Copilot: Next phase")
    st.info("Predictive models: Next phase")
    st.warning("Sample data active unless CSVs are uploaded")

if page == "Executive Control Tower":
    st.subheader("Today's operational picture")
    c1,c2,c3,c4,c5 = st.columns(5)
    metric_card("Warehouses", len(risk))
    metric_card("Critical risks", int((risk.severity=="Critical").sum()))
    metric_card("High risks", int((risk.severity=="High").sum()))
    metric_card("Open maintenance", int(data["maintenance"]["status"].isin(["Open","Overdue"]).sum()))
    metric_card("Lease actions ≤180d", int((lease["days_to_notice"]<=180).sum()))

    st.markdown("### Priority queue")
    q = risk.sort_values("overall_risk", ascending=False).copy()
    q["risk_score"] = q["overall_risk"].round(0).astype(int)
    q["risk_reason"] = np.where(q.capacity_risk>60,"Capacity pressure; ",
                         "") + np.where(q.maintenance_risk>60,"Maintenance/asset risk; ","") + np.where(q.lease_risk>60,"Lease deadline; ","") + np.where(q.logistics_risk>45,"Logistics disruption; ","")
    st.dataframe(q[["warehouse_id","warehouse_name","region","severity","risk_score","risk_reason","utilisation","backlog"]], use_container_width=True, hide_index=True)

    st.markdown("### Risk distribution")
    dist = risk["severity"].value_counts().reindex(["Critical","High","Medium","Low"]).fillna(0)
    st.bar_chart(dist)

elif page == "Warehouse 360":
    st.subheader("Warehouse 360")
    selected = st.selectbox("Select warehouse", risk["warehouse_name"].tolist())
    wh = risk[risk.warehouse_name==selected].iloc[0]
    wid = wh.warehouse_id
    c1,c2,c3,c4 = st.columns(4)
    metric_card("Overall risk", f"{wh.overall_risk:.0f}/100", wh.severity)
    metric_card("Utilisation", f"{wh.utilisation*100:.0f}%")
    metric_card("Backlog", f"{int(wh.backlog):,}")
    metric_card("Delivery SLA", f"{float(data['logistics'].loc[data['logistics'].warehouse_id==wid,'delivery_sla_pct'].iloc[0]):.1f}%")
    st.markdown("### Why is this warehouse at risk?")
    reasons = []
    if wh.capacity_risk>35: reasons.append(f"Capacity pressure: utilisation is {wh.utilisation*100:.0f}%.")
    if wh.maintenance_risk>35: reasons.append(f"Maintenance/asset risk: {int(wh.open_maintenance)} open work orders; asset risk {wh.asset_risk:.0f}.")
    if wh.lease_risk>35: reasons.append(f"Lease action is approaching: lease risk {wh.lease_risk:.0f}.")
    if wh.logistics_risk>35: reasons.append(f"Logistics risk is elevated: backlog {int(wh.backlog):,}.")
    if not reasons: reasons.append("No major rule-based risk signal is currently detected.")
    for r in reasons: st.warning(r)
    st.markdown("### Maintenance")
    st.dataframe(data["maintenance"][data["maintenance"].warehouse_id==wid], use_container_width=True, hide_index=True)
    st.markdown("### Lease")
    st.dataframe(lease[lease.warehouse_id==wid], use_container_width=True, hide_index=True)
    st.markdown("### Operational metrics")
    st.dataframe(data["logistics"][data["logistics"].warehouse_id==wid], use_container_width=True, hide_index=True)

elif page == "Lease Intelligence":
    st.subheader("Lease Intelligence")
    l = lease.copy()
    l["days_to_notice"] = pd.to_numeric(l["days_to_notice"], errors="coerce")
    l["priority"] = pd.cut(l["days_to_notice"], bins=[-99999,0,60,180,365,99999], labels=["Overdue","Critical","High","Watch","Later"])
    c1,c2,c3 = st.columns(3)
    metric_card("Overdue notice", int((l.days_to_notice<0).sum()))
    metric_card("Notice ≤60d", int(l.days_to_notice.between(0,60).sum()))
    metric_card("Notice ≤180d", int(l.days_to_notice.between(0,180).sum()))
    st.dataframe(l[["lease_id","warehouse_id","expiry_date","notice_days","notice_date","days_to_notice","priority","status","monthly_rent"]].sort_values("days_to_notice"), use_container_width=True, hide_index=True)
    st.info("Next AI layer: upload lease PDFs and the system will extract clauses, renewal options, obligations and evidence-backed deadlines.")

elif page == "Maintenance Intelligence":
    st.subheader("Maintenance Intelligence")
    a = data["assets"].copy()
    c1,c2,c3 = st.columns(3)
    metric_card("Assets", len(a))
    metric_card("Warning assets", int((a.status=="Warning").sum()))
    metric_card("Critical assets", int((a.criticality=="Critical").sum()))
    a["risk"] = ((1-a.health_score)*70 + a.failures_90d*5).clip(0,100)
    st.dataframe(a.sort_values("risk",ascending=False), use_container_width=True, hide_index=True)
    st.markdown("### Open work orders")
    st.dataframe(data["maintenance"][data["maintenance"].status.isin(["Open","Overdue"])].sort_values("severity"), use_container_width=True, hide_index=True)
    st.info("Next AI layer: combine maintenance history + telemetry + warehouse criticality to predict failure risk.")

elif page == "Logistics Risk":
    st.subheader("Last-mile & logistics risk")
    g = data["logistics"].copy()
    g["risk_score"] = ((100-g.dispatch_sla_pct)*.7 + (100-g.delivery_sla_pct)*.4 + g.vehicles_unavailable*2 + g.external_disruption_risk*25).clip(0,100)
    st.dataframe(g.sort_values("risk_score",ascending=False), use_container_width=True, hide_index=True)
    st.markdown("### Operational signals")
    st.bar_chart(g.set_index("warehouse_id")[["dispatch_sla_pct","delivery_sla_pct"]])
    st.info("Next AI layer: correlate weather/traffic/events with warehouse capacity and route constraints to generate disruption recommendations.")

elif page == "Data & Configuration":
    st.subheader("Data & Configuration")
    st.markdown("### Required CSV templates")
    for k,df in data.items():
        with st.expander(k.title()):
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            st.caption(f"Rows: {len(df)}")
    st.markdown("### Current risk formula")
    st.code("""overall_risk =
  22% × capacity_risk
+ 28% × maintenance_risk
+ 20% × lease_risk
+ 30% × logistics_risk""")
    st.warning("This is an MVP rules engine, not a validated predictive model. The weights should be calibrated with real operational outcomes.")
    st.markdown("### Build roadmap")
    st.write("1. Connect real CSV/API sources → 2. Add lease document upload/RAG → 3. Add AI daily briefing → 4. Add evaluation harness → 5. Add controlled actions → 6. Add predictive risk.")

st.divider()
st.caption("AI Warehouse Operations Control Tower • MVP v0.1 • Built for iterative development")
