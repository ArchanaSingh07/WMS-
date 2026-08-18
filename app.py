import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="AI Warehouse Operations Control Tower", page_icon="🏭", layout="wide")

BASE = Path(__file__).parent
DATA = BASE / "data"

def demo_data():
    warehouses = pd.DataFrame([
        ["WH-BLR-01","Bangalore South","South",18000,14500,.81,"High"],
        ["WH-BLR-02","Bangalore North","South",22000,20800,.95,"High"],
        ["WH-MUM-01","Mumbai East","West",20000,17200,.86,"High"],
        ["WH-DEL-01","Delhi NCR","North",30000,25800,.86,"Critical"],
        ["WH-HYD-01","Hyderabad","South",16000,10900,.68,"Medium"],
        ["WH-PUN-01","Pune","West",14000,12100,.86,"Medium"],
        ["WH-CHN-01","Chennai","South",19000,15300,.81,"High"],
        ["WH-KOL-01","Kolkata","East",12000,9900,.83,"Medium"],
    ], columns=["warehouse_id","warehouse_name","region","capacity","outbound","utilisation","criticality"])

    leases = pd.DataFrame([
        ["L-001","WH-BLR-01","2027-06-30",180,"Active"],
        ["L-002","WH-BLR-02","2026-12-31",180,"Renewal Review"],
        ["L-003","WH-MUM-01","2028-03-31",120,"Active"],
        ["L-004","WH-DEL-01","2027-03-31",180,"Renewal Review"],
        ["L-005","WH-HYD-01","2029-01-31",90,"Active"],
        ["L-006","WH-PUN-01","2026-11-30",120,"Action Required"],
        ["L-007","WH-CHN-01","2027-09-30",180,"Active"],
        ["L-008","WH-KOL-01","2028-07-31",90,"Active"],
    ], columns=["lease_id","warehouse_id","expiry_date","notice_days","status"])

    assets = pd.DataFrame([
        ["A-001","WH-BLR-01","Conveyor","Critical",.72,2,"Operational"],
        ["A-002","WH-BLR-01","Generator","High",.84,1,"Operational"],
        ["A-003","WH-BLR-02","Sorter","Critical",.42,4,"Warning"],
        ["A-004","WH-MUM-01","Conveyor","Critical",.78,1,"Operational"],
        ["A-005","WH-DEL-01","Sorter","Critical",.35,5,"Warning"],
        ["A-006","WH-DEL-01","HVAC","High",.67,2,"Operational"],
        ["A-007","WH-PUN-01","Dock Door","Medium",.48,3,"Warning"],
        ["A-008","WH-CHN-01","Conveyor","High",.74,1,"Operational"],
    ], columns=["asset_id","warehouse_id","asset_type","criticality","health","failures_90d","status"])

    maintenance = pd.DataFrame([
        ["WO-001","WH-BLR-02","Sorter vibration inspection","Critical","Open","2026-08-18"],
        ["WO-002","WH-DEL-01","Sorter belt replacement","Critical","Open","2026-08-19"],
        ["WO-003","WH-PUN-01","Dock door service","Medium","Open","2026-08-22"],
        ["WO-004","WH-BLR-01","Conveyor preventive maintenance","High","Scheduled","2026-08-20"],
        ["WO-005","WH-DEL-01","HVAC inspection","High","Open","2026-08-21"],
    ], columns=["work_order_id","warehouse_id","description","severity","status","due_date"])

    logistics = pd.DataFrame([
        ["WH-BLR-01",1200,97,96,2,.18],["WH-BLR-02",2300,99,91,4,.31],
        ["WH-MUM-01",1450,96,95,1,.14],["WH-DEL-01",3100,94,88,6,.37],
        ["WH-HYD-01",650,98,97,1,.09],["WH-PUN-01",1100,95,93,3,.22],
        ["WH-CHN-01",980,97,96,1,.12],["WH-KOL-01",720,98,95,1,.10],
    ], columns=["warehouse_id","backlog","dispatch_sla","delivery_sla","vehicles_unavailable","external_risk"])
    return warehouses, leases, assets, maintenance, logistics

@st.cache_data
def read_csv(name):
    p = DATA / name
    return pd.read_csv(p) if p.exists() else None

def load_data():
    required = ["warehouses.csv","leases.csv","assets.csv","maintenance.csv","logistics.csv"]
    loaded = {x[:-4]: read_csv(x) for x in required}
    missing = [x for x in required if loaded[x[:-4]] is None]
    if missing:
        st.sidebar.info("Demo data is active. Missing: " + ", ".join(missing))
        return dict(zip(["warehouses","leases","assets","maintenance","logistics"], demo_data()))
    return loaded

def calculate_risk(d):
    w=d["warehouses"].copy(); a=d["assets"].copy(); m=d["maintenance"].copy()
    l=d["leases"].copy(); g=d["logistics"].copy()
    l["expiry_date"]=pd.to_datetime(l["expiry_date"], errors="coerce")
    today=pd.Timestamp.today().normalize()
    l["notice_date"]=l["expiry_date"]-pd.to_timedelta(l["notice_days"],unit="D")
    l["days_to_notice"]=(l["notice_date"]-today).dt.days
    ar=a.groupby("warehouse_id").apply(lambda x: ((1-x.health.mean())*70+x.failures_90d.sum()*5).clip(0,100), include_groups=False).reset_index(name="asset_risk")
    mr=m[m.status.isin(["Open","Overdue"])].groupby("warehouse_id").size().reset_index(name="open_maintenance")
    g["logistics_risk"]=((100-g.dispatch_sla)*.7+(100-g.delivery_sla)*.4+g.vehicles_unavailable*2+g.external_risk*25).clip(0,100)
    r=w.merge(ar,on="warehouse_id",how="left").merge(mr,on="warehouse_id",how="left").merge(g,on="warehouse_id",how="left")
    r["asset_risk"]=r.asset_risk.fillna(0); r["open_maintenance"]=r.open_maintenance.fillna(0)
    r["capacity_risk"]=((r.utilisation-.8)/.2*100).clip(0,100)
    r["maintenance_risk"]=(r.asset_risk*.65+r.open_maintenance.clip(0,5)*7).clip(0,100)
    r["lease_risk"]=0.0
    for _,x in l.iterrows():
        days=x.days_to_notice
        score=100 if days<0 else 85 if days<=60 else 65 if days<=180 else 30 if days<=365 else 10
        r.loc[r.warehouse_id==x.warehouse_id,"lease_risk"]=score
    r["overall_risk"]=(r.capacity_risk*.22+r.maintenance_risk*.28+r.lease_risk*.20+r.logistics_risk*.30).clip(0,100)
    r["severity"]=pd.cut(r.overall_risk,[-1,35,60,75,101],labels=["Low","Medium","High","Critical"])
    return r,l

def metric(label,value,delta=None):
    st.metric(label,value,delta)

d=load_data()
risk,lease=calculate_risk(d)

st.title("🏭 AI Warehouse Operations Control Tower")
st.caption("MVP v0.1.1 — dashboard-first prototype. Risk scores are illustrative and configurable.")

with st.sidebar:
    st.header("Navigation")
    page=st.radio("Go to",["Executive Control Tower","Warehouse 360","Lease Intelligence","Maintenance Intelligence","Logistics Risk","Data & Configuration"])
    st.divider()
    st.success("Dashboard: Live")
    st.info("AI/RAG: Next phase")

if page=="Executive Control Tower":
    st.subheader("Today's operational picture")
    c=st.columns(5)
    c[0].metric("Warehouses",len(risk))
    c[1].metric("Critical risks",int((risk.severity=="Critical").sum()))
    c[2].metric("High risks",int((risk.severity=="High").sum()))
    c[3].metric("Open maintenance",int(d.maintenance.status.isin(["Open","Overdue"]).sum()))
    c[4].metric("Lease actions ≤180d",int((lease.days_to_notice<=180).sum()))
    st.subheader("Priority queue")
    q=risk.sort_values("overall_risk",ascending=False).copy()
    q["Risk Score"]=q.overall_risk.round(0).astype(int)
    q["Reason"]=np.where(q.capacity_risk>60,"Capacity pressure; ","")+np.where(q.maintenance_risk>60,"Maintenance risk; ","")+np.where(q.lease_risk>60,"Lease deadline; ","")+np.where(q.logistics_risk>45,"Logistics risk; ","")
    st.dataframe(q[["warehouse_id","warehouse_name","region","severity","Risk Score","Reason","utilisation","backlog"]],use_container_width=True,hide_index=True)
    st.subheader("Risk distribution")
    st.bar_chart(risk.severity.value_counts().reindex(["Critical","High","Medium","Low"]).fillna(0))

elif page=="Warehouse 360":
    st.subheader("Warehouse 360")
    name=st.selectbox("Warehouse",risk.warehouse_name.tolist())
    x=risk[risk.warehouse_name==name].iloc[0]
    c=st.columns(4)
    c[0].metric("Overall risk",f"{x.overall_risk:.0f}/100",str(x.severity))
    c[1].metric("Utilisation",f"{x.utilisation*100:.0f}%")
    c[2].metric("Backlog",f"{int(x.backlog):,}")
    c[3].metric("Delivery SLA",f"{x.delivery_sla:.1f}%")
    st.subheader("Why is this warehouse at risk?")
    reasons=[]
    if x.capacity_risk>35: reasons.append(f"Capacity pressure: {x.utilisation*100:.0f}% utilisation.")
    if x.maintenance_risk>35: reasons.append(f"Maintenance/asset risk: {int(x.open_maintenance)} open work orders.")
    if x.lease_risk>35: reasons.append("Lease deadline requires attention.")
    if x.logistics_risk>35: reasons.append(f"Logistics risk: backlog {int(x.backlog):,}.")
    for reason in reasons or ["No major rule-based risk signal detected."]: st.warning(reason)
    st.subheader("Maintenance")
    st.dataframe(d.maintenance[d.maintenance.warehouse_id==x.warehouse_id],use_container_width=True,hide_index=True)
    st.subheader("Lease")
    st.dataframe(lease[lease.warehouse_id==x.warehouse_id],use_container_width=True,hide_index=True)

elif page=="Lease Intelligence":
    st.subheader("Lease Intelligence")
    l=lease.copy()
    l["Priority"]=pd.cut(l.days_to_notice,[-99999,0,60,180,365,99999],labels=["Overdue","Critical","High","Watch","Later"])
    c=st.columns(3)
    c[0].metric("Overdue",int((l.days_to_notice<0).sum()))
    c[1].metric("Notice ≤60d",int(l.days_to_notice.between(0,60).sum()))
    c[2].metric("Notice ≤180d",int(l.days_to_notice.between(0,180).sum()))
    st.dataframe(l.sort_values("days_to_notice"),use_container_width=True,hide_index=True)
    st.info("Next: upload lease PDFs and add clause extraction + RAG.")

elif page=="Maintenance Intelligence":
    st.subheader("Maintenance Intelligence")
    a=d.assets.copy(); a["risk"]=((1-a.health)*70+a.failures_90d*5).clip(0,100)
    c=st.columns(3); c[0].metric("Assets",len(a)); c[1].metric("Warning assets",int((a.status=="Warning").sum())); c[2].metric("Critical assets",int((a.criticality=="Critical").sum()))
    st.dataframe(a.sort_values("risk",ascending=False),use_container_width=True,hide_index=True)
    st.subheader("Open work orders")
    st.dataframe(d.maintenance[d.maintenance.status.isin(["Open","Overdue"])],use_container_width=True,hide_index=True)
    st.info("Next: predictive failure risk using historical maintenance + telemetry.")

elif page=="Logistics Risk":
    st.subheader("Last-mile & logistics risk")
    g=d.logistics.copy()
    g["risk_score"]=((100-g.dispatch_sla)*.7+(100-g.delivery_sla)*.4+g.vehicles_unavailable*2+g.external_risk*25).clip(0,100)
    st.dataframe(g.sort_values("risk_score",ascending=False),use_container_width=True,hide_index=True)
    st.bar_chart(g.set_index("warehouse_id")[["dispatch_sla","delivery_sla"]])

else:
    st.subheader("Data & Configuration")
    st.write("The app can run with built-in demo data. To use real data, add the matching CSV files under the `data/` folder.")
    for key,df in d.items():
        with st.expander(key.title()):
            st.dataframe(df.head(20),use_container_width=True,hide_index=True)
    st.code("""overall_risk =
22% capacity + 28% maintenance + 20% lease + 30% logistics""")
    st.warning("Do not treat these weights as production decision logic until calibrated with real operational outcomes.")

st.divider()
st.caption("AI Warehouse Operations Control Tower • MVP v0.1.1")
