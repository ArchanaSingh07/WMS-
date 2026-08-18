# AI Warehouse Operations Control Tower — MVP v0.1

A free-to-deploy Streamlit MVP for a last-mile logistics company managing distributed warehouses.

## What is included

- Executive Control Tower
- Warehouse 360
- Lease Intelligence
- Maintenance Intelligence
- Logistics Risk
- CSV upload for each data domain
- Rule-based risk engine
- Sample data
- Clear extension points for RAG, AI briefing, evaluation and controlled actions

## Important

This is a decision-support prototype. The risk formula is illustrative and must be calibrated against real operational outcomes before production use.

## Deploy without local coding

### Option A — Streamlit Community Cloud

1. Create a GitHub account.
2. Create a new repository, for example `ai-warehouse-control-tower`.
3. Upload `app.py`, `requirements.txt`, and the `data` folder.
4. Go to Streamlit Community Cloud and create an app.
5. Select your GitHub repository and `app.py`.
6. Deploy.

### Option B — GitHub Codespaces

Open the repository in Codespaces and run the Streamlit app from the browser. This avoids installing Python locally.

## Next build stages

1. Replace sample CSVs with real data.
2. Add lease PDF ingestion and RAG.
3. Add AI daily briefing.
4. Add evaluation and red-team tests.
5. Add authenticated roles.
6. Add controlled action workflows.
7. Add predictive maintenance/capacity models.
