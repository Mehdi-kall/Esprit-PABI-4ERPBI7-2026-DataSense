# ML Automation & ETL Pipeline (n8n)

## Overview

This project implements an end-to-end Machine Learning automation pipeline integrated with an ETL workflow. It is designed to automate data ingestion, execute multiple ML models, store predictions, trigger business alerts, and ensure robustness through monitoring and error handling.

---

## Architecture

```
Trigger → ETL → Validation → DWH Sync → ML Models → Processing → Merge → Decision → Alerting
```

**Key principles:**
- Modular and scalable design
- Fully automated execution with no manual intervention
- Clear separation of concerns across pipeline stages

---

## Pipeline Stages

### 1. Trigger
- Scheduled daily execution at 6 AM via Cron
- Ensures consistent, automated pipeline runs

### 2. ETL Layer
- Executes ETL process via API
- Normalizes ETL status
- Detects and handles failures before downstream execution

### 3. Error Handling
- If ETL fails: alert email is sent and failure is logged
- All execution steps are traceable

### 4. Data Warehouse Sync
- Wait mechanism ensures data availability before ML inference begins

### 5. ML Models
Each model follows the same pattern: **Query → API → Format → Upsert**

| Model | Type |
|---|---|
| Classification | Supervised |
| Regression | Supervised |
| Time Series | Sequential |
| Clustering | Unsupervised |

Each model is independently executed and fully integrated via API.

### 6. Data Processing
JavaScript nodes handle cleaning, structuring, and standardizing outputs from each model.

### 7. Merge Layer
All model outputs are combined into a unified dataset for downstream decision logic.

### 8. Business Logic
```javascript
if (CA > threshold) {
  return "High Performance";
} else {
  return "Low Performance — Alert";
}
```

### 9. Alerting
- High performance → success notification
- Low performance → alert email
- ETL failure → critical alert

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | n8n |
| ETL | REST API / Data Pipeline |
| ML Models | Python APIs (FastAPI / Flask) |
| Database | SQL / Data Warehouse |
| Scripting | JavaScript |
| Alerts | Email API |

### n8n Nodes Used
- `HTTP Request` — API calls
- `IF` — conditional branching
- `Merge` — output aggregation
- `Code` — JavaScript processing
- `Cron` — scheduling
- `Wait` — sync delays
- `Email` — notifications

---

## Monitoring & Robustness

- ETL failure detection with immediate alerting
- API failure tracking
- Email notifications for all critical events
- Fully traceable execution logs
- Retry logic extendable per node

---

## BI Integration

The pipeline output can be connected to:
- Power BI
- SQL-based analyzers
- Custom dashboard tools

Supports KPI monitoring and performance analysis.

---

## Project Structure

```
/workflow
   ├── ETL Execution
   ├── ML Models
   ├── Data Processing
   ├── Merge Logic
   └── Alert System

/scripts
   ├── JS Processing
   └── Business Logic

/docs
   ├── Architecture
   └── Workflow JSON
```

---

## Export

The workflow can be exported as:
- n8n JSON workflow file
- README documentation
- Architecture diagrams

---

## Future Improvements

- Real-time monitoring dashboard
- AutoML integration
- Slack / WhatsApp alerts
- Enhanced retraining pipeline
- Anomaly detection module

---


