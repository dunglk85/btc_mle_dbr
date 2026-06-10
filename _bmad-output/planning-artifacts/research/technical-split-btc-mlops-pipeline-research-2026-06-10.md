---
stepsCompleted: [1]
inputDocuments: []
workflowType: 'research'
lastStep: 1
research_type: 'technical'
research_topic: 'Split BTC MLOps Pipeline into Inference Job and Training Job'
research_goals: 'Evaluate architecture for decoupling hourly inference from on-demand training, triggered by drift detection or manual request'
user_name: 'dunglk85'
date: '2026-06-10'
web_research_enabled: true
source_verification: true
---

# Research Report: technical

**Date:** 2026-06-10
**Author:** dunglk85
**Research Type:** technical

---

## Technical Research Scope Confirmation

**Research Topic:** Split BTC MLOps Pipeline into Inference Job and Training Job
**Research Goals:** Evaluate architecture for decoupling hourly inference from on-demand training, triggered by drift detection or manual request

**Technical Research Scope:**

- Architecture Analysis - design patterns, frameworks, system architecture
- Implementation Approaches - development methodologies, coding patterns
- Technology Stack - languages, frameworks, tools, platforms
- Integration Patterns - APIs, protocols, interoperability
- Performance Considerations - scalability, optimization, patterns

**Research Methodology:**

- Current web data with rigorous source verification
- Multi-source validation for critical technical claims
- Confidence level framework for uncertain information
- Comprehensive technical coverage with architecture-specific insights

**Scope Confirmed:** 2026-06-10

---

## Technology Stack Analysis

### Programming Languages

The pipeline uses **Python** for all notebooks (PySpark, scikit-learn, LightGBM, XGBoost) and **SQL** for Databricks alerts. This is standard for Databricks MLOps.

### Development Frameworks and Libraries

- **PySpark**: Feature engineering, data manipulation
- **MLflow**: Model registry, experiment tracking
- **Optuna**: Hyperparameter optimization
- **LightGBM/XGBoost/RandomForest**: Prediction models
- **scikit-learn**: Feature selection (mutual info, correlation)

### Databricks Platform Capabilities

Key Databricks features relevant to multi-job architecture:

- **Job Dependencies**: Jobs can trigger other jobs via `dbutils.jobs.taskValues` or REST API
- **Databricks Alerts**: SQL alerts can trigger webhooks or notebooks
- **Delta Lake**: Shared tables between jobs (no data transfer needed)
- **Unity Catalog**: Cross-job data governance

### Cloud Infrastructure and Deployment

The pipeline runs on Databricks with:
- **Databricks Jobs**: Scheduled and on-demand execution
- **Databricks SQL Warehouses**: Alert evaluation
- **MLflow Model Registry**: Champion/challenger management
- **Git Integration**: Notebook version control

---

## Architecture Analysis: Two-Job Pipeline Design

### Current Problem

Training on 12,000+ rows every hour is wasteful. The model doesn't need hourly retraining — BTC price patterns change slowly.

### Proposed Architecture

| Job 1: Inference (hourly) | Job 2: Training (on-demand) |
|---|---|
| `data_ingestion → feature_engineering → prediction → monitoring` | `training → champion_challenger` |
| Schedule: `0 0 * * * ?` (every hour) | Trigger: manual OR drift alert |
| Low compute (~5 min) | High compute (~15-30 min) |
| Always running | Only when needed |

### Benefits

1. **Cost reduction**: 95% less compute (no hourly training)
2. **Faster inference**: No training bottleneck
3. **Clean separation**: Inference is predictable, training is experimental
4. **Better monitoring**: Drift detection drives retraining decisions

---

## Integration Patterns: How Job 2 Gets Triggered

### Option 1: Databricks Alerts → Webhook → Job 2 API

**How it works:**
1. Monitoring notebook writes drift metrics to `pipeline_metrics` table
2. Databricks SQL Alert detects drift threshold exceeded
3. Alert webhook calls Databricks Jobs API to start Job 2

**Pros:**
- Fully automated
- Native Databricks feature

**Cons:**
- Requires webhook endpoint (Databricks doesn't support direct job trigger from alerts)
- Needs external service or Databricks SQL Alert → webhook → job API chain

**Implementation:**
```sql
-- Alert query
SELECT COUNT(*) AS drift_count
FROM btc_simply.monitoring.pipeline_metrics
WHERE metric_time >= current_timestamp() - INTERVAL 2 HOURS
  AND status = 'alert'
  AND metric_name RLIKE '^(data_drift|label_drift|prediction_drift)_'
```

Then use Databricks REST API to trigger Job 2:
```bash
curl -X POST https://<databricks-host>/api/2.1/jobs/run-now \
  -H "Authorization: Bearer <token>" \
  -d '{"job_id": <job2_id>}'
```

### Option 2: Monitoring Notebook Calls Job 2 Directly

**How it works:**
1. Job 1 monitoring notebook checks drift metrics
2. If drift exceeds threshold, call `dbutils.notebook.run()` on training notebooks
3. Or call Databricks Jobs API from within the notebook

**Pros:**
- Simple, no external services
- All logic in one place

**Cons:**
- Job 1 runtime increases if training is triggered
- Error handling is more complex

**Implementation:**
```python
# In 06_monitoring.py
drift_alert_count = spark.sql("""
    SELECT COUNT(*) AS cnt FROM ... WHERE status = 'alert'
""").collect()[0]["cnt"]

if drift_alert_count > 0:
    print("DRIFT_DETECTED: Triggering training job")
    # Option A: Run training notebooks inline
    dbutils.notebook.run("notebooks/03_optuna_training", timeout_seconds=1800, arguments={"catalog": catalog, "model_algo": "lightgbm"})
    
    # Option B: Trigger Job 2 via API
    import requests
    requests.post(f"{dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()}/api/2.1/jobs/run-now",
                  json={"job_id": JOB2_ID},
                  headers={"Authorization": f"Bearer {dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()}"})
```

### Option 3: Separate Monitoring Job Triggers Job 2

**How it works:**
1. Job 1: `data_ingestion → feature_engineering → prediction` (no monitoring)
2. Job 3 (hourly): `monitoring` → triggers Job 2 if drift detected
3. Job 2: `training → champion_challenger` (on-demand)

**Pros:**
- Monitoring doesn't slow down inference
- Clean separation

**Cons:**
- More jobs to manage
- More complex scheduling

---

## Recommendation: Option 2 (Monitoring Notebook Triggers Training)

**Why Option 2:**
1. **Simplest implementation**: No external webhooks or additional jobs
2. **Immediate response**: Training starts right after drift detection
3. **No extra cost**: Uses existing Job 1 runtime
4. **Easy to debug**: All logic in one place

**Implementation plan:**

1. Add `trigger_training_on_drift` parameter to `06_monitoring.py` (default: `true`)
2. Add drift threshold logic: if `drift_alert_count >= 2` (not just 1), trigger training
3. Call training notebooks inline via `dbutils.notebook.run()`
4. Job 2 becomes optional — can still be run manually for scheduled retraining

**Job 1 (hourly):**
```yaml
- task_key: data_ingestion
- task_key: feature_engineering
- task_key: prediction
- task_key: monitoring
  base_parameters:
    trigger_training_on_drift: "true"
```

**Job 2 (manual/scheduled weekly):**
```yaml
- task_key: model_training_reg_lgbm
- task_key: model_training_reg_xgb
- task_key: model_training_reg_rf
- task_key: champion_challenger
```

---

## Performance Considerations

### Cost Comparison

| Scenario | Current (hourly training) | Proposed (on-demand training) |
|---|---|---|
| Compute per hour | ~20 min | ~5 min |
| Training runs/day | 24 | 0-2 |
| Cost reduction | - | ~80-90% |

### Drift Threshold Recommendations

| Metric | Warning | Alert (trigger training) |
|---|---|---|
| PSI (Population Stability Index) | 0.1-0.2 | > 0.2 |
| KS Statistic | 0.1-0.2 | > 0.2 |
| Prediction error (MAPE) | > 3% | > 5% |

### Training Frequency Guidelines

- **BTC hourly data**: Train weekly or when drift detected
- **High volatility periods**: May need daily training
- **Stable periods**: Can go 2-4 weeks without retraining

---

## Implementation Steps

1. **Split jobs.yml** into two job definitions
2. **Modify 06_monitoring.py** to trigger training on drift
3. **Add drift threshold parameters** to monitoring notebook
4. **Update alert queries** to reflect new architecture
5. **Test both jobs independently**
6. **Set up manual training schedule** (e.g., weekly Sunday 2AM)

---

## Conclusion

Splitting the pipeline into 2 jobs is **recommended** for cost efficiency and operational clarity. The monitoring notebook should trigger training when drift exceeds thresholds, with manual training as a fallback.

**Next steps:**
1. Implement the split in `jobs.yml`
2. Add drift-triggered training logic to `06_monitoring.py`
3. Test with historical data to validate drift thresholds
