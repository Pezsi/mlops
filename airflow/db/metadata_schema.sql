-- MLOps Metadata Database Schema
-- Tracks all aspects of the ML pipeline: models, experiments, data, lineage

-- Model Training Runs table
CREATE TABLE IF NOT EXISTS model_runs (
    run_id VARCHAR(255) PRIMARY KEY,
    dag_id VARCHAR(255) NOT NULL,
    task_id VARCHAR(255) NOT NULL,
    execution_date TIMESTAMP NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50),
    model_type VARCHAR(100),
    framework VARCHAR(50),
    training_status VARCHAR(50) NOT NULL,  -- started, completed, failed
    training_duration_seconds FLOAT,
    training_start_time TIMESTAMP,
    training_end_time TIMESTAMP,
    deployed BOOLEAN DEFAULT FALSE,
    deployment_stage VARCHAR(50),  -- staging, production, archived
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) REFERENCES model_runs(run_id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_type VARCHAR(50),  -- training, validation, test
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_metrics_run_id ON model_metrics(run_id);
CREATE INDEX idx_metrics_name ON model_metrics(metric_name);

-- Model Parameters/Hyperparameters table
CREATE TABLE IF NOT EXISTS model_parameters (
    param_id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) REFERENCES model_runs(run_id) ON DELETE CASCADE,
    param_name VARCHAR(100) NOT NULL,
    param_value TEXT NOT NULL,
    param_type VARCHAR(50),  -- hyperparameter, config, system
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_params_run_id ON model_parameters(run_id);

-- Dataset Versions table
CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(255) NOT NULL,
    dataset_version VARCHAR(100) NOT NULL,
    dataset_path TEXT NOT NULL,
    dataset_size_bytes BIGINT,
    row_count INTEGER,
    column_count INTEGER,
    dataset_hash VARCHAR(255),  -- MD5 or SHA256 hash
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_name, dataset_version)
);

-- Data Lineage table (which data was used for which model)
CREATE TABLE IF NOT EXISTS data_lineage (
    lineage_id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) REFERENCES model_runs(run_id) ON DELETE CASCADE,
    dataset_id INTEGER REFERENCES dataset_versions(dataset_id) ON DELETE CASCADE,
    usage_type VARCHAR(50),  -- train, validation, test
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_lineage_run_id ON data_lineage(run_id);
CREATE INDEX idx_lineage_dataset_id ON data_lineage(dataset_id);

-- Model Artifacts table
CREATE TABLE IF NOT EXISTS model_artifacts (
    artifact_id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) REFERENCES model_runs(run_id) ON DELETE CASCADE,
    artifact_name VARCHAR(255) NOT NULL,
    artifact_type VARCHAR(100),  -- model_file, scaler, encoder, plot, report
    artifact_path TEXT NOT NULL,
    artifact_size_bytes BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_artifacts_run_id ON model_artifacts(run_id);

-- Model Comparison Results table
CREATE TABLE IF NOT EXISTS model_comparisons (
    comparison_id SERIAL PRIMARY KEY,
    dag_run_id VARCHAR(255),
    new_run_id VARCHAR(255) REFERENCES model_runs(run_id),
    baseline_run_id VARCHAR(255) REFERENCES model_runs(run_id),
    comparison_result VARCHAR(50),  -- better, worse, equal
    metric_compared VARCHAR(100),
    new_metric_value FLOAT,
    baseline_metric_value FLOAT,
    improvement_percentage FLOAT,
    deployed BOOLEAN DEFAULT FALSE,
    comparison_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_comparison_new_run ON model_comparisons(new_run_id);
CREATE INDEX idx_comparison_baseline ON model_comparisons(baseline_run_id);

-- Data Drift Detection table
CREATE TABLE IF NOT EXISTS data_drift_events (
    drift_id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES dataset_versions(dataset_id),
    feature_name VARCHAR(255),
    drift_metric VARCHAR(100),  -- KS_test, PSI, chi_square
    drift_score FLOAT,
    drift_threshold FLOAT,
    drift_detected BOOLEAN,
    reference_period_start TIMESTAMP,
    reference_period_end TIMESTAMP,
    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_drift_dataset_id ON data_drift_events(dataset_id);
CREATE INDEX idx_drift_detected ON data_drift_events(drift_detected);

-- Pipeline Events/Logs table
CREATE TABLE IF NOT EXISTS pipeline_events (
    event_id SERIAL PRIMARY KEY,
    dag_id VARCHAR(255) NOT NULL,
    dag_run_id VARCHAR(255),
    task_id VARCHAR(255),
    event_type VARCHAR(100) NOT NULL,  -- info, warning, error, critical
    event_message TEXT NOT NULL,
    event_data JSONB,  -- Additional structured data
    notification_sent BOOLEAN DEFAULT FALSE,
    notification_channel VARCHAR(100),  -- email, slack, webhook
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_events_dag_id ON pipeline_events(dag_id);
CREATE INDEX idx_events_type ON pipeline_events(event_type);
CREATE INDEX idx_events_created ON pipeline_events(created_at DESC);

-- Notification History table
CREATE TABLE IF NOT EXISTS notification_history (
    notification_id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES pipeline_events(event_id),
    channel VARCHAR(100) NOT NULL,  -- email, slack, webhook
    recipient VARCHAR(255) NOT NULL,
    subject VARCHAR(500),
    message TEXT NOT NULL,
    status VARCHAR(50) NOT NULL,  -- sent, failed, pending
    retry_count INTEGER DEFAULT 0,
    sent_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_notification_status ON notification_history(status);
CREATE INDEX idx_notification_channel ON notification_history(channel);

-- Model Performance Over Time (for monitoring)
CREATE TABLE IF NOT EXISTS model_monitoring (
    monitoring_id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) REFERENCES model_runs(run_id),
    monitoring_timestamp TIMESTAMP NOT NULL,
    prediction_count INTEGER,
    average_prediction_time_ms FLOAT,
    error_count INTEGER,
    metrics JSONB,  -- Dynamic metrics storage
    alerts_triggered JSONB,  -- Array of alerts
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_monitoring_run_id ON model_monitoring(run_id);
CREATE INDEX idx_monitoring_timestamp ON model_monitoring(monitoring_timestamp DESC);

-- Feature Statistics table (for feature store)
CREATE TABLE IF NOT EXISTS feature_statistics (
    stat_id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES dataset_versions(dataset_id),
    feature_name VARCHAR(255) NOT NULL,
    feature_type VARCHAR(50),  -- numerical, categorical
    mean_value FLOAT,
    median_value FLOAT,
    std_dev FLOAT,
    min_value FLOAT,
    max_value FLOAT,
    null_count INTEGER,
    unique_count INTEGER,
    statistics_json JSONB,  -- Additional stats
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feature_stats_dataset ON feature_statistics(dataset_id);
CREATE INDEX idx_feature_stats_name ON feature_statistics(feature_name);

-- AB Test Experiments table
CREATE TABLE IF NOT EXISTS ab_experiments (
    experiment_id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255) NOT NULL,
    control_run_id VARCHAR(255) REFERENCES model_runs(run_id),
    treatment_run_id VARCHAR(255) REFERENCES model_runs(run_id),
    traffic_split FLOAT,  -- Percentage to treatment
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    status VARCHAR(50),  -- running, completed, stopped
    winner VARCHAR(50),  -- control, treatment, inconclusive
    results JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Views for common queries

-- View: Latest Production Models
CREATE OR REPLACE VIEW latest_production_models AS
SELECT
    mr.run_id,
    mr.model_name,
    mr.model_version,
    mr.deployment_stage,
    mr.training_end_time,
    COALESCE(AVG(CASE WHEN mm.metric_name = 'r2_score' THEN mm.metric_value END), 0) as r2_score,
    COALESCE(AVG(CASE WHEN mm.metric_name = 'rmse' THEN mm.metric_value END), 0) as rmse,
    COALESCE(AVG(CASE WHEN mm.metric_name = 'mae' THEN mm.metric_value END), 0) as mae
FROM model_runs mr
LEFT JOIN model_metrics mm ON mr.run_id = mm.run_id
WHERE mr.deployment_stage = 'production'
    AND mr.training_status = 'completed'
GROUP BY mr.run_id, mr.model_name, mr.model_version, mr.deployment_stage, mr.training_end_time
ORDER BY mr.training_end_time DESC;

-- View: Recent Pipeline Events
CREATE OR REPLACE VIEW recent_pipeline_events AS
SELECT
    pe.event_id,
    pe.dag_id,
    pe.task_id,
    pe.event_type,
    pe.event_message,
    pe.notification_sent,
    pe.created_at,
    COUNT(nh.notification_id) as notification_count
FROM pipeline_events pe
LEFT JOIN notification_history nh ON pe.event_id = nh.event_id
WHERE pe.created_at > NOW() - INTERVAL '7 days'
GROUP BY pe.event_id, pe.dag_id, pe.task_id, pe.event_type, pe.event_message, pe.notification_sent, pe.created_at
ORDER BY pe.created_at DESC;

-- View: Model Performance Comparison
CREATE OR REPLACE VIEW model_performance_comparison AS
SELECT
    mr.model_name,
    mr.run_id,
    mr.model_version,
    mr.deployment_stage,
    mr.training_end_time,
    mm.metric_name,
    mm.metric_value,
    RANK() OVER (PARTITION BY mr.model_name, mm.metric_name ORDER BY mm.metric_value DESC) as performance_rank
FROM model_runs mr
JOIN model_metrics mm ON mr.run_id = mm.run_id
WHERE mr.training_status = 'completed'
    AND mm.metric_name IN ('r2_score', 'rmse', 'mae', 'accuracy')
ORDER BY mr.model_name, mm.metric_name, performance_rank;
