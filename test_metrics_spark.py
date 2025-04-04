cutoff: 
  start_date: '2024-11-01T00:00:00.000Z'
  end_date: '2025-03-31T23:59:59.999Z'
end_date: null

# Model metadata
model_name: dcpt_p2p
model_version: v0.0.1

# Field mappings for features and scores
field_mapping:
  model_score: 'response.result.score'
  sec_since_add_payee_rgst_in_l: 'response.result.features.sec_since_add_payee_rgst_in_l'
  sec_since_dvc_use_last_ts: 'response.result.features.sec_since_dvc_use_last_ts'
  sec_since_prty_birth_inc_dt: 'response.result.features.sec_since_prty_birth_inc_dt'
  tran_amt: 'response.result.features.tran_amt'

# Outsort configuration
outsort:
  p2p_fraud:
    score_field: model_score 
    outsort_frac: 0.0010
    threshold: 0.56
  p2p_scam:
    score_field: scam_score
    outsort_frac: 0.001
    threshold: 0.4936

# Model path for SHAP analysis
model_path: s3://csca-rmg-vol/models/dcpt/p2p/v6_5/model_p2p_fraud.json

# S3 location for production data
loc: s3
s3:
  bucket: csca-rmg-vol
  column: null
  endpoint: https://wfps01.wellsfargo.net:10443
  filters:
    actn_intnt_txt_contains_setup: "= 1"
    num_commits: "> 0"
  prefix: raw/dcpt/p2p/v6_5/p2p_v65_full_oot_scored/
  start_date: null

# Time periods for analysis
periods:
  # Rolling 7-day baseline
  - cutoff:
      start_date: '2025-04-01T00:00:00.000Z'
      end_date: '2025-04-07T23:59:59.999Z'
    interval: daily
    lag: 0
    loc: s3
    s3:
      bucket: csca-ml-vol
      column: parsed_value
      endpoint: https://wfps01.wellsfargo.net:10443
      filters:
        payload.baseTransactionC.transactionType: "= P2P"
      prefix: raw/sims/response/data/
    type: baseline

  # Current period for comparison
  - cutoff:
      start_date: '2025-04-08T00:00:00.000Z'
      end_date: '2025-04-08T23:59:59.999Z'
    interval: daily
    lag: 0
    loc: s3
    s3:
      bucket: csca-ml-vol
      column: parsed_value
      endpoint: https://wfps01.wellsfargo.net:10443
      filters:
        payload.baseTransactionC.transactionType: "= P2P"
      prefix: raw/sims/response/data/
    type: current

data:
  start_time: '2025-03-27T16:31:26.391329+00:00'
  id: payload.tran_id

# Field configurations
fields:
  - response.result.score:
      metrics:
        - psi_customized
        - shap
      type: continuous
      value_range: [0.0, 1.0, true]
      
  - response.result.features.sec_since_add_payee_rgst_in_l:
      metrics:
        - psi_customized
        - shap
      type: continuous
      value_range: null
      
  - response.result.features.sec_since_dvc_use_last_ts:
      metrics:
        - psi_customized
        - shap
      type: continuous
      value_range: null
      
  - response.result.features.sec_since_prty_birth_inc_dt:
      metrics:
        - psi_customized
        - shap
      type: continuous
      value_range: null

# Metric configurations
metrics:
  psi_customized:
    type: distribution
    n_bins: 10
    include_special: false  # Exclude OOR and null values
    
  shap:
    type: model_explanation
    abs_val: true
    top_pct: 0.0010  # For the top 0.1%, aligning with p2p_fraud outsort
    reference_period:
      start_date: '2024-11-01T00:00:00.000Z'
      end_date: '2025-03-31T23:59:59.999Z'
    confidence_level: 2.0  # For ~95% confidence intervals

# Global threshold from seg_dict
threshold: 0.56
type: continuous
value_range:
  - 0.0
  - 1.0
  - true

window_size:
  daily: daily
---
---
---
    cutoff:
  start_date: '2024-11-01T00:00:00.000Z'
  end_date: '2025-03-31T23:59:59.999Z'
end_date: null

# Model metadata
model_name: dcpt_p2p
model_version: v0.0.1

# Field mappings for features and scores
field_mapping:
  model_score: 'response.result.score'
  sec_since_add_payee_rgst_in_l: 'response.result.features.sec_since_add_payee_rgst_in_l'
  sec_since_dvc_use_last_ts: 'response.result.features.sec_since_dvc_use_last_ts'
  sec_since_prty_birth_inc_dt: 'response.result.features.sec_since_prty_birth_inc_dt'
  tran_amt: 'response.result.features.tran_amt'

# Outsort configuration
outsort:
  p2p_fraud:
    score_field: model_score 
    outsort_frac: 0.0010
    threshold: 0.56
  p2p_scam:
    score_field: scam_score
    outsort_frac: 0.001
    threshold: 0.4936

# Model path for SHAP analysis
model_path: s3://csca-rmg-vol/models/dcpt/p2p/v6_5/model_p2p_fraud.json

# S3 location for production data
loc: s3
s3:
  bucket: csca-rmg-vol
  column: null
  endpoint: https://wfps01.wellsfargo.net:10443
  filters:
    actn_intnt_txt_contains_setup: "= 1"
    num_commits: "> 0"
  prefix: raw/dcpt/p2p/v6_5/p2p_v65_full_oot_scored/
  start_date: null

# Time periods for analysis
periods:
  # Rolling 7-day baseline
  - cutoff:
      start_date: '2025-04-01T00:00:00.000Z'
      end_date: '2025-04-07T23:59:59.999Z'
    interval: daily
    lag: 0
    loc: s3
    s3:
      bucket: csca-ml-vol
      column: parsed_value
      endpoint: https://wfps01.wellsfargo.net:10443
      filters:
        payload.baseTransactionC.transactionType: "= P2P"
      prefix: raw/sims/response/data/
    type: baseline

  # Current period for comparison
  - cutoff:
      start_date: '2025-04-08T00:00:00.000Z'
      end_date: '2025-04-08T23:59:59.999Z'
    interval: daily
    lag: 0
    loc: s3
    s3:
      bucket: csca-ml-vol
      column: parsed_value
      endpoint: https://wfps01.wellsfargo.net:10443
      filters:
        payload.baseTransactionC.transactionType: "= P2P"
      prefix: raw/sims/response/data/
    type: current

data:
  start_time: '2025-03-27T16:31:26.391329+00:00'
  id: payload.tran_id

# Field configurations
fields:
  - response.result.score:
      metrics:
        - psi_customized  # Using the custom PSI with outsort bins
        - shap           # Add SHAP analysis
      type: continuous
      outsort_frac: 0.0010  # P2P fraud outsort fraction
      value_range: [0.0, 1.0, true]
      
  - response.result.features.sec_since_add_payee_rgst_in_l:
      metrics:
        - psi_customized
        - shap
      type: continuous
      outsort_frac: 0.0010
      value_range: null
      
  - response.result.features.sec_since_dvc_use_last_ts:
      metrics:
        - psi_customized
        - shap
      type: continuous
      outsort_frac: 0.0010
      value_range: null
      
  - response.result.features.sec_since_prty_birth_inc_dt:
      metrics:
        - psi_customized
        - shap
      type: continuous
      outsort_frac: 0.0010
      value_range: null

# Metric configurations
metrics:
  psi_customized:
    type: distribution
    n_bins: 10
    include_special: false  # Exclude OOR and null values
    
  shap:
    type: model_explanation
    abs_val: true
    top_pct: 0.0010  # Use outsort_frac from P2P fraud
    reference_period:
      start_date: '2024-11-01T00:00:00.000Z'
      end_date: '2025-03-31T23:59:59.999Z'
    confidence_level: 2.0  # For 95% confidence intervals

# Global threshold from seg_dict
threshold: 0.56
type: continuous
value_range:
  - 0.0
  - 1.0
  - true

window_size:
  daily: daily
