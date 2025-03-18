Here's how I would modify the `cal_non_gt_metrics_spark` function to accept pre-calculated histograms and avoid redundant calculations:

```python
def cal_non_gt_metrics_spark(self, curr_data, bl_data, fields, pre_calculated_histograms=None):
    """
    Calculate non-ground-truth metrics using Spark.
    
    Args:
        curr_data: Current data frame
        bl_data: Baseline data frame
        fields: Dictionary of fields and their metrics
        pre_calculated_histograms: Dictionary of pre-calculated histograms {field: (hist_curr, hist_bl)}
            
    Returns:
        Dictionary of results
    """
    pre_calculated_histograms = pre_calculated_histograms or {}
    results = {}
    
    for field, metrics in fields.items():
        result = {}
        field_name = field.replace("-", "_")
        field_type = self.config["fields"][field]["type"]
        
        # Data quality validation
        st = time.time()
        r_nm, r_nv, df_filtered = validate_data_quality_spark(
            curr_data.select(field_name),
            field_name,
            field_type,
            self.config["fields"][field]["value_range"]
        )
        result["t_quality"] = time.time() - st
        result["missing_value"] = r_nm
        result["range_violation"] = r_nv
        
        if df_filtered.count() < MIN_SAMPLE_SIZE:
            result["tags"] = [Tag_NOT_ENOUGH_DATA]
            results[field] = result
            continue
        
        # Check if this field needs histogram metrics
        require_hist = False
        for metric in metrics:
            if metric in MetricGroups.HIST_METRICS:
                require_hist = True
                break
        
        hist_curr = None
        hist_bl = None
        
        # Handle histogram calculation or retrieval
        if require_hist:
            st = time.time()
            
            # Check if we have pre-calculated histograms for this field
            if field in pre_calculated_histograms:
                hist_curr, hist_bl = pre_calculated_histograms[field]
                result["histogram_source"] = "pre_calculated"
            else:
                # Calculate histograms from scratch
                bins = MetricsSpark.hist_bins(bl_data, field_name)
                hist_curr = MetricsSpark.hist_density(curr_data, field_name, bins)
                hist_bl = MetricsSpark.hist_density(bl_data, field_name, bins)
                result["histogram_source"] = "calculated"
            
            result["t_histogram"] = time.time() - st
        
        # Calculate each metric
        for metric in metrics:
            func = self.metric_funcs["spark"][metric]
            st = time.time()
            
            try:
                if metric in MetricGroups.HIST_METRICS:
                    # Use histograms for histogram-based metrics
                    val = func(hist_curr, hist_bl)
                else:
                    # Use raw data for other metrics
                    val = func(
                        curr_data.select(field_name),
                        field_name,
                        bl_data
                    )
                
                t = time.time() - st
                result[metric] = val
                result["t_" + metric] = t
            except Exception as e:
                result[f"error_{metric}"] = str(e)
                logging.error(f"Error calculating {metric} for {field}: {e}")
        
        results[field] = result
    
    return results

```

Now, I'll modify the relevant part of the `process_task` function to use this updated method when histograms are available from cache:

```python
# In the process_task function, replace the direct calculations with:

# Process each field
results = {}
pre_calculated_histograms = {}

# Prepare pre-calculated histograms dictionary
for field in hist_requiring_fields:
    if field in hist_data:
        # We need to create the current histogram since we only have baseline
        field_name = field.replace("-", "_").replace(".", "_")
        # Extract bins from baseline histogram
        bins = MetricsSpark.extract_bins(hist_data[field])
        # Calculate current histogram using same bins
        hist_curr = MetricsSpark.hist_density(curr_data, field_name, bins)
        # Store both histograms
        pre_calculated_histograms[field] = (hist_curr, hist_data[field])

# If we have at least some pre-calculated histograms or fields don't need histograms
if pre_calculated_histograms or not hist_requiring_fields:
    # Use the updated method that accepts pre-calculated histograms
    results = analysis.cal_non_gt_metrics_spark(
        curr_data, 
        bl_data, 
        non_gt_fields,
        pre_calculated_histograms=pre_calculated_histograms
    )
else:
    # Fall back to original method if no pre-calculated histograms available
    results = analysis.cal_non_gt_metrics_spark(curr_data, bl_data, non_gt_fields)

```

The key improvements in these changes are:

1. Added a new parameter `pre_calculated_histograms` to `cal_non_gt_metrics_spark` to accept histograms that were already loaded from cache.

2. Modified the function to check if histograms are available for each field before calculating them from scratch.

3. Updated the integration in `process_task` to:
   - Create a dictionary of pre-calculated histograms using the cached baseline histograms
   - Calculate only the current histogram for fields where baseline histogram is already cached
   - Pass this dictionary to the updated function

4. Added tracking of histogram source (pre-calculated vs. calculated) in the results for monitoring and debugging.

These changes ensure that the system properly leverages cached histograms and avoids redundant calculations, maintaining the optimization intent of the original code.
