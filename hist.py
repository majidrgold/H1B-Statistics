# histogram_manager.py
import os
import pickle
import time
import json
from datetime import datetime

class HistogramManager:
    def __init__(self, task_dir):
        self.task_dir = task_dir
        self.hist_dir = os.path.join(task_dir, "histograms")
        os.makedirs(self.hist_dir, exist_ok=True)
        self.hist_metadata_file = os.path.join(self.hist_dir, "metadata.json")
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        if os.path.exists(self.hist_metadata_file):
            with open(self.hist_metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        with open(self.hist_metadata_file, 'w') as f:
            json.dump(self.metadata, f)
    
    def get_histogram_path(self, task, field, bl_start, bl_end):
        # Convert timestamps to string format for filename
        start_str = datetime.fromtimestamp(bl_start).strftime("%Y%m%d")
        end_str = datetime.fromtimestamp(bl_end).strftime("%Y%m%d")
        # Replace any potentially problematic characters in field name
        field_safe = field.replace('.', '_').replace('-', '_')
        filename = f"{task}_{field_safe}_{start_str}_{end_str}.pkl"
        return os.path.join(self.hist_dir, filename)
    
    def save_histograms(self, task, field, bl_start, bl_end, hist_curr, hist_bl):
        hist_path = self.get_histogram_path(task, field, bl_start, bl_end)
        with open(hist_path, "wb") as f:
            pickle.dump((hist_curr, hist_bl), f)
        
        # Update metadata
        if task not in self.metadata:
            self.metadata[task] = {}
        
        self.metadata[task][field] = {
            "path": hist_path,
            "bl_start": bl_start,
            "bl_end": bl_end,
            "created": time.time(),
            "last_accessed": time.time()
        }
        self._save_metadata()
        return hist_path
    
    def load_histograms(self, task, field, bl_start, bl_end):
        # Check if we have exact match in metadata
        if (task in self.metadata and 
            field in self.metadata[task] and 
            self.metadata[task][field]["bl_start"] == bl_start and
            self.metadata[task][field]["bl_end"] == bl_end):
            
            hist_path = self.metadata[task][field]["path"]
            if os.path.exists(hist_path):
                try:
                    with open(hist_path, "rb") as f:
                        histograms = pickle.load(f)
                    
                    # Update last accessed timestamp
                    self.metadata[task][field]["last_accessed"] = time.time()
                    self._save_metadata()
                    return histograms, True
                except (pickle.PickleError, EOFError, IOError) as e:
                    # Handle corrupted pickle file
                    print(f"Error loading histogram from {hist_path}: {e}")
                    return None, False
        
        return None, False
    
    def should_update_baseline(self, task, field, bl_config, bl_start, bl_end):
        """Check if baseline histogram needs updating based on config and timestamps"""
        if task not in self.metadata or field not in self.metadata[task]:
            return True
        
        metadata = self.metadata[task][field]
        
        # If baseline period changed
        if metadata["bl_start"] != bl_start or metadata["bl_end"] != bl_end:
            return True
        
        # Check for forced update based on config
        if "update_frequency" in bl_config:
            # If frequency is "never", never update
            if bl_config["update_frequency"] == "never":
                return False
            
            # If frequency is in days
            if isinstance(bl_config["update_frequency"], int):
                days_since_update = (time.time() - metadata["created"]) / (60 * 60 * 24)
                return days_since_update >= bl_config["update_frequency"]
        
        return False
    
    def cleanup_old_histograms(self, max_age_days=30, unused_days=7):
        """Delete old histograms based on creation date or last access time"""
        current_time = time.time()
        tasks_to_delete = []
        
        for task, fields in self.metadata.items():
            fields_to_delete = []
            
            for field, metadata in fields.items():
                age_days = (current_time - metadata["created"]) / (60 * 60 * 24)
                unused_time_days = (current_time - metadata["last_accessed"]) / (60 * 60 * 24)
                
                if age_days > max_age_days or unused_time_days > unused_days:
                    hist_path = metadata["path"]
                    if os.path.exists(hist_path):
                        try:
                            os.remove(hist_path)
                            print(f"Deleted old histogram: {hist_path}")
                        except OSError as e:
                            print(f"Error deleting {hist_path}: {e}")
                    fields_to_delete.append(field)
            
            for field in fields_to_delete:
                del self.metadata[task][field]
            
            if not self.metadata[task]:
                tasks_to_delete.append(task)
        
        for task in tasks_to_delete:
            del self.metadata[task]
        
        self._save_metadata()
        return len(tasks_to_delete)

# analysis.py
import os
import time
import pickle
from metrics_spark import MetricsSpark
from metric_groups import MetricGroups

MIN_SAMPLE_SIZE = 100  # Minimum sample size for valid analysis
Tag_NOT_ENOUGH_DATA = "not_enough_data"

class Analysis:
    def __init__(self, config, task_dir=None):
        self.config = config
        self.task_dir = task_dir
        # Define your metric functions mapping
        self.metric_funcs = {
            "spark": {
                # Your metric function mapping here
                # Example: "jsd": MetricsSpark.jensen_shannon_divergence,
            }
        }
    
    def cal_non_gt_metrics_spark(self, curr_data, bl_data, fields):
        """
        Calculate non-ground-truth metrics using Spark.
        This function is kept for backward compatibility.
        
        Args:
            curr_data: Current data frame
            bl_data: Baseline data frame
            fields: Dictionary of fields and their metrics
            
        Returns:
            Dictionary of results
        """
        results = {}
        for field, metrics in fields.items():
            result = {}
            field_type = self.config["fields"][field]["type"]
            st = time.time()
            r_nm, r_nv, df_filtered = validate_data_quality_spark(
                curr_data.select(field.replace("-", "_")),
                field.replace("-", "_"),
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
            
            require_hist = False
            for metric in metrics:
                if metric in MetricGroups.HIST_METRICS:
                    require_hist = True
                    break
            
            if require_hist:
                st = time.time()
                # Calculate histograms
                bins = MetricsSpark.hist_bins(bl_data, field.replace(".", "_"))
                hist_curr = MetricsSpark.hist_density(curr_data, field.replace(".", "_"), bins)
                hist_bl = MetricsSpark.hist_density(bl_data, field.replace(".", "_"), bins)
                result["t_histogram"] = time.time() - st
            
            for metric in metrics:
                func = self.metric_funcs["spark"][metric]
                st = time.time()
                if metric in MetricGroups.HIST_METRICS:
                    val = func(hist_curr, hist_bl)
                else:
                    val = func(
                        curr_data.select(field.replace(".", "_")),
                        field.replace(".", "_"),
                        bl_data
                    )
                t = time.time() - st
                result[metric] = val
                result["t_" + metric] = t
            
            results[field] = result
        
        return results

# spm_new.py
import os
import time
import logging
from datetime import datetime
from histogram_manager import HistogramManager
from utils import load_yaml_file, get_bl_config, group_fields
from db import DB
from metric_groups import MetricGroups

logger = logging.getLogger(__name__)

def validate_data_quality_spark(df, field_name, field_type, value_range):
    """
    Validate data quality for a specific field
    
    Args:
        df: Spark DataFrame
        field_name: Name of the field to validate
        field_type: Type of the field (int, float, etc.)
        value_range: Expected range for the field values
        
    Returns:
        Tuple of (missing_rate, violation_rate, filtered_df)
    """
    # Implementation depends on your specific requirements
    # This is a placeholder function
    return 0.0, 0.0, df

def extract_monitor_ranges_wo_gt(curr_time, hist_start=None, hist_end=None, window_size="daily"):
    """Extract monitoring time ranges without ground truth"""
    # Implementation depends on your specific requirements
    # This is a placeholder function
    return [curr_time - 86400], [curr_time]  # Default to last 24 hours

def extract_monitor_ranges_w_gt(curr_time, hist_start=None, hist_end=None, window_size="daily", 
                                interval=None, n=None):
    """Extract monitoring time ranges with ground truth"""
    # Implementation depends on your specific requirements
    # This is a placeholder function
    return [curr_time - 86400], [curr_time]  # Default to last 24 hours

def extract_bl_ranges(monitor_st, bl_config):
    """Extract baseline time ranges"""
    # Implementation depends on your specific requirements
    # This is a placeholder function
    bl_start = monitor_st - (86400 * bl_config.get("days", 7))
    return [bl_start], [monitor_st]

class SPM:
    def __init__(self, task_dir, infra_config, curr_time, hist_tasks=None):
        self.task_dir = task_dir
        self.infra_config = infra_config
        self.curr_time = curr_time
        self.hist_tasks = hist_tasks
    
    def process_task(self, task):
        print(f"Processing task: {task}")
        
        # Initialize histogram manager
        hist_manager = HistogramManager(self.task_dir)
        # Periodically clean up old histograms
        num_deleted = hist_manager.cleanup_old_histograms(max_age_days=30, unused_days=7)
        if num_deleted > 0:
            print(f"Cleaned up {num_deleted} old histogram entries")
        
        # Load task config
        config_filepath = os.path.join(self.task_dir, "spm_configs", task + ".yml")
        config = load_yaml_file(config_filepath)
        if not config:
            logger.error(f"Failed to load config file: {config_filepath}")
            return
        
        # Group fields by whether they require ground truth
        gt_fields, non_gt_fields = group_fields(config.get("fields", {}))
        
        # Find fields that require histograms
        hist_requiring_fields = {}
        for field, metrics in non_gt_fields.items():
            for metric in metrics:
                if metric in MetricGroups.HIST_METRICS:
                    hist_requiring_fields[field] = metrics
                    break
        
        # Connect to data database
        data_db = DB(config["data"], self.infra_config)
        data_db.connect()
        
        # Initialize analysis instance
        analysis = Analysis(config=config, task_dir=self.task_dir)
        
        # Results containers
        non_gt_results = {}
        gt_results = {}
        
        # Process each window size
        for ws in config.get("window_size", ["daily"]):
            # Get historical monitoring period if available
            hist_start, hist_end = None, None
            if self.hist_tasks and ws in self.hist_tasks and task in self.hist_tasks[ws]:
                hist_start = config.get("hist_start")
                hist_end = config.get("hist_end")
            
            # Extract monitoring time ranges
            monitor_sts, monitor_ets = extract_monitor_ranges_wo_gt(
                self.curr_time,
                hist_start=hist_start,
                hist_end=hist_end,
                window_size=ws
            )
            
            print(f"Baseline windows for {ws}:")
            print(f"Start times: {[datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in monitor_sts]}")
            print(f"End times: {[datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in monitor_ets]}")
            
            non_gt_results_ws = {}
            
            # Start timing for non-ground-truth metrics
            st_non_gt = time.time()
            
            # Process each monitoring period
            for i, monitor_st in enumerate(monitor_sts):
                monitor_et = monitor_ets[i]
                print(f"Processing baseline: {datetime.fromtimestamp(monitor_st).strftime('%Y-%m-%d %H:%M:%S')} to "
                      f"{datetime.fromtimestamp(monitor_et).strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Get monitored data for current window
                curr_data = data_db.read_data(
                    st=monitor_st,
                    et=monitor_et,
                    fields=list(non_gt_fields.keys())
                )
                curr_data.persist()
                
                # Get baseline configuration
                bl_config, err = get_bl_config(config["baseline"])
                if err:
                    logger.error(f"Error in baseline config: {err}")
                    continue
                
                # Calculate baseline time ranges
                bl_sts, bl_ets = extract_bl_ranges(monitor_st, bl_config)
                bl_start = min(bl_sts) if bl_sts else None
                bl_end = max(bl_ets) if bl_ets else None
                
                if not bl_start or not bl_end:
                    logger.error("Could not determine baseline period")
                    continue
                
                # Check if we have valid histograms for fields that need them
                fields_needing_bl_data = set()
                hist_data = {}
                
                for field in hist_requiring_fields:
                    # Check if we should update baseline based on config
                    should_update = hist_manager.should_update_baseline(
                        task, field, bl_config, bl_start, bl_end
                    )
                    
                    if not should_update:
                        # Try to load existing histograms
                        histograms, found = hist_manager.load_histograms(task, field, bl_start, bl_end)
                        if found:
                            # We have valid histograms - extract baseline histogram
                            _, hist_bl = histograms
                            hist_data[field] = hist_bl
                            continue
                    
                    # No valid histograms or need to update - need to load baseline data
                    fields_needing_bl_data.add(field)
                
                # Only load baseline data if needed
                bl_data = []
                if fields_needing_bl_data:
                    print(f"Loading baseline data for fields: {fields_needing_bl_data}")
                    for j, bl_st in enumerate(bl_sts):
                        bl_et = bl_ets[j]
                        
                        # Determine which fields to load based on source type
                        if bl_config["type"] == "s3":
                            # For S3, we might need to transform field names
                            bl_fields = list(fields_needing_bl_data)
                            # Special handling for S3 sources if needed
                            bl_path = os.path.join(self.task_dir, "histogram", f"hist-{task}.json")
                        else:
                            # For other sources, use field names as-is
                            bl_fields = list(fields_needing_bl_data)
                        
                        # Load baseline data
                        if bl_config["type"] == "s3":
                            # Special handling for S3 data source
                            bl_data_sub = data_db.read_data(
                                st=bl_st,
                                et=bl_et,
                                fields=bl_fields
                            )
                        else:
                            # Connect to baseline database
                            bl_db = DB(bl_config)
                            bl_db.connect()
                            
                            # Get baseline data
                            bl_data_sub = bl_db.read_data(
                                st=bl_st,
                                et=bl_et,
                                fields=bl_fields
                            )
                        
                        bl_data.append(bl_data_sub)
                    
                    # Persist baseline data for better performance
                    for v in bl_data:
                        v.persist()
                
                # Process each field
                results = {}
                for field, metrics in non_gt_fields.items():
                    field_name = field.replace("-", "_").replace(".", "_")
                    result = {}
                    field_type = config["fields"][field]["type"]
                    
                    # Validate data quality
                    st_quality = time.time()
                    r_nm, r_nv, df_filtered = validate_data_quality_spark(
                        curr_data.select(field_name),
                        field_name,
                        field_type,
                        config["fields"][field]["value_range"]
                    )
                    result["t_quality"] = time.time() - st_quality
                    result["missing_value"] = r_nm
                    result["range_violation"] = r_nv
                    
                    # Check for minimum sample size
                    if df_filtered.count() < MIN_SAMPLE_SIZE:
                        result["tags"] = [Tag_NOT_ENOUGH_DATA]
                        results[field] = result
                        continue
                    
                    # Check if field requires histogram metrics
                    require_hist = field in hist_requiring_fields
                    hist_curr = None
                    hist_bl = None
                    
                    # Process histograms if needed
                    if require_hist:
                        st_hist = time.time()
                        
                        if field in hist_data:
                            # We already have baseline histogram
                            hist_bl = hist_data[field]
                            # Extract bins from baseline histogram
                            bins = MetricsSpark.extract_bins(hist_bl)
                            # Calculate current histogram using same bins
                            hist_curr = MetricsSpark.hist_density(curr_data, field_name, bins)
                        elif field in fields_needing_bl_data and bl_data:
                            # Calculate both histograms
                            bins = MetricsSpark.hist_bins(bl_data, field_name)
                            hist_curr = MetricsSpark.hist_density(curr_data, field_name, bins)
                            hist_bl = MetricsSpark.hist_density(bl_data, field_name, bins)
                            # Save histograms for future use
                            hist_path = hist_manager.save_histograms(
                                task, field, bl_start, bl_end, hist_curr, hist_bl
                            )
                            print(f"Saved new histograms for {field} to {hist_path}")
                        else:
                            # This shouldn't happen, but handle it gracefully
                            logger.error(f"Cannot calculate histogram for {field}: "
                                        f"No baseline data or cached histogram")
                            result["tags"] = result.get("tags", []) + ["histogram_error"]
                            results[field] = result
                            continue
                        
                        result["t_histogram"] = time.time() - st_hist
                    
                    # Calculate metrics for this field
                    for metric in metrics:
                        # Get the metric function
                        if metric not in analysis.metric_funcs["spark"]:
                            logger.error(f"Unknown metric: {metric}")
                            continue
                        
                        func = analysis.metric_funcs["spark"][metric]
                        st_metric = time.time()
                        
                        try:
                            if metric in MetricGroups.HIST_METRICS:
                                # Use histograms for histogram-based metrics
                                if hist_curr is not None and hist_bl is not None:
                                    val = func(hist_curr, hist_bl)
                                else:
                                    logger.error(f"Cannot calculate {metric} for {field}: "
                                                f"Missing histograms")
                                    val = None
                            else:
                                # Use raw data for other metrics
                                if field in fields_needing_bl_data and bl_data:
                                    # We have baseline data
                                    val = func(
                                        curr_data.select(field_name),
                                        field_name,
                                        bl_data
                                    )
                                else:
                                    # No baseline data needed or available
                                    val = func(
                                        curr_data.select(field_name),
                                        field_name,
                                        None
                                    )
                            
                            t_metric = time.time() - st_metric
                            result[metric] = val
                            result["t_" + metric] = t_metric
                        
                        except Exception as e:
                            logger.error(f"Error calculating {metric} for {field}: {e}")
                            result[f"error_{metric}"] = str(e)
                    
                    results[field] = result
                
                # Store results for this monitoring period
                non_gt_results_ws = results
                
                # Clean up to free memory
                curr_data.unpersist()
                for v in bl_data:
                    v.unpersist()
            
            # Print timing information
            print(f"Non-groundtruth metrics time: {time.time() - st_non_gt:.2f}s")
            
            # Store results by window size
            non_gt_results[ws] = non_gt_results_ws
        
        # Process ground truth metrics if needed
        if gt_fields:
            for ws in config.get("window_size", ["daily"]):
                gt_results_ws = {}
                
                for field, metrics in gt_fields.items():
                    gt_config = config["fields"][field]["ground_truth"]
                    
                    # Extract monitoring ranges with ground truth
                    monitor_sts, monitor_ets = extract_monitor_ranges_w_gt(
                        self.curr_time,
                        hist_start=hist_start,
                        hist_end=hist_end,
                        window_size=ws,
                        interval=gt_config["backfill"]["interval"],
                        n=gt_config["backfill"]["n"]
                    )
                    
                    # Process ground truth metrics (implementation not shown)
                    # ...
                
                gt_results[ws] = gt_results_ws
        
        # Return all results
        return {
            "non_gt": non_gt_results,
            "gt": gt_results
        }
