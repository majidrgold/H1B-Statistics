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
            with open(self.hist_metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        with open(self.hist_metadata_file, "w") as f:
            json.dump(self.metadata, f)

    def get_histogram_path(self, task, field, bl_start, bl_end):
        # Convert timestamps to string format for filename
        start_str = datetime.fromtimestamp(bl_start).strftime("%Y%m%d")
        end_str = datetime.fromtimestamp(bl_end).strftime("%Y%m%d")
        # Replace any potentially problematic characters in field name
        field_safe = field.replace(".", "_")
        filename = f"{task}_{field_safe}_{start_str}_{end_str}.pkl"
        return os.path.join(self.hist_dir, filename)

    def save_histograms(self, task, field, bl_start, bl_end, hist_bl, bins):
        """Save histogram data and bins to a pickle file"""
        if bins is None:
            raise ValueError("Bins must be provided when saving histogram data")
            
        hist_path = self.get_histogram_path(task, field, bl_start, bl_end)
        
        # Store both histogram and bins in the same file
        data_to_save = {
            "histogram": hist_bl,
            "bins": bins
        }
        
        with open(hist_path, "wb") as f:
            pickle.dump(data_to_save, f)

        # Update metadata
        if task not in self.metadata:
            self.metadata[task] = {}

        self.metadata[task][field] = {
            "path": hist_path,
            "bl_start": bl_start,
            "bl_end": bl_end,
            "created": time.time(),
            "last_accessed": time.time(),
        }

        self._save_metadata()
        return hist_path

    def load_histograms(self, task, field, bl_start, bl_end):
        """Load histogram data and bins from a pickle file
        Returns: (histogram, bins, success)
        """
        # Check if we have exact match in metadata
        if (
            task in self.metadata
            and field in self.metadata[task]
            and self.metadata[task][field]["bl_start"] == bl_start
            and self.metadata[task][field]["bl_end"] == bl_end
        ):
            hist_path = self.metadata[task][field]["path"]
            if os.path.exists(hist_path):
                with open(hist_path, "rb") as f:
                    try:
                        loaded_data = pickle.load(f)
                        # Update last accessed timestamp
                        self.metadata[task][field]["last_accessed"] = time.time()
                        self._save_metadata()
                        
                        # Handle both old and new format
                        if isinstance(loaded_data, dict) and "histogram" in loaded_data:
                            # New format with histogram and bins
                            return loaded_data["histogram"], loaded_data["bins"], True
                        else:
                            # Old format with just histogram - this should not happen with new code
                            # but keeping for backward compatibility
                            print(f"Warning: Old format histogram found for {field} without bins")
                            return loaded_data, None, False
                    except (pickle.PickleError, EOFError, IOError) as e:
                        print(f"Error loading histogram from {hist_path}: {e}")

        return None, None, False

    def should_update_baseline(self, task, field, bl_config, bl_start, bl_end):
        """Check if baseline histogram needs updating based on config and timestamps"""
        if task not in self.metadata or field not in self.metadata[task]:
            return True

        metadata = self.metadata[task][field]

        # If baseline time period changed
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

    def cleanup_old_histograms(self, max_age_days=90, unused_days=30):
        """Delete old histograms based on creation date or last access time"""
        current_time = time.time()
        tasks_to_delete = []

        for task, fields in self.metadata.items():
            fields_to_delete = []

            for field, metadata in fields.items():
                age_days = (current_time - metadata["created"]) / (60 * 60 * 24)
                unused_time_days = (current_time - metadata["last_accessed"]) / (
                    60 * 60 * 24
                )

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



# ---
# Dictionary to store pre-calculated histograms
pre_calculated_histograms = {}

# For each field that requires histogram metrics
for field, metrics in non_gt_fields.items():
    # Check if any of the metrics require histograms
    requires_hist = any(metric in MetricGroups.HIST_METRICS for metric in metrics)
    
    if requires_hist:
        # Get timestamps for histogram cache
        hist_bl_start = min(bl_sts).timestamp() if bl_sts else None
        hist_bl_end = max(bl_ets).timestamp() if bl_ets else None
        
        if hist_bl_start is not None and hist_bl_end is not None:
            # Try to load cached baseline histogram with bins
            hist_bl, bins, success = self.histogram_manager.load_histograms(
                task, field, hist_bl_start, hist_bl_end
            )
            
            if success:
                print(f"Loaded baseline histogram and bins for {field} from cache")
                # Use the loaded bins for the current histogram
                hist_curr = analysis.metric_funcs["spark"].hist_density(
                    [curr_data], field.replace(".", "_"), bins
                )
                pre_calculated_histograms[field] = (hist_curr, hist_bl)
            else:
                # Calculate histograms and bins if cache miss or bins are missing
                bins = analysis.metric_funcs["spark"].hist_bins(bl_data, field.replace(".", "_"))
                hist_curr = analysis.metric_funcs["spark"].hist_density(
                    [curr_data], field.replace(".", "_"), bins
                )
                hist_bl = analysis.metric_funcs["spark"].hist_density(
                    bl_data, field.replace(".", "_"), bins
                )
                
                # Save both the baseline histogram and bins to cache
                self.histogram_manager.save_histograms(
                    task, field, hist_bl_start, hist_bl_end, hist_bl, bins
                )
                
                pre_calculated_histograms[field] = (hist_curr, hist_bl)
