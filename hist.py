import os
import time
from datetime import timedelta
import dateutil.parser
from dateutil.relativedelta import relativedelta

from dotenv import load_dotenv
from .utils.file import load_yaml_file, load_json_file
from .utils.logger import Logger
from .database.es import init_es, read_from_es, index_document, get_count

from .analysis.analysis_new import Analysis
from .database.db import DB
from .utils.label import label_spark
from .utils.parser_utils import group_fields, get_bl_config
from .utils.datetime_helper import (
    get_gt_ranges,
    extract_monitor_ranges_wo_gt,
    extract_bl_ranges,
)

from .analysis.constants import MetricGroups
from pyspark.sql.functions import regexp_replace, col
from .utils.result import prepare_es_result, prepare_es_result_per_field
from .analysis.histogram_manager import HistogramManager  # Import HistogramManager

import pickle


class SPM:
    def __init__(
        self, task_filename, task_dir, curr_time, infra_path, window_size=None
    ):
        load_dotenv()
        self.task_filename = task_filename
        self.task_dir = task_dir
        self.tasks = load_json_file(os.path.join(task_dir, task_filename))
        self.curr_time = curr_time
        self.window_size = window_size
        self.hist_tasks = load_json_file(os.path.join(task_dir, "hist_tasks.json"))
        self.infra_config = load_yaml_file(infra_path)
        self.histogram_manager = HistogramManager(task_dir)  # Initialize HistogramManager

    def process_tasks(self):
        es, err = init_es(
            self.infra_config["non_prod_es"][0],
            os.getenv("QA_USERNAME"),
            os.getenv("QA_PASSWORD"),
            os.getenv("CA_CERTS"),
        )

        if err is not None:
            logger.error(f"Failed to initiate result ES: {err}")
            return

        if not self.tasks:
            logger.debug(f"No tasks found in {self.task_filename}")
            return

        for task in self.tasks:
            print(task)
            st = time.time()
            res, res_per_field = self.process_task(task)

            # Save results to ES
            for v in res:
                es_resp, err = index_document(es, "spm_result_one_payload", v)
                if err is not None:
                    logger.error(f"Failed to write result ES: {err}")
                print(es_resp, err)

            for v in res_per_field:
                es_resp, err = index_document(es, "spm_result_per_field", v)
                if err is not None:
                    logger.error(f"Failed to write result ES: {err}")
                print(es_resp, err)

        logger.debug(
            f"Metrics processed and the result saved to ES for task file {self.task_filename}"
        )

    def process_task(self, task):
        print("process task: ", task)
        config_filepath = os.path.join(self.task_dir, "spm_configs", task + ".yml")
        config = load_yaml_file(config_filepath)
        if not config:
            logger.error(f"Failed to load config file: {config_filepath}")
            return

        # Extract gt and non-gt metrics
        gt_fields, non_gt_fields = group_fields(config.get("fields", {}))

        # Connect to data_db where to fetch current data
        data_db = DB(config["data"], self.infra_config)
        data_db.connect()
        bl_db, gt_db = None, None

        # Initialize Analysis instance for metrics calculation
        analysis = Analysis(config=config)

        non_gt_results = {}
        gt_results = {}

        for ws in config.get("window_size", ["daily"]):
            hist_start, hist_end = None, None
            hist_start = config["hist_start"]
            hist_end = config["hist_end"]

            # Extract monitoring start and end datetimes
            monitor_sts, monitor_ets = extract_monitor_ranges_wo_gt(
                self.curr_time, hist_start=hist_start, hist_end=hist_end, window_size=ws
            )

            print("bl windows")
            print(monitor_sts)
            print(monitor_ets)

            st_non_gt = time.time()
            for i, st in enumerate(monitor_sts):
                print("process bl:", monitor_sts[i], monitor_ets[i])
                st_process = time.time()
                curr_data = data_db.read_data(
                    st, monitor_ets[i], fields=list(non_gt_fields.keys())
                )

                curr_data.persist()
                curr_data.unpersist()

                bl_config = get_bl_config(st, config["baseline"])
                bl_sts, bl_ets = extract_bl_ranges(st, bl_config)

                if len(bl_sts) == 0:
                    continue

                bl_data = []
                for i, bl_st in enumerate(bl_sts):
                    bl_data_sub = data_db.read_data(
                        bl_st,
                        bl_ets[i],
                        config=bl_config,
                        fields=list(non_gt_fields.keys()),
                    )
                    bl_data.append(bl_data_sub)

                for v in bl_data:
                    v.persist()
                    v.show(1)
                
                # Dictionary to store pre-calculated histograms
                pre_calculated_histograms = {}
                
                # For each field that requires histogram metrics
                for field, metrics in non_gt_fields.items():
                    # Check if any of the metrics require histograms
                    requires_hist = any(metric in MetricGroups.HIST_METRICS for metric in metrics)
                    
                    if requires_hist:
                        # Try to load histogram from cache
                        hist_bl_start = min(bl_sts).timestamp() if bl_sts else None
                        hist_bl_end = max(bl_ets).timestamp() if bl_ets else None
                        
                        if hist_bl_start is not None and hist_bl_end is not None:
                            # Check if we need to update baseline histograms
                            need_update = self.histogram_manager.should_update_baseline(
                                task, field, bl_config, hist_bl_start, hist_bl_end
                            )
                            
                            if not need_update:
                                # Try to load cached histograms
                                hist_bl, success = self.histogram_manager.load_histograms(
                                    task, field, hist_bl_start, hist_bl_end
                                )
                                
                                if success:
                                    print(f"Loaded histogram for {field} from cache")
                                    # Calculate histogram for current data
                                    bins = analysis.metric_funcs["spark"].hist_bins(bl_data, field.replace(".", "_"))
                                    hist_curr = analysis.metric_funcs["spark"].hist_density(
                                        [curr_data], field.replace(".", "_"), bins
                                    )
                                    pre_calculated_histograms[field] = (hist_curr, hist_bl)
                            
                            if field not in pre_calculated_histograms:
                                # Calculate histograms and save to cache
                                bins = analysis.metric_funcs["spark"].hist_bins(bl_data, field.replace(".", "_"))
                                hist_curr = analysis.metric_funcs["spark"].hist_density(
                                    [curr_data], field.replace(".", "_"), bins
                                )
                                hist_bl = analysis.metric_funcs["spark"].hist_density(
                                    bl_data, field.replace(".", "_"), bins
                                )
                                
                                # Save baseline histogram to cache
                                self.histogram_manager.save_histograms(
                                    task, field, hist_bl_start, hist_bl_end, hist_bl
                                )
                                
                                pre_calculated_histograms[field] = (hist_curr, hist_bl)

                # Use pre-calculated histograms in metric calculation
                non_gt_res = analysis.cal_non_gt_metrics_spark(
                    curr_data, bl_data, non_gt_fields, pre_calculated_histograms
                )

                print("before: ", non_gt_results)
                curr_data.unpersist()
                for v in bl_data:
                    v.unpersist()

                non_gt_results[ws] = non_gt_res

                print("after: ", non_gt_results)
                print("non gt time: ", time.time() - st_non_gt)
                print("ws bl: ", non_gt_res)

            # TODO: Load ground-truth
            st_gt = time.time()
            if len(gt_fields) > 0:
                for gt_field in list(gt_fields.keys()):
                    gt_config = config["fields"][gt_field]["ground_truth"]
                    monitor_sts, monitor_ets = extract_monitor_ranges_wo_gt(
                        self.curr_time,
                        hist_start,
                        hist_end,
                        window_size=ws,
                        interval=gt_config["backfill"]["interval"],
                        n=gt_config["backfill"]["n"],
                    )

                    print("gt windows")
                    print(monitor_sts)
                    print(monitor_ets)

                    gt_data = data_db.read_data(
                        min(monitor_sts),
                        max(monitor_ets),
                        config=gt_config,
                        fields=[gt_config["id"]],
                    )

                    gt_data.persist()
                    gt_data = gt_data.withColumn(
                        gt_config["id"] + "_new",
                        regexp_replace(col("TRAN_INITL_ID"), " ", "_"),
                    )

                    for i, st in enumerate(monitor_sts):
                        print("process gt:", monitor_sts[i], monitor_ets[i])
                        st_process = time.time()
                        curr_data = data_db.read_data(
                            st,
                            monitor_ets[i],
                            fields=list(gt_fields.keys())
                            + [
                                gt_config["id"],
                                config["fields"][gt_field]["amt_field"],
                            ],
                        )

                        curr_data.persist()
                        data_labeled = label_spark(
                            curr_data,
                            gt_config["id"].replace(".", "_"),
                            gt_data,
                            (gt_config["id"] + "_new").replace(".", "_"),
                        )

                        data_labeled.persist()
                        data_labeled = data_labeled.withColumnRenamed(
                            "label", gt_field.replace(".", "_") + "_label"
                        )

                        gt_res = analysis.cal_gt_metrics_spark(data_labeled, gt_fields)

                        print("gt time: ", time.time() - st_gt)
                        print(gt_res)

                        data_labeled.unpersist()
                        curr_data.unpersist()
                        gt_results[ws] = gt_res

            # TODO: Save result
            with open("results.pkl", "wb") as f:
                pickle.dump([non_gt_results, gt_results], f)

            es_result = prepare_es_result(
                non_gt_results, gt_results, config, self.curr_time
            )
            es_result_per_field = prepare_es_result_per_field(
                non_gt_results, gt_results, config, self.curr_time
            )

            es, err = init_es(
                self.infra_config["non_prod_es"][0],
                os.getenv("QA_USERNAME"),
                os.getenv("QA_PASSWORD"),
                os.getenv("CA_CERTS"),
            )

            return es_result, es_result_per_field
