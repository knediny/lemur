# ----------------------------------------------------------------------------------
# - Author Contact: wei.zhang, zwpride@buaa.edu.cn (Original code)
# ----------------------------------------------------------------------------------

import os
from utils import evaluator
import pandas as pd


input_dir = "data/loghub_2k_corrected/"  # The input directory of log file
benchmark_settings = {
    "HDFS": {
        "log_file": "HDFS/HDFS_2k.log",
    },
    "Hadoop": {
        "log_file": "Hadoop/Hadoop_2k.log",
    },
    "Spark": {
        "log_file": "Spark/Spark_2k.log",
    },
    "Zookeeper": {
        "log_file": "Zookeeper/Zookeeper_2k.log",
    },
    "BGL": {
        "log_file": "BGL/BGL_2k.log",
    },
    "HPC": {
        "log_file": "HPC/HPC_2k.log",
    },
    "Thunderbird": {
        "log_file": "Thunderbird/Thunderbird_2k.log",
    },
    "Windows": {
        "log_file": "Windows/Windows_2k.log",
    },
    "Linux": {
        "log_file": "Linux/Linux_2k.log",
    },
    "Android": {
        "log_file": "Android/Android_2k.log",
    },
    "HealthApp": {
        "log_file": "HealthApp/HealthApp_2k.log",
    },
    "Apache": {
        "log_file": "Apache/Apache_2k.log",
    },
    "Proxifier": {
        "log_file": "Proxifier/Proxifier_2k.log",
    },
    "OpenSSH": {
        "log_file": "OpenSSH/OpenSSH_2k.log",
    },
    "OpenStack": {
        "log_file": "OpenStack/OpenStack_2k.log",
    },
    "Mac": {
        "log_file": "Mac/Mac_2k.log",
    },
}

before_cot_benchmarks = []
after_cot_benchmarks = []

for dataset, setting in benchmark_settings.items():
    input_file = os.path.join(input_dir, setting["log_file"]+"_structured_corrected.csv")
    log_file = os.path.basename(setting["log_file"])
    before_cot_file = (f"cot/archive/loghub_2k_corrected_before_cot/{log_file}_structured.csv")
    after_cot_file = (f"cot/archive/loghub_2k_corrected_after_cot/{log_file}_structured.csv")

    print(f"===== Dataset: {dataset} =====")
    print("Before COT Evaluation: ", end="")
    before_cot_benchmark = {"Dataset": dataset}
    before_cot_benchmark.update(evaluator.evaluate(
        groundtruth=input_file,
        parsedresult=before_cot_file,
        debug=False
    ))

    print("After COT Evaluation: ", end="")
    after_cot_benchmark = {"Dataset": dataset}
    after_cot_benchmark.update(evaluator.evaluate(
        groundtruth=input_file,
        parsedresult=after_cot_file,
        debug=False
    ))

    before_cot_benchmarks.append(before_cot_benchmark)
    after_cot_benchmarks.append(after_cot_benchmark)

pd.DataFrame(before_cot_benchmarks).to_csv("cot/archive/before_cot.csv", float_format="%.6f")
pd.DataFrame(after_cot_benchmarks).to_csv("cot/archive/after_cot.csv", float_format="%.6f")
