# ----------------------------------------------------------------------------------
# - Author Contact: wei.zhang, zwpride@buaa.edu.cn
#
# - Inherits basic inference framework of LOGAPI.
# - Original code is to enhance the ability to inference.
# ----------------------------------------------------------------------------------

import os
from parser.parser import Parser
from utils import evaluator
import pandas as pd

input_dir = "data/loghub_2k_corrected/"  # The input directory of log file
output_dir = (
    f"result/loghub_2k_corrected/"  # The output directory of parsing results
)

benchmark_settings = {
    "HDFS": {
        "log_file": "HDFS/HDFS_2k.log",
        "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
        "regex": [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"],
        "delimiter": [""],
        "k": 2,
        "entropy_theshold": 2.0,
        "jaccard_similarity_threshold": 0.7,
        "digit_percent": 0,
    },
    "Hadoop": {
        "log_file": "Hadoop/Hadoop_2k.log",
        "log_format": "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
        "regex": [r"(\d+\.){3}\d+"],
        "delimiter": [],
        "k": 8,
        "entropy_theshold": 1.7,
        "jaccard_similarity_threshold": 0.7,
        "digit_percent": 0.0,
    },
    "Spark": {
        "log_file": "Spark/Spark_2k.log",
        "log_format": "<Date> <Time> <Level> <Component>: <Content>",
        "regex": [
            r"(\d+\.){3}\d+",
            r"\b[KGTM]?B\b",
        ],
        # "regex": [r"(\d+\.){3}\d+", r"\b[KGTM]?B\b", r"([\w-]+\.){2,}[\w-]+"],
        "delimiter": [],
        "k": 6,
        "entropy_theshold": 2.10,
        "jaccard_similarity_threshold": 0.6,
        "digit_percent": 0,
    },
    "Zookeeper": {
        "log_file": "Zookeeper/Zookeeper_2k.log",
        "log_format": "<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
        "regex": [r"(/|)(\d+\.){3}\d+(:\d+)?"],
        "delimiter": [],
        "k": 8,
        "entropy_theshold": 2.2,
        "jaccard_similarity_threshold": 0.9,
        "digit_percent": 0,
    },
    "BGL": {
        "log_file": "BGL/BGL_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
        "regex": [r"core\.\d+", r"\/(\w+\/?)+"],
        "delimiter": [],
        "k": 9,
        "entropy_theshold": 5.5,
        "jaccard_similarity_threshold": 0.6,
        "digit_percent": 0,
    },
    "HPC": {
        "log_file": "HPC/HPC_2k.log",
        "log_format": "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
        "regex": [r"\\042[A-Za-z0-9\:\-\\]+"],
        "delimiter": [],
        "k": 9,
        "entropy_theshold": 1.2,
        "jaccard_similarity_threshold": 0.6,
        "digit_percent": 0.4,
    },
    "Thunderbird": {
        "log_file": "Thunderbird/Thunderbird_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"(/[\w-. ]+)+"],
        "delimiter": [],
        "k": 11,
        "entropy_theshold": 4.1,
        "jaccard_similarity_threshold": 0.4,
        "digit_percent": 0.05,
    },
    "Windows": {
        "log_file": "Windows/Windows_2k.log",
        "log_format": "<Date> <Time>, <Level>                  <Component>    <Content>",
        "regex": [r"0x.*?\s"],
        "delimiter": [],
        "k": 8,
        "entropy_theshold": 1.1,
        "jaccard_similarity_threshold": 0.6,
        "digit_percent": 0.0,
    },
    "Linux": {
        "log_file": "Linux/Linux_2k.log",
        "log_format": "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2} \d{4}"],
        # "regex": [r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2}", r"J([a-z]{2})"],
        "delimiter": [r""],
        "k": 25,
        "entropy_theshold": 0.09,
        "jaccard_similarity_threshold": 0.33,
        "digit_percent": 0.3,
    },
    "Android": {
        "log_file": "Android/Android_2k.log",
        "log_format": "<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>",
        "regex": [
            r"(/[\w-]+)+",
            r"([\w-]+\.){2,}[\w-]+",
            r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b",
            r"(true|false)"
        ],
        "delimiter": [r""],
        "k": 9,
        "entropy_theshold": 3.5,
        "jaccard_similarity_threshold": 0.7,
        "digit_percent": 0.0,
    },
    "HealthApp": {
        "log_file": "HealthApp/HealthApp_2k.log",
        "log_format": "<Time>\|<Component>\|<Pid>\|<Content>",
        "regex": [],
        "delimiter": [r""],
        "k": 12,
        "entropy_theshold": 1.8,
        "jaccard_similarity_threshold": 0.7,
        "digit_percent": 0.0,
    },
    "Apache": {
        "log_file": "Apache/Apache_2k.log",
        "log_format": "\[<Time>\] \[<Level>\] <Content>",
        "regex": [r"(\d+\.){3}\d+"],
        "delimiter": [],
        "k": 12,
        "entropy_theshold": 0,
        "jaccard_similarity_threshold": 0.7,
        "digit_percent": 0.0,
    },
    "Proxifier": {
        "log_file": "Proxifier/Proxifier_2k.log",
        "log_format": "\[<Time>\] <Program> - <Content>",
        "regex": [
            r"<\d+\ssec",
            r"([\w-]+\.)+[\w-]+(:\d+)?",
            r"\d{2}:\d{2}(:\d{2})*",
            r"[KGTM]B",
        ],
        "delimiter": [r"\(.*?\)"],
        "k": 12,
        "jaccard_similarity_threshold": 0.7,
        "entropy_theshold": 0.1,
        "digit_percent": 0.0,
    },
    "OpenSSH": {
        "log_file": "OpenSSH/OpenSSH_2k.log",
        "log_format": "<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"([\w-]+\.){2,}[\w-]+"],
        "delimiter": [],
        "k": 4,
        "entropy_theshold": 0.2,
        "jaccard_similarity_threshold": 0.5,
        "digit_percent": 0.3,
    },
    "OpenStack": {
        "log_file": "OpenStack/OpenStack_2k.log",
        "log_format": "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>",
        "regex": [r"((\d+\.){3}\d+,?)+", r"/.+?\s", r"\d+"],
        "delimiter": [],
        "k": 20,
        "entropy_theshold": 2.3,
        "jaccard_similarity_threshold": 0.7,
        "digit_percent": 0.26,
    },
    "Mac": {
        "log_file": "Mac/Mac_2k.log",
        "log_format": "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
        "regex": [r"([\w-]+\.){2,}[\w-]+"],
        "delimiter": [],
        "k": 12,
        "entropy_theshold": 4.7,
        "jaccard_similarity_threshold": 0.7,
        "digit_percent": 0.25,
    },
}


def inference(dataset, setting, debug=False):
    input_file = os.path.join(input_dir, os.path.dirname(setting["log_file"]))
    log_file = os.path.basename(setting["log_file"])
    parser = Parser(
        input_file=input_file,
        output_file=output_dir,
        rex=setting["regex"],
        delimiter=setting["delimiter"],
        log_format=setting["log_format"],
        dataset=dataset,
        k=setting["k"],
        entropy_theshold=setting["entropy_theshold"],
        jaccard_similarity_threshold=setting["jaccard_similarity_threshold"],
        digit_percent=setting["digit_percent"],
        debug=debug
    )
    parser.parse(log_file)

    benchmark_result = {"Dataset": dataset}
    benchmark_result.update(evaluator.evaluate(
        groundtruth=os.path.join(input_file, log_file + "_structured_corrected.csv"),
        parsedresult=os.path.join(output_dir, log_file + "_structured.csv"),
        debug=debug
    ))

    benchmark_results.append(benchmark_result)

benchmark_results = []
for dataset, setting in benchmark_settings.items():
    inference(dataset, setting, False)
df_result = pd.DataFrame(benchmark_results)
df_result.to_csv(f"{output_dir}/benckmark_results.csv", float_format="%.6f")
