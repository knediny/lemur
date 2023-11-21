import sys

sys.path.append(".")

import os
from parser.parser import Parser
from utils import evaluator
import pandas as pd
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
input_dir = "data/loghub_2k_corrected/"  # The input directory of log file
output_dir = (
    f"result/loghub_2k_corrected{timestamp}/"  # The output directory of parsing results
)

benchmark_settings = {
    "HDFS": {
        "log_file": "HDFS/HDFS_2k.log",
        "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
        "regex": [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"],
        "delimiter": [""],
        "k": 2,
        "n_candidate": 32,
        "freq_threshold": 0.3,
        "type_theshold": 0,
        "jaccard_similarity_threshold": 0.7,
        "first_weight": 5,
        "digit_percent": 0,
    },
    "Hadoop": {
        "log_file": "Hadoop/Hadoop_2k.log",
        "log_format": "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
        "regex": [r"(\d+\.){3}\d+"],
        "delimiter": [],
        "k": 12,
        "n_candidate": 128,
        "freq_threshold": 0.5,
        "type_theshold": 0,
        "jaccard_similarity_threshold": 0.6,
        "first_weight": 10,
        "digit_percent": 0,
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
        "n_candidate": 256,
        "freq_threshold": 0.5,
        "type_theshold": 4,
        "jaccard_similarity_threshold": 0.6,
        "first_weight": 10,
        "digit_percent": 0,
    },
    "Zookeeper": {
        "log_file": "Zookeeper/Zookeeper_2k.log",
        "log_format": "<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
        "regex": [r"(/|)(\d+\.){3}\d+(:\d+)?"],
        "delimiter": [],
        "k": 3,
        "n_candidate": 64,
        "freq_threshold": 0.5,
        "type_theshold": 3,
        "jaccard_similarity_threshold": 0.7,
        "first_weight": 10,
        "digit_percent": 0,
    },
    "BGL": {
        "log_file": "BGL/BGL_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
        "regex": [r"core\.\d+", r"\/(\w+\/?)+"],
        "delimiter": [],
        "k": 9,
        "n_candidate": 32,
        "freq_threshold": 0.07,
        "type_theshold": 6,
        "jaccard_similarity_threshold": 0.6,
        "first_weight": 10,
        "digit_percent": 0,
    },
    "HPC": {
        "log_file": "HPC/HPC_2k.log",
        "log_format": "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
        "regex": [r"\\042[A-Za-z0-9\:\-\\]+"],
        "delimiter": [],
        "k": 9,
        "n_candidate": 64,
        "freq_threshold": 0.2,
        "type_theshold": 5,
        "jaccard_similarity_threshold": 0.6,
        "first_weight": 10,
        "digit_percent": 0.4,
    },
    "Thunderbird": {
        "log_file": "Thunderbird/Thunderbird_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"(/[\w-. ]+)+"],
        "delimiter": [],
        "k": 11,
        "n_candidate": 64,
        "freq_threshold": 0.1,
        "type_theshold": 2,
        "jaccard_similarity_threshold": 0.4,
        "first_weight": 10,
        "digit_percent": 0.05,
    },
    "Windows": {
        "log_file": "Windows/Windows_2k.log",
        "log_format": "<Date> <Time>, <Level>                  <Component>    <Content>",
        "regex": [r"0x.*?\s"],
        "delimiter": [],
        "k": 8,
        "n_candidate": 128,
        "freq_threshold": 0.1,
        "type_theshold": 1,
        "jaccard_similarity_threshold": 0.6,
        "first_weight": 10,
        "digit_percent": 0.0,
    },
    "Linux": {
        "log_file": "Linux/Linux_2k.log",
        "log_format": "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2} \d{4}"],
        # "regex": [r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2}", r"J([a-z]{2})"],
        "delimiter": [r""],
        "k": 25,
        "n_candidate": 256,
        "freq_threshold": 0.94,  # 0.35 0.69086 0.9327
        "type_theshold": 2,
        "jaccard_similarity_threshold": 0.3,  # important
        "first_weight": 20,
        "digit_percent": 0.0,
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
        "n_candidate": 32,
        "freq_threshold": 0.1,
        "type_theshold": 3,
        "jaccard_similarity_threshold": 0.7,
        "first_weight": 10,
        "digit_percent": 0.0,
    },
    "HealthApp": {
        "log_file": "HealthApp/HealthApp_2k.log",
        "log_format": "<Time>\|<Component>\|<Pid>\|<Content>",
        "regex": [],
        "delimiter": [r""],
        "k": 12,
        "n_candidate": 32,
        "freq_threshold": 0.1,
        "type_theshold": 4,
        "jaccard_similarity_threshold": 0.7,
        "first_weight": 10,
        "digit_percent": 0.0,
    },
    "Apache": {
        "log_file": "Apache/Apache_2k.log",
        "log_format": "\[<Time>\] \[<Level>\] <Content>",
        "regex": [r"(\d+\.){3}\d+"],
        "delimiter": [],
        "k": 12,
        "n_candidate": 32,
        "freq_threshold": 0.1,
        "type_theshold": 4,
        "jaccard_similarity_threshold": 0.7,
        "first_weight": 10,
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
        "n_candidate": 32,
        "freq_threshold": 0.1,
        "type_theshold": 3,
        "jaccard_similarity_threshold": 0.7,
        "first_weight": 10,
        "digit_percent": 0.0,
    },
    "OpenSSH": {
        "log_file": "OpenSSH/OpenSSH_2k.log",
        "log_format": "<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"([\w-]+\.){2,}[\w-]+"],
        "delimiter": [],
        "k": 4,
        "n_candidate": 128,
        "freq_threshold": 0.98,
        "type_theshold": 5,
        "jaccard_similarity_threshold": 0.5,
        "first_weight": 20,
        "digit_percent": 0.3,
    },
    "OpenStack": {
        "log_file": "OpenStack/OpenStack_2k.log",
        "log_format": "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>",
        "regex": [r"((\d+\.){3}\d+,?)+", r"/.+?\s", r"\d+"],
        "delimiter": [],
        "k": 20,
        "n_candidate": 32,
        "freq_threshold": 0.1,
        "type_theshold": 5,
        "jaccard_similarity_threshold": 0.7,
        "first_weight": 10,
        "digit_percent": 0.26,
    },
    "Mac": {
        "log_file": "Mac/Mac_2k.log",
        "log_format": "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
        "regex": [r"([\w-]+\.){2,}[\w-]+"],
        "delimiter": [],
        "k": 30,
        "n_candidate": 128,
        "freq_threshold": 0.03,
        "type_theshold": 5,
        "jaccard_similarity_threshold": 0.7,
        "first_weight": 3,
        "digit_percent": 0.25,
    },
    # '安徽移动': {
    #     'log_file': 'anhuiyidong_cut_tuomin_2k.csv',
    #     'log_format': '<Content>',
    #     'regex': [],
    #     "min_event_count": 2,
    #     "merge_percent": 0.6,
    #     "n_candidate": 64,
    #     "k": 6,
    #     "m": 3,
    #     "sample_count": 0,
    # },
    # '国网': {
    #     'log_file': 'guowang_cut_tuomin_2k.csv',
    #     'log_format': '<Content>',
    #     'regex': [],
    #     "min_event_count": 2,
    #     "merge_percent": 0.6,
    #     "n_candidate": 64,
    #     "k": 6,
    #     "m": 3,
    #     "sample_count": 0,
    # },
    # '海通': {
    #     'log_file': 'haitong_extend_tuomin_2k.csv',
    #     'log_format': '<Content>',
    #     'regex': [],
    #     "min_event_count": 2,
    #     "merge_percent": 0.6,
    #     "n_candidate": 64,
    #     "k": 6,
    #     "m": 3,
    #     "sample_count": 0,
    # },
    # '河南移动': {
    #     'log_file': 'henanyidong_cut_tuomin_2k.csv',
    #     'log_format': '<Content>',
    #     'regex': [],
    #     "min_event_count": 2,
    #     "merge_percent": 0.6,
    #     "n_candidate": 64,
    #     "k": 6,
    #     "m": 3,
    #     "sample_count": 0,
    # },
    # '农商行': {
    #     'log_file': 'nongshanghang_extend_tuomin_2k.csv',
    #     'log_format': '<Content>',
    #     'regex': [],
    #     "min_event_count": 2,
    #     "merge_percent": 0.6,
    #     "n_candidate": 64,
    #     "k": 6,
    #     "m": 3,
    #     "sample_count": 0,
    # },
    # '上汽大众': {
    #     'log_file': 'shangqidazhong_tuomin_2k.csv',
    #     'log_format': '<Content>',
    #     'regex': [],
    #     "min_event_count": 2,
    #     "merge_percent": 0.6,
    #     "n_candidate": 64,
    #     "k": 6,
    #     "m": 3,
    #     "sample_count": 0,
    # },
}


def inference(dataset, setting):
    input_file = os.path.join(input_dir, os.path.dirname(setting["log_file"]))
    log_file = os.path.basename(setting["log_file"])
    parser = Parser(
        input_file=input_file,
        output_file=output_dir,
        rex=setting["regex"],
        delimiter=setting["delimiter"],
        log_format=setting["log_format"],
        dataset=dataset,
        n_candidate=setting["n_candidate"],
        k=setting["k"],
        type_threshold=setting["type_theshold"],
        freq_threshold=setting["freq_threshold"],
        jaccard_similarity_threshold=setting["jaccard_similarity_threshold"],
        first_weight=setting["first_weight"],
        digit_percent=setting["digit_percent"],
    )
    parser.parse(log_file)

    bechmark_result = {"Dataset": dataset}
    bechmark_result.update(evaluator.evaluate(
        groundtruth=os.path.join(input_file, log_file + "_structured_corrected.csv"),
        parsedresult=os.path.join(output_dir, log_file + "_structured.csv"),
        debug=False
    ))

    bechmark_results.append(bechmark_result)


bechmark_results = []
for dataset, setting in benchmark_settings.items():
    inference(dataset, setting)
df_result = pd.DataFrame(bechmark_results)
df_result.to_csv(f"{output_dir}/{output_dir}_benckmark_results.csv", float_format="%.6f")
