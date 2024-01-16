# ----------------------------------------------------------------------------------
# Code License Notice
# ----------------------------------------------------------------------------------
# - Author Contact: wei.zhang, zwpride@buaa.edu.cn
#
# - Inherits the basic data load algorithm and basic inference framework of LOGAPI.
# - But more is original code.
# ----------------------------------------------------------------------------------
import regex as re
import os
import hashlib
import pandas as pd
from datetime import datetime
from collections import defaultdict
from functools import lru_cache
from parser.vectorizer import Vectorizer
import numpy as np
import math

class Parser:
    def __init__(
        self,
        input_file,
        output_file,
        rex=[],
        delimiter=[],
        log_format=None,
        dataset=None,
        keep_para=True,
        k=2,
        entropy_theshold=0,
        jaccard_similarity_threshold=0.7,
        digit_percent=0.3,
        debug=False,
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.rex = rex
        self.delimiter = delimiter
        self.logformat = log_format
        self.dataset = dataset
        self.logname = None
        self.logname_extension = None
        self.k = k
        self.entropy_theshold = entropy_theshold
        self.jaccard_similarity_threshold = jaccard_similarity_threshold
        self.digit_percent = digit_percent
        self.keep_para = keep_para
        self.df_log = None
        self.token_count_buckets = defaultdict(lambda: [None, None, None])
        self.token_count_tidx_clusters = defaultdict(list)
        self.template_collection = set()
        if debug:
            print(self.dataset, 
                "k: " ,self.k,
                "entropy_theshold: ", self.entropy_theshold,
                    "jaccard_similarity_threshold: ", self.jaccard_similarity_threshold,
                    "digit_percent: ", self.digit_percent)
        
    def parse(self, logname):
        start_time = datetime.now()
        print("Parsing file: " + os.path.join(self.input_file, logname))
        self.logname = logname
        _, self.logname_extension = os.path.splitext(os.path.basename(self.logname))
        self.load_data()
        self.tokenize()
        self.bucket_by_token_count()
        self.split_bucket()
        self.cluster()
        self.retrieve()
        self.reconcile()
        self.dump()
        print("Parsing done. [Time taken: {!s}]".format(datetime.now() - start_time))

    def clean(self, log):
        dataset = self.dataset
        rex = self.rex
        delimiter = self.delimiter
        for crex in rex:
            log = re.sub(crex, "<*>", log)
        for de in delimiter:
            log = re.sub(de, "", log)
        if dataset == "HealthApp":
            log = re.sub(":", ": ", log)
            log = re.sub("=", "= ", log)
            log = re.sub("\|", "| ", log)
        if dataset == "Android":
            log = re.sub("\(", "( ", log)
            log = re.sub("\)", ") ", log)
        if dataset == "Android":
            # log = re.sub(":", ": ", log)
            log = re.sub(" - ", "-", log)
            log = re.sub("=", "= ", log)
            log = re.sub(",", ", ", log)
        if dataset == "HPC":
            log = re.sub("=", "= ", log)
            log = re.sub("-", "- ", log)
            log = re.sub(":", ": ", log)
        if dataset == "BGL":
            log = re.sub("=", "= ", log)
            log = re.sub("\.\.", ".. ", log)
            log = re.sub("\(", "( ", log)
            log = re.sub("\)", ") ", log)
        if dataset == "Hadoop":
            log = re.sub("_", "_ ", log)
            log = re.sub(":", ": ", log)
            log = re.sub("=", "= ", log)
            log = re.sub("\(", "( ", log)
            log = re.sub("\)", ") ", log)
        if dataset == "HDFS":
            log = re.sub(":", ": ", log)
        if dataset == "Linux":
            log = re.sub("=:", "=", log)
            log = re.sub("=", "= ", log)
        if dataset == "Spark":
            log = re.sub(":", ": ", log)
        if dataset == "Thunderbird":
            log = re.sub(":", ": ", log)
            log = re.sub("=", "= ", log)
        if dataset == "Windows":
            log = re.sub(":", ": ", log)
            log = re.sub("=", "= ", log)
            log = re.sub("\[", "[ ", log)
            log = re.sub("]", "] ", log)
        if dataset == "OpenSSH":
            log = re.sub("=", "= ", log)
        if dataset == "Zookeeper":
            log = re.sub(":", ": ", log)
            log = re.sub("=", "= ", log)
            log = re.sub(",", ", ", log)
        log = re.sub(" +", " ", log)
        return log

    def log_to_dataframe(self, log_file, regex, headers):
        log_messages = []
        linecount = 0
        with open(log_file, "r") as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    print("Skip line: " + line)
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, "LineId", None)
        logdf["LineId"] = [i + 1 for i in range(linecount)]
        return logdf

    def generate_logformat_regex(self, logformat):
        headers = []
        splitters = re.split(r"(<[^<>]+>)", logformat)
        regex = ""
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(r" +", r"\s+", splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip("<").strip(">")
                regex += "(?P<%s>.*?)" % header
                headers.append(header)
        regex = re.compile("^" + regex + "$")
        return headers, regex

    def load_data(self):
        if self.logname_extension == ".csv":
            df_log = pd.read_csv(os.path.join(self.input_file, self.logname))
            self.df_log = df_log[["LineId", "Content"]]
            self.df_log["Content_"] = self.df_log["Content"].apply(
                lambda x: pd.Series(self.clean(x))
            )
        elif self.logname_extension == ".log":
            headers, regex = self.generate_logformat_regex(self.logformat)
            self.df_log = self.log_to_dataframe(
                os.path.join(self.input_file, self.logname), regex, headers
            )
            self.df_log["Content_"] = self.df_log["Content"].apply(
                lambda x: pd.Series(self.clean(x))
            )

    def split_to_log_token(self, log):
        log_token = log.split()
        token_count = len(log_token)
        return log_token, token_count

    def tokenize(self):
        self.df_log[["log_token", "token_count"]] = self.df_log["Content_"].apply(
            lambda x: pd.Series(self.split_to_log_token(x))
        )

    def bucket_by_token_count(self):
        for idx, token_count in self.df_log["token_count"].items():
            if token_count not in self.token_count_buckets:
                self.token_count_buckets[token_count][0] = [idx]
            else:
                self.token_count_buckets[token_count][0].append(idx)

    @lru_cache(maxsize=None)
    def compute_distance_between_idx1_and_idx2(self, idx1, idx2):
        log_token1 = self.df_log["log_token"].loc[idx1]
        log_token2 = self.df_log["log_token"].loc[idx2]
        last_len = max(len(log_token1), len(log_token2))
        log_token1 += [""] * (last_len - len(log_token1))
        log_token2 += [""] * (last_len - len(log_token2))
        return sum(self.compute_distance_between_token(t1, t2) for t1, t2 in zip(log_token1, log_token2))

    @lru_cache(maxsize=None)
    def compute_distance_between_token(self, token1, token2):
        # CHINESE_WEIGHT = 2
        # OTHER_WEIGHT = 1
        # @lru_cache(maxsize=None)
        # def char_type(char):
        #     if re.match("[\u4e00-\u9fff]", char):
        #         return CHINESE_WEIGHT
        #     else:
        #         return OTHER_WEIGHT
        return 1 if token1 != token2 else 0

    def jaccard_similarity(self, idx1, idx2):
        s1 = set(self.df_log["log_token"].loc[idx1])
        s2 = set(self.df_log["log_token"].loc[idx2])
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def entropy(self, idx):
        s = self.df_log["Content_"].loc[idx]
        if isinstance(s, int):
            s = str(s)
        prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
        return entropy

    def first_token(self, idx):
        s = self.df_log["Content_"].loc[idx]
        return s.split()[0] if s and " " in s else s

    def process_key(self, key):
        def entropy_sampling(bucket, k=10, n_layers=5):
            entropy_and_token = [(item, self.entropy(item), self.first_token(item)) for item in bucket]
            entropy_and_token.sort(key=lambda x: x[1], reverse=True)
            layers = np.array_split(entropy_and_token, n_layers)
            selected_samples = set()
            tokens_selected = set()
            for layer in layers:
                for item in layer:
                    if item[2] not in tokens_selected or len(tokens_selected) >= k:
                        selected_samples.add(item[0])
                        tokens_selected.add(item[2])
                    if len(selected_samples) >= k:
                        break
                if len(selected_samples) >= k:
                    break
            return [int(idx) for idx in selected_samples]

        def check(selected_samples):
            len_selected_samples = len(selected_samples)
            visit = [False] * len_selected_samples
            merge = []
            for i in range(len_selected_samples):
                if visit[i]:
                    continue
                visit[i] = True
                for j in range(i + 1, len_selected_samples):
                    if visit[j]:
                        continue
                    visit[j] = True
                    if (
                        self.jaccard_similarity(selected_samples[i], selected_samples[j])
                        > self.jaccard_similarity_threshold
                    ):
                        merge.append(selected_samples[j])
            return [s for s in selected_samples if s not in merge]

        value = self.token_count_buckets[key]
        bucket = value[0][:]
        selected_samples = entropy_sampling(bucket, self.k)
        selected_samples = check(selected_samples)
        value[1] = selected_samples
        value[2] = [item for item in bucket if item not in selected_samples]
        return value

    def split_bucket(self):
        for key in self.token_count_buckets.keys():
            self.token_count_buckets[key] = self.process_key(key)
        # check cluster center and other samples code
        # data_path = os.path.join(self.output_file, "detail/")
        # if not os.path.isdir(data_path):
        #     os.makedirs(data_path)
        # for key in self.token_count_buckets.keys():
        #     value = self.token_count_buckets[key]
        #     key_data = [
        #         (
        #             "center" if v in value[1] else "global",
        #             self.df_log["log_token"].loc[v],
        #         )
        #         for v in value[0]
        #     ]
        #     key_data = sorted(key_data, key=lambda x: x[1])
        #     key_data_df = pd.DataFrame(key_data)
        #     key_data_df.to_csv(data_path + f"{self.dataset}_{key}.csv", index=False)

    def cluster(self):
        for key in self.token_count_buckets.keys():
            value = self.token_count_buckets[key]
            for tidx in value[1]:
                self.token_count_tidx_clusters[(key, tidx)] = [tidx]
            for idx in value[2]:
                min_distance = float("inf")
                min_tidx = value[1][0]
                for tidx in value[1]:
                    distance = self.compute_distance_between_idx1_and_idx2(
                        idx, tidx
                    )
                    if distance < min_distance:
                        min_distance = distance
                        min_tidx = tidx
                self.token_count_tidx_clusters[(key, min_tidx)].append(idx)

    def exclude_digits(self, string):
        pattern = r"\d"
        digits = re.findall(pattern, string)
        if len(digits) == 0:
            return False
        return len(digits) / len(string) >= self.digit_percent

    def retrieve(self):
        for key in self.token_count_tidx_clusters.keys():
            value = self.token_count_tidx_clusters[key]
            data = {val: self.df_log["log_token"].loc[val] for val in value}
            vectorizer = Vectorizer().fit(data)
            varas = set()
            for da in data.keys():
                diff = vectorizer.get_difference_from_common_subsequences(da)
                for di in diff:
                    if di[0] in varas:
                        continue
                    p = di[1]
                    e, stat = vectorizer.calculate_entropy_at_position(p)
                    if e < self.entropy_theshold:
                        continue
                    else:
                        for s in stat:
                            varas.add(s[0])
                    
                    # stats = vectorizer.get_tokens_stats_at_position_all_docs(p)
                    # if len(stats) < min(self.type_threshold, len(data.keys()) + 1):
                    #     continue
                    # else:
                    #     observed = [stat[2] for stat in stats]
                    #     range_val = max(observed) - min(observed)
                    #     if range_val > self.freq_threshold:
                    #         continue
                    #     else:
                    #         for stat in stats:
                    #             varas.add(stat[0])

            # print(varas)
            def replace_token(log_token):
                handle_token = []
                for token in log_token:
                    if self.exclude_digits(token):
                        handle_token.append("<*>")
                    elif bool("<*>" in token):
                        handle_token.append("<*>")
                    elif token in varas:
                        handle_token.append("<*>")
                    else:
                        handle_token.append(token)
                return tuple(handle_token)

            for da in data.keys():
                self.df_log.at[da, "log_token"] = replace_token(
                    self.df_log.loc[da, "log_token"]
                )

    def reconcile(self):
        template_list = self.df_log["log_token"].tolist()
        self.template_collection = set(template_list)

    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex:
            return []
        template_regex = re.sub(r"([^A-Za-z0-9])", r"\\\1", template_regex)
        template_regex = re.sub(r"\\ +", r"\\s+", template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = (
            list(parameter_list)
            if isinstance(parameter_list, tuple)
            else [parameter_list]
        )
        return parameter_list


    def dump(self):
        if not os.path.exists(self.output_file):
            os.makedirs(self.output_file)

        templateL = [0] * self.df_log.shape[0]
        idL = [0] * self.df_log.shape[0]
        df_events = []

        for template_token in self.template_collection:
            template_str = " ".join(template_token)
            template_id = hashlib.md5(template_str.encode("utf-8")).hexdigest()[0:8]

            for idx in range(len(idL)):
                if self.df_log["log_token"].loc[idx] == template_token:
                    templateL[idx] = template_str
                    idL[idx] = template_id

            df_events.append([template_id, template_str, templateL.count(template_str)])

        self.df_log["EventId"] = idL
        self.df_log["EventTemplate"] = templateL
        self.df_log.drop(["Content_", "log_token", "token_count"], axis=1, inplace=True)

        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)

        self.df_log.to_csv(
            os.path.join(self.output_file, self.logname + "_structured.csv"), 
            index=False
        )

        df_events.sort(key=lambda x: x[1], reverse=True)
        
        df_event = pd.DataFrame(df_events, columns=["EventId", "EventTemplate", "Occurrences"])
        df_event.to_csv(
            os.path.join(self.output_file, self.logname + "_templates.csv"), 
            index=False, 
            columns=["EventId", "EventTemplate", "Occurrences"]
        )
