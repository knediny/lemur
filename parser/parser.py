import regex as re
import os
import hashlib
import pandas as pd
from datetime import datetime
from collections import defaultdict
import random
from functools import lru_cache
from parser.vectorizer import Vectorizer

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
        n_candidate=2,
        k=2,
        freq_threshold=2,
        type_threshold=0.3,
        digit_percent=0.3,
        jaccard_similarity_threshold=0.7,
        first_weight=10,
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.rex = rex
        self.delimiter = delimiter
        self.logformat = log_format
        self.dataset = dataset
        self.logname = None
        self.logname_extension = None
        self.n_candidate = n_candidate
        self.k = k
        self.freq_threshold = freq_threshold
        self.type_threshold = type_threshold
        self.jaccard_similarity_threshold = jaccard_similarity_threshold
        self.first_weight = first_weight
        self.digit_percent = digit_percent
        self.keep_para = keep_para
        self.df_log = None
        self.token_count_groups = defaultdict(lambda: [None, None, None])
        self.token_count_tidx_clusters = defaultdict(list)
        self.template_collection = set()

    def parse(self, logname):
        start_time = datetime.now()
        print("Parsing file: " + os.path.join(self.input_file, logname))
        self.logname = logname
        _, self.logname_extension = os.path.splitext(os.path.basename(self.logname))
        self.load_data()
        self.tokenize()
        self.group_by_token_count()
        self.split_group()
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

    def group_by_token_count(self):
        for idx, token_count in self.df_log["token_count"].iteritems():
            if token_count not in self.token_count_groups:
                self.token_count_groups[token_count][0] = [idx]
            else:
                self.token_count_groups[token_count][0].append(idx)

    def compute_edit_distance_between_c_and_all_t(self, candidate_set, T):
        cs_ts_distance = []
        for c in candidate_set:
            c_ts_distance = 0
            for t in T:
                c_ts_distance += self.compute_edit_distance_between_idx1_and_idx2(c, t)
            cs_ts_distance.append(c_ts_distance)
        return cs_ts_distance

    @lru_cache(maxsize=None)
    def compute_edit_distance_between_idx1_and_idx2(self, idx1, idx2):
        log_token1 = self.df_log["log_token"].loc[idx1]
        log_token2 = self.df_log["log_token"].loc[idx2]
        last_len = max(len(log_token1), len(log_token2))
        log_token1 += [""] * (last_len - len(log_token1))
        log_token2 += [""] * (last_len - len(log_token2))
        return (self.first_weight if log_token1[0] != log_token2[0] else 1) + sum(
            self.compute_edit_distance_between_token(t1, t2)
            for t1, t2 in zip(log_token1, log_token2)
        )

    @lru_cache(maxsize=None)
    def compute_edit_distance_between_token(self, token1, token2):
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

    def process_key(self, key):
        def inspector_sample_log(group, k, n_candidate=64):
            T = []
            for _ in range(k):
                candidate_set = random.sample(group, min(len(group), n_candidate))
                if len(candidate_set) == 0:
                    break
                if len(T) == 0:
                    selected = random.choice(candidate_set)
                    T.append(selected)
                    group.remove(selected)
                    continue
                all_c_all_t_distance = self.compute_edit_distance_between_c_and_all_t(
                    candidate_set, T
                )
                best_candidate = max(
                    range(len(all_c_all_t_distance)),
                    key=all_c_all_t_distance.__getitem__,
                )
                selected = candidate_set[best_candidate]
                T.append(selected)
                group.remove(selected)
            return T, group

        def check(T):
            len_T = len(T)
            visit_T = [False] * len_T
            details = []
            for i in range(len_T):
                if visit_T[i]:
                    continue
                visit_T[i] = True
                for j in range(i + 1, len_T):
                    if visit_T[j]:
                        continue
                    visit_T[j] = True
                    if (
                        self.jaccard_similarity(T[i], T[j])
                        > self.jaccard_similarity_threshold
                    ):
                        details.append(T[j])
            return details

        def renew(T, group):
            details = check(T)
            T = [t for t in T if t not in details]
            group.extend(details)
            return T, group

        value = self.token_count_groups[key]
        group = value[0][:]
        T, group = inspector_sample_log(group, self.k, self.n_candidate)
        T, group = renew(T, group)
        value[1] = T
        value[2] = group
        return value

    def split_group(self):
        for key in self.token_count_groups.keys():
            self.token_count_groups[key] = self.process_key(key)
        data_path = os.path.join(self.output_file, "detail/")
        if not os.path.isdir(data_path):
            os.makedirs(data_path)
        for key in self.token_count_groups.keys():
            value = self.token_count_groups[key]
            key_data = [
                (
                    "center" if v in value[1] else "global",
                    self.df_log["log_token"].loc[v],
                )
                for v in value[0]
            ]
            key_data = sorted(key_data, key=lambda x: x[1])
            key_data_df = pd.DataFrame(key_data)
            key_data_df.to_csv(data_path + f"{self.dataset}_{key}.csv", index=False)

    def cluster(self):
        for key in self.token_count_groups.keys():
            value = self.token_count_groups[key]
            for tidx in value[1]:
                self.token_count_tidx_clusters[(key, tidx)] = [tidx]
            for idx in value[2]:
                min_distance = float("inf")
                min_tidx = value[1][0]
                for tidx in value[1]:
                    distance = self.compute_edit_distance_between_idx1_and_idx2(
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
                    stats = vectorizer.get_tokens_stats_at_position_all_docs(p)
                    if len(stats) < min(self.type_threshold, len(data.keys()) + 1):
                        continue
                    else:
                        observed = [stat[2] for stat in stats]
                        range_val = max(observed) - min(observed)
                        if range_val > self.freq_threshold:
                            continue
                        else:
                            for stat in stats:
                                varas.add(stat[0])

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
        if not os.path.isdir(self.output_file):
            os.makedirs(self.output_file)
        templateL = [0] * self.df_log.shape[0]
        idL = [0] * self.df_log.shape[0]
        df_events = []
        for template_token in self.template_collection:
            template_str = " ".join(template_token)
            template_id = hashlib.md5(template_str.encode("utf-8")).hexdigest()[0:8]
            oc = 0
            for idx in range(len(idL)):
                if self.df_log["log_token"].loc[idx] == template_token:
                    templateL[idx] = template_str
                    idL[idx] = template_id
                    oc += 1
            df_events.append([template_id, template_str, oc])
        df_event = pd.DataFrame(
            df_events, columns=["EventId", "EventTemplate", "Occurrences"]
        )

        self.df_log["EventId"] = idL
        self.df_log["EventTemplate"] = templateL
        self.df_log.drop("Content_", axis=1, inplace=True)
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(
                self.get_parameter_list, axis=1
            )
        self.df_log.to_csv(
            os.path.join(self.output_file, self.logname + "_structured.csv"),
            index=False,
        )

        occ_dict = dict(self.df_log["EventTemplate"].value_counts())
        df_event = pd.DataFrame()
        df_event["EventTemplate"] = self.df_log["EventTemplate"].unique()
        df_event["EventId"] = df_event["EventTemplate"].map(
            lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[0:8]
        )
        df_event["Occurrences"] = df_event["EventTemplate"].map(occ_dict)
        df_event.to_csv(
            os.path.join(self.output_file, self.logname + "_templates.csv"),
            index=False,
            columns=["EventId", "EventTemplate", "Occurrences"],
        )
