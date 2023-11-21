import pandas as pd
from scipy.special import comb
from nltk.metrics.distance import edit_distance
import numpy as np
from tqdm import tqdm

def evaluate(groundtruth, parsedresult, debug=False):
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult)

    non_empty_log_ids = df_groundtruth[~df_groundtruth["EventId"].isnull()].index
    df_groundtruth = df_groundtruth.loc[non_empty_log_ids]
    df_parsedlog = df_parsedlog.loc[non_empty_log_ids]

    metrics = {}
    metrics.update(get_accuracy(df_groundtruth["EventId"], df_parsedlog["EventId"], debug))
    metrics.update(get_GA(df_groundtruth, df_parsedlog, debug))
    metrics.update(get_editdistance(df_groundtruth, df_parsedlog, debug))
    metrics.update(get_WLA(df_groundtruth, df_parsedlog))

    print(
        "Precision: {:.6f}, Recall: {:.6f}, F1_measure: {:.6f}, Accuracy: {:.6f}, GA: {:.6f}, ED: {:.6f}, ED_: {:.6f}, WLA: {:.6f}".format(
            metrics['Precision'], 
            metrics['Recall'], 
            metrics['F1_measure'], 
            metrics['Accuracy'], 
            metrics['GA'], 
            metrics['ED'], 
            metrics['ED_'],
            metrics['WLA']
        )
    )

    return metrics

def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0
    for parsed_EventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_EventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_EventIds = (
            parsed_EventId,
            series_groundtruth_logId_valuecounts.index.tolist(),
        )
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_EventId = series_groundtruth_logId_valuecounts.index[0]
            if (
                logIds.size
                == series_groundtruth[series_groundtruth == groundtruth_EventId].size
            ):
                accurate_events += logIds.size
                error = False
        if error or debug:
            print(
                "(parsed_EventId, groundtruth_EventId) =",
                error_EventIds,
                "failed",
                logIds.size,
                "messages",
            )
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    metrics = {
        "Precision": precision, 
        "Recall": recall,
        "F1_measure": f_measure, 
        "Accuracy": accuracy
    }
    
    return metrics

def get_GA(ground_list, parsedlog, debug=False):
    correct = 0
    parserdtemplate = parsedlog["EventId"]
    grouped_parsed = parserdtemplate.groupby(parserdtemplate).apply(
        lambda x: x.index.tolist()
    )
    groundtruthtemplate = ground_list["EventId"]
    grouped_groundtruth = groundtruthtemplate.groupby(groundtruthtemplate).apply(
        lambda x: x.index.tolist()
    )
    grouped_groundtruth = grouped_groundtruth.values.tolist()
    for parsedset in grouped_parsed.values:
        if parsedset in grouped_groundtruth:
            correct += len(parsedset)
    GA = correct / len(parsedlog)
    return {"GA": GA}

def get_editdistance(groundtruth, parsedlog, debug=False):
    edit_distance_result = []
    for i, j in tqdm(
        zip(
            np.array(groundtruth.EventTemplate.values, dtype="str"),
            np.array(parsedlog.EventTemplate.values, dtype="str"),
        ),
        desc="Calculating Edit_distance...",
    ):
        edit_distance_result.append(edit_distance(i, j))
    edit_distance_result_mean = np.mean(edit_distance_result)
    edit_distance_result_std = np.std(edit_distance_result)

    return {"ED": edit_distance_result_mean, "ED_": edit_distance_result_std}

def get_WLA(groundtruth, parsedlog, debug=False):
    total_words = 0
    correct_words = 0
    for i in range(len(groundtruth)):
        groundtruth_event = groundtruth.loc[i, 'EventTemplate'].split()
        parsedlog_event = parsedlog.loc[i, 'EventTemplate'].split()
        total_words += len(groundtruth_event)
        correct_words += sum([g == p for g, p in zip(groundtruth_event, parsedlog_event)])
    WLA = correct_words / total_words if total_words > 0 else 0
    return {"WLA": WLA}