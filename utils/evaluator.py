# ----------------------------------------------------------------------------------
# Code License Notice
# ----------------------------------------------------------------------------------
# - Author Contact: wei.zhang, zwpride@buaa.edu.cn
#
# - Inherits the basic evaluation algorithm and basic framework of LOGAPI.
# - Original code is enhanced to improve error and bug output capabilities.
# ----------------------------------------------------------------------------------

import pandas as pd
from scipy.special import comb

def evaluate(groundtruth, parsedresult, debug=False):
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult)
    # Remove invalid groundtruth event Ids
    non_empty_log_ids = df_groundtruth[~df_groundtruth["EventId"].isnull()].index
    df_groundtruth = df_groundtruth.loc[non_empty_log_ids]
    df_parsedlog = df_parsedlog.loc[non_empty_log_ids]
    (precision, recall, f_measure, accuracy) = get_accuracy(
        df_groundtruth["EventId"], df_parsedlog["EventId"], debug
    )
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1_measure: {f_measure:.4f}, Parsing_Accuracy: {accuracy:.4f}")

    # print(
    #     "Precision: {:.4f}, Recall: {:.4f}, F1_measure: {:.4f}, Parsing_Accuracy: {:.4f}".format(
    #         precision, recall, f_measure, accuracy
    #     )
    # )
    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1_measure": f_measure,
        "Accuracy": accuracy,
    }
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
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()

        # Detect parsing errors
        if series_groundtruth_logId_valuecounts.size != 1:
            if debug:
                print(
                    "Parsing Error: parsed_eventId = {}, corresponds to multiple groundtruth_eventIds = {}, affecting log entries = {}".format(
                        parsed_eventId,
                        series_groundtruth_logId_valuecounts.index.tolist(),
                        logIds.size,
                    )
                )
            continue

        groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
        if (
            logIds.size
            != series_groundtruth[series_groundtruth == groundtruth_eventId].size
        ):
            if debug:
                print(
                    "Parsing Error: parsed_eventId = {}, does not completely match groundtruth_eventId = {}, affecting log entries = {}".format(
                        parsed_eventId, groundtruth_eventId, logIds.size
                    )
                )
            continue

        # If no errors, calculate accurate pairs
        accurate_events += logIds.size
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += comb(count, 2)
    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy