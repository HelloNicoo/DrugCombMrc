# -*- coding: utf-8 -*-
# @Time    : 2022/9/23 11:36
# @Author  : sylviazz
# @FileName: convert2final
import csv
import json
import os
from typing import List, Dict

import jsonlines
import numpy as np


def average_pairwise_distance(spans: List[Dict]) -> float:
    '''This function calculates the average distance between pairs of spans in a relation, which may be a useful
    bucketing attribute for error analysis.

    Args:
        spans: List of spans (each represented as a dictionary)

    Returns:
        average pairwise distance between spans in the provided list
    '''
    distances = []
    for i in range(len(spans)):
        for j in range(i+1, len(spans)):
            span_distance = spans[j]["token_start"] > spans[i]["token_end"]
            if span_distance >= 0:
                distances.append(span_distance)
            else:
                span_distance = spans[i]["token_start"] - spans[j]["token_end"]
                assert span_distance >= 0
                distances.append(span_distance)
    assert len(distances) >= 1
    return np.mean(distances)
class ErrorAnalysisAttributes:
    def __init__(self, dataset_row: Dict, full_document: Dict, prediction: int):
        self.row_id = dataset_row["row_id"]
        self.sentence = full_document["sentence"]
        self.paragraph = full_document["paragraph"]
        self.sentence_length = len(full_document["sentence"].split())
        self.paragraph_length = len(full_document["paragraph"].split())
        spans = full_document["spans"]

        self.entities = [span["text"] for span in spans]
        spans_in_relation = [spans[idx] for idx in dataset_row["drug_indices"]]

        self.num_spans_in_ground_truth_relation = len(spans_in_relation)
        self.avg_span_distance_in_ground_truth_relation  = average_pairwise_distance(spans_in_relation)
        self.ground_truth_label = dataset_row["target"]
        self.predicted_label = prediction

    def get_row(self):
        return [self.row_id, self.sentence, self.entities, self.paragraph, self.ground_truth_label, self.predicted_label, self.sentence_length, self.paragraph_length, self.num_spans_in_ground_truth_relation, self.avg_span_distance_in_ground_truth_relation]
def write_error_analysis_file(dataset: List[Dict], test_data_raw: List[Dict], test_row_ids: List[str], test_predictions: List[int], fname: str):
    '''Write out all test set rows and their predictions to a TSV file, which will let us connect easily with ExplainaBoard.

    Args:
        dataset: List of row dictionaries representing the test dataset
        test_row_ids: List of row identifiers in the test set
        test_predictions: List of integer predictions corresponding to the test rows
        fname: String file to write the TSV output
    '''
    test_data_raw = {doc["doc_id"]:doc for doc in test_data_raw}
    row_predictions = dict(zip(test_row_ids, test_predictions))

    header = [
                "Row ID",
                "Sentence",
                "Entities",
                "Paragraph",
                "True Relation Label",
                "Predicted Relation Label",
                "Sentence Length",
                "Paragraph Length",
                "Number of Entities in Ground Truth Relation",
                "Average Distance of Entities"
            ]

    with open(fname, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(header)
        for dataset_row in dataset:
            prediction = row_predictions[dataset_row["row_id"]]
            doc_id = json.loads(dataset_row["row_id"])["doc_id"]
            full_document = test_data_raw[doc_id]
            error_analysis_attributes = ErrorAnalysisAttributes(dataset_row, full_document, prediction)
            tsv_writer.writerow(error_analysis_attributes.get_row())
    print(f"Wrote error analysis file to {fname}")

def write_jsonl(data: List[Dict], fname: str):
    with open(fname, 'wb') as fp:
        writer = jsonlines.Writer(fp)
        writer.write_all(data)
    writer.close()
    print(f"Wrote {len(data)} json lines to {fname}")
def filter_overloaded_predictions(preds):
    def do_filtering(d):
        # our filtering algorithm:
        #   1. we assume each sentence gets predictions for each subset of drugs in the sentence
        #   2. we assume these are too many and probably conflicting predictions, so they need to be filtered
        #   3. we use a very simple (greedy) heuristic in which we look for the biggest (by drug-count) combination,
        #       that has a non NO_COMB prediction, and we take it.
        #   4. we try to get as large a coverage (on the drugs) as possible while maintaining
        #       a minimalistic list of predictions as possible, so we do this repeatedly on the remaining drugs
        out = d[0]
        for j, e in d:
            if e["relation_label"]:
                out = (j, e)
                break
        send_to_further = []
        for j, e in d:
            # store all non intersecting predictions with the chosen one, so we can repeat the filtering process on them
            if len(set(out[1]["drug_idxs"]).intersection(set(e["drug_idxs"]))) == 0:
                send_to_further.append((j, e))
        return [out] + (do_filtering(send_to_further) if send_to_further else [])

    # we sort here so it would be easier to group by sentence,
    #   and to have the high-drug-count examples first for the filtering process
    sorted_test = sorted(enumerate(preds), key=lambda x: (x[1]["doc_id"], len(x[1]["drug_idxs"]), str(x[1]["drug_idxs"])), reverse=True)
    test_pred_output = "test_pred.jsonl"
    write_jsonl(sorted_test, test_pred_output)
    # aggregate predictions by the sentence and filter each prediction group
    final_test = []
    doc = []
    for i, (original_idx, example) in enumerate(sorted_test):
        doc.append((original_idx, example))
        # reached the last one in the list, or last one for this sentence
        if (i + 1 == len(sorted_test)) or (sorted_test[i + 1][1]["doc_id"] != example["doc_id"]):
            final_test.extend(do_filtering(doc))
            doc = []
    # reorder the filtered list according to original indices, and get rid of the these indices
    return [x[1] for x in sorted(final_test, key=lambda x: x[0])]
# final_data = []
# file_name = '/home/zpq/PythonProject/drug_combo_spert_final1/epoch_5'
# with open(file_name, 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         temp_dict = {}
#         data = json.loads(line)
#         temp_dict['doc_id'] = data['doc_id']
#         temp_dict['drug_idxs'] = data['drug_idx']
#         temp_dict['relation_label'] = data['pred']
#         final_data.append(temp_dict)
# fixed_test = filter_overloaded_predictions(final_data)
# print(fixed_test)
# os.makedirs("outputs", exist_ok=True)
# test_output = os.path.join("outputs", "final_predictions.jsonl")
# write_jsonl(fixed_test, test_output)
# write_error_analysis_file(test_data, test_data_raw, test_row_ids, test_predictions, os.path.join("outputs", args.model_name + ".tsv"))
