import re
import collections
import numpy as np

from itertools import groupby

from llm_access import *
import time

from sklearn.cluster import KMeans

import pickle



DOCUMENT_TITLE_REGEX=r"(.+)\s+n?º\s*([0-9\.]+)[,\s]+de\s+([0-9]+)º?\s+de\s+(\w+)\s+de\s+([0-9]{4})"

def return_unique_references(questions, which_field="formatted_references"):

    all_references = {}
    
    questions_without_annotated_references = []
    
    for question in questions:
        if which_field not in question:
            print(f"{question} ― No {which_field} field.\n")
            questions_without_annotated_references.append(question['question_number'])
        else:
            if question[which_field] is not None:
                for each_reference in question[which_field]:
                    if 'título' not in each_reference:
                        print(f"{question['question_number']} = {each_reference}\n")
                    else:
                        if each_reference['título'] not in all_references:
                            all_references[each_reference['título']] = []
                
                        all_references[each_reference['título']].append(question['question_number'])
            else:
                print(f"{question} ― Empty {which_field} field.\n")
                questions_without_annotated_references.append(question['question_number'])

    return all_references, questions_without_annotated_references



def normalize_document_title(which_title):
    m = re.match(DOCUMENT_TITLE_REGEX, which_title.lower())

    if m is None:
        normalized_title = which_title.lower()
    else:
        normalized_title = "_".join(m.groups())

    return normalized_title



NAME_SPLITTER_BY_NUMBER_REGEX=r"([nN]?º\s*([0-9\.]+))"
MAME_SPLITTER_BY_YEAR_REGEX=r",?(\sde\s[0-9]{4})"
NAME_SPLITTER_BY_DATE_REGEX=r",?(\sde\s.+[0-9]{4})"

def split_document_name(document_name):
    name_parts = re.split(NAME_SPLITTER_BY_NUMBER_REGEX, document_name)

    if len(name_parts) == 1:
        name_parts = re.split(NAME_SPLITTER_BY_DATE_REGEX, document_name)

        if len(name_parts) == 1:
            name_parts = re.split(MAME_SPLITTER_BY_YEAR_REGEX, document_name)

            if len(name_parts) == 1:
                name_parts += ['', '']

        name_parts = name_parts[0:1] + ['', ''] + name_parts[1:-1]
    else:
        name_parts[2] = name_parts[2].replace('.', '')

    return name_parts



def get_tokens(s):
  if not s: return []

  reduced_s = re.sub(r"\be\b|\bda\b|\bdo\b|\bde\b", ' ', s)

  return re.split(r"\s+|-", reduced_s.lower())



def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    
    # print(f"gold_toks={gold_toks}, pred_toks={pred_toks}")
    
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    
    if len(gold_toks) == 0 or len(pred_toks) == 0 or num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1 



class legalDocumentsMatcher:

    def __init__(self,
                 documents_reference_filename):

        self.documents_reference_filename = documents_reference_filename

        with open(documents_reference_filename, "rb") as input_file:
            self.reference_titles = pickle.load(input_file)
    
        self.reference_documents_parts = [split_document_name(which_document) for which_document in self.reference_titles]


    
    def get_best_match_for_parts(self, title_parts):
        
        title_f1 = np.array([compute_f1(title_parts[0], 
                             reference_doc[0]) for reference_doc in self.reference_documents_parts])

        number_f1 = np.array([compute_f1(title_parts[2], 
                                         reference_doc[2]) for reference_doc in self.reference_documents_parts])
    
        
        date_f1 = np.array([compute_f1(title_parts[3], 
                                       reference_doc[3]) for reference_doc in self.reference_documents_parts])
    
        final_f1 = title_f1 + number_f1 + date_f1
        reverse_ordered_final_f1 = np.argsort(final_f1)[::-1]
    
        for i in range(reverse_ordered_final_f1.shape[0]):
            if final_f1[reverse_ordered_final_f1[0]] != final_f1[reverse_ordered_final_f1[i]]:
                break

        return {"title_f1": title_f1[reverse_ordered_final_f1[0]],
                "number_f1": number_f1[reverse_ordered_final_f1[0]],
                "date_f1": date_f1[reverse_ordered_final_f1[0]],
                "best_matches": [self.reference_titles[j] for j in reverse_ordered_final_f1[:i]]}

    
    
    def get_best_match(self, document_title):

        title_parts = split_document_name(document_title.split(".txt")[0])

        return self.get_best_match_for_parts(title_parts)
    


def execute_references_match(matcher, references_to_match):

    extracted_documents_parts = [split_document_name(which_document.split(".txt")[0]) for which_document in references_to_match]

    extracted_documents_matches = []

    for extracted_id, extracted_doc in enumerate(extracted_documents_parts):

        result = matcher.get_best_match_for_parts(extracted_doc)

        extracted_documents_matches.append({"extracted_title": references_to_match[extracted_id],
                                            "title_f1": result["title_f1"],
                                            "number_f1": result["number_f1"],
                                            "date_f1": result["date_f1"],
                                            "best_matches": result["best_matches"],}) 

    exact_matches = []
    multiple_matches = []

    for doc_match in extracted_documents_matches:
        if len(doc_match['best_matches']) == 1:
            exact_matches.append(doc_match)
        else:
            multiple_matches.append(doc_match)

    print(f"Number of exact matches={len(exact_matches)}; number of multiple matches={len(multiple_matches)}")

    return {"extracted_references_matches": extracted_documents_matches,
            "exact_matches": exact_matches,
            "multiple_matches": multiple_matches}



def get_llm_interface(api_keys_file="/work/api_keys_20240427.json",
                      which_llm_provider='groq',
                      which_llm=GROQ_LLAMA3_3_70B_MODEL):

    llm_key = json.load(open(api_keys_file))[which_llm_provider]

    if which_llm_provider == 'groq':
        llm_interface = groq_access(llm_key, which_llm)
    else:
        raise NotImplementedError(f"{which_llm_provider} LLM provider not supported in this version")

    return llm_interface



DOCUMENTS_LIST_MAX_LENGTH=20

def cluster_legal_references_by_number(documents_list, 
                                       max_cluster_size_reference=DOCUMENTS_LIST_MAX_LENGTH,
                                       verbose=True):

    splitted_list = [split_document_name(which_doc) for which_doc in documents_list]

    # Get document numbers

    converted = []

    for element in splitted_list:
        if element[2] == '':
            converted.append(-1)
        else:
            converted.append(int(element[2].replace('.', '')))

    kmeans = KMeans(n_clusters=len(converted) // max_cluster_size_reference + 1, 
                    random_state=0, 
                    n_init=10).fit(np.array(converted).reshape(-1, 1))

    clusters = {}
    for i, label in enumerate(kmeans.labels_):
        clusters.setdefault(label, []).append(documents_list[i])

    cluster_sizes = []

    for cluster_id, cluster in clusters.items():
        cluster_sizes.append(len(cluster))

        if verbose:
            print(f"Cluster {cluster_id}: {cluster}\n")

    return clusters, cluster_sizes



def deduplicate_legal_references_list(legal_references_list, 
                                      llm_interface):

    unique_references_by_initial_letter = {key: list(group) for key, group in groupby(legal_references_list, key=lambda x: x[0])}

    group_counts = {}

    for key, value in unique_references_by_initial_letter.items():
        group_counts[key] = len(value)

    print(group_counts)

    processing_start = time.time()

    deduplicated_reference_titles = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for initial_letter, references in unique_references_by_initial_letter.items():
        print("\n*****************************************")
        print(f"Processing titles starting with {initial_letter}\n")
        print("*****************************************\n")

        if len(references) > 1:
            if len(references) > DOCUMENTS_LIST_MAX_LENGTH:
                print(f">> Titles list longer than {DOCUMENTS_LIST_MAX_LENGTH} elements. Clustering by document number to process.")

                references_clusters, clusters_sizes = cluster_legal_references_by_number(references)
            
                references_lists = list(references_clusters.values())

                print(f">> Resulting clusters sizes: {clusters_sizes}")
            else:
                references_lists = [references]

            for references_cluster in references_lists:

                if len(references_cluster) > 1:
                    result = legal_refereces_titles_deduplication(llm_interface, references_cluster)
                
                    print(f">> Original titles count={len(references_cluster)}; deduplicated titles count={len(result['deduplicated-references'])}")
                
                    deduplicated_reference_titles += result["deduplicated-references"]
                
                    total_prompt_tokens += result['prompt_tokens']
                    total_completion_tokens += result['completion_tokens']
                else:
                    print(f">> Single document directly added to the list")
                    deduplicated_reference_titles += references_cluster                    
        else:
            print(f">> Single document directly added to the list")
            deduplicated_reference_titles += references

    processing_end = time.time()

    print(f"\n\nTotal time to process the legal references: {processing_end - processing_start}")
    print(f"total_prompt_tokens={total_prompt_tokens}, total_completion_tokens={total_completion_tokens}")

    return deduplicated_reference_titles, {"total_time": processing_end - processing_start,
                                           "total_prompt_tokens": total_prompt_tokens,
                                           "total_completion_tokens": total_completion_tokens}

