import re
import collections
import numpy as np

DOCUMENT_TITLE_REGEX=r"(.+)\s+n?º\s*([0-9\.]+)[,\s]+de\s+([0-9]+)º?\s+de\s+(\w+)\s+de\s+([0-9]{4})"

def return_unique_references(questions):

    all_references = {}
    
    questions_without_annotated_references = []
    
    for question in questions:
        if 'formatted_references' not in question:
            print(f"{question} ― No formatted_references field.\n")
            questions_without_annotated_references.append(question['question_number'])
        else:
            if question['formatted_references'] is not None:
                for each_reference in question['formatted_references']:
                    if 'título' not in each_reference:
                        print(f"{question['question_number']} = {each_reference}\n")
                    else:
                        if each_reference['título'] not in all_references:
                            all_references[each_reference['título']] = []
                
                        all_references[each_reference['título']].append(question['question_number'])
            else:
                print(f"{question} ― Empty formatted_references field.\n")
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
    

    return name_parts



def get_tokens(s):
  if not s: return []
  return re.split(r"\s|-", s.lower())



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