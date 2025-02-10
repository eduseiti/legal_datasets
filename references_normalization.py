import re
import collections

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
  return re.split(r"\s|-", s)



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