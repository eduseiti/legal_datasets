import ast
import re
import pandas as pd



SPLITTING_HIERARCHY=[
    "LIVRO ([IVX]+)",
    "TÍTULO ([IVX]+)",
    "(CAPÍTULO)|(Capítulo)\s([IVX]+|ÚNICO)",
    "Seção ([IVX]+)",
    "Subseção (.+)",
    "Art\. (\d+)[º\. ]*",
    "§ (\d+)[º\. ]*",
    "([IVX])+[-\s]*",
]

HIERARCHY_NAMES=[
    "livro",
    "título",
    "capítulo",
    "seção",
    "subseção",
    "artigo",
    "parágrafo",
    "inciso"
]



def extract_question_references(rag_output_df, 
                                question_number_field='question_number', 
                                rag_context_field='contexto',
                                verbose=True):

    questions = rag_output_df[question_number_field].to_numpy()
    contexts = rag_output_df[rag_context_field].to_numpy()

    questions_references = []

    for i in range(len(contexts)):
        if type(contexts[i]) == str:
            passages_list = ast.literal_eval(contexts[i])
        else:
            passages_list = contexts[i]

        if passages_list is not None: 
            for passage in passages_list:

                passage_info = {}
                passage_info['question'] = questions[i]        
                
                passage_info['path'] = re.split(": \n|: ", passage)[0]

                passage_path_parts = re.split("_", passage_info['path'])

                if verbose:
                    print(f"passage_path_parts={passage_path_parts}")

                passage_info['nome'] = passage_path_parts[0]
                
                current_level = 0
                for path_details in passage_path_parts[1:]:
                    for j, hierarchy_pattern in enumerate(SPLITTING_HIERARCHY[current_level:]):

                        if verbose:
                            print(f"current_level={current_level}, j={j}")
                        
                        passage_info[HIERARCHY_NAMES[current_level + j]] = ""
                        
                        if re.match(hierarchy_pattern, path_details) is not None:
                            parts = re.split(hierarchy_pattern, path_details)

                            if verbose:
                                print(f"{HIERARCHY_NAMES[current_level + j]} = {parts}")
                            
                            if HIERARCHY_NAMES[current_level + j] == 'capítulo':
                                passage_info[HIERARCHY_NAMES[current_level + j]] = parts[-1]
                            else:
                                passage_info[HIERARCHY_NAMES[current_level + j]] = parts[1]

                            current_level += j + 1

                            break

                for current_level in range(current_level, len(HIERARCHY_NAMES)):
                    passage_info[HIERARCHY_NAMES[current_level]] = ""                        
                    
                questions_references.append(passage_info)

    return questions_references



def count_rag_matches(rag_questions_references, 
                      annotated_references, 
                      file_to_reference,
                      match_mode='all_occurrences'):

    context_matches = {}

    for context in rag_questions_references:
        context_document_filename = context['nome'] + ".txt"
        question_id = context['question']
        question_annotated_references = list(annotated_references[question_id - 1]['references'].keys())

        if question_id not in context_matches:
            context_matches[question_id] = {'correct_references': 0,
                                            'wrong_references': 0,
                                            'missing_references': 0,
                                            'correct_titles_count': {},
                                            'wrong_titles_count': {},
                                            'total_references_count': len(question_annotated_references)}  
        
        if context_document_filename in file_to_reference:
            context_reference = file_to_reference[context_document_filename]
        
            if match_mode == 'all_occurrences':
                if context_reference in question_annotated_references:
                    context_matches[question_id]['correct_references'] += 1
                else:
                    context_matches[question_id]['wrong_references'] += 1

            elif match_mode == 'title_occurrences':
                if context_reference in question_annotated_references:
                    if context_reference not in context_matches[question_id]['correct_titles_count']:
                        context_matches[question_id]['correct_titles_count'][context_reference] = 0
                        context_matches[question_id]['correct_references'] += 1

                    context_matches[question_id]['correct_titles_count'][context_reference] += 1
                else:
                    if context_reference not in context_matches[question_id]['wrong_titles_count']:
                        context_matches[question_id]['wrong_titles_count'][context_reference] = 0
                        context_matches[question_id]['wrong_references'] += 1

                    context_matches[question_id]['wrong_titles_count'][context_reference] += 1

            elif match_mode == 'complete_occurrences':
                raise NotImplementedError("\"complete_occurences\" match_mode not yet implemented...")
            else:
                raise ValueError(f"Invalid match_mode == \"{match_mode}\"")
        else:
            context_matches[question_id]['missing_references'] += 1

    context_matches_df = pd.DataFrame.from_dict(context_matches, orient='index')
    context_matches_df['correct_ratio'] = context_matches_df['correct_references'] / context_matches_df['total_references_count']
    context_matches_df['wrong_ratio'] = context_matches_df['wrong_references'] / context_matches_df['total_references_count']
    context_matches_df['missing_ratio'] = context_matches_df['missing_references'] / context_matches_df['total_references_count']

    print(context_matches_df['correct_ratio'].clip(upper=1).describe())
    print("\n")
    print(context_matches_df['wrong_ratio'].clip(upper=1).describe())
    print("\n")
    print(context_matches_df['missing_ratio'].clip(upper=1).describe())

    return context_matches_df


