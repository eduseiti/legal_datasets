import fitz
import re
import numpy as np

### Regular expressions to handle each question

QUESTIONS_PROCESSING_PATTERNS={
    "NEW_QUESTION": "^([0-9]{3}[0-9]?)\s?[—–-]\s?(.+)",
    "MULTI_LINE_QUESTION":"^(.+\?)",
    "END_OF_QUESTION": "^Retorno ao sumário"
}

ANSWER_REFERENCES_PATTERN=".+\n\s?\((.+)\)\.?$|.+\n\s?\((.+)\)\.?\s*\n.*([Cc]onsulte.+pergunta.+)|.+([Cc]onsulte.+pergunta.+)$"
# ANSWER_LEGAL_REFERENCES_PATTERN="\n\s?\(([^)]+)\)\.?\n"
ANSWER_LEGAL_REFERENCES_PATTERN="\n\s?\((.+)\)\.?\n"

# ANSWER_QUESTION_REFERENCES_PATTERN="[^\n]*\s?([Cc]onsulte[\s,]*(ainda)?[,]*[\sas]+pergunta[s]?[e ,0-9]+)"
ANSWER_QUESTION_REFERENCES_PATTERN="[^\n]*\s?([Cc]onsulte[\s,]*(ainda)?[,]*[\sas]+pergunta[s]?[e\s,0-9(itens)(item)s]+)"

QUESTION_REFERENCES_LIST_SPLIT_PATTERN="[Cc]onsulte[\s,]*(ainda)?[,]*[\sas]+pergunta[s]?\s?"
QUESTION_REFERENCE_SPLIT_PATTERN="\d+ \(itens[\s,\de]+\)|\d+ \(item \d+\)|\d+"

### Functions to process a single question

def process_answer_body(which_answer):

    answer_string = "\n".join(which_answer) + "\n"

    legal_references = re.findall(ANSWER_LEGAL_REFERENCES_PATTERN, answer_string, flags=re.DOTALL)
    question_references = re.findall(ANSWER_QUESTION_REFERENCES_PATTERN, answer_string, flags=re.DOTALL)

    # Remove legal references from answer body

    answer_cleaned = which_answer

    for reference in legal_references:
        answer_cleaned = re.sub('\(.*' + re.escape(reference) + '.*\)', "", "\n".join(answer_cleaned)).split("\n")
        # answer_cleaned = "\n".join(answer_cleaned).replace(reference, '').split("\n")

    # print(answer_cleaned)

    print(question_references)

    print(answer_cleaned)

    # Remove question references from answer body, consolidating them in a single list

    linked_questions = []

    for reference in question_references:
        referred_questions = re.split(QUESTION_REFERENCES_LIST_SPLIT_PATTERN, reference[0])[-1]
        referred_questions = re.findall(QUESTION_REFERENCE_SPLIT_PATTERN, referred_questions)

        linked_questions = np.union1d(linked_questions, referred_questions)

        print(reference[0])
        print("\n".join(answer_cleaned) + "\n")

        answer_cleaned = re.sub("\n.*" + re.escape(reference[0]), "\n", "\n".join(answer_cleaned) + "\n").split("\n")

    return {"answer_cleaned": answer_cleaned,
            "references": legal_references,
            "linked_questions": linked_questions}



def process_single_page(page_lines, 
                        current_question,
                        state,
                        processed_questions,
                        verbose=True):

    for line in page_lines[2:]:
    
        m = re.match(QUESTIONS_PROCESSING_PATTERNS[state['current_pattern']], line)
    
        if m is not None:
            if state['current_pattern'] == "NEW_QUESTION":
                if len(m.groups()) > 0:
                    current_question['question_number'] = m.group(1)
                    current_question['question_summary'] = state['current_last_line']
                    current_question['question_text'] = m.group(2).strip()
                    current_question['answer'] = []

                    if verbose:
                        print("\n")
                        print(current_question)
                        print("\n")
                    
                    if current_question['question_text'][-1] != "?":
                        state['current_pattern'] = "MULTI_LINE_QUESTION"
                    else:
                        state['current_pattern'] = "END_OF_QUESTION"

                    if verbose:
                        print(f"Começo pergunta. questão={current_question['question_number']}")
            
            elif state['current_pattern'] == "MULTI_LINE_QUESTION":
                if len(m.groups()) > 0:
                    current_question['question_text'] += " " + m.group(1)
    
                    state['current_pattern'] = "END_OF_QUESTION"

                    if verbose:
                        print(f"Achou fim pergunta. questão={current_question['question_number']}")
                else:
                    current_question['question_text'] += " " + line
    
            elif state['current_pattern'] == "END_OF_QUESTION": 

                processed_answer = process_answer_body(current_question['answer'])

                current_question['answer_cleaned'] = processed_answer['answer_cleaned']
                current_question['references'] = processed_answer['references']
                current_question['linked_questions'] = processed_answer['linked_questions']
                
                processed_questions.append(current_question)

                if verbose:
                    print(f"Achou fim. questão={current_question['question_number']}. Total={len(processed_questions)}")

                current_question = {}
                state['current_pattern'] = "NEW_QUESTION"
        
            else:
                raise ValueError(f"Invalid pattern {state['current_pattern']}")
        else:
            if len(line.strip()) > 0:
                state['current_last_line'] = line.strip()
        
                if state['current_pattern'] == "END_OF_QUESTION":
                    current_question['answer'].append(line.strip())

    return current_question, state



def print_questions(questions_list):
    for which_question in questions_list:
        print("\n-----------------------------------------------\n")
        print(f"Question number: {which_question['question_number']}")
        print(f"Question summary: {which_question['question_summary']}")
        print(f"Question text: {which_question['question_text']}\n")
        
        whole_answer = "\n".join(which_question['answer'])
        answer_cleaned = "\n".join(which_question['answer_cleaned'])
    
        print(f"Answer:\n{whole_answer}\n")
        print(f"Answer cleaned:\n{answer_cleaned}\n")
        print(f"References:\n{which_question['references']}\n")
        print(f"Linked questions:\n{which_question['linked_questions']}\n")



def extract_answers(which_file, pages_list=None, verbose=True):

    qna_irpf = fitz.open(which_file)
    print(f"Processing file {which_file} with {qna_irpf.page_count} pages...\n")

    questions = []
    current_question = {}
    processing_state={"current_pattern": "NEW_QUESTION",
                      "current_last_line": ""}

    if pages_list is None:
        pages_list = range(qna_irpf.page_count)

    for which_page in pages_list:
        current_question, processing_state = process_single_page(qna_irpf.load_page(which_page).get_text("text").split("\n"),
                                                                 current_question,
                                                                 processing_state,
                                                                 questions,
                                                                 verbose=verbose)

    return questions