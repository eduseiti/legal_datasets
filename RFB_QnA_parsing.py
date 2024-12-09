import fitz
import re

### Regular expressions to handle each question

QUESTIONS_PROCESSING_PATTERNS={
    "NEW_QUESTION": "^([0-9]{3}[0-9]?)\s?[—–-]\s?(.+)",
    "MULTI_LINE_QUESTION":"^(.+\?)",
    "END_OF_QUESTION": "^Retorno ao sumário"
}

ANSWER_REFERENCES_PATTERN=".+\n\s?\((.+)\)\.?$|.+\n\s?\((.+)\)\.?\s*\n.*([Cc]onsulte.+pergunta.+)|.+([Cc]onsulte.+pergunta.+)$"



### Functions to process a single page

def process_answer_body(which_answer):
    m = re.match (ANSWER_REFERENCES_PATTERN, "\n".join(which_answer), flags=re.DOTALL)

    references = ""
    linked_questions = ""
    end_of_answer_offset = 0
    
    if m is not None:
        if m.group(1) is not None:
            references = m.group(1)
            end_of_answer_offset = m.group(1).count('\n') + 1
            
        elif m.group(2) is not None:
            references = m.group(2)
            
            linked_questions = re.findall("\d+", m.group(3))

            end_of_answer_offset = m.group(2).count('\n') + m.group(3).count('\n') + 2
        elif m.group(4) is not None:
            linked_questions = re.findall("\d+", m.group(4))
            
            end_of_answer_offset = m.group(4).count('\n') + 1
            
    return {"answer_cleaned": which_answer[:-end_of_answer_offset] if end_of_answer_offset > 0 else which_answer,
            "references": references,
            "linked_questions": linked_questions}



def process_single_page(page_lines, 
                        current_question,
                        state,
                        processed_questions):

    for line in page_lines[2:]:
    
        m = re.match(QUESTIONS_PROCESSING_PATTERNS[state['current_pattern']], line)
    
        if m is not None:
            if state['current_pattern'] == "NEW_QUESTION":
                if len(m.groups()) > 0:
                    current_question['question_number'] = m.group(1)
                    current_question['question_summary'] = state['current_last_line']
                    current_question['question_text'] = m.group(2).strip()
                    current_question['answer'] = []

                    print("\n")
                    print(current_question)
                    print("\n")
                    
                    if current_question['question_text'][-1] != "?":
                        state['current_pattern'] = "MULTI_LINE_QUESTION"
                    else:
                        state['current_pattern'] = "END_OF_QUESTION"

                    print(f"Começo pergunta. questão={current_question['question_number']}")
            
            elif state['current_pattern'] == "MULTI_LINE_QUESTION":
                if len(m.groups()) > 0:
                    current_question['question_text'] += " " + m.group(1)
    
                    state['current_pattern'] = "END_OF_QUESTION"

                    print(f"Achou fim pergunta. questão={current_question['question_number']}")
                else:
                    current_question['question_text'] += " " + line
    
            elif state['current_pattern'] == "END_OF_QUESTION": 

                processed_answer = process_answer_body(current_question['answer'])

                current_question['answer_cleaned'] = processed_answer['answer_cleaned']
                current_question['references'] = processed_answer['references']
                current_question['linked_questions'] = processed_answer['linked_questions']
                
                processed_questions.append(current_question)

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



def extract_answers(which_file, pages_list=None):

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
                                                                 questions)

    return questions