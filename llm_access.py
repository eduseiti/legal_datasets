from groq import Groq

import time
import json
import re

GROQ_LLAMA3_2_90B_MODEL="llama-3.2-90b-vision-preview"
GROQ_LLAMA3_70B_MODEL="llama3-70b-8192"
GROQ_LLAMA3_8B_MODEL="llama3-8b-8192"
GROQ_LLAMA3_3_70B_MODEL="llama-3.3-70b-versatile"



#
# Prompt for legal references formatting
#

LEGAL_REFERENCES_FORMATTING=(
    "Leia a lista de referências jurídicas e processe as informações, "
    "separando-as de maneira estruturada. Produza uma resposta apenas com "
    "o JSON no formato a seguir, sem incluir comentários ou mensagens de erro adicionais: "
    "{\"referências\":[{ "
                      "\"título\": \"<nome-completo-da-lei-ou-documento-jurídico-incluindo-instrumento-aprovação>\", "
                      "\"artigos\": [{ "
                                      "\"artigo\": \"<número-do-artigo>\", "
                                      "\"incisos\": [\"<número-romano-inciso-1\">, ..., "
                                                    "\"<número-romano-inciso-n>\"], " 
                                      "\"parágrafos\": [\"único\" | \"<número-parágrafo-1>\", ..., "
                                                       "\"<número-parágrafo-n>\"]"
                                    "}..."
                                   "], "
                       "\"anexos\": [\"único\" | \"<número-romano-anexo-1>\", ..., "
                                    "\"<número-romano-anexo-n>\"]"
                      "}..."
                     "]"
    "}"
)



#
# Prompt for legal reference titles deduplication
#

LEGAL_REFERENCES_TITLES_DEDUPLICATION=(
    "Você recebe uma lista de título de documentos jurídicos do Brasil "
    "e deve retornar uma nova lista retirando documentos duplicados. Os "
    "documentos estarão listados 1 por linha. Documentos com numeração "
    "diferente, ano diferente, ou mesmo dia diferente são documentos "
    "diferentes e não devem ser removidos. Um mesmo documento pode estar "
    "listado de maneira mais extensa, incluindo informações adicionais; "
    "para esses casos, mantenha na nova lista apenas a versão mais completa; "
    "mas se ano for diferente, mantenha ambas versões.  Sua resposta deve "
    "ser apenas o JSON no formato a seguir, nada mais: {\"deduplicated-references\":"
                                                        "[\"<reference-1>\", ..., \"<reference-n>\"]}"
)



#
# Prompt to extract embedded legal references
#

EMBEDDED_LEGAL_REFERENCES_EXTRACTION=(
    "Você recebe um texto que responde a uma pergunta no domínio jurídico "
    "e deve gerar uma lista incluindo somente os documentos jurídicos oficiais "
    "que são explicitamente referenciados no texto. Inclua detalhes de artigos, "
    "parágrafos, incisos, anexos ou qualquer outro detalhe incluído das referências. "
    "Ignore referências que são apenas outras perguntas. Sua resposta deve ser "
    "apenas o JSON no formato a seguir, nenhum comentário a mais: "
    "{\"referências_citadas\":[<\"referência-1\">, ...., <\"referência-n\">]}"
)



#
# Class defining the access to Groq models.
#

class groq_access:

    def __init__(self,
                 api_key,
                 model):

        self.model = model
        self.client = Groq(api_key=api_key)
        

    def send_request(self, messages, temperature=0, max_tokens=2048):
        
        completed_request = False

        while not completed_request:
            try:
                completion = self.client.chat.completions.create(model=self.model,
                                                                 messages=messages,
                                                                 temperature=temperature,
                                                                 max_tokens=max_tokens,
                                                                 top_p=1,
                                                                 stream=True,
                                                                 stop=None)
    
                generated_text = ""
    
                for i, chunk in enumerate(completion):
                    generated_text += chunk.choices[0].delta.content or ""
    
                if generated_text == "":
                    print("\n\nQuota exceeded!!! Waiting for 30 seconds")
    
                    time.sleep(30)
                else:
                    try:
                        # Basic output cleanup
                        print("\n\n---------------------")
                        print(generated_text)
                        print("---------------------\n\n")

                        # Remove json markdown, if any
                        cleaned_text = generated_text.replace("```json", "")
                        cleaned_text = cleaned_text.replace("```", "")

                        # Remove new lines
                        cleaned_text = cleaned_text.replace("\n", "")

                        if cleaned_text[-1] != "}":
                            # Add missing closing curly bracket
                            cleaned_text += "}"
                        elif cleaned_text[-2] == "}":
                            # Remove duplicated closing curly bracket
                            cleaned_text = cleaned_text[:-1]

                        # Remove extra comma at the end of a list
                        cleaned_text = re.sub(r",\s*([\]\}])", r"\1", cleaned_text)

                        print("\n\n---------------------")
                        print(cleaned_text)
                        print("---------------------\n\n")
                        
                        response = json.loads(cleaned_text)
                    except Exception as e:
                        print(e)
                        print("\nError while parsing the response to JSON={}\n".format(generated_text)) 
                    
                    response['generated_text'] = generated_text
                    response['prompt_tokens'] = chunk.x_groq.usage.prompt_tokens
                    response['completion_tokens'] = chunk.x_groq.usage.completion_tokens
                    response['total_tokens'] = chunk.x_groq.usage.total_tokens
                    response['total_time'] = chunk.x_groq.usage.total_time
    
                    completed_request = True
                    
            except Exception as e:
                print(e)
                print("\nError while interacting with Groq API\n")

                time.sleep(10)

        return response



#
# Function to format a message into chat format, according to the
# given role.
#

def format_message(which_role: str, which_message: str):
    return {"role": which_role,
            "content": which_message}



#
# Function to execute legal references formatting
#

def legal_references_formatting(LLM_access: groq_access, 
                                which_text: str, 
                                verbose=True):
    
    messages = [format_message("system", LEGAL_REFERENCES_FORMATTING)]

    user_message = which_text
    
    if verbose:
        print("\n{}".format(user_message))

    messages.append(format_message("user", user_message))
    
    print(messages)

    result = LLM_access.send_request(messages)

    if verbose:
        print("\n{}".format(result))
    
    return result



#
# Function to execute legal reference titles deduplication
#

def legal_refereces_titles_deduplication(LLM_access: groq_access,
                                         titles_list: list[str],
                                         verbose=True):

    if verbose:
        print(f"\n\nTitles deduplication. {len(titles_list)} elements\n")    

    messages = [format_message("system", LEGAL_REFERENCES_TITLES_DEDUPLICATION)]

    user_message = "Lista de documentos:\n" + "\n".join(titles_list)

    if verbose:
        print("\n{}".format(user_message))

    messages.append(format_message("user", user_message))

    print(messages)

    result = LLM_access.send_request(messages, max_tokens=32768)

    if verbose:
        print("\n{}".format(result))
    
    return result



#
# Function to extract embedded legal references from answers
#

def embedded_legal_references_extraction(LLM_access: groq_access,
                                         answer_text: str,
                                         verbose=True):

    if verbose:
        print(f"\n\nEmbedded references extraction. Answer size={len(answer_text)}\n")    

    messages = [format_message("system", EMBEDDED_LEGAL_REFERENCES_EXTRACTION)]

    user_message = "Texto:\n" + answer_text

    if verbose:
        print("\n{}\n".format(user_message))

    messages.append(format_message("user", user_message))

    print(messages)

    result = LLM_access.send_request(messages, max_tokens=5096)

    if verbose:
        print("\n{}".format(result))
    
    return result