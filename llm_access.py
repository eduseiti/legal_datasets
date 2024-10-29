from groq import Groq

import time
import json

GROQ_LLAMA3_2_90B_MODEL="llama-3.2-90b-text-preview"
GROQ_LLAMA3_70B_MODEL="llama3-70b-8192"
GROQ_LLAMA3_8B_MODEL="llama3-8b-8192"


#
# Prompt for legal references formatting
#

LEGAL_REFERENCES_FORMATTING=(
    "Leia a lista de referências jurídicas e processe as informações, "
    "separando-as de maneira estruturada. Produza uma resposta apenas com "
    "o JSON no formato a seguir, sem incluir comentários ou mensagens de erro adicionais: "
    "{\"referências\":[{\"título\": <nome-completo-da-lei-ou-documento-jurídico-incluindo-instrumento-aprovação>, "
                       "\"artigos\": [{\"artigo\": <número-do-artigo>, "
                                      "\"incisos\": [<número-romano-inciso-1>, ..., "
                                                    "<número-romano-inciso-n>], " 
                                      "\"parágrafos\": [\"único\" | <número-parágrafo-1>, ..., "
                                                       "<número-parágrafo-n>]}, ..."
                                    "], "
                       "\"anexos\": [\"único\" | <número-romano-anexo-1>, ..., "
                                    "<número-romano-anexo-n>]}, ..."
                     "]"
    "}"
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
        

    def send_request(self, messages, temperature=0):
        
        completed_request = False

        while not completed_request:
            try:
                completion = self.client.chat.completions.create(model=self.model,
                                                                 messages=messages,
                                                                 temperature=temperature,
                                                                 max_tokens=2048,
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

                        cleaned_text = generated_text.replace("\n", "")

                        # if cleaned_text.rfind("}") > 0:
                        #     if cleaned_text.rfind("}") < len(cleaned_text) - 5:
                        #         cleaned_text += "}"
                        #     else:
                        #         cleaned_text = cleaned_text[:cleaned_text.rfind("}") + 1]
                        # else:
                        #     cleaned_text += "}"

                        if cleaned_text[-1] != "}":
                            cleaned_text += "}"

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
