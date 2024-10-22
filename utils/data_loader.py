import json
import sys
import os
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader, Dataset
from config.config import tokenizer, data_loader_config, train_config, IGNORE_INDEX
from utils.template import system_message
from utils.function import evaluate_expression
import re

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.function import extract_GSM8K_numerical_final_answer_using_regex, extract_MATH_numerical_final_answer_using_regex, extract_last_number



def llemma_7b_muinstruct_camelmath_complete_instruction(instruction):
    """
    Takes an instruction as input and returns a formatted response that
    guides the user through a step-by-step process.
    
    Parameters:
        instruction (str): The task description or question provided.
        
    Returns:
        str: A formatted response.
    """
    # Define the response template
    
    response_template = """Input:{}


Response:"""

    # Format the response with the given instruction
    formatted_response = response_template.format(instruction)
    
    return formatted_response


def meta_math_complete_instruction(instruction):
    response_template = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
Let's think step by step.
"""

    # Format the response with the given instruction
    formatted_response = response_template.format(instruction)
    
    return formatted_response



def load_MATH(path, n_row, zeroshot = False, meta_math_template = False, llemma_7b_muinstruct_camelmath = False, self_distillation = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            # groundtruth = line['solution']
            # num = find_last_boxed_number_with_simple_format(groundtruth)
            num = line['numerical_final_answer']
            original_question = line['question']
            self_distillation_question = \
f"""You are an expert in math. Below is a math question. Write a response that appropriately answers the question.

{original_question}
"""
            line['self_distillation_question'] = self_distillation_question
            # num = extract_last_number(groundtruth)
            if num:
                line['question'] = line['question'] + """

Please provide the final answer (a number) at the end, after 'Final Answer:'
"""
            
                if zeroshot:
                    line['question'] += f"""

Please put the final digital answer at the end after you finish inference in this format FINAL ANSWER: final neumerical answer

Format:
SOME_INFERENCE

FINAL ANSWER: """

                if meta_math_template:
                    q = line['question']
                    q = meta_math_complete_instruction(q)
                    line['question'] = q

                if llemma_7b_muinstruct_camelmath:
                    q = llemma_7b_muinstruct_camelmath_complete_instruction(q)
                    line['question'] = q
                line['numerical_final_answer'] = str(evaluate_expression(num))
                line['original_question'] = original_question
                if self_distillation:
                    line['question'] = self_distillation_question
                data_list.append(line)
            else:
                a =1 

    data_list = data_list[:n_row]
    return data_list

def load_MATH_mc_on_gpt4(path, n_row, extract_numerical_final_answer = False, zeroshot = False):
    question_list = []
    answer_list = []
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            answer = line['answer']
            line['question'] = line['question'] + """

Please provide the final answer (a number) at the end, after 'Final Answer:'.
"""
            
            if zeroshot:
                line['question'] += f"""

Please put the final digital answer at the end after you finish inference in this format FINAL ANSWER: final neumerical answer

Format:
SOME_INFERENCE

FINAL ANSWER: """
            question_list.append(line['question'])
            answer_list.append(answer)
            data_list.append(line)
        
    if extract_numerical_final_answer:
        data_list = extract_MATH_numerical_final_answer_using_regex(question_list, answer_list)
        data_list_temp = []
        for item in data_list:
            if item['numerical_final_answer']:
                data_list_temp.append(item)
        data_list = data_list_temp
    data_list = data_list[:n_row]
    return data_list

import re

def find_last_boxed_number_with_simple_format(text):
    # Regex pattern to match \boxed{17}, \boxed{-17}, \boxed{1.7}, \boxed{-5/4}
    pattern = r'\\boxed{(-?\d+(\.\d+)?|-\d+/\d+)}'
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    
    # Check if there are any matches
    if matches:
        # Select the last match found
        last_match = matches[-1][0]  # [-1] gets the last match, [0] gets the full match ignoring capturing groups
        return last_match
    else:
        # Return None or an appropriate value if no matches are found
        return None

    
def load_GSM8K_learn_from_mistakes(path, n_row):
    question_list = []
    answer_list = []
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            answer = line['answer']
            answer = answer.replace('####', 'The answer is')
            temp = {}
            question = line['question']

            temp['question'] = f"""

Please solve the following math problem.
Question: {question}

Answer: Let's think step by step.
"""
            
            temp['input'] = ''
            temp['answer'] = answer
            question_list.append(temp['question'])
            answer_list.append(answer)
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list
    
def load_GSM8K_lets_think_step_by_step(path, n_row, extract_numerical_final_answer = False, zeroshot = False):
    question_list = []
    answer_list = []
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}

            temp['question'] = line['question'] 
            temp['question'] += f"""

Let's think step by step.

Please put the final digital answer at the end after you finish thinking step by step in this format FINAL ANSWER: final_numerical_answer_here
"""
            temp['input'] = ''
            temp['answer'] = line['answer']
            question_list.append(temp['question'])
            answer_list.append(temp['answer'])
            data_list.append(temp)
        
    if extract_numerical_final_answer:
        data_list = extract_GSM8K_numerical_final_answer_using_regex(question_list, answer_list)
    data_list = data_list[:n_row]
    return data_list

def load_GSM8K_straight_forward(path, n_row, extract_numerical_final_answer = False, zeroshot = False):
    question_list = []
    answer_list = []
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}

            temp['question'] = line['question'] + """

Please solve the problem straightforwardly.

After you have solved the problem straightforwardly, please place the final numerical answer at the end, using the format: FINAL ANSWER: final_numerical_answer_here.
"""
            temp['input'] = ''
            temp['answer'] = line['answer']
            question_list.append(temp['question'])
            answer_list.append(temp['answer'])
            data_list.append(temp)
        
    if extract_numerical_final_answer:
        data_list = extract_GSM8K_numerical_final_answer_using_regex(question_list, answer_list)
    data_list = data_list[:n_row]
    return data_list

def load_GSM8K_very_clear(path, n_row, extract_numerical_final_answer = False, zeroshot = False):
    question_list = []
    answer_list = []
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}

            temp['question'] = line['question'] + """

Please provide very clear solution.

Please put the final digital answer at the end after you done in this format FINAL ANSWER: final_numerical_answer_here

"""
            temp['input'] = ''
            temp['answer'] = line['answer']
            question_list.append(temp['question'])
            answer_list.append(temp['answer'])
            data_list.append(temp)
        
    if extract_numerical_final_answer:
        data_list = extract_GSM8K_numerical_final_answer_using_regex(question_list, answer_list)
    data_list = data_list[:n_row]
    return data_list

def load_GSM8K_solve_without_explaination(path, n_row, extract_numerical_final_answer = False, zeroshot = False):
    question_list = []
    answer_list = []
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}

            temp['question'] = line['question'] + """

Please solve directly. No need to explain

Please put the final digital answer at the end after you done in this format FINAL ANSWER: final_numerical_answer_here
"""
            temp['input'] = ''
            temp['answer'] = line['answer']
            question_list.append(temp['question'])
            answer_list.append(temp['answer'])
            data_list.append(temp)
        
    if extract_numerical_final_answer:
        data_list = extract_GSM8K_numerical_final_answer_using_regex(question_list, answer_list)
    data_list = data_list[:n_row]
    return data_list

def load_GSM8K_calculation_only(path, n_row, extract_numerical_final_answer = False, zeroshot = False):
    question_list = []
    answer_list = []
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}

            temp['question'] = line['question'] + """

Please solve directly. Just show the calculation. No need to explain

Please put the final digital answer at the end after you done in this format FINAL ANSWER: final_numerical_answer_here
"""
            temp['input'] = ''
            temp['answer'] = line['answer']
            question_list.append(temp['question'])
            answer_list.append(temp['answer'])
            data_list.append(temp)
        
    if extract_numerical_final_answer:
        data_list = extract_GSM8K_numerical_final_answer_using_regex(question_list, answer_list)
    data_list = data_list[:n_row]
    return data_list

def load_GSM8K(path, n_row, zeroshot = False, meta_math_template = False, llemma_7b_muinstruct_camelmath = False, self_distillation = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            question = line['question'] 
            original_question = question
            if zeroshot:
                question += f"""

Please put the final digital answer at the end after you finish inference in this format Final Answer: final neumerical answer

Format:
SOME_INFERENCE

Final Answer: """
            else:
                question += """

Please provide the final answer (a number) at the end, after 'Final Answer:'
"""
            if meta_math_template:
                question = meta_math_complete_instruction(question)
            if llemma_7b_muinstruct_camelmath:
                question = llemma_7b_muinstruct_camelmath_complete_instruction(question)
            


            self_distillation_question = \
f"""You are an expert in math. Below is a math question. Write a response that appropriately answers the question.

{original_question}
"""

            answer = line['answer']
            if zeroshot:
                answer = answer.replace('####', 'Final Answer: ')
            line['answer'] = answer
            line['question'] = question
            line['original_question'] = original_question
            line['self_distillation_question'] = self_distillation_question
            if self_distillation:
                line['question'] = self_distillation_question
            data_list.append(line)
        
    data_list = data_list[:n_row]
    return data_list



def load_AQuaRAT(path, n_row, zeroshot = False, meta_math_template = False, llemma_7b_muinstruct_camelmath = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            question = line['question'] 
            A = '(' + line['options'][0] 
            B = '(' + line['options'][1] 
            C = '(' + line['options'][2] 
            D = '(' + line['options'][3] 
            E = '(' + line['options'][4] 
            if zeroshot:
                question += f"""

Please put the final digital answer at the end after you finish inference in this format Final Answer: final neumerical answer

Format:
SOME_INFERENCE

Final Answer: """
            else:
                question += f"""

Options:
{A}
{B}
{C}
{D} 
{E}
Please choose the correct answer (A)/(B)/(C)/(D)/(E) and place it at the end, after '\n\nFinal Answer: '
"""
            if meta_math_template:
                question = meta_math_complete_instruction(question)
            if llemma_7b_muinstruct_camelmath:
                question = llemma_7b_muinstruct_camelmath_complete_instruction(question)

            answer = line['answer']
            if zeroshot:
                answer = answer.replace('####', 'Final Answer: ')
            line['answer'] = answer
            line['question'] = question
            data_list.append(line)
        
    data_list = data_list[:n_row]
    return data_list



def load_GSM8K_march_27(path, n_row, extract_numerical_final_answer = False, zeroshot = False):
    question_list = []
    answer_list = []
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}
            temp['question'] = line['question'] + """

Please provide the final answer (a number) at the end, after 'Final Answer:'
"""
            if zeroshot:
                temp['question'] += f"""

Please put the final digital answer at the end after you finish inference in this format FINAL ANSWER: final neumerical answer

Format:
SOME_INFERENCE

FINAL ANSWER: """
            answer = line['answer']

            answer = answer.replace('####', 'Final Answer:')
            temp['input'] = ''
            temp['answer'] = line['answer']
            question_list.append(temp['question'])
            answer_list.append(temp['answer'])
            data_list.append(temp)
        
    if extract_numerical_final_answer:
        data_list = extract_GSM8K_numerical_final_answer_using_regex(question_list, answer_list)
    for i in range(len(data_list)):
        data_list[i]['answer'] = data_list[i]['answer'].replace('####', 'Final Answer:')
    data_list = data_list[:n_row]


    return data_list





def load_GSM8K_change_model_internal_preference(path, n_row, extract_numerical_final_answer = False, zeroshot = False):
    question_list = []
    answer_list = []
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}

            temp['question'] = line['question'] + """

Assume you are a math expert and you are doing a math exam. You care so much about details and you want to make sure everything you write down is correct.
"""
            temp['input'] = ''
            temp['answer'] = line['answer']
            question_list.append(temp['question'])
            answer_list.append(temp['answer'])
            data_list.append(temp)
        
    if extract_numerical_final_answer:
        data_list = extract_GSM8K_numerical_final_answer_using_regex(question_list, answer_list)
    data_list = data_list[:n_row]
    return data_list
    


def load_GSM8K_nov_25(path, n_row, extract_numerical_final_answer = False, minimum_change_or_zero_shot = False):
    question_list = []
    answer_list = []
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}
            if not minimum_change_or_zero_shot: #means you are loading finetuning dataset
                temp['question'] = line['question']
            else:
                temp['question'] = line['question'] + """

Please inference then provide the final answer. 

"""
            temp['input'] = ''
            temp['answer'] = line['answer']
            question_list.append(line['question'])
            answer_list.append(line['answer'])
            data_list.append(temp)
        
    if extract_numerical_final_answer:
        data_list = extract_GSM8K_numerical_final_answer_using_regex(question_list, answer_list)
    data_list = data_list[:n_row]
    return data_list

def load_API_BANK_aug_2(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}
            prompt  = line['instruction'] + line['input']
            prompt = prompt.replace('Generate API Request', 'Generate next API Request')
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_API_BANK_vanilla(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}
            prompt  = line['instruction'] + line['input']
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_API_BANK(path, n_row, minimum_change_or_zero_shot = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}
            if not minimum_change_or_zero_shot: #means you are loading finetuning dataset
                if '->' in line['input']:
                    prompt  = f"""The instruction requires multi-step API-CALL. The input contains the history of previous API-Request with the received data in this format API-Request -> Received data. Based on the instruction, previous API-Request and the users' utterance, Please decide the next API-Request to generate. 

                    {line['instruction'] + line['input']}"""
                else:
                    prompt  = f"""The instruction requires multi-step API-CALL. Based on the instruction and the available APIs, please decide the next API-Request to generate. 

                    {line['instruction'] + line['input']}"""
            else:
                # prompt = f"""{line['instruction'] + line['input']}
                if '->' in line['input']:
#                     prompt  = f"""Given the instruction and the API descriptions, please analyze step by step then generate the correct next API-request.

# To help the user, it may requires multi-step inference calls. The input contains the history of previous API calls with
# Each time only generate the next api call. The input contains the history of previous API-Request with the received data in this format API-Request -> Received data

# The input contains the available API-Request options. You need to select the best API-Request that should be generated at the next time. You also need fill out the API-request based on the description, input parameters, input types, etc.

# You only need to generate the next API-request based on the previous API-Request history at each time 

# Please think step by step, then generate the final answer. Here is the proper thinking steps.

# 1. Understand the user's utterance.
# 2. Identify the previous API-Request call history 
# 3. Based on the previous API-Request and the received data, select the best API request that should be generated next step
# 4. Based on the selected API request, fill in the required parameters
# 5. Combine the API request name and the filled-in parameters to generate the final API request.

# Instruction: {line['instruction'] + line['input']}
# """
                    prompt  = \
f"""The instruction requires multi-step API-CALL. The input contains the history of previous API-Request with the received data in this format API-Request -> Received data. Based on the previous API-Request and the users' utterance, Please decide the next API-Request to generate. 
                    
Let's solve the problem step by step.

1. Understand the user's utterance.
2. Identify the API-Request called previously.
3. Based on the available API options, previous API-Request and the received data, select the best API request that should be generated next step
4. Identify the input to for the selected API.
5. Fillin the input for the selected API-Request to generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)]

Instruction: {line['instruction'] + line['input']}"""
                else:
                    prompt  = f"""The instruction requires multi-step API-CALL. Please decide the next API-Request to generate. Let's solve the problem step by step.

1. Understand the user's utterance.
2. Based on the available API options, select the best API request that should be generated next step
3. Identify the input to for the selected API.
4. Fillin the input for the selected API-Request to generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)]

Instruction: {line['instruction'] + line['input']}"""
                    # prompt  = f"""According to the instruction and the API description, please generate the next API-Request. Instruction: {line['instruction'] + line['input']}"""
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_API_BANK_step_by_step(path, n_row, minimum_change_or_zero_shot = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            if minimum_change_or_zero_shot: #means you are loading finetuning dataset
                if '->' in line['input']:
                    prompt  = \
f"""The instruction requires multi-step API-CALL. The input contains the history of previous API-Request with the received data in this format API-Request -> Received data. Based on the instruction, previous API-Request and the users' utterance, Please decide the next API-Request to generate. 

{line['instruction'] + line['input']}

Let's solve the problem step by step."""
                else:
                    prompt  = \
f"""The instruction requires multi-step API-CALL. Based on the instruction and the available APIs, please decide the next API-Request to generate. 

{line['instruction'] + line['input']}

Let's solve the problem step by step."""
            else:
                if '->' in line['input']:
                    prompt  = \
f"""The instruction requires multi-step API-CALL. The input contains the history of previous API-Request with the received data in this format API-Request -> Received data. Based on the previous API-Request and the users' utterance, Please decide the next API-Request to generate. 
                    
Instruction: {line['instruction'] + line['input']}"""
                else:
                    prompt  = \
f"""The instruction requires multi-step API-CALL. Please decide the next API-Request to generate.

Instruction: {line['instruction'] + line['input']}"""
                    # prompt  = f"""According to the instruction and the API description, please generate the next API-Request. Instruction: {line['instruction'] + line['input']}"""
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_API_BANK_simplified(path, n_row, minimum_change_or_zero_shot = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}
            if not minimum_change_or_zero_shot: #means you are loading finetuning dataset
                prompt  = line['instruction'] + line['input']
            else:
                if '->' in line['input']:
                    prompt  = f"""Instruction: {line['instruction'] + line['input']}

The instruction requires multi-step API-CALL. The input contains the history of previous API-Request with the received data in this format API-Request -> Received data. Based on the previous API-Request and the users' utterance, Please decide the next API-Request to generate. 
                    
Let's solve the problem step by step.
1. Identify the user's utterance and previous API-Requst
2. Based on the available API options, the previous API-Request call history and user's utterance select the next API to call.
3. Determine the input value to fill in the selcted API according to its description and users' utterance.
4. Generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)]"""
                else:
                    prompt  = f"""Instruction: {line['instruction'] + line['input']}


The instruction requires multi-step API-CALL. Please decide the next API-Request to generate. Let's solve the problem step by step.

1. Identify the user's utterance and previous API-Requst
2. Based on the available API options and user's utterance select the next API to call.
3. Determine the input value to fill in the selcted API according to its description and users' utterance.
4. Generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)]"""
                    # prompt  = f"""Instruction: {line['instruction'] + line['input']}"""
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_API_BANK_vanilla(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}
            prompt  = line['instruction'] + line['input']
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list



def load_API_BANK_vanilla_step2(path, n_row, minimum_change_or_zero_shot = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}
            prompt  = line['instruction'] + line['input']
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


# def load_API_BANK_minimum_change_data_creation(path, n_row, minimum_change_or_zero_shot = False):
#     data_list = []
#     with open(path, 'r') as file:
#         data = json.load(file)
        
#         for line in data:
#             temp = {}
            
#             # prompt = f"""{line['instruction'] + line['input']}
#             if '->' in line['input']:
#                 prompt  = f"""Given the instruction and the API descriptions, please analyze step by step then generate the correct next API-request.

# To help the user, it may requires multi-step inference calls. The input contains the history of previous API calls with
# Each time only generate the next api call. The input contains the history of previous API-Request with the received data in this format API-Request -> Received data

# The input contains the available API-Request options. You need to select the best API-Request that should be generated at the next time. You also need fill out the API-request based on the description, input parameters, input types, etc.

# You only need to generate the next API-request based on the previous API-Request history at each time 

# Please think step by step, then generate the final answer. Here is the proper thinking steps.

# 1. Understand the user's utterance.
# 2. Identify the previous API-Request call history 
# 3. Based on the previous API-Request and the received data, select the best API request that should be generated next step
# 4. Based on the selected API request, fill in the required parameters
# 5. Combine the API request name and the filled-in parameters to generate the final API request.

# Instruction: {line['instruction'] + line['input']}
# """
#             else:
#                 prompt = f"""Given the instruction and the API descriptions, please analyze step by step then generate the correct next API-request.

# To help the user, it may requires multi-step inference calls. The input contains the history of previous API calls with
# Each time only generate the next api call.

# The input contains the available API-Request options. You need to select the best API-Request that should be generated at the next time. You also need fill out the API-request based on the description, input parameters, input types, etc.

# You only need to generate the next API-request.

# Please think step by step, then generate the final answer. Here is the proper thinking steps.

# 1. Understand the user's utterance.
# 2. select the best API request that should be generated next step
# 3. Based on the selected API request, fill in the required parameters
# 4. Combine the API request name and the filled-in parameters to generate the final API request.

# Instruction: {line['instruction'] + line['input']}
# """
#             temp['question'] = prompt
#             temp['input'] = ''
#             temp['answer'] = line['output']
#             try:
#                 temp['sample_id'] = line['sample_id']
#                 temp['api_id'] = line['api_id']
#             except:
#                 a = 1
#             data_list.append(temp)
        
#     data_list = data_list[:n_row]
#     return data_list


def load_API_BANK_Rationale_creation(path, n_row, minimum_change_or_zero_shot = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            temp = {}
            if not minimum_change_or_zero_shot: #means you are loading finetuning dataset
                prompt  = line['instruction'] + line['input']
            else:
#                 task_intro = f"""Given the instruction and the API input, please Analyze then generate the correct 
# API-request.
# Each time only generate the next api call. For example, you may need to generate API A given user request. Next time, you may need to generate API call B given API call A and user request."""

                prompt = f"""
given the instruction and the answer, what do you think should be the thinking steps to inference to finally get the answer ? Keep in mind that each api-request require specific input.


"instruction": {line['instruction'] + line['input']}
"groundtruth": {line['output']}
"""
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_ANLI_sftgt(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context. Analyze then answer. 

Format:
Answer: 
"""
#             answer_item = f""""Analyzation: {line['reason']}
# Answer: {line['gold_label']}"""
            answer_item = f""""Answer: {line['gold_label']}"""
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer_item
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            temp['reason'] = line['reason']
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_API_BANK_optimized(path, n_row, minimum_change_or_zero_shot = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            
            if '->' in line['input']:
                content = line['input']
                api_request_regex = r"API-Request:"
                match = re.search(api_request_regex, content)
                api_history = content[match.start():]
                cc = api_history.count('API-Request:')
                for i in range(1, api_history.count('API-Request:') + 1):
                    if cc > 1:
                        api_history = api_history.replace('API-Request:', f"API-Request{i}:", i)
                for i in range(1, cc+ 1):
                    api_history = api_history.replace(f'API-Request{i}:', f"""{i}. 
API-Request:""")

                api_history = api_history.replace('->', """
Received API Response:""")
                api_history = api_history.replace("\nGenerate API Request: ", '')

                pattern = r"\nUser:(.*?)\nAPI-Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" + \
f"""


Previous API-Request History:

{api_history}



Available information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response"""
            else:
                content = line['input']
                pattern = r"\nUser:(.*?)\nGenerate API Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()
                user_utterance = user_utterance.replace('\nGenerate API Request:', '')

                prompt = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" 
                
            if minimum_change_or_zero_shot: 
                if '->' in line['input']:
                    end_prompt = f"""

1. The previous API-request has help solve part of job specified in user's utterance. What should be done next to satisfy the user? Please solve the rest of the problem step by step. 
2. Remember to generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
                else:
                    end_prompt = f"""

1. The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be done next to satisfy the user? Please solve the rest of the problem step by step. 
2. Remember to generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
            else:
                if '->' in line['input']:
                    end_prompt = f"""

1. The previous API-request has help solve part of job specified in user's utterance. What should be done next to satisfy the user? Please generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
                else:
                    end_prompt = f"""

1. The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be done next to satisfy the user? Please generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
            temp['original_question'] = prompt
            prompt += end_prompt
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_API_BANK_march_1_step1(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            grundtruth = line['output']
            empty_input = False
            if '()]' in grundtruth:
                empty_input = True
            if '->' in line['input']:
                content = line['input']
                api_request_regex = r"API-Request:"
                match = re.search(api_request_regex, content)
                api_history = content[match.start():]
                cc = api_history.count('API-Request:')
                if cc > 2:
                    a = 1
                API_REQUEST_HISTORY = "Previous API-Request History:"
                if cc > 1:
                    api_history = api_history.replace('API-Request:', f"{API_REQUEST_HISTORY}")
                    last_occurrence_index = api_history.rfind(f"{API_REQUEST_HISTORY}")
                    # Split the string into the part before and after the last occurrence
                    before_part = api_history[:last_occurrence_index]
                    after_part = api_history[last_occurrence_index+len(f"{API_REQUEST_HISTORY}"):]

                    # Create the replacement string with the index
                    replacement = f"The last previous API-Request:"

                    # Reconstruct the string with the replacement
                    api_history = before_part + replacement + after_part

                api_history = api_history.replace('->', """
Received API Response:""")
                api_history = api_history.replace("\nGenerate API Request: ", '')

                pattern = r"\nUser:(.*?)\nAPI-Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" + \
f"""


{api_history}



Available information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response"""
            else:
                content = line['input']
                pattern = r"\nUser:(.*?)\nGenerate API Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()
                user_utterance = user_utterance.replace('\nGenerate API Request:', '')

                prompt = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" 
                
            if '->' in line['input']:
                if 'ToolSearcher' in grundtruth:
                    end_prompt = f"""

1. The task requires multi steps API-Request generation. The previous API-request has help solve part of job specified in user's utterance. What API-Request should be called in the next step?  We only generate the next API-Request at this time.
2. We already know the answer is {grundtruth}. Please show how you get the answer."""
                else:
                    end_prompt = f"""

1. The task requires multi steps API-Request generation. The previous API-request has help solve part of job specified in user's utterance. What API-Request should be called in the next step?  We only generate the next API-Request at this time.
2. We already know the answer is {grundtruth}. Please show how you get the answer."""
                    if empty_input:
                        end_prompt += f"""
3. Before show the final answer, make sure to explain why you fill in the API-Request with the empty inputs according to the API description. You have to explain why the input filled in is empty before you generate the final API-request.
"""
                    else:
                        end_prompt += f"""
3. Before show the final answer, make sure to explain how you fill in the API-Request with the proper inputs according to the API description. You have to explain how to fill in the inputs before you generate the final API-request.
"""
            else:
                end_prompt = f"""

1. The task requires multi steps API-Request generation. We only generate the next API-Request at this time. What API-Request should be called in the next step?
2. We already know the answer is {grundtruth}. Please show how you get the answer."""
            prompt += end_prompt
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_API_BANK_march_1(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            grundtruth = line['output']
            empty_input = False
            if '()]' in grundtruth:
                empty_input = True
            if '->' in line['input']:
                content = line['input']
                api_request_regex = r"API-Request:"
                match = re.search(api_request_regex, content)
                api_history = content[match.start():]
                cc = api_history.count('API-Request:')
                if cc > 2:
                    a = 1
                API_REQUEST_HISTORY = "Previous API-Request History:"
                if cc > 1:
                    api_history = api_history.replace('API-Request:', f"{API_REQUEST_HISTORY}")
                    last_occurrence_index = api_history.rfind(f"{API_REQUEST_HISTORY}")
                    # Split the string into the part before and after the last occurrence
                    before_part = api_history[:last_occurrence_index]
                    after_part = api_history[last_occurrence_index+len(f"{API_REQUEST_HISTORY}"):]

                    # Create the replacement string with the index
                    replacement = f"The last previous API-Request:"

                    # Reconstruct the string with the replacement
                    api_history = before_part + replacement + after_part

                api_history = api_history.replace('->', """
Received API Response:""")
                api_history = api_history.replace("\nGenerate API Request: ", '')

                pattern = r"\nUser:(.*?)\nAPI-Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" + \
f"""


{api_history}



Available information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response"""
            else:
                content = line['input']
                pattern = r"\nUser:(.*?)\nGenerate API Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()
                user_utterance = user_utterance.replace('\nGenerate API Request:', '')

                prompt = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" 
                
            if '->' in line['input']:
                if 'ToolSearcher' in grundtruth:
                    end_prompt = f"""

1. The task requires multi steps API-Request generation. The previous API-request has help solve part of job specified in user's utterance. What API-Request should be called in the next step?  We only generate the next API-Request at this time."""
                else:
                    end_prompt = f"""

1. The task requires multi steps API-Request generation. The previous API-request has help solve part of job specified in user's utterance. What API-Request should be called in the next step?  We only generate the next API-Request at this time."""
            else:
                end_prompt = f"""

1. The task requires multi steps API-Request generation. We only generate the next API-Request at this time. What API-Request should be called in the next step?"""
            prompt += end_prompt
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_API_BANK_march_5_step1(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            grundtruth = line['output']
            empty_input = False
            if '()]' in grundtruth:
                empty_input = True
            if '->' in line['input']:
                content = line['input']
                api_request_regex = r"API-Request:"
                match = re.search(api_request_regex, content)
                api_history = content[match.start():]
                cc = api_history.count('API-Request:')
                if cc > 2:
                    a = 1
                API_REQUEST_HISTORY = "Previous API-Request History:"
                if cc > 1:
                    api_history = api_history.replace('API-Request:', f"{API_REQUEST_HISTORY}")
                    last_occurrence_index = api_history.rfind(f"{API_REQUEST_HISTORY}")
                    # Split the string into the part before and after the last occurrence
                    before_part = api_history[:last_occurrence_index]
                    after_part = api_history[last_occurrence_index+len(f"{API_REQUEST_HISTORY}"):]

                    # Create the replacement string with the index
                    replacement = f"The last previous API-Request:"

                    # Reconstruct the string with the replacement
                    api_history = before_part + replacement + after_part

                api_history = api_history.replace('->', """
Received API Response:""")
                api_history = api_history.replace("\nGenerate API Request: ", '')

                pattern = r"\nUser:(.*?)\nAPI-Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" + \
f"""


{api_history}



Available information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response"""
            else:
                content = line['input']
                pattern = r"\nUser:(.*?)\nGenerate API Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()
                user_utterance = user_utterance.replace('\nGenerate API Request:', '')

                prompt = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" 
                
            if '->' in line['input']:
                if 'ToolSearcher' in grundtruth:
                    end_prompt = f"""

1. The task requires multi steps API-Request generation. The previous API-request has help solve part of job specified in user's utterance. What API-Request should be called in the next step?  We only generate the next API-Request at this time.
2. Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)].
3. Before show the final answer, make sure to explain why you fill in the API-Request with the the inputs according to the API description. You have to explain how to fill in the inputs before you generate the final API-request.
"""
                else:
                    end_prompt = f"""

1. The task requires multi steps API-Request generation. The previous API-request has help solve part of job specified in user's utterance. What API-Request should be called in the next step?  We only generate the next API-Request at this time.
2. Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] """
                    if empty_input:
                        end_prompt += f"""
3. Before show the final answer, make sure to explain why you fill in the API-Request with the empty inputs according to the API description. You have to explain why the input filled in is empty before you generate the final API-request.
"""
                    else:
                        end_prompt += f"""
3. Before show the final answer, make sure to explain how you fill in the API-Request with the proper inputs according to the API description. You have to explain how to fill in the inputs before you generate the final API-request.
"""
            else:
                end_prompt = f"""

1. The task requires multi steps API-Request generation. We only generate the next API-Request at this time. What API-Request should be called in the next step?
2. Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)].
3. Before show the final answer, make sure to explain why you fill in the API-Request with the the inputs according to the API description. You have to explain how to fill in the inputs before you generate the final API-request.
"""
            prompt += end_prompt
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_API_BANK_march_8_step1(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            grundtruth = line['output']
            empty_input = False
            if '()]' in grundtruth:
                empty_input = True
            if '->' in line['input']:
                content = line['input']
                api_request_regex = r"API-Request:"
                match = re.search(api_request_regex, content)
                api_history = content[match.start():]
                cc = api_history.count('API-Request:')
                if cc > 2:
                    a = 1
                API_REQUEST_HISTORY = "Previous API-Request History:"
                if cc > 1:
                    api_history = api_history.replace('API-Request:', f"{API_REQUEST_HISTORY}")
                    last_occurrence_index = api_history.rfind(f"{API_REQUEST_HISTORY}")
                    # Split the string into the part before and after the last occurrence
                    before_part = api_history[:last_occurrence_index]
                    after_part = api_history[last_occurrence_index+len(f"{API_REQUEST_HISTORY}"):]

                    # Create the replacement string with the index
                    replacement = f"The last previous API-Request:"

                    # Reconstruct the string with the replacement
                    api_history = before_part + replacement + after_part

                api_history = api_history.replace('->', """
Received API Response:""")
                api_history = api_history.replace("\nGenerate API Request: ", '')

                pattern = r"\nUser:(.*?)\nAPI-Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" + \
f"""


{api_history}



Available information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response"""
            else:
                content = line['input']
                pattern = r"\nUser:(.*?)\nGenerate API Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()
                user_utterance = user_utterance.replace('\nGenerate API Request:', '')

                prompt = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" 
                
            if '->' in line['input']:
                if 'ToolSearcher' in grundtruth:
                    end_prompt = f"""

1. The task requires multi steps API-Request generation. The previous API-request has help solve part of job specified in user's utterance. What API-Request should be called in the next step?  We only generate the next API-Request at this time.
2. Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)].
3. Before show the final answer, make sure to explain why you fill in the API-Request with the the inputs according to the API description. You have to explain how to fill in the inputs before you generate the final API-request.
4. This is how you should approach the problem. You should first identify the API from the available API options.
Then you need to figure out the correct input according to the API description. Please notice that you may only use the input specified by the API description for input
Then you need to generate the API request
"""
                else:
                    end_prompt = f"""

1. The task requires multi steps API-Request generation. The previous API-request has help solve part of job specified in user's utterance. What API-Request should be called in the next step?  We only generate the next API-Request at this time.
2. Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] """
                    if empty_input:
                        end_prompt += f"""
3. Before show the final answer, make sure to explain why you fill in the API-Request with the empty inputs according to the API description. You have to explain why the input filled in is empty before you generate the final API-request.
4. This is how you should approach the problem. You should first identify the API from the available API options.
Then you need to figure out the correct input according to the API description. Please notice that you may only use the input specified by the API description for input
Then you need to generate the API request
"""
                    else:
                        end_prompt += f"""
3. Before show the final answer, make sure to explain how you fill in the API-Request with the proper inputs according to the API description. You have to explain how to fill in the inputs before you generate the final API-request.
4. This is how you should approach the problem. You should first identify the API from the available API options.
Then you need to figure out the correct input according to the API description. Please notice that you may only use the input specified by the API description for input
Then you need to generate the API request
"""
            else:
                end_prompt = f"""

1. The task requires multi steps API-Request generation. We only generate the next API-Request at this time. What API-Request should be called in the next step?
2. Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)].
3. Before show the final answer, make sure to explain why you fill in the API-Request with the the inputs according to the API description. You have to explain how to fill in the inputs before you generate the final API-request.
4. This is how you should approach the problem. You should first identify the API from the available API options.
Then you need to figure out the correct input according to the API description. Please notice that you may only use the input specified by the API description for input
Then you need to generate the API request
"""
            prompt += end_prompt
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list




def load_API_BANK_march_9_step1(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            grundtruth = line['output']
            empty_input = False
            if '()]' in grundtruth:
                empty_input = True
            if '->' in line['input']:
                content = line['input']
                api_request_regex = r"API-Request:"
                match = re.search(api_request_regex, content)
                api_history = content[match.start():]
                cc = api_history.count('API-Request:')
                if cc > 2:
                    a = 1
                API_REQUEST_HISTORY = "Previous API-Request History:"
                if cc > 1:
                    api_history = api_history.replace('API-Request:', f"{API_REQUEST_HISTORY}")
                    last_occurrence_index = api_history.rfind(f"{API_REQUEST_HISTORY}")
                    # Split the string into the part before and after the last occurrence
                    before_part = api_history[:last_occurrence_index]
                    after_part = api_history[last_occurrence_index+len(f"{API_REQUEST_HISTORY}"):]

                    # Create the replacement string with the index
                    replacement = f"The last previous API-Request:"

                    # Reconstruct the string with the replacement
                    api_history = before_part + replacement + after_part

                api_history = api_history.replace('->', """
Received API Response:""")
                api_history = api_history.replace("\nGenerate API Request: ", '')

                pattern = r"\nUser:(.*?)\nAPI-Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" + \
f"""


{api_history}



Available information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response"""
            else:
                content = line['input']
                pattern = r"\nUser:(.*?)\nGenerate API Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()
                user_utterance = user_utterance.replace('\nGenerate API Request:', '')

                prompt = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" 
                
            if '->' in line['input']:
                if 'ToolSearcher' in grundtruth:
                    end_prompt = f"""

1. The task requires multi steps API-Request generation. The previous API-request has help solve part of job specified in user's utterance. What API-Request should be called in the next step?  We only generate the next API-Request at this time.
2. Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)].
3. This is how you should approach the problem: First, decide which API to use for the next step, considering the history of previous API requests. Then, determine the correct input based on the API description, noting that you may only use the inputs specified in the API description. Finally, generate the API request. Before showing the final answer, ensure you explain why you filled in the API request with the specified inputs, according to the API description. It's important to explain how to input the data before generating the final API request.
"""
                else:
                    end_prompt = f"""

1. The task requires multi steps API-Request generation. The previous API-request has help solve part of job specified in user's utterance. What API-Request should be called in the next step?  We only generate the next API-Request at this time.
2. Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] """
                    if empty_input:
                        end_prompt += f"""
3. This is how you should approach the problem: First, decide which API to use for the next step, considering the history of previous API requests. Then, determine the correct input based on the API description, noting that you may only use the inputs specified in the API description. Finally, generate the API request. Before showing the final answer, ensure you explain why you filled in the API request with the specified inputs, according to the API description. It's important to explain how to input the data before generating the final API request.
"""
                    else:
                        end_prompt += f"""
3. This is how you should approach the problem: First, decide which API to use for the next step, considering the history of previous API requests. Then, determine the correct input based on the API description, noting that you may only use the inputs specified in the API description. Finally, generate the API request. Before showing the final answer, ensure you explain why you filled in the API request with the specified inputs, according to the API description. It's important to explain how to input the data before generating the final API request.
"""
            else:
                end_prompt = f"""

1. The task requires multi steps API-Request generation. We only generate the next API-Request at this time. What API-Request should be called in the next step?
2. Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)].
3. This is how you should approach the problem: First, decide which API to use for the next step, considering the available API. Then, determine the correct input based on the API description, noting that you may only use the inputs specified in the API description. Finally, generate the API request. Before showing the final answer, ensure you explain why you filled in the API request with the specified inputs, according to the API description. It's important to explain how to input the data before generating the final API request.
"""
            prompt += end_prompt
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list



def load_API_BANK_feb_8(path, n_row, minimum_change_or_zero_shot = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            
            if '->' in line['input']:
                content = line['input']
                api_request_regex = r"API-Request:"
                match = re.search(api_request_regex, content)
                api_history = content[match.start():]
                cc = api_history.count('API-Request:')
                for i in range(1, api_history.count('API-Request:') + 1):
                    if cc > 1:
                        api_history = api_history.replace('API-Request:', f"API-Request{i}:", i)
                for i in range(1, cc+ 1):
                    api_history = api_history.replace(f'API-Request{i}:', f"""{i}. 
API-Request:""")

                api_history = api_history.replace('->', """
Received API Response:""")
                api_history = api_history.replace("\nGenerate API Request: ", '')

                pattern = r"\nUser:(.*?)\nAPI-Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" + \
f"""


Previous API-Request History:

{api_history}



Available information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response"""
            else:
                content = line['input']
                pattern = r"\nUser:(.*?)\nGenerate API Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()
                user_utterance = user_utterance.replace('\nGenerate API Request:', '')

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" 
                
            if minimum_change_or_zero_shot: 
                if '->' in line['input']:
                    end_prompt = f"""

1. The previous API-request has help solve part of job specified in user's utterance. What should be done next to satisfy the user?
2. Remember to generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
                else:
                    end_prompt = f"""

1. The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be done next to satisfy the user? 
2. Remember to generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
            else:
                if '->' in line['input']:
                    end_prompt = f"""

1. The previous API-request has help solve part of job specified in user's utterance. What should be done next to satisfy the user? Please generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
                else:
                    end_prompt = f"""

1. The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be done next to satisfy the user? Please generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
            prompt += end_prompt
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_API_BANK_jan_30(path, n_row, minimum_change_or_zero_shot = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            
            if '->' in line['input']:
                content = line['input']
                api_request_regex = r"API-Request:"
                match = re.search(api_request_regex, content)
                api_history = content[match.start():]
                cc = api_history.count('API-Request:')
                for i in range(1, api_history.count('API-Request:') + 1):
                    if cc > 1:
                        api_history = api_history.replace('API-Request:', f"API-Request{i}:", i)
                for i in range(1, cc+ 1):
                    api_history = api_history.replace(f'API-Request{i}:', f"""{i}. 
API-Request:""")

                api_history = api_history.replace('->', """
Received API Response:""")
                api_history = api_history.replace("\nGenerate API Request: ", '')

                pattern = r"\nUser:(.*?)\nAPI-Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" + \
f"""


Previous API-Request History:

{api_history}



Available information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response"""
            else:
                content = line['input']
                pattern = r"\nUser:(.*?)\nGenerate API Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()
                user_utterance = user_utterance.replace('\nGenerate API Request:', '')

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" 
                
            if minimum_change_or_zero_shot: 
                if '->' in line['input']:
                    end_prompt = f"""
Requirement:
1. The previous API-request has help solve part of job specified in user's utterance. What should be done next to satisfy the user? Please solve the problem. 
2. Remember to generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
                else:
                    end_prompt = f"""

1. The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be done next to satisfy the user? Please solve the problem. 
2. Remember to generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
            else:
                if '->' in line['input']:
                    end_prompt = f"""

1. The previous API-request has help solve part of job specified in user's utterance. What should be done next to satisfy the user? Please generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
                else:
                    end_prompt = f"""

1. The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be done next to satisfy the user? Please generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
            prompt += end_prompt
            temp['question'] = prompt
            temp['input'] = ''
            temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_API_BANK_optimized_simplified(path, n_row, minimum_change_or_zero_shot = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            
            if '->' in line['input']:
                content = line['input']
                api_request_regex = r"API-Request:"
                match = re.search(api_request_regex, content)
                api_history = content[match.start():]
                cc = api_history.count('API-Request:')
                for i in range(1, api_history.count('API-Request:') + 1):
                    if cc > 1:
                        api_history = api_history.replace('API-Request:', f"API-Request{i}:", i)
                for i in range(1, cc+ 1):
                    api_history = api_history.replace(f'API-Request{i}:', f"""{i}. 
API-Request:""")

                api_history = api_history.replace('->', """
Received API Response:""")
                api_history = api_history.replace("\nGenerate API Request: ", '')

                pattern = r"\nUser:(.*?)\nAPI-Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" + \
f"""


Previous API-Request History:

{api_history}



Available information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response"""
            else:
                content = line['input']
                pattern = r"\nUser:(.*?)\nGenerate API Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()
                user_utterance = user_utterance.replace('\nGenerate API Request:', '')

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" 
                
            if minimum_change_or_zero_shot: 
                if '->' in line['input']:
                    end_prompt = f"""

1. The previous API-request has help solve part of job specified in user's utterance. What should be done next to satisfy the user? Please solve the rest of the problem step by step. 
2. Remember to generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
                else:
                    end_prompt = f"""

1. The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be done next to satisfy the user? Please solve the rest of the problem step by step. 
2. Remember to generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
            else:
                if '->' in line['input']:
                    end_prompt = f"""

1. The previous API-request has help solve part of job specified in user's utterance. What should be done next to satisfy the user? Please generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
                else:
                    end_prompt = f"""

1. The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be done next to satisfy the user? Please generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
            prompt += end_prompt
            temp['question'] = prompt
            temp['input'] = ''
            if 'ToolSearcher' in line['groundtruth']:
                temp['answer'] = line['output']
            else:
                temp['answer'] = line['groundtruth']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list





def load_API_BANK_plan_only(path, n_row, minimum_change_or_zero_shot = False):
    data_list = []
    count = 0
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            
            if '->' in line['input']:
                content = line['input']
                api_request_regex = r"API-Request:"
                match = re.search(api_request_regex, content)
                api_history = content[match.start():]
                cc = api_history.count('API-Request:')
                for i in range(1, api_history.count('API-Request:') + 1):
                    if cc > 1:
                        api_history = api_history.replace('API-Request:', f"API-Request{i}:", i)
                for i in range(1, cc+ 1):
                    api_history = api_history.replace(f'API-Request{i}:', f"""{i}. 
API-Request:""")

                api_history = api_history.replace('->', """
Received API Response:""")
                api_history = api_history.replace("\nGenerate API Request: ", '')

                pattern = r"\nUser:(.*?)\nAPI-Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" + \
f"""


Previous API-Request History:

{api_history}



Available information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response"""
            else:
                content = line['input']
                pattern = r"\nUser:(.*?)\nGenerate API Request:"
                user_utterance = re.findall(pattern, content, re.DOTALL)
                user_utterance = user_utterance[0].strip()
                user_utterance = user_utterance.replace('\nGenerate API Request:', '')

                prompt  = \
f"""
The current time is {{time}}.

Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.

User's utterance: {user_utterance}
""" + \
"""
Available API descriptions:
{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}
""" 
                
            if minimum_change_or_zero_shot: 
                if '->' in line['input']:
                    end_prompt = f"""

1. The previous API-request has help solve part of job specified in user's utterance. What should be done next to satisfy the user? Please solve the rest of the problem step by step. 
2. Remember to generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
                else:
                    end_prompt = f"""

1. The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be done next to satisfy the user? Please solve the rest of the problem step by step. 
2. Remember to generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
            else:
                if '->' in line['input']:
                    end_prompt = f"""

1. The previous API-request has help solve part of job specified in user's utterance. What should be done next to satisfy the user? Please generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
                else:
                    end_prompt = f"""

1. The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be done next to satisfy the user? Please generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] according to the API description. """
            prompt += end_prompt
            temp['question'] = prompt
            temp['input'] = ''
            try:
                if 'ToolSearcher' in line['groundtruth']:
                    temp['answer'] = line['output']
                    count += 1
                else:
                    temp['answer'] = line['groundtruth']
            except:
                if 'ToolSearcher' in line['output']:
                    count += 1
                temp['answer'] = line['output']
            try:
                temp['sample_id'] = line['sample_id']
                temp['api_id'] = line['api_id']
            except:
                a = 1
            data_list.append(temp)
            if count == n_row:
                break
    return data_list




def load_ANLI(path, n_row, zeroshot = False, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            
            prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

Determine whether the statement is entailment, contradiction, or neutral given the context.
Context: {line['premise']}
Statement: {line['hypothesis']}

Please inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'
"""
            if zeroshot:
                prompt += \
f"""

Please put the final answer after Final Answer(answer eigher entailment,contradiction or neutral): after you finish inferencing. 

Format:
INFERENCE

Final Answer (Entailment/Neutral/Contradiction):
"""         
            try:
                answer_item = line['answer']
            except:
                answer_item = line['gold_label']
            if finetune:
                answer_item = f"""The answer is {answer_item}, because {line['reason']}

Final Answer: {line['gold_label']}"""
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer_item
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            temp['reason'] = line['reason']
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_ANLI_vanilla_initial_prediction(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            
            prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

Determine whether the statement is entailment, contradiction, or neutral given the context.
Context: {line['premise']}
Statement: {line['hypothesis']}
"""
            try:
                answer_item = line['answer']
            except:
                answer_item = line['gold_label']
            if finetune:
                answer_item = f"""The answer is {answer_item}, because {line['reason']}

Final Answer: {line['gold_label']}"""
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer_item
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            temp['reason'] = line['reason']
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list



def load_ANLI_low_p(path, n_row, finetune_on_gt = False, zeroshot = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            
            prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

Determine whether the statement is entailment, contradiction, or neutral given the context.
Context: {line['premise']}
Statement: {line['hypothesis']}
"""
                        
            try:
                answer_item = line['output']
            except:
                answer_item = line['gold_label']
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer_item
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            temp['reason'] = line['reason']
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_ANLI_jan_30_try(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

Determine whether the statement is entailment, contradiction, or neutral given the context.
Context: {line['premise']}
Statement: {line['hypothesis']}
"""

# You already know the answer is {line['gold_label']}. Please solve this problem while assuming you don’t know the answer."""
            answer_item = line['prediction']
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer_item
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            temp['reason'] = line['reason']
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_ANLI_sft_gtreason(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context. Analyze then answer. 

Format:
Analyzation: 
Answer: 
"""
            answer_item = f""""Analyzation: {line['reason']}
Answer: {line['gold_label']}"""
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer_item
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            temp['reason'] = line['reason']
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_ANLI_simplified(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            prompt = f"""We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

This is a NLI task. Please determine whether the statement is entailment, contradiction, or neutral given the context.
"""
#             answer_item = f""""Analyzation: {line['reason']}
# Answer: {line['gold_label']}"""
            try:
                answer_item = line['output']
            except:
                answer_item = line['gold_label']
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer_item
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            temp['reason'] = line['reason']
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list



# def load_ANLI(path, n_row):
#     data_list = []
#     with open(path, 'r') as file:
#         data = json.load(file)
#         for line in data:
#             temp = {}
#             prompt = f"""We know the definetion of entailment, contradiction and neutral is
# Entailment: The statement is definitely true given the context.
# Contradiction: The statement is definitely false given the context.
# Neutral: The truth of the statement is undetermined or irrelevant given the context.

# We have 
# Context: {line['premise']}
# Statement: {line['hypothesis']}

# Determine whether the statement is entailment, contradiction, or neutral given the context. Analyze then answer. 

# Format:
# ANALYZATION: 
# ANSWER: 
# """
# #             answer_item = f""""Analyzation: {line['reason']}
# # Answer: {line['gold_label']}"""
#             answer_item = line['output']
#             temp['question'] = prompt
#             temp['input'] = ''
#             temp['gold_label'] = line['gold_label']
#             temp['answer'] = answer_item
#             temp['premise'] = line['premise']
#             temp['hypothesis'] = line['hypothesis']
#             temp['reason'] = line['reason']
#             data_list.append(temp)
        
#     data_list = data_list[:n_row]
#     return data_list



def load_MNLI(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            if finetune:
                prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context. You may answer directly. 
"""
            else:
                prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

1. Determine whether the statement is entailment, contradiction, or neutral given the context. 
2. Analyze then answer. 
"""
#             answer_item = f""""Analyzation: {line['reason']}
# Answer: {line['gold_label']}"""
            try:
                answer_item = line['answer']
            except:
                answer_item = line['gold_label']
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer_item
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list



def load_MNLI_step_1(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            groundtruth = line['gold_label']
            if finetune:
                prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context. You may answer directly. 
"""
            else:
                prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

1. Determine whether the statement is entailment, contradiction, or neutral given the context. Analyze then answer. 
2. You already know the answer is {groundtruth}. Please solve this problem while assuming you do not know the answer.
"""
#             answer_item = f""""Analyzation: {line['reason']}
# Answer: {line['gold_label']}"""
            try:
                answer_item = line['answer']
            except:
                answer_item = line['gold_label']
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer_item
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_ESNLI(path, n_row, finetune = False, zeroshot = False, meta_math_template = False, self_distillation = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            if finetune:
                prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context. You may answer (Entailment/Neutral/Contradiction) directly.
"""
            elif zeroshot:
                prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context. You may answer (Entailment/Neutral/Contradiction) directly.

Format:
Inference: INFERENCE_HERE
Final Answer: (Entailment/Neutral/Contradiction)_HERE
"""

            else:
                prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context. 
"""
            if finetune:
                prompt += \
f"""
Please choose the option directly. Final Answer: (entailment/contradiction/neutral)
"""
            else:
                prompt += \
f"""
Please inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'
"""
            if finetune:
                answer_item = line['answer']
                explaination = line['explanation_1']
                explaination = explaination.lower()
                explaination = explaination.replace('.', ',')
                answer = f"""Because {explaination} the answer is {answer_item}.
Final Answer: {answer_item}"""
            else:
                answer = line['answer']

            if meta_math_template:
                explanation_1 = line['explanation_1']
#                 prompt = f"""
# We know the definetion of entailment, contradiction and neutral is
# Entailment: The statement is definitely true given the context.
# Contradiction: The statement is definitely false given the context.
# Neutral: The truth of the statement is undetermined or irrelevant given the context.

# We have
# Context: {line['premise']}
# Statement: {line['hypothesis']}

# Please determine whether the statement is entailment, contradiction, or neutral given the context.

# You have to inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'
# """
                prompt = f"""
Given the context: "{line['premise']}" and the statement: "{line['hypothesis']}"

Please determine whether the statement is true, false, or undetermined given the context.

You have to inference first, then provide the answer (true/false/undetermined).
"""
                prompt = meta_math_complete_instruction(prompt)
#                 prompt += f"""

# Hint: {explanation_1}"""

            temp['question'] = prompt
            original_question = f"""We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context.'
"""
            temp['original_question'] = original_question
            
            self_distillation_question = \
f"""Below are an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{original_question}
### Response:"""
            line['self_distillation_question'] = self_distillation_question
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            if self_distillation:
                line['question'] = self_distillation_question
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list



def load_SCITAIL(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            if finetune:
                prompt = f"""We know the definetion of entailment and neutral are
Entailment: The context support the statement
Neutral: The context not support the statement

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, or neutral given the context. You may answer directly (Entailment/Neutral).
"""
            else:
                prompt = f"""We know the definetion of entailment and neutral are
Entailment: The context support the statement
Neutral: The context not support the statement

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, or neutral given the context. 
Make sure to analyze first, then provide the answer (Entailment/Neutral).
"""
            try:
                answer_item = line['answer']
            except:
                answer_item = line['gold_label']
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer_item
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_PIQA(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            goal = line['goal']
            sol1 = line['sol1']
            sol2 = line['sol2']
            label = line['gold_label']
            prompt = \
f"""Given the question: {goal}

What option is correct?
Option 1: {sol1}
Option 2: {sol2}
"""
            if finetune:
                prompt += \
f"""
Please choose the option directly. Answer: (1/2)
"""
            else:
                prompt += \
f"""
Please inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'
"""
                
            temp['question'] = prompt
            temp['input'] = ''
            temp['sol1'] = line['sol1']
            temp['sol2'] = line['sol2']
            temp['gold_label'] = str(label)
            temp['answer'] = str(label)
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_BOOLQ(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            question = line['question']
            passage = line['passage']
            label = line['gold_label']
            prompt = \
f"""Given the context: {passage}

{question}?
"""
            temp['original_question'] = prompt
            if finetune:
                prompt += \
f"""
Please answer True or False directly. Answer: (True/False)
"""
            else:
                prompt += \
f"""
Please inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'
"""
                
            temp['question'] = prompt
            temp['input'] = ''
            temp['passage'] = passage
            temp['gold_label'] = str(label)
            temp['answer'] = str(label)
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_WINOGRANDE(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        data = data[:n_row]
        for line in data:
            temp = {}
            sentence = line['sentence']
            option1 = line['option1']
            option2 = line['option2']
            label = line['gold_label']
            prompt = \
f"""Given the question: {sentence}

What option is correct?
Option 1: {option1}
Option 2: {option2}
"""
            temp['original_question'] = prompt
            if finetune:
                prompt += \
f"""
Please choose the option directly (1/2).
"""
            else:
                prompt += \
f"""
Please inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'
"""
                
            temp['question'] = prompt
            temp['input'] = ''
            temp['option1'] = line['option1']
            temp['option2'] = line['option2']
            temp['gold_label'] = str(label)
            temp['answer'] = str(label)
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list



def load_TRIVIAQA(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            question = line['question']
            gold_label = line['gold_label']
            evidence = line['evidence']

            if finetune:
                prompt = \
f"""We have a question and we found some relevant context from wikipedia. Please try to answer the question given the context. Please notice that the context might not all be correct and there might not necessarily be answer in the context.

We have 
Question: {question}
Context: {evidence}

You may provide the final answer directly.
"""
            else:
                prompt = \
f"""We have a question and we found some relevant context from wikipedia. Please try to answer the question given the context. Please notice that the context might not all be correct and there might not necessarily be answer in the context.

We have 
Question: {question}
Context: {evidence}

You may inference first, then provide the final answer.
"""
            try:
                answer_item = line['answer']
            except:
                answer_item = gold_label
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = gold_label
            temp['answer'] = answer_item
            temp['evidence'] = evidence
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_SQUAD(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for i, line in enumerate(data):
            temp = {}
            question = line['question']
            context = line['context']
            answer = line['gold_label']['text']
            if answer == []:
                gold_label = 'No answer'
            else:
                gold_label = answer[0]
                prompt = \
f"""Given the question: {question} and the context: {context}

What is the answer?
"""
                if finetune:
                    prompt += f"""
Please directly provide the final answer (text span) at the end, after 'Final Answer:'
"""
                else:
                    prompt += \
f"""
Please inference first, then provide the final answer (text span) at the end, after 'Final Answer:'
"""
    #             else:
    #                 prompt += \
    # f"""
    # Please inference first, then provide the answer.
    # """
                if finetune:
                    answer = 'Final Answer: ' + gold_label
                temp['question'] = prompt
                temp['input'] = ''
                temp['context'] = context
                temp['gold_label'] = gold_label
                temp['answer'] = gold_label
                temp['answer_list'] = answer
                data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_NATURAL_QUESTIONS(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            question = line['question']
            gold_label = line['gold_label']
            evidence = line['evidence']

            if finetune:
                prompt = \
f"""We have a question and we found some relevant context from wikipedia. Please try to answer the question given the context. Please notice that the context might not all be correct and there might not necessarily be answer in the context.

We have 
Question: {question}
Context: {evidence}

You may provide the final answer directly.
"""
            else:
                prompt = \
f"""We have a question and we found some relevant context from wikipedia. Please try to answer the question given the context. Please notice that the context might not all be correct and there might not necessarily be answer in the context.

We have 
Question: {question}
Context: {evidence}

You may inference first, then provide the final answer.
"""
            try:
                answer_item = line['answer']
            except:
                answer_item = gold_label
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = gold_label
            temp['answer'] = answer_item
            temp['evidence'] = evidence
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


# def load_MMLU_old(path, n_row, finetune = False):
#     data_list = []
#     with open(path, 'r') as file:
#         data = json.load(file)
#         for line in data:
#             temp = {}
#             question = line['question']
#             A = line['A']
#             B = line['B']
#             C = line['C']
#             D = line['D']
#             gold_label = line['answer']
#             prompt = \
# f"""Given the question: {question}

# and the options:
# A: {A}
# B: {B}
# C: {C}
# D: {D}

# What is the answer?
# """
#             if finetune:
#                 prompt += \
# f"""
# Please answer directly (A/B/C/D).
# """
#             else:
#                 prompt += \
# f"""
# Please inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'
# """
                
#             temp['question'] = prompt
#             temp['input'] = ''
#             temp['A'] = line['A']
#             temp['B'] = line['B']
#             temp['C'] = line['C']
#             temp['D'] = line['D']
#             temp['gold_label'] = gold_label
#             temp['answer'] = gold_label
#             data_list.append(temp)
        
#     data_list = data_list[:n_row]
#     return data_list

def load_MMLU(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            question = line['question']
            subject = line['subject']
            A = line['choices'][0]
            B = line['choices'][1]
            C = line['choices'][2]
            D = line['choices'][3]
            gold_label = line['answer']
            prompt = \
f"""Given the question: {question}

and the options:
A: {A}
B: {B}
C: {C}
D: {D}

What is the answer?
"""
            if finetune:
                prompt += \
f"""
Please answer directly (A/B/C/D).
"""
            else:
                prompt += \
f"""
Please inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'
"""
                
            temp['question'] = prompt
            temp['input'] = ''
            temp['subject'] = subject
            temp['A'] = A
            temp['B'] = B
            temp['C'] = C
            temp['D'] = D
            temp['gold_label'] = gold_label
            temp['answer'] = gold_label
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_AGIEVAL(path, n_row, finetune = False, category = 'logiqa'):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            passage = line['passage']
            question = line['question']
            options = line['options']
            option_item = \
f"""{options[0]}
{options[1]}
{options[2]}
{options[3]}
"""
            gold_label = line['label']
            prompt = \
f"""Given the statement: {passage}

and the question: {question}

and the options:
{option_item}

What is the answer?

"""
            if category == 'sat':
                prompt = \
f"""Given the context: {passage}

and the question: {question}

and the options:
{option_item}

What is the answer?

"""
            if finetune:
                prompt += \
f"""
Please answer directly. Answer: (A/B/C/D)
"""
            else:
                prompt += \
f"""
Please inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'
"""
                
            temp['question'] = prompt
            temp['passage'] = passage
            temp['input'] = ''
            temp['A'] = options[0]
            temp['B'] = options[1]
            temp['C'] = options[2]
            temp['D'] = options[3]
            temp['gold_label'] = gold_label
            temp['answer'] = gold_label
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_ECQA(path, n_row, finetune = False, use_gt_rationale = False, zeroshot = False, meta_math_template = False, self_distillation = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            question = line['question']
            o1 = line['1']
            o2 = line['2']
            o3 = line['3']
            o4 = line['4']
            o5 = line['5']
            gold_label = line['gold_label']
            answer = line['answer']
            pos_explaination = line['pos_explaination']
            neg_explaination = line['neg_explaination']
            combined_explaination = line['combined_explaination']
            prompt = \
f"""We have the question: {question}
and the options:
(1): {o1}
(2): {o2}
(3): {o3}
(4): {o4}
(5): {o5}

what is the correct option?
"""
            self_distillation_question = \
f"""Below are an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{prompt}
### Response:"""
            line['self_distillation_question'] = self_distillation_question
            line['original_question'] = prompt
            if finetune and not use_gt_rationale:
                prompt += \
f"""
Please answer True or False directly. Answer: (1/2/3/4/5)
"""
            else:
                prompt += \
f"""
Please inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'
"""
            if zeroshot: 
                prompt += """
Format:
Inference: INFERENCE_HERE
Final Answer: (1/2/3/4/5)_HERE"""
            if finetune and use_gt_rationale:
                answer = f"""Inference: {combined_explaination}

Final Answer: """ + answer
            
            # prompt = prompt.replace("\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'\n", '')
            if meta_math_template:
                prompt = meta_math_complete_instruction(prompt)

            line['answer'] = answer
            line['question'] = prompt
            if self_distillation:
                line['question'] = self_distillation_question
            data_list.append(line)
        
    data_list = data_list[:n_row]
    return data_list

def load_ANLI_march_9(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            if finetune:
                prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context. You may answer (Entailment/Neutral/Contradiction) directly.
"""
            else:
                prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context. 

You should analyze first

Please provide the final answer (Entailment/Neutral/Contradiction) at the end, after 'Final Answer:'
"""
            try:
                answer_item = line['answer']
            except:
                answer_item = line['gold_label']
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer_item
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list




def load_ANLI_march_9_old(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            if finetune:
                prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context. You may answer (Entailment/Neutral/Contradiction) directly.
"""
            else:
                prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context. 
Make sure to analyze first, then provide the answer (Entailment/Neutral/Contradiction).
"""
            try:
                answer_item = line['answer']
            except:
                answer_item = line['gold_label']
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer_item
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list

def load_MNLI_vanilla(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            prompt = f"""We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.

We have 
Context: {line['premise']}
Statement: {line['hypothesis']}

Determine whether the statement is entailment, contradiction, or neutral given the context. You may answer directly. 
"""
            try:
                answer_item = line['answer']
            except:
                answer_item = line['gold_label']
            temp['question'] = prompt
            temp['input'] = ''
            temp['gold_label'] = line['gold_label']
            temp['answer'] = answer_item
            temp['premise'] = line['premise']
            temp['hypothesis'] = line['hypothesis']
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_APPS(file_path, n_row, minimum_change_or_zero_shot = False):
    data = []
    with open(file_path, 'r') as file:
        if not minimum_change_or_zero_shot:
            for line in file:
                json_object = json.loads(line)
                temp = {}
                difficulty = json_object['difficulty']
                starter_code = json_object['starter_code']
                if difficulty == 'introductory':
                    if starter_code != '':
                        id = json_object['id']
                        question = json_object['question']
                        solutions = json_object['solutions']
                        solutions = json.loads(solutions)
                        solutions = solutions[0]

                        temp['original_question'] = question
                        temp['input'] = ''
                        temp['id'] = id
                        temp['solutions'] = solutions
                        temp['answer'] = solutions
                        temp['starter_code'] = starter_code
                        
                        modified_question = f"""Please try your best to solve this code puzzle. Show me the code.

# Puzzle: {question}"""
                        temp['question'] = modified_question
                        data.append(temp)
        else:
            data = []
            data = json.load(file)
    return data[:n_row]

#             if minimum_change:
#                 modified_prompt = f"""
# 1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.
# 2. You need to try your best to solve this problem. 
# 3. Since entry point is already given, you only need to fill out the context under the entry point. Remember that you need to start with 4 white space(indentation) since you your answer will be placed under the entry point. Place your code after FINAL ANSWER:

# Problem: {prompt}
# Entry point: {entry_point}"""
#             else:
#                 modified_prompt = f"""
# 1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.
# 2. You need to try your best to solve this problem. 
# 3. Since entry point is already given, you only need to fill out the context under the entry point. Remember that you need to start with 4 white space(indentation) since you your answer will be placed under the entry point.

# Example1:
# Example1-Promblem: \n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n
# Example1-Entry point: truncate_number
# Example1-ANSWER: 
#     return number % 1.0\n


# Example2:
# Example2-Promblem: from typing import List\n\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n
# Example2-Entry point: below_zero
# Example2-ANSWER: 
#     balance = 0\n\n    for op in operations:\n        balance += op\n        if balance < 0:\n            return True\n\n    return False\n


# Now, please solve the following problem. Remember that you will write the code under the given entrypoint starting with four white space(indentation)

# Problem: {prompt}
# Entry point: {entry_point}"""
#             if not minimum_change:
#                 modified_prompt += '\n'
#                 modified_prompt += \
# f"""ANSWER(directly output the code under the entry point {entry_point} with proper indentation as shown in the example):\n"""

#             else:
#                 modified_prompt = f"""Please try your best to solve this code puzzle. Show me the code.

# Puzzle: {prompt}

# Entry point: {entry_point}"""
                # answer_item = canonical_solution
            
def load_CODE(path, n_row, finetune = False, meta_math_template = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            prompt = line['prompt']
            entry_point = line['entry_point']
            canonical_solution = line['canonical_solution']
            
            modified_prompt = f"""
1. Please solve the following problem using only the information and resources provided in the question. Do not use external functions not mentioned or implied in the problem statement.
2. You need to try your best to solve this problem. 
3. Since entry point is already given, you only need to fill out the context under the entry point. Remember that you need to start with 4 white space(indentation) since you your answer will be placed under the entry point. Place your code after 'Final Answer:'

Problem: {prompt}
Entry point: {entry_point}

"""
            if finetune:
                modified_prompt += \
f"""
Provide your answer directly. Please provide the final answer (code) at the end, after 'Final Answer:'
Be careful about the spacing. The code after 'Final Answer: ' should start with 4 whitespace for indentation because it is under the entrypoint.
"""
                answer = line['answer']
            else:
                modified_prompt += \
f"""
Please inference first, then provide the final answer (code) at the end, after 'Final Answer:'
Be careful about the spacing. The code after 'Final Answer: ' should start with 4 whitespace for indentation because it is under the entrypoint.
"""
                answer = canonical_solution

            if meta_math_template:
                modified_prompt = meta_math_complete_instruction(modified_prompt)
            line['question'] = modified_prompt
            line['ground_truth'] = canonical_solution
            line['input'] = ''
            line['answer'] = answer
            data_list.append(line)
        
    data_list = data_list[:n_row]
    return data_list

# def load_CODE_code_only(path, n_row, finetune = False):
#     data_list = []
#     with open(path, 'r') as file:
#         data = json.load(file)
#         for line in data:
#             prompt = line['prompt']
#             entry_point = line['entry_point']
#             canonical_solution = line['canonical_solution']
            
#             modified_prompt = f"""
# 1. Please solve the following problem using only the information and resources provided in the question. Do not use external functions not mentioned or implied in the problem statement.
# 2. You need to try your best to solve this problem. 
# 3. Since entry point is already given, you only need to fill out the context under the entry point. Remember that you need to start with 4 white space(indentation) since you your answer will be placed under the entry point. Place your code after 'Final Answer:'

# Problem: {prompt}
# Entry point: {entry_point}

# """
#             if finetune:
#                 modified_prompt += \
# f"""
# Provide your answer directly. Please provide the final answer (code) at the end, after 'Final Answer:'
# Be careful about the spacing. The code after 'Final Answer: ' should start with 4 whitespace for indentation because it is under the entrypoint.
# """
#                 answer = line['answer']
#             else:
#                 modified_prompt += \
# f"""
# Please inference first, then provide the final answer (code) at the end, after 'Final Answer:'
# Be careful about the spacing. The code after 'Final Answer: ' should start with 4 whitespace for indentation because it is under the entrypoint.
# """
#                 answer = canonical_solution
#             line['question'] = modified_prompt
#             line['ground_truth'] = canonical_solution
#             line['input'] = ''
#             line['answer'] = answer
#             data_list.append(line)
        
#     data_list = data_list[:n_row]
#     return data_list

def load_CODE_code_only(path, n_row, meta_math_template = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            prompt = line['prompt']
            entry_point = line['entry_point']            
            modified_prompt = f"""
1. Please solve the following problem using only the information and resources provided in the question. Do not use external functions not mentioned or implied in the problem statement.
2. You need to try your best to solve this problem. 
3. Since entry point is already given, you only need to fill out the context under the entry point. Remember that you need to start with 4 white space(indentation) since you your answer will be placed under the entry point. Place your code after 'Final Answer:'

Problem: {prompt}
Entry point: {entry_point}

Provide your answer directly. Please provide the final answer (code) at the end, after 'Final Answer:'
Be careful about the spacing. The code after 'Final Answer:' should start with 4 whitespace for indentation because it is under the entry point.
"""
            if meta_math_template:
                modified_prompt = meta_math_complete_instruction(modified_prompt)
            line['question'] = modified_prompt
            data_list.append(line)
    return data_list[:n_row]


# def load_CODE_code_only(path, n_row, minimum_change = False):
#     data_list = []
#     with open(path, 'r') as file:
#         data = json.load(file)
#         for line in data:
#             modified_prompt = f"""
# 1. Please solve the promblem. 
# 2. Since entry point is already given, you only need to fill out the context under the entry point. Remember that you need to start with 4 white space(indentation) since you your answer will be placed under the entry point. Place your code after 'Final Answer:'
# 3. Directly generate the code with no explaination.

# Problem: {line['prompt']}
# Entry point: {line['entry_point']}

# """
            
#             line['question'] = modified_prompt
#             line['input'] = ''
#             line['answer'] = ''
#             data_list.append(line)
        
#     data_list = data_list[:n_row]
#     return data_list


def load_CODE_Step_by_step(path, n_row, minimum_change = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            task_id = line['task_id']
            prompt = line['prompt']
            entry_point = line['entry_point']
            canonical_solution = line['canonical_solution']
            test = line['test']

            modified_prompt = f"""
1. let's solve the problem step by step. 
2. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.
2. You need to try your best to solve this problem. 


Problem: {prompt}"""
            if minimum_change:
                answer = line['answer']
                answer_item = answer
            else:
                answer_item = canonical_solution
            
            temp['question'] = modified_prompt
            temp['prompt'] = prompt 
            temp['input'] = ''
            temp['task_id'] = task_id
            temp['ground_truth'] = canonical_solution
            temp['answer'] = answer_item
            temp['entry_point'] = entry_point
            temp['test'] = test
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


def load_CODE_vanilla_total(path, n_row, minimum_change = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for i, line in enumerate(data):
            temp = {}
            task_id = line['task_id']
            prompt = line['prompt']
            entry_point = line['entry_point']
            canonical_solution = line['canonical_solution']
            test = line['test']

            modified_prompt = \
f"""1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.
2. You need to try your best to solve this problem. 
3. You only have to complete and show the code inside the entrypoint {entry_point}. Please be careful about the spacing. The code inside the entrypoint should start with 4 whitespace for indentation.

Problem: {prompt}"""
            if minimum_change:
                answer = line['answer']
                answer_item = answer
            else:
                answer_item = canonical_solution
            
            temp['question'] = modified_prompt
            temp['prompt'] = prompt 
            temp['input'] = ''
            temp['task_id'] = task_id
            temp['ground_truth'] = canonical_solution
            temp['answer'] = answer_item
            temp['entry_point'] = entry_point
            temp['test'] = test
            data_list.append(temp)
        
    data_list = data_list[:n_row]
    return data_list


# def load_CODE_vanilla_total_pure(path, n_row, minimum_change = False):
#     data_list = []
#     with open(path, 'r') as file:
#         data = json.load(file)
#         for line in data:
#             temp = {}
#             task_id = line['task_id']
#             prompt = line['prompt']
#             entry_point = line['entry_point']
#             canonical_solution = line['canonical_solution']
#             test = line['test']

#             modified_prompt = f"""
# 1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.
# 2. You need to try your best to solve this problem. 


# Problem: {prompt}"""
#             if minimum_change:
#                 answer = line['answer']
#                 answer_item = answer
#             else:
#                 answer_item = canonical_solution
            
#             temp['question'] = modified_prompt
#             temp['prompt'] = prompt 
#             temp['input'] = ''
#             temp['task_id'] = task_id
#             temp['ground_truth'] = canonical_solution
#             temp['answer'] = answer_item
#             temp['entry_point'] = entry_point
#             temp['test'] = test
#             data_list.append(temp)
        
#     data_list = data_list[:n_row]
#     return data_list

def load_APPS_Step_by_step(file_path, n_row, minimum_change_or_zeroshot = False):
    data = []
    with open(file_path, 'r') as file:
        data_list = json.load(file)
        count = 0
        for i, json_object in enumerate(data_list):
            temp = {}
            contains_difficulty_level = False
            difficulty = ''
            if 'difficulty' in json_object:
                contains_difficulty_level = True
                difficulty = json_object['difficulty']
            starter_code = json_object['starter_code']
            if difficulty == 'introductory' or not contains_difficulty_level:
            # if difficulty == 'interview' or not contains_difficulty_level:
                if starter_code == '':
                # if 1:
                    id = json_object['id']
                    question = json_object['question']
                    if minimum_change_or_zeroshot:
                        modified_prompt = \
f"""1. When solving the code challenge, you cannot import the library except the library in the AVAILABLE LIBRARY. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.
2. Please try your best to solve the python code challenge step by step. 


AVAILABLE LIBRARY:
import sysimport time
import itertools
from itertools import accumulate, product, permutations, combinations
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
import math
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
import fractions
from typing import List, Tuple
import numpy as np
import random
import heapq
from heapq import *

Python Code Challenge: {question}"""
                    else:
                        modified_prompt = \
f"""When solving the code challenge, you cannot import the library except the library in the AVAILABLE LIBRARY. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.

AVAILABLE LIBRARY:
import sysimport time
import itertools
from itertools import accumulate, product, permutations, combinations
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
import math
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
import fractions
from typing import List, Tuple
import numpy as np
import random
import heapq
from heapq import *

Python Code Challenge: {question}"""
                    
                    solutions = json_object['solutions']
                    temp['original_question'] = question
                    temp['input'] = ''
                    temp['id'] = id
                    temp['solutions'] = solutions
                    if minimum_change_or_zeroshot:
                        try:
                            temp['answer'] = json_object['answer'] # when trainning
                        except:
                            temp['answer'] = solutions # when testing
                    else:
                        temp['answer'] = solutions
                    temp['starter_code'] = starter_code
                    temp['question'] = modified_prompt
                    data.append(temp)
                    if count > n_row:
                        break
                    count += 1
    return data[:n_row]


def load_APPS_vanilla(file_path, n_row, minimum_change_or_zeroshot = False):
    data = []
    with open(file_path, 'r') as file:
        data_list = json.load(file)
        count = 0
        for i, json_object in enumerate(data_list):
            temp = {}
            contains_difficulty_level = False
            difficulty = ''
            if 'difficulty' in json_object:
                contains_difficulty_level = True
                difficulty = json_object['difficulty']
            starter_code = json_object['starter_code']
            if difficulty == 'introductory' or not contains_difficulty_level:
            # if difficulty == 'interview' or not contains_difficulty_level:
                if starter_code == '':
                # if 1:
                    id = json_object['id']
                    question = json_object['question']
                    if minimum_change_or_zeroshot:
                        modified_prompt = \
f"""Python Code Challenge: {question}"""
                    else:
                        modified_prompt = \
f"""Python Code Challenge: {question}"""
                    
                    solutions = json_object['solutions']
                    temp['original_question'] = question
                    temp['input'] = ''
                    temp['id'] = id
                    temp['solutions'] = solutions
                    if minimum_change_or_zeroshot:
                        try:
                            temp['answer'] = json_object['answer'] # when trainning
                        except:
                            temp['answer'] = solutions # when testing
                    else:
                        temp['answer'] = solutions
                    temp['starter_code'] = starter_code
                    temp['question'] = modified_prompt
                    data.append(temp)
                    if count > n_row:
                        break
                    count += 1
    return data[:n_row]

def load_MBPP(file_path, n_row):
    with open(file_path, 'r') as file:
        data_list = json.load(file) 
    return data_list[:n_row]

def load_MBPP_march_8(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            prompt = line['original_question']
            test_example = line['test']
            modified_prompt = \
f"""You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code.

Task: {prompt}
Test Example: {test_example[0]}

Let's solve the task step by step.
"""
            temp['question'] = modified_prompt
            temp['original_question'] = line['original_question'] 
            temp['answer'] = line['answer'] 
            temp['input'] = ''
            data_list.append(temp)
    return data_list[:n_row]

def load_MBPP_Step_by_step(path, n_row, minimum_change = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            prompt = line['prompt']
            modified_prompt = f"""
1. let's solve the problem step by step. 
2. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.
2. You need to try your best to solve this problem. 


Problem: {prompt}"""
            if minimum_change:
                answer = line['answer']
                answer_item = answer
            else:
                # answer_item = canonical_solution
                a = 1
            
            temp['question'] = modified_prompt
            temp['prompt'] = prompt 
            temp['input'] = ''
            data_list.append(temp)
    return data_list[:n_row]



def load_MBPP_march_9(path, n_row):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            prompt = line['original_question']
            test_example = line['test']
            modified_prompt = \
f"""You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code.

Task: {prompt}
Test Example: {test_example[0]}


You should analyze first

Then provide the answer at the end.
"""
            temp['question'] = modified_prompt
            temp['original_question'] = line['original_question'] 
            temp['answer'] = line['answer'] 
            temp['input'] = ''
            data_list.append(temp)
    return data_list[:n_row]


def load_MBPP_april_5(path, n_row, finetune = False, data_creation = False, meta_math_template = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            prompt = line['original_question']
            test_example = line['test']
            modified_prompt = \
f"""You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. The test is only used to show you the input structure. You do not need to run the test.


Task: {prompt}
Test Example: {test_example[0]}
"""
            if data_creation:
                modified_prompt = modified_prompt
            else:
                if finetune:
                    modified_prompt += \
f"""
Provide your answer directly. Please provide the final answer (code) at the end, after 'Final Answer:'
"""
                else:
                    modified_prompt += \
f"""
Please inference first, then provide the final answer (code) at the end, after 'Final Answer:'
"""
            if meta_math_template:
                modified_prompt = meta_math_complete_instruction(modified_prompt)

            line['question'] = modified_prompt
            data_list.append(line)
    return data_list[:n_row]

def load_MBPP_code_only(path, n_row, meta_math_template = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            prompt = line['original_question']
            test_example = line['test_case']
            
            modified_prompt = \
f"""You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. The test is only used to show you the input structure. You do not need to run the test.


Task: {prompt}
Test Example: {test_example}

Provide your answer directly without any explaination. Please provide the final answer (code) at the end, after 'Final Answer:'
"""
            
            if meta_math_template:
                modified_prompt = meta_math_complete_instruction(modified_prompt)

            line['question'] = modified_prompt
            data_list.append(line)
    return data_list[:n_row]

def find_break_ids(input_ids):
    for i in range(len(input_ids)):
        if input_ids[i] == 25580:
            if input_ids[i-1] == 29914:
                if input_ids[i+1] == 29962:
                    return i + 2
    return False

class TRAIN_Dataset(Dataset):
    def __init__(self, data, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.ignore_index = IGNORE_INDEX
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_too_long = False
        item = self.data[idx]
        question = item['question']
        answer = item['answer'] + '</s>'
        question = system_message + f"\n\n {question} [/INST] "
        input = question + answer 

        encoding = self.tokenizer(input, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_length)
        input_ids = encoding['input_ids'][0]

        # Check and adjust the break_id
        try:
            break_id = find_break_ids(input_ids)
        except Exception as e:
            print(f"Error finding break_id: {e}")
            break_id = None

        labels = input_ids.clone()
        if break_id is not None:
            for i in range(break_id):
                labels[i] = self.ignore_index
        labels[:-1] = labels[1:].clone()
        labels[-1] = self.ignore_index

        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'][0],
            'labels': labels,
            'data_too_long': data_too_long
        }

class TEST_Dataset(Dataset):
    def __init__(self, data, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_too_long = False
        item = self.data[idx]
        question = item['question']
        question = system_message + f"\n\n {question} [/INST] "

        encoding = self.tokenizer(question, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        input_ids = encoding['input_ids'][0]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            data_too_long = True

        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'][0],
            'data_too_long': data_too_long
        }

def dataloader(data_list, type, use_trainner = False):
    if type == 'test':
        dataset = TEST_Dataset(data_list, data_loader_config['input_length']) 
    elif type == 'train':
        dataset = TRAIN_Dataset(data_list, train_config['max_length'])
    
    if use_trainner:
        reconstructed_data_list = []
        for item in dataset:
            reconstructed_data_list_temp = {}
            reconstructed_data_list_temp['input_ids'] = item['input_ids']
            reconstructed_data_list_temp['attention_mask'] = item['attention_mask']
            reconstructed_data_list_temp['labels'] = item['labels']
            reconstructed_data_list.append(reconstructed_data_list_temp)
        return reconstructed_data_list
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX
    )
    a = 1
    data_loader = DataLoader(
        dataset,
        batch_size=data_loader_config['batch_size'],
        shuffle=data_loader_config['shuffle'],  # Shuffle data for better training
        num_workers=data_loader_config['num_workers'],  # Adjust based on your machine's capability
        pin_memory=data_loader_config['pin_memory'],  # Faster data transfer to CUDA
        drop_last=data_loader_config['drop_last'],  # Drop the last incomplete batch if dataset size isn't divisible by the batch size
        collate_fn=data_collator
    )
    return data_loader



def extract_after_last_occurrence(text, keyword):
    # Find the last occurrence of the keyword
    last_index = text.rfind(keyword)
    if last_index == -1:
        return text # Return a message if the keyword is not found
    else:
        a = 1
    # Extract the context after the keyword
    extracted_text = text[last_index + len(keyword):]
    return extracted_text.strip()  # Strip any leading or trailing whitespace



def eval_MATH_correctness(predict_item, correct_number):


    def extract_boxed_content(s):
        start = s.rfind('\\boxed{')
        if start == -1:
            return None
        
        end = s.rfind('}')
            
        if end != 0:
            answer = s[start + 7 : end]
            return answer  # 7 is the length of '\\boxed{'
    
    def evaluate_expression_try_best(expr):
        try:
            # Handle LaTeX-style fractions and square roots
            expr = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
            expr = re.sub(r'\\dfrac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
            expr = re.sub(r'\\sqrt\{(.*?)\}', r'(\1) ** 0.5', expr)
            expr = re.sub(r'\\(cfrac|dfrac|frac)\{(.*?)\}\{(.*?)\}', r'(\2) / (\3)', expr)

            expr = re.sub(r'(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)', r'(\1) / (\2)', expr)



            # Evaluate the expression
            result = eval(expr)
            result = float(result)
            return str(result)
        except:
            return False

    
    def extract_last_number(text):
        # New pattern to include LaTeX-style expressions
        # pattern = r'(-?\d+(?:,\d{3})*(?:\.\d+)?%?|-?\d+\/-?\d+%?|-?\d+%?|\\frac\{-?\d+\}\{-?\d+\}|\\dfrac\{.*?\}\{.*?\}|\\cfrac\{.*?\}\{.*?\}|\\sqrt\{.*?\})'
        pattern = r'(-?\d+\/-?\d+|-?\d+(?:,\d{3})*(?:\.\d+)?%?|-?\d+\/-?\d+%?|-?\d+%?|\\frac\{-?\d+\}\{-?\d+\}|\\dfrac\{.*?\}\{.*?\}|\\cfrac\{.*?\}\{.*?\}|\\sqrt\{.*?\})'


        founded_text = extract_boxed_content(text)
        if founded_text:
            if '\\frac' in founded_text or '\\dfrac' in founded_text or '\\cfrac' in founded_text or '\\sqrt' in founded_text or '/' in founded_text:
                extracted_num = evaluate_expression_try_best(founded_text)
                if not extracted_num:
                    return -3333333333333 
                else:
                    return extracted_num
            else: 
                text = founded_text

        # Find all numbers and expressions in the string
        all_numbers = re.findall(pattern, text)

        # Process the last number or expression
        if all_numbers:
            number = all_numbers[-1]
            # Evaluate LaTeX-style expressions
            if '\\frac' in number or '\\dfrac' in number or '\\cfrac' in number or '\\sqrt' in number or '/' in number:
                extracted_num = evaluate_expression_try_best(number)
                if not extracted_num:
                    return -3333333333333 
                else:
                    return extracted_num
            
            # Handle percentages and remove commas from numbers
            is_percentage = '%' in number
            number = number.replace('%', '').replace(',', '')
            
            # Convert to float and adjust for percentage if needed
            number = float(number)
            if is_percentage:
                number /= 100

            return str(number)
        else:
            return -3333333333333 


    extracted_final_answer = extract_last_number(predict_item)
    correct_number = extract_last_number(correct_number)

    if extracted_final_answer == correct_number:
    
        return True
    else:
        return False


def extract_nli_answer(text):
    text = text.lower()
    found_answer_num = text.count('answer:')
    if found_answer_num > 0:
        text = text.split("answer:")[-1].strip()
    neutral_count = 0
    entail_count = 0
    contradiction_count = 0
    for item in text.split():
        if 'neutr' in item:
            neutral_count += 1
        if 'entail' in item:
            entail_count += 1
        if 'contrad' in item:
            contradiction_count += 1
    max_num = max(neutral_count, entail_count, contradiction_count)

    final_answer = ''
    if max_num == 0:
        final_answer = 'null'
    else:
        if max_num == neutral_count:
            final_answer = 'neutral'
            if max_num == entail_count or max_num == contradiction_count:
                final_answer = 'null'
        elif max_num == entail_count:
            final_answer = 'entailment'
            if max_num == neutral_count or max_num == contradiction_count:
                final_answer = 'null'
        else:
            final_answer = 'contradiction'
            if max_num == neutral_count or max_num == entail_count:
                final_answer = 'null'
    return final_answer




def find_first_number(text, numbers=[1, 2, 3, 4, 5]):
    # Initialize variables to store the first number and its index
    first_number = None
    first_index = len(text)  # Start with the highest possible index

    # Loop through the numbers to find which appears first
    for number in numbers:
        index = text.find(str(number))
        # Check if the number is found and appears before the current first number
        if index != -1 and index < first_index:
            first_number = str(number)
            first_index = index

    # Return the first number, or "null" if none of the numbers are found
    return first_number if first_number else "null"


def extract_option_1_to_5(text):
    text = text.lower()
    found_answer_num = text.count('answer:')
    if found_answer_num > 0:
        text = text.split("answer:")[-1].strip()
    else:
        found_answer_num = text.count('answer')
        if found_answer_num > 0:
            text = text.split("answer")[-1].strip()
    
    return_text = find_first_number(text)
    return return_text


def load_dataset(HOME_DIRECTORY, train_task_name, data_name, n_train):
    if 'math' in train_task_name.lower():
        train_task_name = 'MATH'
    
    with open(f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/{data_name}.json', 'r') as f:
        train_data_list = json.load(f)
    train_data_list = train_data_list[:n_train]
    
    minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/{data_name}.json'
    with open(minimum_change_train_path, 'r') as file:
        minimum_change_train_data_list = json.load(file)
    minimum_change_train_data_list = minimum_change_train_data_list[:n_train]

    gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/{data_name}.json'
    with open(gpt4_generated_data_train_path, 'r') as file:
        gpt4_generated_train_data_list = json.load(file)
    gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
    
    intermediate_finetune_file_name = f'{train_task_name}_{data_name}_{n_train}'
    intermediate_finetune_file_path = f"{HOME_DIRECTORY}/intermediate_data/{intermediate_finetune_file_name}.json"
    train_data = []
    for item in train_data_list:
        temp = {}
        temp['instruction'] = item['question']
        temp['output'] = item['answer']
        temp['input'] = ''
        train_data.append(temp)
    with open(intermediate_finetune_file_path, 'w') as json_file:
        json.dump(train_data, json_file, indent=4)

    intermediate_minimum_change_train_file_name = f'{train_task_name}_{data_name}_{n_train}'
    train_data = []
    for item in minimum_change_train_data_list:
        temp = {}
        try:
            temp['instruction'] = item['question']
        except: 
            temp['instruction'] = item['instruction']
        try:
            temp['output'] = item['answer']
        except: 
            temp['output'] = item['output']
        temp['input'] = ''
        train_data.append(temp)
    intermediate_minimum_change_train_file_path = f"{HOME_DIRECTORY}/intermediate_data/{intermediate_minimum_change_train_file_name}.json"
    with open(intermediate_minimum_change_train_file_path, 'w') as json_file:
        json.dump(train_data, json_file, indent=4)

    intermediate_gpt4_generated_data_train_file_name = f'{train_task_name}_{data_name}_{n_train}'
    train_data = []
    for item in gpt4_generated_train_data_list:
        temp = {}
        temp['instruction'] = item['question']
        temp['output'] = item['answer']
        temp['input'] = ''
        train_data.append(temp)
    intermediate_gpt4_generated_train_file_path = f"{HOME_DIRECTORY}/intermediate_data/{intermediate_gpt4_generated_data_train_file_name}.json"
    with open(intermediate_gpt4_generated_train_file_path, 'w') as json_file:
        json.dump(train_data, json_file, indent=4)
    return intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path 


