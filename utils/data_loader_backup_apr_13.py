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



def load_MATH(path, n_row, zeroshot = False):
    question_list = []
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        
        for line in data:
            # groundtruth = line['solution']
            # num = find_last_boxed_number_with_simple_format(groundtruth)
            num = line['numerical_final_answer']
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
                question_list.append(line['question'])
                line['numerical_final_answer'] = str(evaluate_expression(num))
                data_list.append(line)

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

def load_GSM8K(path, n_row, zeroshot = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            question = line['question'] 
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

def load_API_BANK_vanilla(path, n_row, minimum_change_or_zero_shot = False):
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




def load_ANLI_jan_30(path, n_row, finetune_on_gt = False, zeroshot = False):
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
            if not finetune_on_gt:
                prompt += '\nPlease analyze then answer (Entailment/Neutral/Contradiction). '

# You already know the answer is {line['gold_label']}. Please solve this problem while assuming you don’t know the answer."""
            if zeroshot:
                prompt += \
f"""

Please put the final answer after Final Answer(answer eigher entailment,contradiction or neutral): after you finish inferencing. 

Format:
INFERENCE

Final Answer (Entailment/Neutral/Contradiction):
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



def load_ANLI(path, n_row):
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
ANALYZATION: 
ANSWER: 
"""
#             answer_item = f""""Analyzation: {line['reason']}
# Answer: {line['gold_label']}"""
            answer_item = line['output']
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



def load_SNLI(path, n_row, finetune = False):
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
        for line in data:
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
                prompt += \
f"""
Please answer directly.
"""
            else:
                prompt += \
f"""
Please inference first, then provide the answer.
"""
                
            temp['question'] = prompt
            temp['input'] = ''
            temp['context'] = context
            temp['gold_label'] = gold_label
            temp['answer'] = gold_label
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


def load_MMLU(path, n_row, finetune = False):
    data_list = []
    with open(path, 'r') as file:
        data = json.load(file)
        for line in data:
            temp = {}
            question = line['question']
            A = line['A']
            B = line['B']
            C = line['C']
            D = line['D']
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
            temp['A'] = line['A']
            temp['B'] = line['B']
            temp['C'] = line['C']
            temp['D'] = line['D']
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


def load_ECQA(path, n_row, finetune = False, use_gt_rationale = False):
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
            if finetune and use_gt_rationale:
                answer = line['answer']
                answer = f"""Inference: {combined_explaination}

Final Answer: """ + answer
            line['question'] = prompt
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
            
def load_CODE(path, n_row, finetune = False):
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

def load_CODE_code_only(path, n_row):
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


def load_MBPP_april_5(path, n_row, finetune = False, data_creation = False):
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

            line['question'] = modified_prompt
            data_list.append(line)
    return data_list[:n_row]

def load_MBPP_code_only(path, n_row):
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



def dpo_dataloader(minimum_change_train_path, minimum_change_train_data_list, train_data_list):
    previous_prediction_list = []
    with open(minimum_change_train_path, 'r') as file:
        data = json.load(file)
        for line in data:
            try:
                previous_prediction = line['previous_prediction']
            except:
                previous_prediction = line['original_prediction']
            previous_prediction_list.append(previous_prediction)
    minimum_change_answer_list = []
    for item in minimum_change_train_data_list:
        minimum_change_answer_list.append(item['answer'])
    groundtruth_answer_list = []
    for item in train_data_list:
        groundtruth_answer_list.append(item['answer'])

    dpo_minimum_change = []
    for ii in range(len(minimum_change_train_data_list)):
        temp = {}
        temp['question'] =  minimum_change_train_data_list[ii]['question']
        temp['input'] =  ''
        temp['answer'] =  [minimum_change_train_data_list[ii]['answer'], previous_prediction_list[ii]]
        dpo_minimum_change.append(temp)

    dpo_groundtruth = []
    for ii in range(len(train_data_list)):
        temp = {}
        temp['question'] =  train_data_list[ii]['question']
        temp['input'] =  ''
        temp['answer'] =  [groundtruth_answer_list[ii], previous_prediction_list[ii]]
        dpo_groundtruth.append(temp)

    return dpo_minimum_change, dpo_groundtruth





















def prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name, n_train, multi_task_or_continual_learning, variation_suffix = '', train_data_list = [], minimum_change_train_data_list = [], gpt4_generated_train_data_list = [], paraphrased_data_train_list = [], dpo_mc_train_data_list = [], dpo_gt_train_data_list = [], dpo_sample_10_train_data_list = [], dpo_gpt4_train_data_list = [], sample_10_train_data_list = [], continual_learning = False):
    intermediate_finetune_file_name = f'{train_task_name}_finetune_{n_train}'
    intermediate_finetune_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_finetune_file_name}.json"
    train_data = []
    for item in train_data_list:
        temp = {}
        temp['instruction'] = item['question']
        temp['output'] = item['answer']
        temp['input'] = item['input']
        train_data.append(temp)
    with open(intermediate_finetune_file_path, 'w') as json_file:
        json.dump(train_data, json_file, indent=4)

    intermediate_minimum_change_train_file_name = f'{train_task_name}_minimum_change_{n_train}'
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
        temp['input'] = item['input']
        train_data.append(temp)
    intermediate_minimum_change_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_minimum_change_train_file_name}.json"
    with open(intermediate_minimum_change_train_file_path, 'w') as json_file:
        json.dump(train_data, json_file, indent=4)

    intermediate_gpt4_generated_data_train_file_name = f'{train_task_name}_{variation_suffix}_gpt4_generated_data_{n_train}'
    train_data = []
    for item in gpt4_generated_train_data_list:
        temp = {}
        temp['instruction'] = item['question']
        temp['output'] = item['answer']
        temp['input'] = item['input']
        train_data.append(temp)
    intermediate_gpt4_generated_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_gpt4_generated_data_train_file_name}.json"
    with open(intermediate_gpt4_generated_train_file_path, 'w') as json_file:
        json.dump(train_data, json_file, indent=4)
    if continual_learning:
        return intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path

    # --------------------------
    if not continual_learning:
        intermediate_paraphrased_data_train_file_path = ''

        intermediate_dpo_finetune_file_path = ''
        intermediate_dpo_minimum_change_train_file_path = ''
        intermediate_dpo_sample_10_train_file_path = ''
        intermediate_sample_10_train_file_path = ''
        intermediate_dpo_gpt4_train_file_path = ''

        if train_task_name.lower() == 'anli' or train_task_name.lower() == 'mnli' or train_task_name.lower() == 'snli' or train_task_name.lower() == 'scitail' or train_task_name.lower() == 'gsm8k' or train_task_name.lower() == 'api_bank' or train_task_name.lower() == 'code' or 'math' in train_task_name.lower():
            intermediate_paraphrased_data_train_file_name = f'{train_task_name}_paraphrased_data_{n_train}'
            intermediate_paraphrased_data_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_paraphrased_data_train_file_name}.json"
            with open(intermediate_paraphrased_data_train_file_path, 'w') as json_file:
                json.dump(paraphrased_data_train_list, json_file, indent=4)

        if train_task_name.lower() != 'code' and train_task_name.lower() != 'api_bank' and train_task_name.lower() != 'mbpp' and train_task_name.lower() != 'mnli' and train_task_name.lower() != 'snli' and train_task_name.lower() != 'scitail' and train_task_name.lower() != 'boolq' and train_task_name.lower() != 'piqa' and train_task_name.lower() != 'winogrande' and not multi_task_or_continual_learning:
            intermediate_dpo_minimum_change_train_file_name = f'dpo_{train_task_name}_minimum_change_{n_train}'
            intermediate_dpo_minimum_change_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_dpo_minimum_change_train_file_name}.json"
            with open(intermediate_dpo_minimum_change_train_file_path, 'w') as json_file:
                json.dump(dpo_mc_train_data_list, json_file, indent=4)

            intermediate_dpo_finetune_file_name = f'dpo_{train_task_name}_finetune_{n_train}'
            intermediate_dpo_finetune_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_dpo_finetune_file_name}.json"
            with open(intermediate_dpo_finetune_file_path, 'w') as json_file:
                json.dump(dpo_gt_train_data_list, json_file, indent=4)

            intermediate_dpo_sample_10_train_file_name = f'dpo_{train_task_name}_sample_10_{n_train}'
            intermediate_dpo_sample_10_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_dpo_sample_10_train_file_name}.json"
            with open(intermediate_dpo_sample_10_train_file_path, 'w') as json_file:
                json.dump(dpo_sample_10_train_data_list, json_file, indent=4)

            intermediate_gpt4_train_file_name = f'dpo_{train_task_name}_{variation_suffix}_gpt4_{n_train}'
            intermediate_dpo_gpt4_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_gpt4_train_file_name}.json"
            with open(intermediate_dpo_gpt4_train_file_path, 'w') as json_file:
                json.dump(dpo_gpt4_train_data_list, json_file, indent=4)
            intermediate_sample_10_train_file_name = f'{train_task_name}_sample_10_{n_train}'
            intermediate_sample_10_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_sample_10_train_file_name}.json"
            with open(intermediate_sample_10_train_file_path, 'w') as json_file:
                json.dump(sample_10_train_data_list, json_file, indent=4)

        return intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path, intermediate_paraphrased_data_train_file_path, intermediate_dpo_finetune_file_path, intermediate_dpo_minimum_change_train_file_path, intermediate_dpo_gpt4_train_file_path, intermediate_dpo_sample_10_train_file_path, intermediate_sample_10_train_file_path



def load_dataset(HOME_DIRECTORY, continual_learning, multi_task_learning, train_task_name, n_train, enable_learn_from_mistakes, enable_REST_em_initial_500, enable_REST_em_initial_100, enable_change_model_internal_preference, enable_step_by_step_gpt4, variation_suffix_code, variation_suffix, enable_dpo, task_sequence_list = ['all'], model_name = ''):
    import random
    intermediate_dpo_finetune_file_path = ''
    intermediate_dpo_minimum_change_train_file_path = ''
    intermediate_dpo_gpt4_train_file_path = ''
    intermediate_dpo_sample_10_train_file_path = ''
    intermediate_sample_10_train_file_path = ''
    intermediate_paraphrased_data_train_file_path = ''

    if 'mistral' in model_name:
        model_name = '_mistral'

    multi_task_or_continual_learning = False
    if continual_learning or multi_task_learning:
        multi_task_or_continual_learning = True
    if not multi_task_or_continual_learning:
        if train_task_name.upper() =='GSM8K':
            train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/train_filtered.json'
            train_data_list = load_GSM8K(train_path, n_train)
            # train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/minimum_change_train_dec11.json'
            
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/minimum_change_train_nov_25.json'
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gsm8k_proof_read_100.json'
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gsm8k_proof_read_100_creation_num_3.json'
            if variation_suffix == 'paraphrased_question':
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gsm8k{model_name}_minimum_change_100_march_27_question_paraphrased.json'
            else:
                # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gsm8k{model_name}_minimum_change_1000_march_27.json'
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gsm8k_mistral_minimum_change_1000_apr_5.json'
            
            if model_name == '':
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gsm8k_minimum_change_1000_march_27.json'
            elif 'mistral' in model_name:
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gsm8k_mistral_minimum_change_1000_apr_5.json'

            # minimum_change_train_data_list = load_GSM8K(minimum_change_train_path, n_train)#, extract_numerical_final_answer = True)
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]

            if enable_learn_from_mistakes:
                minimum_change_train_data_list = minimum_change_train_data_list + train_data_list
                random.shuffle(minimum_change_train_data_list)
            if enable_REST_em_initial_500:
                with open(f'{HOME_DIRECTORY}/dataset/GSM8K/REST_em_initial_500.json', 'r') as file:
                    train_data_list = json.load(file)
                train_data_list = train_data_list[:n_train]
            
            if enable_REST_em_initial_100:
                with open(f'{HOME_DIRECTORY}/dataset/GSM8K/REST_em_initial_100.json', 'r') as file:
                    train_data_list = json.load(file)
                train_data_list = train_data_list[:n_train]
            
            if enable_learn_from_mistakes:
                train_data_list_old = train_data_list[:n_train]
                with open(f'{HOME_DIRECTORY}/dataset/GSM8K/gsm8k{model_name}_minimum_change_100_learn_from_mistakes_cleaned.json', 'r') as file:
                    train_data_list = json.load(file)
                
                for i in range(len(train_data_list)):
                    question_item = train_data_list[i]['question']
                    previous_prediction = train_data_list[i]['previous_prediction']
                    answer = train_data_list[i]['answer']
                    question_item = \
f"""For the following math problem, the original solution is incorrect. Please identify the incorrect step, explain why it is incorrect, and correct the original solution starting from the incorrect step.
Question: {question_item}
Original Solution:{previous_prediction}

Incorrect Step: """
                    train_data_list[i]['question'] = question_item 
                    train_data_list[i]['answer'] = answer

                train_data_list = train_data_list[:n_train]
                train_data_list = train_data_list_old+ train_data_list
                random.shuffle(train_data_list)
            


            if enable_change_model_internal_preference:
                minimum_change_train_data_list = load_GSM8K_change_model_internal_preference(minimum_change_train_path, n_train)

            # if enable_step_by_step_gpt4:
            #     gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gpt4_generated_{train_task_name.lower()}_data_step_by_step_True_500.json'
            # else:

            if variation_suffix == '' or variation_suffix == 'paraphrased_question':
                # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gpt4_generated_{train_task_name.lower()}_data_step_by_step_False_500.json'
                # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gpt4_generated_gsm8k_False_999999.json'
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gpt4_generated_gsm8k_False_999999_march_27.json'
            if variation_suffix == 'calculation_only':
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gpt4_generated_gsm8k_calculation_only_False_100.json'
            elif variation_suffix == 'lets_think_step_by_step':
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gpt4_generated_gsm8k_lets_think_step_by_step_False_100.json'
            elif variation_suffix == 'solve_without_explaination':
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gpt4_generated_gsm8k_solve_without_explaination_False_100.json'
            elif variation_suffix == 'straight_forward':
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gpt4_generated_gsm8k_straight_forward_False_100.json'
            elif variation_suffix == 'very_clear':
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gpt4_generated_gsm8k_very_clear_False_100.json'

            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gpt4_generated_gsm8k_False_999999_march_27.json'
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]


            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/paraphrased_data_gsm8k_False_99999999.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]
            
            # if 'mistral' in model_name:
            #     mc_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/mc_dpo_5_mistral.json'
            #     gt_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/gt_dpo_5_mistral.json'
            #     sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/sample_5_mistral_dpo.json'
            #     gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/_gpt4_data_dpo_5_mistral.json'
            # else:
            #     mc_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/mc_dpo_5.json'
            #     gt_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/gt_dpo_5.json'
            #     sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/sample_5_dpo.json'
            #     gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/_gpt4_data_dpo_5.json'
            

            # with open(sample_10_train_path, 'r') as file:
            #     sample_10_train_data_list = json.load(file)
            #     sample_10_train_data_list = sample_10_train_data_list[:n_train]
            # for kkk in range(len(sample_10_train_data_list)):
            #     output = sample_10_train_data_list[kkk]['output']
            #     sample_10_train_data_list[kkk]['output'] = output[0]

            # with open(gpt4_train_path, 'r') as file:
            #     gpt4_train_data_list = json.load(file)
            #     gpt4_train_data_list = gpt4_train_data_list[:n_train]
            # for kkk in range(len(gpt4_train_data_list)):
            #     output = gpt4_train_data_list[kkk]['output']
            #     gpt4_train_data_list[kkk]['output'] = output[0]
            
            # # if enable_dpo:
            # with open(mc_train_path, 'r') as file:
            #     dpo_mc_train_data_list = json.load(file)
            #     dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            # with open(gt_train_path, 'r') as file:
            #     dpo_gt_train_data_list = json.load(file)
            #     dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            # with open(sample_10_train_path, 'r') as file:
            #     dpo_sample_10_train_data_list = json.load(file)
            #     dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            # with open(gpt4_train_path, 'r') as file:
            #     dpo_gpt4_train_data_list = json.load(file)
            #     dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]

        elif train_task_name.lower() == 'math_algebra':
            # train_path = f'{HOME_DIRECTORY}/dataset/MATH/train_algebra.json'
            train_path = f'{HOME_DIRECTORY}/dataset/MATH/train_algebra_total_filtered.json'
            
            # train_path = f'{HOME_DIRECTORY}/dataset/MATH/train_easy_algebra_total_filtered.json'
            # train_data_list = load_MATH(train_path, n_train, extract_numerical_final_answer = True, minimum_change_or_zero_shot = False)
            train_data_list = load_MATH(train_path, n_train, zeroshot = False)

            if enable_REST_em_initial_100:
                with open(f'{HOME_DIRECTORY}/dataset/MATH/math_algebra_REST_em_initial_100.json', 'r') as file:
                    train_data_list = json.load(file)
                train_data_list = train_data_list[:n_train]

            # train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/minimum_change_train_dec11.json'
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MATH/minimum_change_train_algebra.json'
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MATH/math_algebra_total_minimum_change_100.json'
            if variation_suffix == 'given_answer':
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MATH/math_algebra_total{model_name}_given_answer_minimum_change_100.json'
            if variation_suffix == 'paraphrased_question':
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MATH/math{model_name}_minimum_change_100_march_27_question_paraphrased.json'
            else:
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MATH/math_algebra_total_easy{model_name}_minimum_change_422.json'


            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if variation_suffix == 'easy':
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MATH/math_algebra_total_mistral_minimum_change_9999_april_5_easy.json'
            else:
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MATH/math_algebra_total_mistral_minimum_change_9999_april_5_clean.json'
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


            if model_name == 'mistral':
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MATH/math_algebra_total_mistral_minimum_change_9999_april_5_clean.json'
            else:
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MATH/math_algebra_total_minimum_change_1000_april_12.json'

            # minimum_change_train_data_list = load_MATH(minimum_change_train_path, n_train, minimum_change_or_zero_shot = True)
            # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MATH/gpt4_generated_{train_task_name.lower()}_data_step_by_step_False_500.json'

            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            if variation_suffix == 'easy':
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MATH/gpt4_generated_math_algebra_total_easy.json'
            else:
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MATH/gpt4_generated_math_algebra_total.json'
            # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MATH/gpt4_generated_math_algebra_total_easy.json'
            
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]

            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/MATH/paraphrased_data_math_algebra_False_500.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]

            # math_category = train_task_name.lower()
            # math_category = math_category.replace('math_', '')
            # mc_train_path = f'{HOME_DIRECTORY}/dpo_data/MATH/_mistral_mc_dpo_5.json'
            # gt_train_path = f'{HOME_DIRECTORY}/dpo_data/MATH/_mistral_gt_dpo_5.json'
            # sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/MATH/_mistral_sample_5_dpo.json'
            # gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/MATH/_mistral_gpt4_data_dpo_5.json'

            # with open(sample_10_train_path, 'r') as file:
            #     sample_10_train_data_list = json.load(file)
            #     sample_10_train_data_list = sample_10_train_data_list[:n_train]
            # for kkk in range(len(sample_10_train_data_list)):
            #     output = sample_10_train_data_list[kkk]['output']
            #     sample_10_train_data_list[kkk]['output'] = output[0]

            # with open(gpt4_train_path, 'r') as file:
            #     gpt4_train_data_list = json.load(file)
            #     gpt4_train_data_list = gpt4_train_data_list[:n_train]
            # for kkk in range(len(gpt4_train_data_list)):
            #     output = gpt4_train_data_list[kkk]['output']
            #     gpt4_train_data_list[kkk]['output'] = output[0]

            # # if enable_dpo:
            # with open(mc_train_path, 'r') as file:
            #     dpo_mc_train_data_list = json.load(file)
            #     dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            # with open(gt_train_path, 'r') as file:
            #     dpo_gt_train_data_list = json.load(file)
            #     dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            # with open(sample_10_train_path, 'r') as file:
            #     dpo_sample_10_train_data_list = json.load(file)
            #     dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            # with open(gpt4_train_path, 'r') as file:
            #     dpo_gpt4_train_data_list = json.load(file)
            #     dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]

        elif train_task_name.lower() == 'math_counting_and_probability':
            train_path = f'{HOME_DIRECTORY}/dataset/MATH/train_counting_and_probability.json'
            train_data_list = load_MATH(train_path, n_train, zeroshot = False)
            # train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/minimum_change_train_dec11.json'
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MATH/minimum_change_train_counting_and_probability{model_name}.json'
            # minimum_change_train_data_list = load_MATH(minimum_change_train_path, n_train, minimum_change_or_zero_shot = True)
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]

            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MATH/gpt4_generated_{train_task_name.lower()}_data_step_by_step_False_500.json'
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
            math_category = train_task_name.lower()
            math_category = math_category.replace('math_', '')
            mc_train_path = f'{HOME_DIRECTORY}/dpo_data/MATH/{math_category}_mc_dpo_10.json'
            gt_train_path = f'{HOME_DIRECTORY}/dpo_data/MATH/{math_category}_gt_dpo_10.json'
            sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/MATH/{math_category}_sample_10_dpo.json'
            gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/MATH/{math_category}_gpt4_data_dpo_10.json'
            
            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/MATH/paraphrased_data_math_counting_and_probability_False_100.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]



            if enable_REST_em_initial_100:
                with open(f'{HOME_DIRECTORY}/dataset/MATH/math_counting_probability_REST_em_initial_100.json', 'r') as file:
                    train_data_list = json.load(file)
                # train_data_list = train_data_list[:n_train]


            with open(sample_10_train_path, 'r') as file:
                sample_10_train_data_list = json.load(file)
                sample_10_train_data_list = sample_10_train_data_list[:n_train]
            for kkk in range(len(sample_10_train_data_list)):
                output = sample_10_train_data_list[kkk]['output']
                sample_10_train_data_list[kkk]['output'] = output[0]

            with open(gpt4_train_path, 'r') as file:
                gpt4_train_data_list = json.load(file)
                gpt4_train_data_list = gpt4_train_data_list[:n_train]
            for kkk in range(len(gpt4_train_data_list)):
                output = gpt4_train_data_list[kkk]['output']
                gpt4_train_data_list[kkk]['output'] = output[0]

            # if enable_dpo:
            with open(mc_train_path, 'r') as file:
                dpo_mc_train_data_list = json.load(file)
                dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            with open(gt_train_path, 'r') as file:
                dpo_gt_train_data_list = json.load(file)
                dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            with open(sample_10_train_path, 'r') as file:
                dpo_sample_10_train_data_list = json.load(file)
                dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            with open(gpt4_train_path, 'r') as file:
                dpo_gpt4_train_data_list = json.load(file)
                dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]
        elif train_task_name.lower() == 'math_number_theory':
            train_path = f'{HOME_DIRECTORY}/dataset/MATH/train_number_theory.json'
            train_data_list = load_MATH(train_path, n_train, zeroshot = False)
            # train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/minimum_change_train_dec11.json'
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MATH/minimum_change_train_number_theory{model_name}.json'
            # minimum_change_train_data_list = load_MATH(minimum_change_train_path, n_train, minimum_change_or_zero_shot = True)

            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MATH/gpt4_generated_{train_task_name.lower()}_data_step_by_step_False_500.json'
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
            math_category = train_task_name.lower()
            math_category = math_category.replace('math_', '')
            mc_train_path = f'{HOME_DIRECTORY}/dpo_data/MATH/{math_category}_mc_dpo_10.json'
            gt_train_path = f'{HOME_DIRECTORY}/dpo_data/MATH/{math_category}_gt_dpo_10.json'
            sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/MATH/{math_category}_sample_10_dpo.json'
            gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/MATH/{math_category}_gpt4_data_dpo_10.json'

            with open(sample_10_train_path, 'r') as file:
                sample_10_train_data_list = json.load(file)
                sample_10_train_data_list = sample_10_train_data_list[:n_train]
            for kkk in range(len(sample_10_train_data_list)):
                output = sample_10_train_data_list[kkk]['output']
                sample_10_train_data_list[kkk]['output'] = output[0]

            with open(gpt4_train_path, 'r') as file:
                gpt4_train_data_list = json.load(file)
                gpt4_train_data_list = gpt4_train_data_list[:n_train]
            for kkk in range(len(gpt4_train_data_list)):
                output = gpt4_train_data_list[kkk]['output']
                gpt4_train_data_list[kkk]['output'] = output[0]
            
            # if enable_dpo:
            with open(mc_train_path, 'r') as file:
                dpo_mc_train_data_list = json.load(file)
                dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            with open(gt_train_path, 'r') as file:
                dpo_gt_train_data_list = json.load(file)
                dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            with open(sample_10_train_path, 'r') as file:
                dpo_sample_10_train_data_list = json.load(file)
                dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            with open(gpt4_train_path, 'r') as file:
                dpo_gpt4_train_data_list = json.load(file)
                dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]

        elif train_task_name.upper() =='API_BANK':
            # train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/train/training-data_lv3-train.json'
            # train_data_list = load_API_BANK_optimized(train_path, n_train, minimum_change_or_zero_shot = False)
            # train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/gpt4_reason_train.json'
            # gpt4_reason_train_data_list = load_API_BANK(train_path, n_train, minimum_change_or_zero_shot = True)
            # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/gpt4_generated_{train_task_name.lower()}_data_step_by_step_False_500.json'
            # with open(gpt4_generated_data_train_path, 'r') as file:
            #     gpt4_generated_train_data_list = json.load(file)
            # gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
            # train_data_list_temp = []
            # for i in range(len(gpt4_reason_train_data_list)):
            #     temp = {}
            #     temp['question'] = train_data_list[i]['question']
            #     temp['input'] = ''
            #     temp['answer'] = gpt4_reason_train_data_list[i]['answer']
            #     train_data_list_temp.append(temp)
            # gpt4_reason_train_data_list = train_data_list_temp
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/minimum_change_train_optimized_new.json'
            # if note == '_non_simplified':
            #     minimum_change_train_data_list = load_API_BANK_optimized(minimum_change_train_path, n_train, minimum_change_or_zero_shot = True)
            # else:
            #     minimum_change_train_data_list = load_API_BANK_optimized_simplified(minimum_change_train_path, n_train, minimum_change_or_zero_shot = True)

            # if enable_dpo:
            #     dpo_minimum_change_train_data_list, dpo_train_data_list = dpo_dataloader(minimum_change_train_path, minimum_change_train_data_list, train_data_list)       
            #     dpo_minimum_change_train_data_list = dpo_minimum_change_train_data_list[:n_train]
            #     dpo_train_data_list = dpo_train_data_list[:n_train]         
            
            # a = 1



            train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/train/training-data_lv3-train.json'
            train_data_list = load_API_BANK_optimized(train_path, n_train, minimum_change_or_zero_shot = False)
    #        api_bank_minimum_change_100_feb8
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/api_bank_minimum_change_100_feb8.json'
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/api_bank_minimum_change_100_mar_8.json'
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/api_bank_minimum_change_100_mar_9.json'
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/api_bank{model_name}_minimum_change_100_mar_17.json'
            

            
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/minimum_change_train_optimized_new.json'
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list_temp = json.load(file)
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            
            # minimum_change_train_data_list = load_API_BANK_march_1(minimum_change_train_path, n_train)
            # minimum_change_train_data_list = load_API_BANK_march_5_step1(minimum_change_train_path, n_train)
            # minimum_change_train_data_list = load_API_BANK_march_8_step1(minimum_change_train_path, n_train)
            
            
            # minimum_change_train_data_list_temp = load_API_BANK_optimized(minimum_change_train_path, n_train, minimum_change_or_zero_shot = True)
            minimum_change_train_data_list = []
            for i, item in enumerate(minimum_change_train_data_list_temp):
                item['question'] = item['instruction']
                item['answer'] = item['output']
                # item['answer'] = item['output']

                # if 'ToolSearcher' not in train_data_list[i]['answer']:
                #     item['answer'] = 'We should call the API found by the previous ToolSearch API request. ' + item['answer']
                minimum_change_train_data_list.append(item)


            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)     
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            # for i, item in enumerate(train_data_list):
            #     answer = train_data_list[i]['answer']
            #     if 'ToolSearcher' in answer:
            #         minimum_change_train_data_list[i]['answer'] = answer

            if enable_step_by_step_gpt4:
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/gpt4_generated_{train_task_name.lower()}_data_step_by_step_True_500.json'
            else:
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/gpt4_generated_{train_task_name.lower()}_data_step_by_step_False_500.json'
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]

            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/paraphrased_data_api_bank_False_100.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)

            # mc_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/mc_dpo_10.json'
            # gt_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/gt_dpo_10.json'
            # gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/_gpt4_data_dpo_10.json'

            # with open(gpt4_train_path, 'r') as file:
            #     gpt4_train_data_list = json.load(file)
            #     gpt4_train_data_list = gpt4_train_data_list[:n_train]
            # for kkk in range(len(gpt4_train_data_list)):
            #     output = gpt4_train_data_list[kkk]['output']
            #     gpt4_train_data_list[kkk]['output'] = output[0]
            
            # if enable_dpo:
            # with open(mc_train_path, 'r') as file:
            #     dpo_mc_train_data_list = json.load(file)
            #     dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            # with open(gt_train_path, 'r') as file:
            #     dpo_gt_train_data_list = json.load(file)
            #     dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]
            
            # with open(gpt4_train_path, 'r') as file:
            #     dpo_gpt4_train_data_list = json.load(file)
            #     dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]
        elif train_task_name.upper() =='ANLI': # load_ANLI load_ANLI_sft_gtreason load_ANLI_sftgt
            train_path = f'{HOME_DIRECTORY}/dataset/ANLI/train.json'
            train_data_list = load_ANLI_jan_30(train_path, n_train, finetune_on_gt = True)
            # train_data_low_p_list = load_ANLI_low_p(train_path, n_train)
            
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/minimum_change_train_feb_1.json'
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/minimum_change_train_jan_30.json'
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/anli_minimum_change_9999_march_9.json'
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/anli_minimum_change_100_march_9.json'

            if variation_suffix == '':
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/anli{model_name}_minimum_change_100_march_9_300.json'
            elif variation_suffix == 'old':
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/anli{model_name}_minimum_change_9999_march_9_old.json'

            
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list_temp = json.load(file)
            minimum_change_train_data_list = []
            for item in minimum_change_train_data_list_temp:
                question = item['question']
                question = question.replace('\n\nYou already know the answer is neutral. Please solve this problem while assuming you don’t know the answer.', '')
                question = question.replace('\n\nYou already know the answer is entailment. Please solve this problem while assuming you don’t know the answer.', '')
                question = question.replace('\n\nYou already know the answer is contradiction. Please solve this problem while assuming you don’t know the answer.', '')
                item['question'] = question# + '\nPlease analyze then answer.'
                item['answer'] = item['output']
                minimum_change_train_data_list.append(item)
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/gpt4_generated_{train_task_name.lower()}_data_step_by_step_False_500.json'
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
            mc_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/mc_dpo_10.json'
            gt_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/gt_dpo_10.json'
            sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/sample_10_dpo.json'
            gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/_gpt4_data_dpo_10.json'

            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/paraphrased_data_anli_False_500.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]

            with open(sample_10_train_path, 'r') as file:
                sample_10_train_data_list = json.load(file)
                sample_10_train_data_list = sample_10_train_data_list[:n_train]
            for kkk in range(len(sample_10_train_data_list)):
                output = sample_10_train_data_list[kkk]['output']
                sample_10_train_data_list[kkk]['output'] = output[0]

            with open(gpt4_train_path, 'r') as file:
                gpt4_train_data_list = json.load(file)
                gpt4_train_data_list = gpt4_train_data_list[:n_train]
            for kkk in range(len(gpt4_train_data_list)):
                output = gpt4_train_data_list[kkk]['output']
                gpt4_train_data_list[kkk]['output'] = output[0]
            
            # if enable_dpo:
            with open(mc_train_path, 'r') as file:
                dpo_mc_train_data_list = json.load(file)
                dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            with open(gt_train_path, 'r') as file:
                dpo_gt_train_data_list = json.load(file)
                dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            with open(sample_10_train_path, 'r') as file:
                dpo_sample_10_train_data_list = json.load(file)
                dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            with open(gpt4_train_path, 'r') as file:
                dpo_gpt4_train_data_list = json.load(file)
                dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]
        
        elif train_task_name.upper() =='MNLI': 
            train_path = f'{HOME_DIRECTORY}/dataset/MNLI/train.json'
            train_data_list = load_MNLI_vanilla(train_path, n_train)
            
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/mnli{model_name}_minimum_change.json'
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list_temp = json.load(file)       

            
            minimum_change_train_data_list = []
            for item in minimum_change_train_data_list_temp:
                item['answer'] = item['output']
                minimum_change_train_data_list.append(item)
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/ANLI/gpt4_generated_anli_data_step_by_step_False_500.json'
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
            mc_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/mc_dpo_10.json'
            gt_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/gt_dpo_10.json'
            sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/sample_10_dpo.json'
            gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/_gpt4_data_dpo_10.json'

            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/ANLI/paraphrased_data_anli_False_500.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]

            with open(sample_10_train_path, 'r') as file:
                sample_10_train_data_list = json.load(file)
                sample_10_train_data_list = sample_10_train_data_list[:n_train]
            for kkk in range(len(sample_10_train_data_list)):
                output = sample_10_train_data_list[kkk]['output']
                sample_10_train_data_list[kkk]['output'] = output[0]

            with open(gpt4_train_path, 'r') as file:
                gpt4_train_data_list = json.load(file)
                gpt4_train_data_list = gpt4_train_data_list[:n_train]
            for kkk in range(len(gpt4_train_data_list)):
                output = gpt4_train_data_list[kkk]['output']
                gpt4_train_data_list[kkk]['output'] = output[0]
            
            # if enable_dpo:
            with open(mc_train_path, 'r') as file:
                dpo_mc_train_data_list = json.load(file)
                dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            with open(gt_train_path, 'r') as file:
                dpo_gt_train_data_list = json.load(file)
                dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            with open(sample_10_train_path, 'r') as file:
                dpo_sample_10_train_data_list = json.load(file)
                dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            with open(gpt4_train_path, 'r') as file:
                dpo_gpt4_train_data_list = json.load(file)
                dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]
        elif train_task_name.upper() =='SNLI': 
            train_path = f'{HOME_DIRECTORY}/dataset/SNLI/train.json'
            train_data_list = load_SNLI(train_path, n_train, finetune = True)
            
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/snli{model_name}_minimum_change_100.json'
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list_temp = json.load(file)       

            minimum_change_train_data_list = []
            for item in minimum_change_train_data_list_temp:
                item['answer'] = item['output']
                minimum_change_train_data_list.append(item)
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/gpt4_generated_snli_False_999999.json'
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
            mc_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/mc_dpo_10.json'
            gt_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/gt_dpo_10.json'
            sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/sample_10_dpo.json'
            gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/_gpt4_data_dpo_10.json'

            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/ANLI/paraphrased_data_anli_False_500.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]

            with open(sample_10_train_path, 'r') as file:
                sample_10_train_data_list = json.load(file)
                sample_10_train_data_list = sample_10_train_data_list[:n_train]
            for kkk in range(len(sample_10_train_data_list)):
                output = sample_10_train_data_list[kkk]['output']
                sample_10_train_data_list[kkk]['output'] = output[0]

            with open(gpt4_train_path, 'r') as file:
                gpt4_train_data_list = json.load(file)
                gpt4_train_data_list = gpt4_train_data_list[:n_train]
            for kkk in range(len(gpt4_train_data_list)):
                output = gpt4_train_data_list[kkk]['output']
                gpt4_train_data_list[kkk]['output'] = output[0]
            
            # if enable_dpo:
            with open(mc_train_path, 'r') as file:
                dpo_mc_train_data_list = json.load(file)
                dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            with open(gt_train_path, 'r') as file:
                dpo_gt_train_data_list = json.load(file)
                dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            with open(sample_10_train_path, 'r') as file:
                dpo_sample_10_train_data_list = json.load(file)
                dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            with open(gpt4_train_path, 'r') as file:
                dpo_gpt4_train_data_list = json.load(file)
                dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]
        elif train_task_name.upper() =='SCITAIL': 
            train_path = f'{HOME_DIRECTORY}/dataset/SCITAIL/train.json'
            train_data_list = load_SCITAIL(train_path, n_train, finetune = True)
            
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/scitail{model_name}_minimum_change_100.json'
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/scitail{model_name}_minimum_change_100_step_2.json'
            
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list_temp = json.load(file)       

            minimum_change_train_data_list = []
            for item in minimum_change_train_data_list_temp:
                item['answer'] = item['output']
                minimum_change_train_data_list.append(item)
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/SCITAIL/gpt4_generated_scitail_False_100.json'
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
            mc_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/mc_dpo_10.json'
            gt_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/gt_dpo_10.json'
            sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/sample_10_dpo.json'
            gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/_gpt4_data_dpo_10.json'

            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/ANLI/paraphrased_data_anli_False_500.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]

            with open(sample_10_train_path, 'r') as file:
                sample_10_train_data_list = json.load(file)
                sample_10_train_data_list = sample_10_train_data_list[:n_train]
            for kkk in range(len(sample_10_train_data_list)):
                output = sample_10_train_data_list[kkk]['output']
                sample_10_train_data_list[kkk]['output'] = output[0]

            with open(gpt4_train_path, 'r') as file:
                gpt4_train_data_list = json.load(file)
                gpt4_train_data_list = gpt4_train_data_list[:n_train]
            for kkk in range(len(gpt4_train_data_list)):
                output = gpt4_train_data_list[kkk]['output']
                gpt4_train_data_list[kkk]['output'] = output[0]
            
            # if enable_dpo:
            with open(mc_train_path, 'r') as file:
                dpo_mc_train_data_list = json.load(file)
                dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            with open(gt_train_path, 'r') as file:
                dpo_gt_train_data_list = json.load(file)
                dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            with open(sample_10_train_path, 'r') as file:
                dpo_sample_10_train_data_list = json.load(file)
                dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            with open(gpt4_train_path, 'r') as file:
                dpo_gpt4_train_data_list = json.load(file)
                dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]
        elif train_task_name.upper() == 'CODE':
            train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/train{variation_suffix_code}.json'
            # train_data_list = load_CODE(train_path, n_train, minimum_change = False)
            train_data_list = load_CODE_code_only(train_path, n_train)

            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/minimum_change_train_jan_30{variation_suffix_code}.json'
            if 'mistral' in model_name:
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/code_mistral_minimum_change_100_code_only.json'
            else:
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/code_minimum_change_100_code_only_2_cleaned.json'
                
            
            # minimum_change_train_data_list = load_CODE(minimum_change_train_path, n_train, minimum_change = True)
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/gpt4_generated_{train_task_name.lower()}_data_step_by_step_False_500{variation_suffix_code}_code_only.json'
            
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/gpt4_generated_code_False_10000_code_only_answer_directly.json'
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)

            if variation_suffix_code == '_2':
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[-101:]
            elif variation_suffix_code == '_1':
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:100]
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:100]
            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/paraphrased_data_{train_task_name.lower()}_False_100{variation_suffix_code}.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]

        elif train_task_name.upper() =='APPS':
            train_path = f'{HOME_DIRECTORY}/dataset/APPS/train.json'
            train_data_list = load_APPS_vanilla(train_path, n_train, minimum_change_or_zeroshot = False)
            # train_data_list = load_APPS_Step_by_step(train_path, n_train, minimum_change_or_zeroshot = False)

            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/APPS/minimum_change_train_step_by_step.json'
            # minimum_change_train_data_list = load_APPS_Step_by_step(minimum_change_train_path, n_train, minimum_change_or_zeroshot = True)
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]

            if enable_dpo:
                dpo_minimum_change_train_data_list, dpo_train_data_list = dpo_dataloader(minimum_change_train_path, minimum_change_train_data_list, train_data_list)        
                dpo_minimum_change_train_data_list = dpo_minimum_change_train_data_list[:n_train]
                dpo_train_data_list = dpo_train_data_list[:n_train]
        
        elif train_task_name.upper() == 'MBPP':
            # train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/train.json'
            train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/train.json'
            # train_data_list = load_MBPP(train_path, n_train)
            train_data_list = load_MBPP_code_only(train_path, n_train)
            
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/minimum_change_train.json'
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/mbpp{model_name}_minimum_change_100_code_only.json'
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/mbpp_mistral_minimum_change_300_march_27.json'

            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/mbpp_mistral_minimum_change_100_april_7.json'
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/mbpp_mistral_minimum_change_100_not_changing_anything_apr_7.json'
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/mbpp_mistral_minimum_change_100_code_only.json'
            if 'mistral' in model_name:
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/mbpp_mistral_minimum_change_9999_code_only.json'
            else:
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/mbpp_minimum_change_9999_code_only.json'
                
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/mbpp_minimum_change_100.json'
            
            # minimum_change_train_data_list = load_MBPP(minimum_change_train_path, n_train)

            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MBPP/gpt4_generated_mbpp_False_999999.json'
            # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MBPP/gpt4_generated_mbpp_False_code_only.json'
            # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MBPP/gpt4_generated_mbpp_False_100_code_only.json'
            # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MBPP/gpt4_generated_mbpp_False_300_code_only_cleaned.json'
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MBPP/gpt4_generated_mbpp_False_10000_code_only.json'
            
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]

            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/MBPP/fake_paraphrased.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]

        elif train_task_name.upper() =='PIQA': 
            train_path = f'{HOME_DIRECTORY}/dataset/PIQA/train.json'
            train_data_list = load_PIQA(train_path, n_train, finetune = True)
            
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/piqa{model_name}_minimum_change_1000_march_27.json'
            
            # with open(minimum_change_train_path, 'r') as file:
            #     minimum_change_train_data_list_temp = json.load(file)       

            # minimum_change_train_data_list = []
            # for item in minimum_change_train_data_list_temp:
            #     item['answer'] = item['output']
            #     minimum_change_train_data_list.append(item)
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)     
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/PIQA/gpt4_generated_piqa_False_1000.json'
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
            mc_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/mc_dpo_10.json'
            gt_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/gt_dpo_10.json'
            sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/sample_10_dpo.json'
            gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/_gpt4_data_dpo_10.json'

            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/ANLI/paraphrased_data_anli_False_500.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]

            # with open(sample_10_train_path, 'r') as file:
            #     sample_10_train_data_list = json.load(file)
            #     sample_10_train_data_list = sample_10_train_data_list[:n_train]
            # for kkk in range(len(sample_10_train_data_list)):
            #     output = sample_10_train_data_list[kkk]['output']
            #     sample_10_train_data_list[kkk]['output'] = output[0]

            # with open(gpt4_train_path, 'r') as file:
            #     gpt4_train_data_list = json.load(file)
            #     gpt4_train_data_list = gpt4_train_data_list[:n_train]
            # for kkk in range(len(gpt4_train_data_list)):
            #     output = gpt4_train_data_list[kkk]['output']
            #     gpt4_train_data_list[kkk]['output'] = output[0]
            
            # # if enable_dpo:
            # with open(mc_train_path, 'r') as file:
            #     dpo_mc_train_data_list = json.load(file)
            #     dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            # with open(gt_train_path, 'r') as file:
            #     dpo_gt_train_data_list = json.load(file)
            #     dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            # with open(sample_10_train_path, 'r') as file:
            #     dpo_sample_10_train_data_list = json.load(file)
            #     dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            # with open(gpt4_train_path, 'r') as file:
            #     dpo_gpt4_train_data_list = json.load(file)
            #     dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]

        elif train_task_name.upper() =='BOOLQ': 
            train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/train.json'
            train_data_list = load_BOOLQ(train_path, n_train, finetune = True)
            
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/boolq_minimum_change_100.json'
            if 'mistral' in model_name:
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/boolq{model_name}_minimum_change_1000_march_27.json'
            if model_name == '':
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/boolq_minimum_change_1000_april_13_1000.json'
            
            
            # with open(minimum_change_train_path, 'r') as file:
            #     minimum_change_train_data_list_temp = json.load(file)       

            # minimum_change_train_data_list = []
            # for item in minimum_change_train_data_list_temp:
            #     item['answer'] = item['output']
            #     minimum_change_train_data_list.append(item)
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)     
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/gpt4_generated_boolq_False_100.json'
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/gpt4_generated_boolq_False_1000.json'
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
            mc_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/mc_dpo_10.json'
            gt_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/gt_dpo_10.json'
            sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/sample_10_dpo.json'
            gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/_gpt4_data_dpo_10.json'

            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/ANLI/paraphrased_data_anli_False_500.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]

            # with open(sample_10_train_path, 'r') as file:
            #     sample_10_train_data_list = json.load(file)
            #     sample_10_train_data_list = sample_10_train_data_list[:n_train]
            # for kkk in range(len(sample_10_train_data_list)):
            #     output = sample_10_train_data_list[kkk]['output']
            #     sample_10_train_data_list[kkk]['output'] = output[0]

            # with open(gpt4_train_path, 'r') as file:
            #     gpt4_train_data_list = json.load(file)
            #     gpt4_train_data_list = gpt4_train_data_list[:n_train]
            # for kkk in range(len(gpt4_train_data_list)):
            #     output = gpt4_train_data_list[kkk]['output']
            #     gpt4_train_data_list[kkk]['output'] = output[0]
            
            # # if enable_dpo:
            # with open(mc_train_path, 'r') as file:
            #     dpo_mc_train_data_list = json.load(file)
            #     dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            # with open(gt_train_path, 'r') as file:
            #     dpo_gt_train_data_list = json.load(file)
            #     dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            # with open(sample_10_train_path, 'r') as file:
            #     dpo_sample_10_train_data_list = json.load(file)
            #     dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            # with open(gpt4_train_path, 'r') as file:
            #     dpo_gpt4_train_data_list = json.load(file)
            #     dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]

        elif train_task_name.upper() =='WINOGRANDE': 
            train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/train.json'
            train_data_list = load_WINOGRANDE(train_path, n_train, finetune = True)
            
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/winogrande_minimum_change_100.json'
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/winogrande{model_name}_minimum_change_100_march_27.json'
            # with open(minimum_change_train_path, 'r') as file:
            #     minimum_change_train_data_list_temp = json.load(file) 
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)       

            # minimum_change_train_data_list = []
            # for item in minimum_change_train_data_list_temp:
            #     item['answer'] = item['output']
            #     minimum_change_train_data_list.append(item)
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/gpt4_generated_winogrande_False_100.json'
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/gpt4_generated_winogrande_False_100_march_27.json'
            
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
            mc_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/mc_dpo_10.json'
            gt_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/gt_dpo_10.json'
            sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/sample_10_dpo.json'
            gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/_gpt4_data_dpo_10.json'

            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/ANLI/paraphrased_data_anli_False_500.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]

            with open(sample_10_train_path, 'r') as file:
                sample_10_train_data_list = json.load(file)
                sample_10_train_data_list = sample_10_train_data_list[:n_train]
            for kkk in range(len(sample_10_train_data_list)):
                output = sample_10_train_data_list[kkk]['output']
                sample_10_train_data_list[kkk]['output'] = output[0]

            with open(gpt4_train_path, 'r') as file:
                gpt4_train_data_list = json.load(file)
                gpt4_train_data_list = gpt4_train_data_list[:n_train]
            for kkk in range(len(gpt4_train_data_list)):
                output = gpt4_train_data_list[kkk]['output']
                gpt4_train_data_list[kkk]['output'] = output[0]
            
            # if enable_dpo:
            with open(mc_train_path, 'r') as file:
                dpo_mc_train_data_list = json.load(file)
                dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            with open(gt_train_path, 'r') as file:
                dpo_gt_train_data_list = json.load(file)
                dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            with open(sample_10_train_path, 'r') as file:
                dpo_sample_10_train_data_list = json.load(file)
                dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            with open(gpt4_train_path, 'r') as file:
                dpo_gpt4_train_data_list = json.load(file)
                dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]

        # elif train_task_name.upper() =='TRIVIAQA': 
        #     train_path = f'{HOME_DIRECTORY}/dataset/TRIVIA_QA/train.json'
        #     train_data_list = load_TRIVIAQA(train_path, n_train, finetune = True)
            
        #     minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/TRIVIA_QA/triviaqa_minimum_change_100.json'
        #     with open(minimum_change_train_path, 'r') as file:
        #         minimum_change_train_data_list_temp = json.load(file)       

        #     minimum_change_train_data_list = []
        #     for item in minimum_change_train_data_list_temp:
        #         item['answer'] = item['output']
        #         minimum_change_train_data_list.append(item)
        #     minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
        #     gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/TRIVIA_QA/gpt4_generated_triviaqa_False_100.json'
        #     with open(gpt4_generated_data_train_path, 'r') as file:
        #         gpt4_generated_train_data_list = json.load(file)
        #     gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
        #     mc_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/mc_dpo_10.json'
        #     gt_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/gt_dpo_10.json'
        #     sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/sample_10_dpo.json'
        #     gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/_gpt4_data_dpo_10.json'

        #     paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/ANLI/paraphrased_data_anli_False_500.json'
        #     with open(paraphrased_data_train_path, 'r') as file:
        #         paraphrased_data_train_list = json.load(file)
        #     paraphrased_data_train_list = paraphrased_data_train_list[:n_train]

        #     with open(sample_10_train_path, 'r') as file:
        #         sample_10_train_data_list = json.load(file)
        #         sample_10_train_data_list = sample_10_train_data_list[:n_train]
        #     for kkk in range(len(sample_10_train_data_list)):
        #         output = sample_10_train_data_list[kkk]['output']
        #         sample_10_train_data_list[kkk]['output'] = output[0]

        #     with open(gpt4_train_path, 'r') as file:
        #         gpt4_train_data_list = json.load(file)
        #         gpt4_train_data_list = gpt4_train_data_list[:n_train]
        #     for kkk in range(len(gpt4_train_data_list)):
        #         output = gpt4_train_data_list[kkk]['output']
        #         gpt4_train_data_list[kkk]['output'] = output[0]
            
            # # if enable_dpo:
            # with open(mc_train_path, 'r') as file:
            #     dpo_mc_train_data_list = json.load(file)
            #     dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            # with open(gt_train_path, 'r') as file:
            #     dpo_gt_train_data_list = json.load(file)
            #     dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            # with open(sample_10_train_path, 'r') as file:
            #     dpo_sample_10_train_data_list = json.load(file)
            #     dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            # with open(gpt4_train_path, 'r') as file:
            #     dpo_gpt4_train_data_list = json.load(file)
            #     dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]

        elif train_task_name.upper() =='MMLU': 
            train_path = f'{HOME_DIRECTORY}/dataset/MMLU/train.json'
            train_data_list = load_MMLU(train_path, n_train, finetune = True)
            
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MMLU/mmlu_minimum_change_100.json'
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MMLU/mmlu{model_name}_minimum_change_100_march_27.json'
            
            # with open(minimum_change_train_path, 'r') as file:
            #     minimum_change_train_data_list_temp = json.load(file)       

            # minimum_change_train_data_list = []
            # for item in minimum_change_train_data_list_temp:
            #     item['answer'] = item['output']
            #     minimum_change_train_data_list.append(item)
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)     
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MMLU/gpt4_generated_mmlu_False_100.json'
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MMLU/gpt4_generated_mmlu_False_100_march_27.json'
            
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
            mc_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/mc_dpo_10.json'
            gt_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/gt_dpo_10.json'
            sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/sample_10_dpo.json'
            gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/_gpt4_data_dpo_10.json'

            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/ANLI/paraphrased_data_anli_False_500.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]

            # with open(sample_10_train_path, 'r') as file:
            #     sample_10_train_data_list = json.load(file)
            #     sample_10_train_data_list = sample_10_train_data_list[:n_train]
            # for kkk in range(len(sample_10_train_data_list)):
            #     output = sample_10_train_data_list[kkk]['output']
            #     sample_10_train_data_list[kkk]['output'] = output[0]

            # with open(gpt4_train_path, 'r') as file:
            #     gpt4_train_data_list = json.load(file)
            #     gpt4_train_data_list = gpt4_train_data_list[:n_train]
            # for kkk in range(len(gpt4_train_data_list)):
            #     output = gpt4_train_data_list[kkk]['output']
            #     gpt4_train_data_list[kkk]['output'] = output[0]
            
            # # if enable_dpo:
            # with open(mc_train_path, 'r') as file:
            #     dpo_mc_train_data_list = json.load(file)
            #     dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            # with open(gt_train_path, 'r') as file:
            #     dpo_gt_train_data_list = json.load(file)
            #     dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            # with open(sample_10_train_path, 'r') as file:
            #     dpo_sample_10_train_data_list = json.load(file)
            #     dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            # with open(gpt4_train_path, 'r') as file:
            #     dpo_gpt4_train_data_list = json.load(file)
            #     dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]

        elif train_task_name.upper() =='AGIEVAL': 
            train_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/train_march_27.json'
            train_data_list = load_AGIEVAL(train_path, n_train, finetune = True)
            
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/agieval_minimum_change_100.json'
            if model_name == '':
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/agieval_minimum_change_300_march_27_with_examples.json'
            if 'mistral' in model_name:
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/agieval_mistral_minimum_change_1000_march_27.json'
            
            # with open(minimum_change_train_path, 'r') as file:
            #     minimum_change_train_data_list_temp = json.load(file)       

            # minimum_change_train_data_list = []
            # for item in minimum_change_train_data_list_temp:
            #     item['answer'] = item['output']
            #     minimum_change_train_data_list.append(item)
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)     
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]

            # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/gpt4_generated_agieval_False_100.json'
            # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/gpt4_generated_agieval_False_100_march_27.json'
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/gpt4_generated_agieval_False_300.json'
            
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
            mc_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/mc_dpo_10.json'
            gt_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/gt_dpo_10.json'
            sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/sample_10_dpo.json'
            gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/ANLI/_gpt4_data_dpo_10.json'

            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/ANLI/paraphrased_data_anli_False_500.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]

            # with open(sample_10_train_path, 'r') as file:
            #     sample_10_train_data_list = json.load(file)
            #     sample_10_train_data_list = sample_10_train_data_list[:n_train]
            # for kkk in range(len(sample_10_train_data_list)):
            #     output = sample_10_train_data_list[kkk]['output']
            #     sample_10_train_data_list[kkk]['output'] = output[0]

            # with open(gpt4_train_path, 'r') as file:
            #     gpt4_train_data_list = json.load(file)
            #     gpt4_train_data_list = gpt4_train_data_list[:n_train]
            # for kkk in range(len(gpt4_train_data_list)):
            #     output = gpt4_train_data_list[kkk]['output']
            #     gpt4_train_data_list[kkk]['output'] = output[0]
            
            # # if enable_dpo:
            # with open(mc_train_path, 'r') as file:
            #     dpo_mc_train_data_list = json.load(file)
            #     dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            # with open(gt_train_path, 'r') as file:
            #     dpo_gt_train_data_list = json.load(file)
            #     dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            # with open(sample_10_train_path, 'r') as file:
            #     dpo_sample_10_train_data_list = json.load(file)
            #     dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            # with open(gpt4_train_path, 'r') as file:
            #     dpo_gpt4_train_data_list = json.load(file)
            #     dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]

        elif train_task_name.upper() =='ECQA': 
            train_path = f'{HOME_DIRECTORY}/dataset/ECQA/train.json'
            if variation_suffix == 'use_gt_rationale':
                train_data_list = load_ECQA(train_path, n_train, finetune = True, use_gt_rationale = True)
            else:
                train_data_list = load_ECQA(train_path, n_train, finetune = True)
            
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/ECQA/ecqa{model_name}_minimum_change_100_march_27.json'
            if 'mistral' in model_name:
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/ECQA/ecqa_mistral_minimum_change_1000_march_27.json'
            else:
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/ECQA/ecqa_minimum_change_1000_march_27.json'
            
            
            # with open(minimum_change_train_path, 'r') as file:
            #     minimum_change_train_data_list_temp = json.load(file)       

            # minimum_change_train_data_list = []
            # for item in minimum_change_train_data_list_temp:
            #     item['answer'] = item['output']
            #     minimum_change_train_data_list.append(item)
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)     
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/ECQA/gpt4_generated_ecqa_False_1000_march_27.json'
            
            with open(gpt4_generated_data_train_path, 'r') as file:
                gpt4_generated_train_data_list = json.load(file)
            gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]

            if 'mistral' in model_name:
                mc_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/mc_dpo_5_mistral.json'
                gt_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/gt_dpo_5_mistral.json'
                sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/sample_5_mistral_dpo.json'
                gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/_gpt4_data_dpo_5_mistral.json'
            else:
                mc_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/mc_dpo_5.json'
                gt_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/gt_dpo_5.json'
                sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/sample_5_dpo.json'
                gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/_gpt4_data_dpo_5.json'

            paraphrased_data_train_path = f'{HOME_DIRECTORY}/dataset/ANLI/paraphrased_data_anli_False_500.json'
            with open(paraphrased_data_train_path, 'r') as file:
                paraphrased_data_train_list = json.load(file)
            paraphrased_data_train_list = paraphrased_data_train_list[:n_train]

            # with open(sample_10_train_path, 'r') as file:
            #     sample_10_train_data_list = json.load(file)
            #     sample_10_train_data_list = sample_10_train_data_list[:n_train]
            # for kkk in range(len(sample_10_train_data_list)):
            #     output = sample_10_train_data_list[kkk]['output']
            #     sample_10_train_data_list[kkk]['output'] = output[0]

            # with open(gpt4_train_path, 'r') as file:
            #     gpt4_train_data_list = json.load(file)
            #     gpt4_train_data_list = gpt4_train_data_list[:n_train]
            # for kkk in range(len(gpt4_train_data_list)):
            #     output = gpt4_train_data_list[kkk]['output']
            #     gpt4_train_data_list[kkk]['output'] = output[0]
            
            # # if enable_dpo:
            # with open(mc_train_path, 'r') as file:
            #     dpo_mc_train_data_list = json.load(file)
            #     dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

            # with open(gt_train_path, 'r') as file:
            #     dpo_gt_train_data_list = json.load(file)
            #     dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

            # with open(sample_10_train_path, 'r') as file:
            #     dpo_sample_10_train_data_list = json.load(file)
            #     dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
            
            # with open(gpt4_train_path, 'r') as file:
            #     dpo_gpt4_train_data_list = json.load(file)
            #     dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]


        if 'mistral' in model_name:
            mc_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/mc_dpo_5_mistral.json'
            gt_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/gt_dpo_5_mistral.json'
            sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/sample_5_mistral_dpo.json'
            gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/_gpt4_data_dpo_5_mistral.json'
        else:
            mc_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/mc_dpo_5.json'
            gt_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/gt_dpo_5.json'
            sample_10_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/sample_5_dpo.json'
            gpt4_train_path = f'{HOME_DIRECTORY}/dpo_data/{train_task_name.upper()}/_gpt4_data_dpo_5.json'
        

        with open(sample_10_train_path, 'r') as file:
            sample_10_train_data_list = json.load(file)
            sample_10_train_data_list = sample_10_train_data_list[:n_train]
        for kkk in range(len(sample_10_train_data_list)):
            output = sample_10_train_data_list[kkk]['output']
            sample_10_train_data_list[kkk]['output'] = output[0]

        with open(gpt4_train_path, 'r') as file:
            gpt4_train_data_list = json.load(file)
            gpt4_train_data_list = gpt4_train_data_list[:n_train]
        for kkk in range(len(gpt4_train_data_list)):
            output = gpt4_train_data_list[kkk]['output']
            gpt4_train_data_list[kkk]['output'] = output[0]
        
        # if enable_dpo:
        with open(mc_train_path, 'r') as file:
            dpo_mc_train_data_list = json.load(file)
            dpo_mc_train_data_list = dpo_mc_train_data_list[:n_train]

        with open(gt_train_path, 'r') as file:
            dpo_gt_train_data_list = json.load(file)
            dpo_gt_train_data_list = dpo_gt_train_data_list[:n_train]

        with open(sample_10_train_path, 'r') as file:
            dpo_sample_10_train_data_list = json.load(file)
            dpo_sample_10_train_data_list = dpo_sample_10_train_data_list[:n_train]
        
        with open(gpt4_train_path, 'r') as file:
            dpo_gpt4_train_data_list = json.load(file)
            dpo_gpt4_train_data_list = dpo_gpt4_train_data_list[:n_train]




        intermediate_finetune_file_name = f'{train_task_name}_finetune_{n_train}'
        intermediate_finetune_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_finetune_file_name}.json"
        train_data = []
        for item in train_data_list:
            temp = {}
            temp['instruction'] = item['question']
            temp['output'] = item['answer']
            temp['input'] = item['input']
            train_data.append(temp)
        with open(intermediate_finetune_file_path, 'w') as json_file:
            json.dump(train_data, json_file, indent=4)

        intermediate_minimum_change_train_file_name = f'{train_task_name}{model_name}_minimum_change_{n_train}'
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
            temp['input'] = item['input']
            train_data.append(temp)
        intermediate_minimum_change_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_minimum_change_train_file_name}.json"
        with open(intermediate_minimum_change_train_file_path, 'w') as json_file:
            json.dump(train_data, json_file, indent=4)

        intermediate_gpt4_generated_data_train_file_name = f'{train_task_name}_{variation_suffix}{model_name}_gpt4_generated_data_{n_train}'
        train_data = []
        for item in gpt4_generated_train_data_list:
            temp = {}
            temp['instruction'] = item['question']
            temp['output'] = item['answer']
            temp['input'] = item['input']
            train_data.append(temp)
        intermediate_gpt4_generated_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_gpt4_generated_data_train_file_name}.json"
        with open(intermediate_gpt4_generated_train_file_path, 'w') as json_file:
            json.dump(train_data, json_file, indent=4)

        if train_task_name.lower() == 'anli' or train_task_name.lower() == 'mnli' or train_task_name.lower() == 'snli' or train_task_name.lower() == 'scitail' or train_task_name.lower() == 'gsm8k' or train_task_name.lower() == 'api_bank' or train_task_name.lower() == 'code' or 'math' in train_task_name.lower():
            intermediate_paraphrased_data_train_file_name = f'{train_task_name}_paraphrased_data_{n_train}'
            intermediate_paraphrased_data_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_paraphrased_data_train_file_name}.json"
            with open(intermediate_paraphrased_data_train_file_path, 'w') as json_file:
                json.dump(paraphrased_data_train_list, json_file, indent=4)
        
        # if train_task_name.lower() != 'code' and train_task_name.lower() != 'api_bank' and train_task_name.lower() != 'mbpp' and train_task_name.lower() != 'mnli' and train_task_name.lower() != 'snli' and train_task_name.lower() != 'scitail' and train_task_name.lower() != 'boolq' and train_task_name.lower() != 'piqa' and train_task_name.lower() != 'winogrande' and train_task_name.lower() != 'mmlu' and train_task_name.lower() != 'agieval' and train_task_name.lower() != 'ecqa' and not multi_task_or_continual_learning:
        #     intermediate_dpo_minimum_change_train_file_name = f'dpo_{train_task_name}{model_name}_minimum_change_{n_train}'
        #     intermediate_dpo_minimum_change_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_dpo_minimum_change_train_file_name}.json"
        #     with open(intermediate_dpo_minimum_change_train_file_path, 'w') as json_file:
        #         json.dump(dpo_mc_train_data_list, json_file, indent=4)

        #     intermediate_dpo_finetune_file_name = f'dpo_{train_task_name}_finetune_{n_train}'
        #     intermediate_dpo_finetune_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_dpo_finetune_file_name}.json"
        #     with open(intermediate_dpo_finetune_file_path, 'w') as json_file:
        #         json.dump(dpo_gt_train_data_list, json_file, indent=4)

        #     intermediate_dpo_sample_10_train_file_name = f'dpo_{train_task_name}_sample_10_{n_train}'
        #     intermediate_dpo_sample_10_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_dpo_sample_10_train_file_name}.json"
        #     with open(intermediate_dpo_sample_10_train_file_path, 'w') as json_file:
        #         json.dump(dpo_sample_10_train_data_list, json_file, indent=4)

        #     intermediate_gpt4_train_file_name = f'dpo_{train_task_name}_{variation_suffix}_gpt4_{n_train}'
        #     intermediate_dpo_gpt4_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_gpt4_train_file_name}.json"
        #     with open(intermediate_dpo_gpt4_train_file_path, 'w') as json_file:
        #         json.dump(dpo_gpt4_train_data_list, json_file, indent=4)
        #     intermediate_sample_10_train_file_name = f'dpo_{train_task_name}_sample_10_{n_train}'
        #     intermediate_sample_10_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_sample_10_train_file_name}.json"
        #     with open(intermediate_sample_10_train_file_path, 'w') as json_file:
        #         json.dump(sample_10_train_data_list, json_file, indent=4)

        intermediate_dpo_minimum_change_train_file_name = f'dpo_{train_task_name}{model_name}_minimum_change_{n_train}'
        intermediate_dpo_minimum_change_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_dpo_minimum_change_train_file_name}.json"
        with open(intermediate_dpo_minimum_change_train_file_path, 'w') as json_file:
            json.dump(dpo_mc_train_data_list, json_file, indent=4)

        intermediate_dpo_finetune_file_name = f'dpo_{train_task_name}_finetune_{n_train}'
        intermediate_dpo_finetune_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_dpo_finetune_file_name}.json"
        with open(intermediate_dpo_finetune_file_path, 'w') as json_file:
            json.dump(dpo_gt_train_data_list, json_file, indent=4)

        intermediate_dpo_sample_10_train_file_name = f'dpo_{train_task_name}_sample_10_{n_train}'
        intermediate_dpo_sample_10_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_dpo_sample_10_train_file_name}.json"
        with open(intermediate_dpo_sample_10_train_file_path, 'w') as json_file:
            json.dump(dpo_sample_10_train_data_list, json_file, indent=4)

        intermediate_gpt4_train_file_name = f'dpo_{train_task_name}_{variation_suffix}_gpt4_{n_train}'
        intermediate_dpo_gpt4_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_gpt4_train_file_name}.json"
        with open(intermediate_dpo_gpt4_train_file_path, 'w') as json_file:
            json.dump(dpo_gpt4_train_data_list, json_file, indent=4)
        intermediate_sample_10_train_file_name = f'dpo_{train_task_name}_sample_10_{n_train}'
        intermediate_sample_10_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_sample_10_train_file_name}.json"
        with open(intermediate_sample_10_train_file_path, 'w') as json_file:
            json.dump(sample_10_train_data_list, json_file, indent=4)


        return intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path, intermediate_paraphrased_data_train_file_path, intermediate_dpo_finetune_file_path, intermediate_dpo_minimum_change_train_file_path, intermediate_dpo_gpt4_train_file_path, intermediate_dpo_sample_10_train_file_path, intermediate_sample_10_train_file_path
    else:
        gpt4_generated_train_data_total_list = []
        minimum_change_train_data_total_list = []
        train_data_total_list = []

        intermediate_finetune_file_path_list, intermediate_minimum_change_train_file_path_list, intermediate_gpt4_generated_train_file_path_list = [], [], []
        for task_item in task_sequence_list:
            if task_item == 'gsm8k' or task_item == 'all':
                train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/train.json'
                train_data_list = load_GSM8K(train_path, n_train, zeroshot = False)        
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/minimum_change_train_nov_25.json'
                minimum_change_train_data_list = load_GSM8K(minimum_change_train_path, n_train)
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gpt4_generated_gsm8k_False_999999.json'
                with open(gpt4_generated_data_train_path, 'r') as file:
                    gpt4_generated_train_data_list = json.load(file)
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
                train_data_total_list.append(train_data_list)
                minimum_change_train_data_total_list.append(minimum_change_train_data_list)
                gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

                if continual_learning:
                    intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_gsm8k', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
                    intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
                    intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
                    intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)

            if task_item == 'train_algebra' or task_item == 'all':
                train_path = f'{HOME_DIRECTORY}/dataset/MATH/train_algebra.json'
                train_data_list = load_MATH(train_path, n_train, zeroshot = False)
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MATH/minimum_change_train_algebra.json'
                minimum_change_train_data_list = load_MATH(minimum_change_train_path, n_train, minimum_change_or_zero_shot = True)
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MATH/gpt4_generated_math_algebra_data_step_by_step_False_500.json'
                with open(gpt4_generated_data_train_path, 'r') as file:
                    gpt4_generated_train_data_list = json.load(file)
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
                train_data_total_list.append(train_data_list)
                minimum_change_train_data_total_list.append(minimum_change_train_data_list)
                gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

                if continual_learning:
                    intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_math_algebra', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
                    intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
                    intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
                    intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)

            if task_item == 'train_counting_and_probability' or task_item == 'all':
                train_path = f'{HOME_DIRECTORY}/dataset/MATH/train_counting_and_probability.json'
                train_data_list = load_MATH(train_path, n_train, zeroshot = False)
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MATH/minimum_change_train_counting_and_probability.json'
                minimum_change_train_data_list = load_MATH(minimum_change_train_path, n_train, minimum_change_or_zero_shot = True)
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MATH/gpt4_generated_math_counting_and_probability_data_step_by_step_False_500.json'
                with open(gpt4_generated_data_train_path, 'r') as file:
                    gpt4_generated_train_data_list = json.load(file)
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
                train_data_total_list.append(train_data_list)
                minimum_change_train_data_total_list.append(minimum_change_train_data_list)
                gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

                if continual_learning:
                    intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_math_counting_and_probabilitiy', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
                    intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
                    intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
                    intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)

            if task_item == 'anli' or task_item == 'all':
                train_path = f'{HOME_DIRECTORY}/dataset/ANLI/train.json'
                train_data_list = load_ANLI_jan_30(train_path, n_train, finetune_on_gt = True)
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/ANLI/anli_minimum_change_9999_march_9_old.json'
                with open(minimum_change_train_path, 'r') as file:
                    minimum_change_train_data_list_temp = json.load(file)
                minimum_change_train_data_list = []
                for item in minimum_change_train_data_list_temp:
                    item['answer'] = item['output']
                    minimum_change_train_data_list.append(item)
                minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/ANLI/gpt4_generated_anli_data_step_by_step_False_500.json'
                with open(gpt4_generated_data_train_path, 'r') as file:
                    gpt4_generated_train_data_list = json.load(file)
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
                train_data_total_list.append(train_data_list)
                minimum_change_train_data_total_list.append(minimum_change_train_data_list)
                gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

                if continual_learning:
                    intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_anli', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
                    intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
                    intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
                    intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)

            if task_item == 'snli' or task_item == 'all':
                train_path = f'{HOME_DIRECTORY}/dataset/SNLI/train.json'
                train_data_list = load_SNLI(train_path, n_train, finetune = True)
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/SNLI/snli_minimum_change_100.json'
                with open(minimum_change_train_path, 'r') as file:
                    minimum_change_train_data_list_temp = json.load(file)       
                minimum_change_train_data_list = []
                for item in minimum_change_train_data_list_temp:
                    item['answer'] = item['output']
                    minimum_change_train_data_list.append(item)
                minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/SNLI/gpt4_generated_snli_False_999999.json'
                with open(gpt4_generated_data_train_path, 'r') as file:
                    gpt4_generated_train_data_list = json.load(file)
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
                train_data_total_list.append(train_data_list)
                minimum_change_train_data_total_list.append(minimum_change_train_data_list)
                gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

                if continual_learning:
                    intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_snli', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
                    intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
                    intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
                    intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)

            if task_item == 'scitail' or task_item == 'all':
                train_path = f'{HOME_DIRECTORY}/dataset/SCITAIL/train.json'
                train_data_list = load_SCITAIL(train_path, n_train, finetune = True)
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/SCITAIL/scitail_minimum_change_100.json'
                with open(minimum_change_train_path, 'r') as file:
                    minimum_change_train_data_list_temp = json.load(file)       
                minimum_change_train_data_list = []
                for item in minimum_change_train_data_list_temp:
                    item['answer'] = item['output']
                    minimum_change_train_data_list.append(item)
                minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/SCITAIL/gpt4_generated_scitail_False_100.json'
                with open(gpt4_generated_data_train_path, 'r') as file:
                    gpt4_generated_train_data_list = json.load(file)
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
                train_data_total_list.append(train_data_list)
                minimum_change_train_data_total_list.append(minimum_change_train_data_list)
                gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

                if continual_learning:
                    intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_scitail', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
                    intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
                    intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
                    intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)

            if task_item == 'code' or task_item == 'all':
                train_path = f'{HOME_DIRECTORY}/dataset/CODE/train{variation_suffix_code}.json'
                # train_data_list = load_CODE(train_path, n_train, minimum_change = False)
                train_data_list = load_CODE_code_only(train_path, n_train, minimum_change = False)
                
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/CODE/minimum_change_train_jan_30{variation_suffix_code}.json'
                # minimum_change_train_data_list = load_CODE(minimum_change_train_path, n_train, minimum_change = True)
                minimum_change_train_data_list = load_CODE_code_only(minimum_change_train_path, n_train, minimum_change = True)
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/CODE/gpt4_generated_code_data_step_by_step_False_500{variation_suffix_code}.json'
                with open(gpt4_generated_data_train_path, 'r') as file:
                    gpt4_generated_train_data_list = json.load(file)
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
                train_data_total_list.append(train_data_list)
                minimum_change_train_data_total_list.append(minimum_change_train_data_list)
                gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

                if continual_learning:
                    intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_code', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
                    intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
                    intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
                    intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)

            if task_item == 'mbpp' or task_item == 'all':
                train_path = f'{HOME_DIRECTORY}/dataset/MBPP/train.json'
                train_data_list = load_MBPP(train_path, n_train)
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MBPP/minimum_change_train.json'
                minimum_change_train_data_list = load_MBPP(minimum_change_train_path, n_train)
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MBPP/gpt4_generated_mbpp_False_999999.json'
                with open(gpt4_generated_data_train_path, 'r') as file:
                    gpt4_generated_train_data_list = json.load(file)
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
                train_data_total_list.append(train_data_list)
                minimum_change_train_data_total_list.append(minimum_change_train_data_list)
                gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

                if continual_learning:
                    intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_mbpp', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
                    intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
                    intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
                    intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)

            if task_item == 'boolq' or task_item == 'all':
                train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/train.json'
                train_data_list = load_BOOLQ(train_path, n_train, finetune = True)
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/boolq_minimum_change_100.json'
                with open(minimum_change_train_path, 'r') as file:
                    minimum_change_train_data_list_temp = json.load(file)       
                minimum_change_train_data_list = []
                for item in minimum_change_train_data_list_temp:
                    item['answer'] = item['output']
                    minimum_change_train_data_list.append(item)
                minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/gpt4_generated_boolq_False_100.json'
                with open(gpt4_generated_data_train_path, 'r') as file:
                    gpt4_generated_train_data_list = json.load(file)
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
                train_data_total_list.append(train_data_list)
                minimum_change_train_data_total_list.append(minimum_change_train_data_list)
                gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

                if continual_learning:
                    intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_boolq', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
                    intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
                    intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
                    intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)
            
            if task_item == 'piqa' or task_item == 'all':
                train_path = f'{HOME_DIRECTORY}/dataset/PIQA/train.json'
                train_data_list = load_PIQA(train_path, n_train, finetune = True)
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/PIQA/piqa_minimum_change_100.json'
                with open(minimum_change_train_path, 'r') as file:
                    minimum_change_train_data_list_temp = json.load(file)       
                minimum_change_train_data_list = []
                for item in minimum_change_train_data_list_temp:
                    item['answer'] = item['output']
                    minimum_change_train_data_list.append(item)
                minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/PIQA/gpt4_generated_piqa_False_100.json'
                with open(gpt4_generated_data_train_path, 'r') as file:
                    gpt4_generated_train_data_list = json.load(file)
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
                train_data_total_list.append(train_data_list)
                minimum_change_train_data_total_list.append(minimum_change_train_data_list)
                gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

                if continual_learning:
                    intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_piqa', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
                    intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
                    intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
                    intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)

            if task_item == 'winogrande' or task_item == 'all':
                train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/train.json'
                train_data_list = load_WINOGRANDE(train_path, n_train, finetune = True)
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/winogrande_minimum_change_100.json'
                with open(minimum_change_train_path, 'r') as file:
                    minimum_change_train_data_list_temp = json.load(file)       
                minimum_change_train_data_list = []
                for item in minimum_change_train_data_list_temp:
                    item['answer'] = item['output']
                    minimum_change_train_data_list.append(item)
                minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/gpt4_generated_winogrande_False_100.json'
                with open(gpt4_generated_data_train_path, 'r') as file:
                    gpt4_generated_train_data_list = json.load(file)
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
                train_data_total_list.append(train_data_list)
                minimum_change_train_data_total_list.append(minimum_change_train_data_list)
                gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

                if continual_learning:
                    intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_winogrande', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
                    intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
                    intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
                    intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)

            if task_item == 'mmlu' or task_item == 'all':
                train_path = f'{HOME_DIRECTORY}/dataset/MMLU/train.json'
                train_data_list = load_MMLU(train_path, n_train, finetune = True)
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MMLU/mmlu_minimum_change_100.json'
                with open(minimum_change_train_path, 'r') as file:
                    minimum_change_train_data_list_temp = json.load(file)       
                minimum_change_train_data_list = []
                for item in minimum_change_train_data_list_temp:
                    item['answer'] = item['output']
                    minimum_change_train_data_list.append(item)
                minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MMLU/gpt4_generated_mmlu_False_100.json'
                with open(gpt4_generated_data_train_path, 'r') as file:
                    gpt4_generated_train_data_list = json.load(file)
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
                train_data_total_list.append(train_data_list)
                minimum_change_train_data_total_list.append(minimum_change_train_data_list)
                gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

                if continual_learning:
                    intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_mmlu', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
                    intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
                    intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
                    intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)

            if task_item == 'agieval' or task_item == 'all':
                train_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/train.json'
                train_data_list = load_AGIEVAL(train_path, n_train, finetune = True)
                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/agieval_minimum_change_100.json'
                with open(minimum_change_train_path, 'r') as file:
                    minimum_change_train_data_list_temp = json.load(file)       
                minimum_change_train_data_list = []
                for item in minimum_change_train_data_list_temp:
                    item['answer'] = item['output']
                    minimum_change_train_data_list.append(item)
                minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/AGIEVAL/gpt4_generated_agieval_False_100.json'
                with open(gpt4_generated_data_train_path, 'r') as file:
                    gpt4_generated_train_data_list = json.load(file)
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
                train_data_total_list.append(train_data_list)
                minimum_change_train_data_total_list.append(minimum_change_train_data_list)
                gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

                if continual_learning:
                    intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_agieval', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
                    intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
                    intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
                    intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)
            
            if task_item == 'ecqa' or task_item == 'all':
                train_path = f'{HOME_DIRECTORY}/dataset/ECQA/train.json'
                if variation_suffix == 'use_gt_rationale':
                    train_data_list = load_ECQA(train_path, n_train, finetune = True, use_gt_rationale = True)
                else:
                    train_data_list = load_ECQA(train_path, n_train, finetune = True)

                minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/ECQA/ecqa_minimum_change_100.json'
                with open(minimum_change_train_path, 'r') as file:
                    minimum_change_train_data_list = json.load(file)       
                minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
                gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/ECQA/gpt4_generated_ecqa_False_100.json'
                with open(gpt4_generated_data_train_path, 'r') as file:
                    gpt4_generated_train_data_list = json.load(file)
                gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
                train_data_total_list.append(train_data_list)
                minimum_change_train_data_total_list.append(minimum_change_train_data_list)
                gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

                if continual_learning:
                    intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_ecqa', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
                    intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
                    intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
                    intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)



            # train_path = f'{HOME_DIRECTORY}/dataset/TRIVIAQA/train.json'
            # train_data_list = load_TRIVIAQA(train_path, n_train, finetune = True)
            # minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/TRIVIAQA/triviaqa_minimum_change_100.json'
            # with open(minimum_change_train_path, 'r') as file:
            #     minimum_change_train_data_list_temp = json.load(file)       
            # minimum_change_train_data_list = []
            # for item in minimum_change_train_data_list_temp:
            #     item['answer'] = item['output']
            #     minimum_change_train_data_list.append(item)
            # minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            # gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/TRIVIAQA/gpt4_generated_triviaqa_False_100.json'
            # with open(gpt4_generated_data_train_path, 'r') as file:
            #     gpt4_generated_train_data_list = json.load(file)
            # gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]
            # train_data_total_list.append(train_data_list)
            # minimum_change_train_data_total_list.append(minimum_change_train_data_list)
            # gpt4_generated_train_data_total_list.append(gpt4_generated_train_data_list)

            # if continual_learning:
            #     intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = prepare_llama_factory_readable_files(HOME_DIRECTORY, train_task_name + '_triviaqa', n_train, multi_task_or_continual_learning, variation_suffix = variation_suffix, train_data_list = train_data_list, minimum_change_train_data_list = minimum_change_train_data_list, gpt4_generated_train_data_list = gpt4_generated_train_data_list, continual_learning = True)
            #     intermediate_finetune_file_path_list.append(intermediate_finetune_file_path)
            #     intermediate_minimum_change_train_file_path_list.append(intermediate_minimum_change_train_file_path)
            #     intermediate_gpt4_generated_train_file_path_list.append(intermediate_gpt4_generated_train_file_path)


            if multi_task_learning:
                train_data_list = [item for sublist in train_data_total_list for item in sublist]
                minimum_change_train_data_list = [item for sublist in minimum_change_train_data_total_list for item in sublist]
                gpt4_generated_train_data_list = [item for sublist in gpt4_generated_train_data_total_list for item in sublist]

                random.shuffle(train_data_list)
                random.shuffle(minimum_change_train_data_list)
                random.shuffle(gpt4_generated_train_data_list)
            

        return intermediate_finetune_file_path_list, intermediate_minimum_change_train_file_path_list, intermediate_gpt4_generated_train_file_path_list





# intermediate_finetune_file_name = f'{train_task_name}_finetune_{n_train}'
# intermediate_finetune_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_finetune_file_name}.json"
# train_data = []
# for item in train_data_list:
#     temp = {}
#     temp['instruction'] = item['question']
#     temp['output'] = item['answer']
#     temp['input'] = item['input']
#     train_data.append(temp)
# with open(intermediate_finetune_file_path, 'w') as json_file:
#     json.dump(train_data, json_file, indent=4)

# intermediate_minimum_change_train_file_name = f'{train_task_name}_minimum_change_{n_train}'
# train_data = []
# for item in minimum_change_train_data_list:
#     temp = {}
#     try:
#         temp['instruction'] = item['question']
#     except: 
#         temp['instruction'] = item['instruction']
#     try:
#         temp['output'] = item['answer']
#     except: 
#         temp['output'] = item['output']
#     temp['input'] = item['input']
#     train_data.append(temp)
# intermediate_minimum_change_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_minimum_change_train_file_name}.json"
# with open(intermediate_minimum_change_train_file_path, 'w') as json_file:
#     json.dump(train_data, json_file, indent=4)

# intermediate_gpt4_generated_data_train_file_name = f'{train_task_name}_{variation_suffix}_gpt4_generated_data_{n_train}'
# train_data = []
# for item in gpt4_generated_train_data_list:
#     temp = {}
#     temp['instruction'] = item['question']
#     temp['output'] = item['answer']
#     temp['input'] = item['input']
#     train_data.append(temp)
# intermediate_gpt4_generated_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_gpt4_generated_data_train_file_name}.json"
# with open(intermediate_gpt4_generated_train_file_path, 'w') as json_file:
#     json.dump(train_data, json_file, indent=4)

# if train_task_name.lower() == 'anli' or train_task_name.lower() == 'mnli' or train_task_name.lower() == 'snli' or train_task_name.lower() == 'scitail' or train_task_name.lower() == 'gsm8k' or train_task_name.lower() == 'api_bank' or train_task_name.lower() == 'code' or 'math' in train_task_name.lower():
#     intermediate_paraphrased_data_train_file_name = f'{train_task_name}_paraphrased_data_{n_train}'
#     intermediate_paraphrased_data_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_paraphrased_data_train_file_name}.json"
#     with open(intermediate_paraphrased_data_train_file_path, 'w') as json_file:
#         json.dump(paraphrased_data_train_list, json_file, indent=4)

# if train_task_name.lower() != 'code' and train_task_name.lower() != 'api_bank' and train_task_name.lower() != 'mbpp' and train_task_name.lower() != 'mnli' and train_task_name.lower() != 'snli' and train_task_name.lower() != 'scitail' and not multi_task_or_continual_learning:
#     intermediate_dpo_finetune_file_path = ''
#     intermediate_dpo_minimum_change_train_file_path = ''

#     intermediate_dpo_minimum_change_train_file_name = f'dpo_{train_task_name}_minimum_change_{n_train}'
#     intermediate_dpo_minimum_change_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_dpo_minimum_change_train_file_name}.json"
#     with open(intermediate_dpo_minimum_change_train_file_path, 'w') as json_file:
#         json.dump(dpo_mc_train_data_list, json_file, indent=4)

#     intermediate_dpo_finetune_file_name = f'dpo_{train_task_name}_finetune_{n_train}'
#     intermediate_dpo_finetune_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_dpo_finetune_file_name}.json"
#     with open(intermediate_dpo_finetune_file_path, 'w') as json_file:
#         json.dump(dpo_gt_train_data_list, json_file, indent=4)

#     intermediate_dpo_sample_10_train_file_name = f'dpo_{train_task_name}_sample_10_{n_train}'
#     intermediate_dpo_sample_10_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_dpo_sample_10_train_file_name}.json"
#     with open(intermediate_dpo_sample_10_train_file_path, 'w') as json_file:
#         json.dump(dpo_sample_10_train_data_list, json_file, indent=4)

#     intermediate_gpt4_train_file_name = f'dpo_{train_task_name}_{variation_suffix}_gpt4_{n_train}'
#     intermediate_dpo_gpt4_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_gpt4_train_file_name}.json"
#     with open(intermediate_dpo_gpt4_train_file_path, 'w') as json_file:
#         json.dump(dpo_gpt4_train_data_list, json_file, indent=4)
#     intermediate_sample_10_train_file_name = f'dpo_{train_task_name}_sample_10_{n_train}'
#     intermediate_dpo_sample_10_train_file_path = f"{HOME_DIRECTORY}/alpaca_data/{intermediate_sample_10_train_file_name}.json"
#     with open(intermediate_dpo_sample_10_train_file_path, 'w') as json_file:
#         json.dump(sample_10_train_data_list, json_file, indent=4)
