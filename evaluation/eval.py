import os
import re
import gc
import sys
import json
import torch
from fractions import Fraction
import subprocess
import shutil
import random
from multiprocessing import Process, Manager



# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.log_writter import *
from config.config import HOME_DIRECTORY, OUTPUT_RECORD_DIRECTORY, HOME_DIRECTORY, MODEL_DIRECTORY
from utils.function import prediction, record_json_data_to_file, extract_gsm8k_num
# from alpaca.alpaca_predict import alpaca_predict
from utils.llama_factory_data_file_processor import put_json_list_to_data_info
# from APPS_data.eval.test_one_solution_modified import run_main

# def extract_last_number(text):
#     # Define a regex pattern for matching integers, decimals, fractions, and percentages
#     # pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\d+\.\d+%?|\d+\/\d+%?|\d+%?)'

#     # Remove commas if present
#     text = text.replace(',', '')
#     # pattern = r'(\d+(?:\.\d+)?%?|\d+\/\d+%?|\d+%?)'
#     pattern = r'(-?\d+(?:\.\d+)?%?|-?\d+\/\d+%?|-?\d+%?)'

    
#     # Find all numbers in the string
#     all_numbers = re.findall(pattern, text)
    
#     # If any numbers are found, return the last one
#     if all_numbers:
#         number = all_numbers[-1]
        
#         # Handle fractions
#         if '/' in number:
#             number = float(Fraction(number.replace('%', '')))  # remove '%' if present and convert to float
#             if '%' in all_numbers[-1]:  # if it was a percentage
#                 number /= 100
#             return str(number)
        
#         # # Remove commas if present
#         # number = number.replace(',', '')
        
#         # Handle percentages
#         is_percentage = False
#         if '%' in number:
#             is_percentage = True
#             number = number.replace('%', '')
        
#         # Convert to float
#         number = float(number)
        
#         # Handle percentage adjustment
#         if is_percentage:
#             number /= 100
        
#         return str(number)
#     else:
#         return -3333333333333  # or any other suitable default value

#
#def extract_last_number(text):
#    # Regex pattern to handle negative numbers, simple fractions, LaTeX-style fractions, and percentages
#    pattern = r'(-?\d+(?:,\d{3})*(?:\.\d+)?%?|-?\d+\/-?\d+%?|-?\d+%?|\\frac\{-?\d+\}\{-?\d+\})'
#
#    # Find all numbers in the string
#    all_numbers = re.findall(pattern, text)
#
#    # Process the last number
#    if all_numbers:
#        number = all_numbers[-1]
#        # Handle LaTeX-style fractions
#        frac_pattern = r'\\frac\{(-?\d+)\}\{(-?\d+)\}'
#        frac_match = re.search(frac_pattern, number)
#        if frac_match:
#            numerator, denominator = frac_match.groups()
#            number = str(float(Fraction(int(numerator), int(denominator))))
#            return number
#
#        # Handle percentages and remove commas from numbers
#        is_percentage = '%' in number
#        number = number.replace('%', '').replace(',', '')
#
#        # Convert to float and adjust for percentage if needed
#        number = float(number)
#        if is_percentage:
#            number /= 100
#
#        return str(number)
#    else:
#        return -3333333333333

# def extract_boxed_content(text):
#     boxed_content = text
#     if 'boxed' in text:
#         boxed_content = re.findall(r"\\boxed{((?:[^{}]|{[^{}]*})+)}", text)
#         if boxed_content:
#             return boxed_content[0], True
#         else:
#             return text, False
#     else:
#         return text, False

def extract_boxed_content(s):
    start = s.rfind('\\boxed{')
    if start == -1:
        return None
    
    end = s.rfind('}')
        
    if end != 0:
        answer = s[start + 7 : end]
        return answer  # 7 is the length of '\\boxed{'

# def extract_last_number(text):
#     # Regex pattern to handle negative numbers, simple fractions, LaTeX-style fractions, and percentages
#     return_num = -3333333333333
#     pattern = r'(-?\d+(?:,\d{3})*(?:\.\d+)?%?|-?\d+\/-?\d+%?|-?\d+%?|\\frac\{-?\d+\}\{-?\d+\})'

#     text, Found_boxed_content = extract_boxed_content(text)
# #    if not text:
# #        return return_num
#     if Found_boxed_content:
#         if 'sqrt' in text or '^' in text or '(' in text:
#             return return_num

#     # Find all numbers in the string
#     all_numbers = re.findall(pattern, text)

#     # Process the last number
#     if all_numbers:
#         number = all_numbers[-1]
#         # Handle LaTeX-style fractions
#         frac_pattern = r'\\frac\{(-?\d+)\}\{(-?\d+)\}'
#         frac_match = re.search(frac_pattern, number)
#         if frac_match:
#             numerator, denominator = frac_match.groups()
#             number = str(float(Fraction(int(numerator), int(denominator))))
#             return number

#         # Handle percentages and remove commas from numbers
#         is_percentage = '%' in number
#         number = number.replace('%', '').replace(',', '')
        
#         # Convert to float and adjust for percentage if needed
#         number = float(number)
#         if is_percentage:
#             number /= 100

#         return str(number)
#     else:
#         return return_num  # No number found
    
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



# if 'sqrt' in number:
#     return None

# if '^' in number:
#     return None

# if re.search(r'boxed\{\([^)]+\)\}', number):
#     return None

# 4x(8x^2-x+5



def calc_accuracy_GSM8K(question_list, output_list, groundtruth_list, output_folder_name, file_name, task_name = '', sampling_num = 1, train_method = ''):
    eval_data_list_updated = []
    mispredict_eval_data_list_updated = []
    eval_num = 0 # how many data is evaluated?
    accuracy = 0
    cover_ratio = 0

    for i in range(len(output_list)):
        if i % 50 ==0:
            print(i)
        temp = output_list[i]
        # extracted_final_answer = extract_boxed_content(temp) 
        extracted_final_answer = extract_last_number(temp)
        final_answer = extracted_final_answer

        # temp = groundtruth_list[i]
        # extracted_groundtruth = extract_gsm8k_num(temp)
        # extracted_groundtruth = extract_last_number(temp)
        # extracted_groundtruth = extract_boxed_content(temp) 
        # groundtruth_num = extracted_groundtruth
        groundtruth_num = groundtruth_list[i]

        result = float(groundtruth_num)
        groundtruth_num = f"{result:.2f}"
        final_answer = float(final_answer)
        final_answer = f"{final_answer:.2f}"
        eval_num += 1
        if final_answer == groundtruth_num:
            item_temp = {}
            item_temp['question'] = question_list[i]
            item_temp['groundtruth number'] = groundtruth_num
            item_temp['numerical final answer'] = final_answer
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = output_list[i]
            item_temp['extracted groundtruth'] = groundtruth_num
            item_temp['extracted final answer'] = extracted_final_answer
            item_temp['correctness'] = 'Correct'
            eval_data_list_updated.append(item_temp)
        else:
            item_temp = {}
            item_temp['question'] = question_list[i]
            item_temp['groundtruth number'] = groundtruth_num
            item_temp['numerical final answer'] = final_answer
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = output_list[i]
            item_temp['extracted groundtruth'] = groundtruth_num
            item_temp['extracted final answer'] = extracted_final_answer
            item_temp['correctness'] = 'Incorrect'
            mispredict_eval_data_list_updated.append(item_temp)
           
    accuracy = len(eval_data_list_updated)/eval_num      
    cover_ratio = eval_num/len(output_list)

    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/gsm8k_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
        json.dump(eval_data_list_updated, file, indent=4)
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/gsm8k_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
        json.dump(mispredict_eval_data_list_updated, file, indent=4)
    return accuracy, cover_ratio

def calc_accuracy_MATH(question_list, output_list, groundtruth_list, output_folder_name, file_name, task_name = '', sampling_num = 1, train_method = '', math_task_name = ''):
    eval_data_list_updated = []
    mispredict_eval_data_list_updated = []
    eval_num = 0 # how many data is evaluated?
    accuracy = 0
    cover_ratio = 0

    for i in range(len(output_list)):
        if i % 50 ==0:
            print(i)
        temp = output_list[i]
        extracted_final_answer = extract_last_number(temp)
        final_answer = extracted_final_answer

        temp = groundtruth_list[i]
        extracted_groundtruth = extract_last_number(temp)
        groundtruth_num = extracted_groundtruth

        result = float(groundtruth_num)
        groundtruth_num = f"{result:.2f}"
        final_answer = float(final_answer)
        final_answer = f"{final_answer:.2f}"

        eval_num += 1
        if final_answer == groundtruth_num:
            item_temp = {}
            item_temp['question'] = question_list[i]
            item_temp['groundtruth number'] = groundtruth_num
            item_temp['numerical final answer'] = final_answer
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = output_list[i]
            item_temp['extracted groundtruth'] = extracted_groundtruth
            item_temp['extracted final answer'] = extracted_final_answer
            item_temp['correctness'] = 'Correct'
            eval_data_list_updated.append(item_temp)
        else:
            item_temp = {}
            item_temp['question'] = question_list[i]
            item_temp['groundtruth number'] = groundtruth_num
            item_temp['numerical final answer'] = final_answer
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = output_list[i]
            item_temp['extracted groundtruth'] = extracted_groundtruth
            item_temp['extracted final answer'] = extracted_final_answer
            item_temp['correctness'] = 'Incorrect'
            mispredict_eval_data_list_updated.append(item_temp)
           
    accuracy = len(eval_data_list_updated)/eval_num      
    cover_ratio = eval_num/len(output_list)

    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{math_task_name}_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
        json.dump(eval_data_list_updated, file, indent=4)
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{math_task_name}_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
        json.dump(mispredict_eval_data_list_updated, file, indent=4)
    return accuracy, cover_ratio

def calc_accuracy_API_BANK(API_BANK_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = '', prompt_style = ''):
    sys.path.append(f'{HOME_DIRECTORY}/DAMO_ConvAI')
    from api_bank.lv3_evaluator_new import eval_api_bank

    for i in range(len(API_BANK_test_data_list)):
        API_BANK_test_data_list[i]['pred'] = predict_list[i]
    task_name = task_name.upper()
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/api_bank_{prompt_style}_{task_name}_{file_name}_{train_method}_predictions.json", 'w') as file:
        json.dump(API_BANK_test_data_list, file, indent=4)

    accuracy, lv12_accuracy, lv3_accuracy = eval_api_bank(API_BANK_test_data_list, HOME_DIRECTORY) 
    cover_ratio = 1
    return accuracy, cover_ratio, lv12_accuracy, lv3_accuracy

def calc_accuracy_PLAN_BENCH(PLAN_BENCH_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = '', prompt_style = ''):
    sys.path.append(f'{HOME_DIRECTORY}/LLMs-Planning-main/plan-bench')
    from response_evaluation_modified import eval_plan_generation

    def extract_llm_raw_response(answer):
        # Split the answer by 'Final Answer', case-insensitive
        parts = re.split(r'(?i)final answer\s*:', answer)
        if len(parts) > 1:
            # Get the content after the last 'Final Answer'
            llm_raw_response = parts[-1].strip()
        else:
            # If 'Final Answer' not found, return the whole answer or handle accordingly
            llm_raw_response = answer.strip()
        return llm_raw_response

    # Load the initial JSON data
    output_data = {
        "task": PLAN_BENCH_test_data_list[0]["task"],
        "prompt_type": PLAN_BENCH_test_data_list[0]["prompt_type"],
        "domain": PLAN_BENCH_test_data_list[0]["domain"],
        "instances": []
    }

    # Process each instance in the initial data
    for i, instance in enumerate(PLAN_BENCH_test_data_list):
        answer = predict_list[i]
        answer = extract_llm_raw_response(answer)
        answer = answer + '\n'

        new_instance = {
            "instance_id": instance["instance_id"],
            "example_instance_ids": instance["example_instance_ids"],
            "query": instance["question"],
            "ground_truth_plan": instance["gold_label"],
            "llm_raw_response": answer
        }
        output_data["instances"].append(new_instance)

    task_name = task_name.upper()
    modified_path = f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/plan_bench_{file_name}_{train_method}_predictions.json"

    # modified_path = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/output/plan_benchmistral_gpt4__8_1_sep_16_1000_300_20_20_2e-05_10_-1/intermediate_results/plan_bench_mistral_gpt4__8_sep_16_False__1_1000_300_20_20_2e-05_10_-1_log_gpt4_train_predictions"

    with open(modified_path, 'w') as file:
        json.dump(output_data, file, indent=4)
    modified_path = modified_path.replace('.json', '')
    accuracy = eval_plan_generation(modified_path, HOME_DIRECTORY, task = 't1', config = 'blocksworld', engine = "gpt-3.5-turbo_chat")
    cover_ratio = 1
    return accuracy, cover_ratio

def calc_accuracy_ANLI(ANLI_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = ''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    ANLI_test_data_mispredict_list = []
    ANLI_test_data_correct_predict_list = []
    for i in range(len(ANLI_test_data_list)):
        ANLI_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = ANLI_test_data_list[i]['gold_label']
        if answer:
            final_answer = extract_nli_answer(answer)
            ANLI_test_data_item = ANLI_test_data_list[i]
            ANLI_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                ANLI_test_data_correct_predict_list.append(ANLI_test_data_item)
            else:
                ANLI_test_data_mispredict_list.append(ANLI_test_data_item)
    # with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/anli_{task_name}_{file_name}_{train_method}_predictions.json", 'w') as file:
    #     json.dump(ANLI_test_data_list, file, indent=4)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/anli_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
        json.dump(ANLI_test_data_correct_predict_list, file, indent=4)
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/anli_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
        json.dump(ANLI_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio


def calc_accuracy_MNLI(MNLI_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = ''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    MNLI_test_data_mispredict_list = []
    MNLI_test_data_correct_predict_list = []
    for i in range(len(MNLI_test_data_list)):
        MNLI_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = MNLI_test_data_list[i]['gold_label']
        if answer:
            final_answer = extract_nli_answer(answer)
            MNLI_test_data_item = MNLI_test_data_list[i]
            MNLI_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                MNLI_test_data_correct_predict_list.append(MNLI_test_data_item)
            else:
                MNLI_test_data_mispredict_list.append(MNLI_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/mnli_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
        json.dump(MNLI_test_data_correct_predict_list, file, indent=4)
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/mnli_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
        json.dump(MNLI_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio



def calc_accuracy_ESNLI(ESNLI_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = ''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    ESNLI_test_data_mispredict_list = []
    ESNLI_test_data_correct_predict_list = []
    for i in range(len(ESNLI_test_data_list)):
        ESNLI_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = ESNLI_test_data_list[i]['gold_label']
        if answer:
            final_answer = extract_nli_answer(answer)
            ESNLI_test_data_item = ESNLI_test_data_list[i]
            ESNLI_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                ESNLI_test_data_correct_predict_list.append(ESNLI_test_data_item)
            else:
                ESNLI_test_data_mispredict_list.append(ESNLI_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/esnli_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
        json.dump(ESNLI_test_data_correct_predict_list, file, indent=4)
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/esnli_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
        json.dump(ESNLI_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio



def calc_accuracy_SCITAIL(SCITAIL_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = ''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    SCITAIL_test_data_mispredict_list = []
    SCITAIL_test_data_correct_predict_list = []
    for i in range(len(SCITAIL_test_data_list)):
        SCITAIL_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = SCITAIL_test_data_list[i]['gold_label']
        if answer:
            final_answer = extract_nli_answer(answer)
            SCITAIL_test_data_item = SCITAIL_test_data_list[i]
            SCITAIL_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                SCITAIL_test_data_correct_predict_list.append(SCITAIL_test_data_item)
            else:
                SCITAIL_test_data_mispredict_list.append(SCITAIL_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/scitail_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
        json.dump(SCITAIL_test_data_correct_predict_list, file, indent=4)
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/scitail_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
        json.dump(SCITAIL_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio


def calc_accuracy_BOOLQ(BOOLQ_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = ''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    BOOLQ_test_data_mispredict_list = []
    BOOLQ_test_data_correct_predict_list = []
    for i in range(len(BOOLQ_test_data_list)):
        BOOLQ_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(BOOLQ_test_data_list[i]['gold_label'])
        if answer:
            final_answer = extract_bool(answer)
            BOOLQ_test_data_item = BOOLQ_test_data_list[i]
            BOOLQ_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                BOOLQ_test_data_correct_predict_list.append(BOOLQ_test_data_item)
            else:
                BOOLQ_test_data_mispredict_list.append(BOOLQ_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/boolq_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
        json.dump(BOOLQ_test_data_correct_predict_list, file, indent=4)
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/boolq_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
        json.dump(BOOLQ_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio

# def calc_accuracy_PLAN_BENCH(PLAN_BENCH_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = ''):
#     correct_count = 0
#     cover_ratio = 0
#     count = 0
#     PLAN_BENCH_test_data_mispredict_list = []
#     PLAN_BENCH_test_data_correct_predict_list = []
#     for i in range(len(PLAN_BENCH_test_data_list)):
#         PLAN_BENCH_test_data_list[i]['pred'] = predict_list[i]
#     for i, answer in enumerate(predict_list):
#         gold_label = PLAN_BENCH_test_data_list[i]['gold_label']
#         gold_label = gold_label.strip()
#         final_answer = extract_text_span(answer)

#         if final_answer != 'null':
#             count += 1
#         found_answer = False

#         if answer[0] == '"':  # remove the last period
#             answer = answer[1:]
#         if final_answer[0] == '"':  # remove the last period
#             final_answer = final_answer[1:]

#         if answer[-1] == '"':  # remove the last period
#             answer = answer[:-1]
#         if final_answer[-1] == '"':  # remove the last period
#             final_answer = final_answer[:-1]
        
#         if answer[-1] == '.':  # remove the last period
#             answer = answer[:-1]
#         if final_answer[-1] == '.':  # remove the last period
#             final_answer = final_answer[:-1]
#         if gold_label.lower() == final_answer.lower():
#             correct_count += 1
#             found_answer = True
#             break
#         else:
#             found_answer = False
        
#         plan_bench_test_data_item = PLAN_BENCH_test_data_list[i]
#         plan_bench_test_data_item['extracted_answer'] = final_answer
#         if found_answer:
#             PLAN_BENCH_test_data_correct_predict_list.append(plan_bench_test_data_item)
#         else:
#             PLAN_BENCH_test_data_mispredict_list.append(plan_bench_test_data_item)
            
#     accuracy = correct_count/len(predict_list)
#     cover_ratio = count/len(predict_list)

#     with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/plan_bench_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
#         json.dump(PLAN_BENCH_test_data_correct_predict_list, file, indent=4)
#     with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/plan_bench_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
#         json.dump(PLAN_BENCH_test_data_mispredict_list, file, indent=4)

#     return accuracy, cover_ratio


def calc_accuracy_SQUAD(SQUAD_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = ''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    SQUAD_test_data_mispredict_list = []
    SQUAD_test_data_correct_predict_list = []
    for i in range(len(SQUAD_test_data_list)):
        SQUAD_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        answer_list = SQUAD_test_data_list[i]['answer_list']
        final_answer = extract_text_span(answer)

        if final_answer != 'null':
            count += 1
        found_answer = False

        if answer[0] == '"':  # remove the last period
            answer = answer[1:]
        if final_answer[0] == '"':  # remove the last period
            final_answer = final_answer[1:]

        if answer[-1] == '"':  # remove the last period
            answer = answer[:-1]
        if final_answer[-1] == '"':  # remove the last period
            final_answer = final_answer[:-1]
        
        if answer[-1] == '.':  # remove the last period
            answer = answer[:-1]
        if final_answer[-1] == '.':  # remove the last period
            final_answer = final_answer[:-1]
        for answer in answer_list:
            if answer.lower() == final_answer.lower():
                correct_count += 1
                found_answer = True
                break
            else:
                found_answer = False
        
        squad_test_data_item = SQUAD_test_data_list[i]
        squad_test_data_item['extracted_answer'] = final_answer
        if found_answer:
            SQUAD_test_data_correct_predict_list.append(squad_test_data_item)
        else:
            SQUAD_test_data_mispredict_list.append(squad_test_data_item)
            
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/squad_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
        json.dump(SQUAD_test_data_correct_predict_list, file, indent=4)
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/squad_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
        json.dump(SQUAD_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio


def calc_accuracy_PIQA(PIQA_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = ''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    PIQA_test_data_mispredict_list = []
    PIQA_test_data_correct_predict_list = []
    for i in range(len(PIQA_test_data_list)):
        PIQA_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(PIQA_test_data_list[i]['gold_label'])

        

        sol12_content = PIQA_test_data_list[i]['sol'+gold_label]
        gold_label = gold_label.lower()
        option1 = PIQA_test_data_list[i]['sol1']
        option2 = PIQA_test_data_list[i]['sol2']

        sol12_content = sol12_content.strip().lower().rstrip('.')
        option1 = option1.strip().lower().rstrip('.')
        option2 = option2.strip().lower().rstrip('.')


        if answer:
            final_answer = extract_option(answer)
            PIQA_test_data_item = PIQA_test_data_list[i]
            PIQA_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                PIQA_test_data_correct_predict_list.append(PIQA_test_data_item)
            else:
                final_answer = extract_context_after_answer(answer)
                if final_answer == option1.lower() or final_answer == option2.lower():
                    count += 1
                    if final_answer == option1.lower():
                        PIQA_test_data_item['extracted_answer'] = '1'
                    if final_answer == option2.lower():
                        PIQA_test_data_item['extracted_answer'] = '2'
                    if final_answer == sol12_content:
                        correct_count += 1
                        PIQA_test_data_correct_predict_list.append(PIQA_test_data_item)
                    else:
                        PIQA_test_data_mispredict_list.append(PIQA_test_data_item)
                else:
                    PIQA_test_data_mispredict_list.append(PIQA_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/piqa_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
        json.dump(PIQA_test_data_correct_predict_list, file, indent=4)
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/piqa_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
        json.dump(PIQA_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio





def calc_accuracy_AQuaRAT(test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = '', full_task_name = ''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    test_data_mispredict_list = []
    test_data_correct_predict_list = []
    for i in range(len(test_data_list)):
        test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(test_data_list[i]['gold_label'])
        # abcd_content = test_data_list[i][gold_label]
        # gold_label = gold_label.lower()
        # a_content = test_data_list[i]['A']
        # b_content = test_data_list[i]['B']
        # c_content = test_data_list[i]['C']
        # d_content = test_data_list[i]['D']

        # abcd_content = abcd_content.strip().lower().rstrip('.')
        # a_content = a_content.strip().lower().rstrip('.')
        # b_content = b_content.strip().lower().rstrip('.')
        # c_content = c_content.strip().lower().rstrip('.')
        # d_content = d_content.strip().lower().rstrip('.')

        if answer:
            final_answer = extract_answer_aquarat(answer)
            test_data_item = test_data_list[i]
            test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                test_data_correct_predict_list.append(test_data_item)
            else:
                # final_answer = extract_context_after_answer(answer)
                # if final_answer == a_content.lower() or final_answer == b_content.lower() or final_answer == c_content.lower() or final_answer == d_content.lower():
                #     count += 1
                #     if final_answer == a_content.lower():
                #         test_data_item['extracted_answer'] = 'A'
                #     if final_answer == b_content.lower():
                #         test_data_item['extracted_answer'] = 'B'
                #     if final_answer == c_content.lower():
                #         test_data_item['extracted_answer'] = 'C'
                #     if final_answer == d_content.lower():
                #         test_data_item['extracted_answer'] = 'D'
                #     if final_answer == abcd_content:
                #         correct_count += 1
                #         test_data_correct_predict_list.append(test_data_item)
                #     else:
                #         test_data_mispredict_list.append(test_data_item)
                # else:
                test_data_mispredict_list.append(test_data_item)
                
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/AQuaRAT_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
        json.dump(test_data_correct_predict_list, file, indent=4)
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/AQuaRAT_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
        json.dump(test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio


def calc_accuracy_WINOGRANDE(WINOGRANDE_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = ''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    WINOGRANDE_test_data_mispredict_list = []
    WINOGRANDE_test_data_correct_predict_list = []
    for i in range(len(WINOGRANDE_test_data_list)):
        WINOGRANDE_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(WINOGRANDE_test_data_list[i]['gold_label'])
        op12_content = WINOGRANDE_test_data_list[i]['option'+gold_label]
        gold_label = gold_label.lower()
        option1 = WINOGRANDE_test_data_list[i]['option1']
        option2 = WINOGRANDE_test_data_list[i]['option2']

        op12_content = op12_content.strip().lower().rstrip('.')
        option1 = option1.strip().lower().rstrip('.')
        option2 = option2.strip().lower().rstrip('.')

        if answer:
            final_answer = extract_option(answer)
            WINOGRANDE_test_data_item = WINOGRANDE_test_data_list[i]
            WINOGRANDE_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                WINOGRANDE_test_data_correct_predict_list.append(WINOGRANDE_test_data_item)
            else:
                final_answer = extract_context_after_answer(answer)
                if final_answer == option1.lower() or final_answer == option2.lower():
                    count += 1
                    if final_answer == option1.lower():
                        WINOGRANDE_test_data_item['extracted_answer'] = '1'
                    if final_answer == option2.lower():
                        WINOGRANDE_test_data_item['extracted_answer'] = '2'
                    if final_answer == op12_content:
                        correct_count += 1
                        WINOGRANDE_test_data_correct_predict_list.append(WINOGRANDE_test_data_item)
                    else:
                        WINOGRANDE_test_data_mispredict_list.append(WINOGRANDE_test_data_item)
                else:
                    WINOGRANDE_test_data_mispredict_list.append(WINOGRANDE_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/winogrande_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
        json.dump(WINOGRANDE_test_data_correct_predict_list, file, indent=4)
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/winogrande_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
        json.dump(WINOGRANDE_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio



def calc_accuracy_ECQA(ECQA_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = ''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    ECQA_test_data_mispredict_list = []
    ECQA_test_data_correct_predict_list = []
    for i in range(len(ECQA_test_data_list)):
        ECQA_test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(ECQA_test_data_list[i]['gold_label'])
        gold_label_content = ECQA_test_data_list[i][gold_label]
        gold_label_content = gold_label_content.lower()
        option1 = ECQA_test_data_list[i]['1']
        option2 = ECQA_test_data_list[i]['2']
        option3 = ECQA_test_data_list[i]['3']
        option4 = ECQA_test_data_list[i]['4']
        option5 = ECQA_test_data_list[i]['5']

        gold_label_content = gold_label_content.strip().lower().rstrip('.')
        option1 = option1.strip().lower().rstrip('.')
        option2 = option2.strip().lower().rstrip('.')
        option3 = option3.strip().lower().rstrip('.')
        option4 = option4.strip().lower().rstrip('.')
        option5 = option5.strip().lower().rstrip('.')

        if answer:
            final_answer = extract_option_1_to_5(answer)
            ECQA_test_data_item = ECQA_test_data_list[i]
            ECQA_test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label.lower() == final_answer.lower():
                correct_count += 1
                ECQA_test_data_correct_predict_list.append(ECQA_test_data_item)
            else:
                final_answer = extract_context_after_answer(answer)
                if final_answer == option1.lower() or final_answer == option2.lower() or final_answer == option3.lower() or final_answer == option4.lower() or final_answer == option5.lower():
                    count += 1
                    if final_answer == option1.lower():
                        ECQA_test_data_item['extracted_answer'] = '1'
                    if final_answer == option2.lower():
                        ECQA_test_data_item['extracted_answer'] = '2'
                    if final_answer == option1.lower():
                        ECQA_test_data_item['extracted_answer'] = '3'
                    if final_answer == option2.lower():
                        ECQA_test_data_item['extracted_answer'] = '4'
                    if final_answer == option1.lower():
                        ECQA_test_data_item['extracted_answer'] = '5'
                    if final_answer == gold_label_content:
                        correct_count += 1
                        ECQA_test_data_correct_predict_list.append(ECQA_test_data_item)
                    else:
                        ECQA_test_data_mispredict_list.append(ECQA_test_data_item)
                else:
                    ECQA_test_data_mispredict_list.append(ECQA_test_data_item)
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/ecqa_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
        json.dump(ECQA_test_data_correct_predict_list, file, indent=4)
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/ecqa_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
        json.dump(ECQA_test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio

def calc_accuracy_MMLU_AGI(test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = '', full_task_name = ''):
    correct_count = 0
    cover_ratio = 0
    count = 0
    test_data_mispredict_list = []
    test_data_correct_predict_list = []
    for i in range(len(test_data_list)):
        test_data_list[i]['pred'] = predict_list[i]
    for i, answer in enumerate(predict_list):
        gold_label = str(test_data_list[i]['gold_label'])
        abcd_content = test_data_list[i][gold_label]
        gold_label = gold_label.lower()
        a_content = test_data_list[i]['A']
        b_content = test_data_list[i]['B']
        c_content = test_data_list[i]['C']
        d_content = test_data_list[i]['D']

        abcd_content = abcd_content.strip().lower().rstrip('.')
        a_content = a_content.strip().lower().rstrip('.')
        b_content = b_content.strip().lower().rstrip('.')
        c_content = c_content.strip().lower().rstrip('.')
        d_content = d_content.strip().lower().rstrip('.')

        if answer:
            final_answer = extract_option_mmlu_agi(answer)
            test_data_item = test_data_list[i]
            test_data_item['extracted_answer'] = final_answer
            if final_answer != 'null':
                count += 1
            if gold_label == final_answer.lower():
                correct_count += 1
                test_data_correct_predict_list.append(test_data_item)
            else:
                final_answer = extract_context_after_answer(answer)
                if final_answer == a_content.lower() or final_answer == b_content.lower() or final_answer == c_content.lower() or final_answer == d_content.lower():
                    count += 1
                    if final_answer == a_content.lower():
                        test_data_item['extracted_answer'] = 'A'
                    if final_answer == b_content.lower():
                        test_data_item['extracted_answer'] = 'B'
                    if final_answer == c_content.lower():
                        test_data_item['extracted_answer'] = 'C'
                    if final_answer == d_content.lower():
                        test_data_item['extracted_answer'] = 'D'
                    if final_answer == abcd_content:
                        correct_count += 1
                        test_data_correct_predict_list.append(test_data_item)
                    else:
                        test_data_mispredict_list.append(test_data_item)
                else:
                    test_data_mispredict_list.append(test_data_item)
                
    accuracy = correct_count/len(predict_list)
    cover_ratio = count/len(predict_list)

    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{full_task_name}_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
        json.dump(test_data_correct_predict_list, file, indent=4)
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/{full_task_name}_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
        json.dump(test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio

def calc_accuracy_CODE(CODE_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = '', checkpoint_num = 0):
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'  # Sets timeout to 2 seconds
    cover_ratio = 0

    prediction_list = []
    for i, item in enumerate(CODE_test_data_list):
        temp = {}
        completion = predict_list[i]
        completion = completion.replace("```python", '\n')
        completion = completion.replace("```", '\n')

        if re.search('ANSWER:', completion, re.IGNORECASE):
            
            # Regular expression pattern to match everything after "ANSWER:" with ignored case
            pattern = r"ANSWER:(.*)"

            # Using regex to find the matched part with re.IGNORECASE flag for case-insensitive matching
            # re.DOTALL flag allows '.' to match newline characters as well
            match = re.search(pattern, completion, re.DOTALL | re.IGNORECASE)


            # Extract the matched portion
            if match:
                completion = match.group(1) 
            else:
                completion = completion

        temp['question'] = item['question']
        temp['prompt'] = item['prompt']
        temp['task_id'] = f'test/{i}'
        temp['input'] = ''
        temp['test'] = item['test']
        temp['entry_point'] = item['entry_point']
        temp['canonical_solution'] = item['ground_truth']
        temp['completion'] = completion
        temp['original_prediction'] = predict_list[i]
        prediction_list.append(temp)

    # Path for saving the file
    task_name = task_name[:3]
    sample_path = f'{HOME_DIRECTORY}/code_eval/intermediate_file/HumanEval_{task_name}_{file_name}_{train_method}_{checkpoint_num}.jsonl' 
    problem_path = f'{HOME_DIRECTORY}/code_eval/intermediate_file/HumanEval_{task_name}_{file_name}_{train_method}_{checkpoint_num}.jsonl'
   
    with open(problem_path, 'w') as file:
        for item in prediction_list: #problem_list:
            json_line = json.dumps(item)
            file.write(json_line + '\n')

    # Writing the code snippet to a JSONL file
    with open(sample_path, 'w') as file:
        for item in prediction_list:
            json_line = json.dumps(item)
            file.write(json_line + '\n')
    
    cmd = [
        "evaluate_functional_correctness",
        f"{sample_path}",
        "--problem_file", f"{problem_path}"
    ]

    subprocess.run(" ".join(cmd), shell=True, cwd=f'{HOME_DIRECTORY}/code_eval')
    # try:
    log_file_path = sample_path.replace('jsonl', 'jsonl_results.jsonl')
    with open(f'{log_file_path}', 'r') as file:
        # Read the contents of the file
        content = file.read()
    # except:
    #     log_file_path = sample_path.replace('jsonl_results.jsonl', 'txt')
    #     with open(f'{log_file_path}', 'r') as file:
    #         # Read the contents of the file
    #         content = file.read()
    aaa = content.count('"passed": true')
    bbb = content.count('"passed": false')
    ccc = float(aaa/(aaa+bbb))
    accuracy = ccc
    # accuracy = entry_point(sample_file = file_path, problem_file = gzip_path)
    cover_ratio = int(1)
    return accuracy, cover_ratio


def calc_accuracy_APPS(CODE_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = ''):
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'  # Sets timeout to 2 seconds
    cover_ratio = 0

    data = {}
    CODE_test_data_path = []
    for i, item in enumerate(CODE_test_data_list):
        completion = predict_list[i]
        completion = completion.replace("```", '')

        if 'ANSWER:' in completion:
            
            # Regular expression pattern to match everything after "FINAL ANSWER:"
            pattern = r"ANSWER:(.*)"

            # Using regex to find the matched part
            # re.DOTALL flag allows '.' to match newline characters as well
            match = re.search(pattern, completion, re.DOTALL)

            # Extract the matched portion
            if match:
                completion = match.group(1) 
            else:
                completion = completion
            completion = completion.lstrip()
        item_id = item['id']
        data[f'{i}'] = completion
        formatted_id = f"{item_id:04d}"
        CODE_test_data_path.append(f"{HOME_DIRECTORY}/APPS_data/train/APPS/test/{formatted_id}")

    accuracy = run_main(CODE_test_data_path, data)
    cover_ratio = int(1)
    return accuracy, cover_ratio


def run_dynamic_test(test_case, completion):
    # Create isolated namespaces for this execution
    local_namespace = {}
    global_namespace = {}  # Optional: Use if you want to completely isolate from real globals
    
    # Assume globals are needed (like built-in functions, etc.)
    global_namespace.update(globals())  # Comment this if you don't want to include real globals

    PASS = False

    def execute_assertion(test_case_code, local_namespace):
        for test_case_item in test_case_code:
            try:
                exec(test_case_item, global_namespace, local_namespace)
            except AssertionError:
                return False
            except Exception as e:
                return False
        return True

    try:
        exec(completion, global_namespace, local_namespace)
        PASS = execute_assertion(test_case, local_namespace)
    except AssertionError:
        PASS = False
    except Exception as e:
        PASS = False

    return PASS




def run_dynamic_test_with_timeout(test_case, completion, timeout_seconds=5):
    def run_test(completion, test_case, namespace):
        local_namespace = {}
        global_namespace = globals()  # Use real globals or a custom isolated namespace
        
        try:
            exec(completion, global_namespace, local_namespace)
            PASS = True
            for test_case_item in test_case:
                try:
                    exec(test_case_item, global_namespace, local_namespace)
                except AssertionError:
                    PASS = False
                    break  # Stop at the first failed assertion
                except Exception as e:
                    PASS = False
                    break  # Stop at the first exception
            namespace['PASS'] = PASS
        except Exception as e:
            namespace['PASS'] = False
    
    # Use Manager from multiprocessing to create a shared namespace
    with Manager() as manager:
        namespace = manager.dict()
        # Define and start a new process for running the test with the provided code and test cases
        process = Process(target=run_test, args=(completion, test_case, namespace))
        process.start()
        
        # Wait for the process to complete or for the timeout
        process.join(timeout_seconds)
        
        # If the process is still alive after the timeout, it means it's likely stuck in an infinite loop
        if process.is_alive():
            process.terminate()  # Terminate the stuck process
            process.join()  # Ensure process resources are cleaned up
            return False  # Return False to indicate the test did not pass (due to timeout)
        
        # Fetch and return the result from the shared namespace
        return namespace.get('PASS', False)



def calc_accuracy_MBPP(MBPP_test_data_list, predict_list, output_folder_name, file_name, task_name='', sampling_num = 1, train_method = ''):
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'  # Sets timeout to 2 seconds
    cover_ratio = 0

    test_data_mispredict_list = []
    test_data_correct_predict_list = []
    correct_count = 0
    for i, item in enumerate(MBPP_test_data_list):
        completion = predict_list[i]
        # completion = \
        # """Final Answer:\ndef remove_Occ(s,ch): \r\n    for i in range(len(s)): \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    for i in range(len(s) - 1,-1,-1):  \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    return s """
        completion = completion.replace("```python", '')
        completion = completion.replace("```", '')
        original_completion = predict_list[i]

        # Modified pattern to be case-insensitive
        pattern = r"ANSWER:(.*)"

        # Using re.IGNORECASE to make the search case-insensitive
        match = re.search(pattern, completion, re.DOTALL | re.IGNORECASE)

        # Extract the matched portion
        if match:
            completion = match.group(1).lstrip()
        else:
            completion = completion.lstrip()
        passed = True
        # for test_item in item['test_list']:

        test_list = item['test_list']
        challenge_test_list = item['challenge_test_list']
        if challenge_test_list != []:
            test_list += challenge_test_list
        passed = run_dynamic_test_with_timeout(test_list, completion)
        if not passed:
            passed = False
            # break
        if passed:
            correct_count += 1
            item['prediction'] = original_completion
            test_data_correct_predict_list.append(item)
        else:
            item['prediction'] = original_completion
            test_data_mispredict_list.append(item)


    cover_ratio = int(1)
    accuracy = correct_count/len(MBPP_test_data_list)

    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/mbpp_{task_name}_{file_name}_{train_method}_correct_predictions.json", 'w') as file:
        json.dump(test_data_correct_predict_list, file, indent=4)
    with open(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results/mbpp_{task_name}_{file_name}_{train_method}_mispredictions.json", 'w') as file:
        json.dump(test_data_mispredict_list, file, indent=4)

    return accuracy, cover_ratio

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


def extract_option(text):
    text = text.lower()
    found_answer_num = text.count('answer:')
    if found_answer_num > 0:
        text = text.split("answer:")[-1].strip()
    else:
        found_answer_num = text.count('answer')
        if found_answer_num > 0:
            text = text.split("answer")[-1].strip()
    if "1" in text or "2" in text:
        index_1 = text.find("1")
        index_2 = text.find("2")
        if index_1 != -1 and (index_2 == -1 or index_1 < index_2):
            return "1"
        elif index_2 != -1:
            return "2"
    else:
        return "null"
    

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

# Example usage:
text = "This is a sample text with 3 and 2 and 5."
print(find_first_number(text))


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
    
    
def extract_option_mmlu_agi(text):
    text = text.lower()
    abcd = extract_answer_mmlu_agi(text)
    if abcd != "null":
        return abcd
    else:
        return "null"




# def extract_answer_mmlu_agi(text):
#     # Define a regex pattern to match the standard answer formats
#     standard_pattern = r'Answer:\s*\(?(A|B|C|D)\)?(?:,|\s|$)'
    
#     # Define a simpler pattern for standalone letters
#     simple_pattern = r'\b(A|B|C|D)\b'
    
#     # First, try to match the standard pattern
#     match = re.search(standard_pattern, text, re.IGNORECASE)
    
#     # If no match is found, try the simpler pattern
#     if not match:
#         match = re.search(simple_pattern, text, re.IGNORECASE)
    
#     # If a match is found in either case, return the matched group in uppercase
#     if match:
#         return match.group(1).upper()
#     else:
#         return "null"
    

def extract_answer_mmlu_agi(text):
    pattern = r'^\s*(?:\(([A-Da-d])\)|([A-Da-d])\.?)\s*$'
    
    # Search for a match using the defined pattern
    match = re.search(pattern, text)

    # If a match is found, return it in uppercase
    if match:
        return (match.group(1) or match.group(2)).upper()
    
    pattern_direct = r'(?:answer:|the\sanswer\sis)\s*\b([A-D])\b'
    match_direct = re.search(pattern_direct, text, re.IGNORECASE)
    
    if match_direct:
        return match_direct.group(1).upper()

    # If no direct mention is found, look for the first occurrence of A, B, C, or D after 'answer'
    pattern_fallback = r'\banswer\b[^A-D]*\b([A-D])\b'
    match_fallback = re.search(pattern_fallback, text, re.IGNORECASE)
    
    if match_fallback:
        return match_fallback.group(1).upper()

    # If no match is found by either pattern, return "null"
    return "null"

def extract_answer_aquarat(text):
    pattern = r'^\s*(?:\(([A-Ea-e])\)|([A-Ea-e])\.?)\s*$'
    
    # Search for a match using the defined pattern
    match = re.search(pattern, text)

    # If a match is found, return it in uppercase
    if match:
        return (match.group(1) or match.group(2)).upper()
    
    pattern_direct = r'(?:answer:|the\sanswer\sis)\s*\b([A-E])\b'
    match_direct = re.search(pattern_direct, text, re.IGNORECASE)
    
    if match_direct:
        return match_direct.group(1).upper()

    # If no direct mention is found, look for the first occurrence of A, B, C, or D after 'answer'
    pattern_fallback = r'\banswer\b[^A-E]*\b([A-E])\b'
    match_fallback = re.search(pattern_fallback, text, re.IGNORECASE)
    
    if match_fallback:
        return match_fallback.group(1).upper()

    # If no match is found by either pattern, return "null"
    return "null"


# a = '\n\nLet the length of the original piece of cloth be L.\nAfter cutting it lengthwise, we have two smaller rectangular pieces with lengths L1 and L2.\n\nWe know that the shorter piece is one-third of the length of the longer of the 2 new pieces.\nSo, we have the equation L1 = (1/3) * L2.\n\nWe also know that the area of the shorter piece is 12 square feet.\nSo, we have the equation (L1 * W) = 12, where W is the width of the shorter piece.\n\nSince the width of the shorter piece is 2 feet, we can substitute W = 2 into the equation.\n\nNow we have two equations:\nL1 = (1/3) * L2\n(L1 * 2) = 12\n\nSimplifying the second equation, we get:\nL1 = 6\n\nSubstituting this value into the first equation, we get:\n6 = (1/3) * L2\n\nMultiplying both sides by 3, we get:\n18 = L2\n\nSo, the length of the original piece of cloth before cutting is L = L1 + L2 = 6 + 18 = 24 feet.\n\nThe answer: (C)'

# kkk = extract_answer_aquarat(a)
# a = 1


def extract_context_after_answer(text):
    
    # Convert the text to lowercase to ignore case
    text_lower = text.lower()

    # Regular expression to find all occurrences of 'answer:' followed by any content
    matches = re.findall(r'answer:\s*(.*)', text_lower)

    # Extract the last occurrence, if any
    if matches:
        last_answer = matches[-1]
        return_null_list = ['Neither Option 1', 'correct answer is not provided', 'There is no correct answer', 'The given options are incorrect.']
        for return_null in return_null_list:
            return_null = return_null.lower()
            if return_null in last_answer.lower():
                return 'null'
        # Remove a period at the end if present
        if last_answer.endswith('.'):
            last_answer = last_answer[:-1]
        last_answer = last_answer.lower().strip()
        return last_answer
    else:    
        return "null"

def find_smallest_index(index_1, index_2, index_3, index_4):
    # Assuming index_1, index_2, index_3, and index_4 are defined
    indices = [index_1, index_2, index_3, index_4]

    # Filter out any indices that are -1, indicating the string wasn't found
    filtered_indices = [index for index in indices if index != -1]

    # Find the smallest index if there are any indices left after filtering
    if filtered_indices:
        smallest_index = min(filtered_indices)
    else:
        smallest_index = None  # or -1, depending on how you want to handle no matches

    # smallest_index now holds the smallest index that isn't -1, or None if all were -1
    return smallest_index
    
def extract_bool(text):
    text = text.lower()
    found_answer_num = text.count('answer:')
    if found_answer_num > 0:
        text = text.split("answer:")[-1].strip()
    else:
        found_answer_num = text.count('answer')
        if found_answer_num > 0:
            text = text.split("answer")[-1].strip()
    if "false" in text or "true" in text or "yes" in text or "no" in text:
        index_1 = text.find("true")
        index_2 = text.find("false")
        index_3 = text.find("yes")
        index_4 = text.find("no")

        smallest_index = find_smallest_index(index_1, index_2, index_3, index_4)
        if not smallest_index and smallest_index != 0:
            return 'null'
        if smallest_index == index_1 or smallest_index == index_3:
            return "true"
        else:
            return "false"
    else:
        return "null"
    
def extract_text_span(text):
    text = text.lower()
    found_answer_num = text.count('answer:')
    if found_answer_num > 0:
        text = text.split("answer:")[-1].strip()
    else:
        found_answer_num = text.count('answer')
        if found_answer_num > 0:
            text = text.split("answer")[-1].strip()
    if found_answer_num != 0:
        return text
    else:
        return 'null'

def Evaluation(task_name, test_data, test_list, test_config, output_folder_name, lora_path = '', model = None):
    output_list = prediction(test_data, test_list, test_config, lora_path = lora_path, model = model)
    
    torch.cuda.empty_cache()
    gc.collect()
    question_list = []
    groundtruth_list = []
    for i in range(len(test_list)):
        question_list.append(test_list[i]['question'])
        groundtruth_list.append(test_list[i]['answer'])
    if task_name.lower() == 'gsm8k':
        accuracy, cover_ratio = calc_accuracy_GSM8K(question_list, output_list, groundtruth_list, output_folder_name, 'finetune')

    if task_name.lower() == 'api_bank':
        accuracy, cover_ratio = calc_accuracy_API_BANK(question_list, output_list, groundtruth_list, output_folder_name, 'finetune')
    return accuracy, cover_ratio
    

# def EVALUATION_ALPACA(GSM8K_test_data_list, API_BANK_test_data_list, test_task_name_list, train_config, test_config, output_folder_name, file_name, intermediate_test_file_name_suffix, check_point_folder_name = '', data_loader_config = {}, task_name = ''):
#     torch.cuda.empty_cache()
#     gc.collect()
#     test_config['seed_num'] = train_config['seed_num']

#     for test_task_name in test_task_name_list:
#         # test_file_name = f'{test_task_name.upper()}{intermediate_test_file_name_suffix}'
#         # test_data_path = f"{HOME_DIRECTORY}/alpaca_data/{test_file_name}.json"
#         if test_task_name.lower() == 'gsm8k':
#             test_data_list = GSM8K_test_data_list
#         if test_task_name.lower() == 'api_bank':
#             test_data_list = API_BANK_test_data_list

#         predict_list = do_predict_llama_alpaca(test_data_list, output_folder_name, test_task_name, test_config, check_point_folder_name = check_point_folder_name, data_loader_config = data_loader_config)

#         if test_task_name.lower() == 'gsm8k':
#             test_data_list = GSM8K_test_data_list
#         if test_task_name.lower() == 'api_bank':
#             test_data_list = API_BANK_test_data_list

#         question_list = []
#         groundtruth_list = []
#         for i in range(len(test_data_list)):
#             if test_task_name.lower() == 'api_bank':
#                 question = test_data_list[i]['question'] + ' ' + test_data_list[i]['input']
#             else: 
#                 question = test_data_list[i]['question']
#             question_list.append(question)
#             groundtruth_list.append(test_data_list[i]['answer'])

#         record_json_data_to_file(output_folder_name, test_data_list, test_config, 'evaluation_' + test_task_name)
    
#         if test_task_name.lower() == 'gsm8k':
#             accuracy, cover_ratio = calc_accuracy_GSM8K(question_list, predict_list, groundtruth_list, output_folder_name, f'{file_name}_evaluate_on_gsm8k', task_name = task_name)
#         if test_task_name.lower() == 'api_bank':
#             accuracy, cover_ratio = calc_accuracy_API_BANK(API_BANK_test_data_list, predict_list, output_folder_name, f'{file_name}_evaluate_on_api_bank', task_name = task_name)
#         log_line = 'Evaluation for ' + test_task_name
#         write_log(file_name, output_folder_name, log_line)
#         log_line = 'Evaluation Accuracy: ' + str(accuracy) + ' Cover Ratio: ' + str(cover_ratio) + ' for ' + test_task_name
#         write_log(file_name, output_folder_name, log_line)


def EVALUATION_ALPACA(test_data_list, test_task_name, test_config, output_folder_name, file_name, check_point_folder_name = '', data_loader_config = {}, task_name = '', sampling_num = 1):
    predict_list = do_predict_llama_alpaca(test_data_list, output_folder_name, test_config, check_point_folder_name = check_point_folder_name, data_loader_config = data_loader_config, sampling_num = sampling_num)

    question_list = []
    groundtruth_list = []
    for i in range(len(test_data_list)):
        question = test_data_list[i]['question']
        question_list.append(question)
        groundtruth_list.append(test_data_list[i]['answer'])

    if test_task_name.lower() == 'gsm8k':
        accuracy, cover_ratio = calc_accuracy_GSM8K(question_list, predict_list, groundtruth_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num)
    if test_task_name.lower() == 'api_bank':
        accuracy, cover_ratio = calc_accuracy_API_BANK(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num)
    if test_task_name.lower() == 'anli':
        accuracy, cover_ratio = calc_accuracy_ANLI(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num)
    if test_task_name.lower() == 'code':
        accuracy, cover_ratio = calc_accuracy_CODE(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num)
    
    log_line = f'{task_name} Evaluation for ' + test_task_name
    write_log(file_name, output_folder_name, log_line)
    log_line = f'{task_name} Accuracy: ' + str(accuracy) + ', Cover Ratio: ' + str(cover_ratio) + ' for ' + test_task_name
    write_log(file_name, output_folder_name, log_line)
    
    return accuracy, cover_ratio


def EVALUATION_LLAMA_FACTORY(test_data_list, test_task_name, test_config, output_folder_name, file_name, check_point_folder_name = '', data_loader_config = {}, task_name = '', sampling_num = 1, train_method = '', merged_base_model_dir = '', checkpoint_num = 0, data_name = '', LLAMA_FACTORY_DIRECTORY = '', SFT_Best_lora_dir = '', prompt_style = ''):
    if SFT_Best_lora_dir != '':
        predict_list = do_predict_llama_factory_unify(test_data_list, output_folder_name, test_config, file_name, check_point_folder_name = check_point_folder_name, data_loader_config = data_loader_config, sampling_num = sampling_num, data_name = data_name, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY, SFT_for_dpo_lora_dir = SFT_Best_lora_dir)
    else:
        predict_list = do_predict_llama_factory_unify(test_data_list, output_folder_name, test_config, file_name, check_point_folder_name = check_point_folder_name, data_loader_config = data_loader_config, sampling_num = sampling_num, merged_base_model_dir = merged_base_model_dir, data_name = data_name, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY)

    # predict_list = ["""Sure! I'd be happy to help you with that.\n\nSo, based on the test example provided, it seems that we need to find the shared elements between two lists. The function we need to write is called `similar_elements`, and it should return a set of the shared elements.\n\nHere's my inference:\n\nThe function `similar_elements` should take two lists as input, and it should return a set of the elements that are present in both lists. To do this, we can use the `set` data structure, which can uniquely identify the elements in the lists.\n\nFinal Answer:\n\ndef similar_elements(list1, list2):\n    return set(list1).intersection(list2) Final Answer: 10th and 11th centuries""", """Sure! I'd be happy to help you with that.\n\nSo, based on the test example provided, it seems that we need to find the shared elements between two lists. The function we need to write is called `similar_elements`, and it should return a set of the shared elements.\n\nHere's my inference:\n\nThe function `similar_elements` should take two lists as input, and it should return a set of the elements that are present in both lists. To do this, we can use the `set` data structure, which can uniquely identify the elements in the lists.\n\nFinal Answer:\n\ndef similar_elements(list1, list2):\n    return set(list1).intersection(list2)"""]

    # for i, k in enumerate(predict_list):
    #     print(f'-------------------------{i}-------------------------')
    #     print()
    #     print(k)
    #     print()
    question_list = []
    groundtruth_list = []
    for i in range(len(test_data_list)):
        question = test_data_list[i]['question']
        question_list.append(question)
        if test_task_name.lower() == 'piqa' or test_task_name.lower() == 'boolq' or test_task_name.lower() == 'winogrande' or test_task_name.lower() == 'ecqa' or test_task_name.lower() == 'squad' or test_task_name.lower() == 'aquarat' or test_task_name.lower() == 'plan_bench':
            groundtruth_list.append(test_data_list[i]['gold_label'])
        elif 'math' in test_task_name.lower() or 'gsm8k' in test_task_name.lower():
            groundtruth_list.append(test_data_list[i]['numerical_final_answer'])
        else:
            groundtruth_list.append(test_data_list[i]['answer'])
    

    if 'gsm8k' in test_task_name.lower():
        accuracy, cover_ratio = calc_accuracy_GSM8K(question_list, predict_list, groundtruth_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method)
    if 'math_algebra' in test_task_name.lower():
        accuracy, cover_ratio = calc_accuracy_MATH(question_list, predict_list, groundtruth_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method, math_task_name = 'math_algebra')
    if 'math_geometry' in test_task_name.lower():
        accuracy, cover_ratio = calc_accuracy_MATH(question_list, predict_list, groundtruth_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method, math_task_name = 'math_geometry')
    if test_task_name.lower() == 'aquarat':
        # predict_list = ["a = '\n\nLet the length of the original piece of cloth be L.\nAfter cutting it lengthwise, we have two smaller rectangular pieces with lengths L1 and L2.\n\nWe know that the shorter piece is one-third of the length of the longer of the 2 new pieces.\nSo, we have the equation L1 = (1/3) * L2.\n\nWe also know that the area of the shorter piece is 12 square feet.\nSo, we have the equation (L1 * W) = 12, where W is the width of the shorter piece.\n\nSince the width of the shorter piece is 2 feet, we can substitute W = 2 into the equation.\n\nNow we have two equations:\nL1 = (1/3) * L2\n(L1 * 2) = 12\n\nSimplifying the second equation, we get:\nL1 = 6\n\nSubstituting this value into the first equation, we get:\n6 = (1/3) * L2\n\nMultiplying both sides by 3, we get:\n18 = L2\n\nSo, the length of the original piece of cloth before cutting is L = L1 + L2 = 6 + 18 = 24 feet.\n\nThe answer: (C)", "a = '\n\nLet the length of the original piece of cloth be L.\nAfter cutting it lengthwise, we have two smaller rectangular pieces with lengths L1 and L2.\n\nWe know that the shorter piece is one-third of the length of the longer of the 2 new pieces.\nSo, we have the equation L1 = (1/3) * L2.\n\nWe also know that the area of the shorter piece is 12 square feet.\nSo, we have the equation (L1 * W) = 12, where W is the width of the shorter piece.\n\nSince the width of the shorter piece is 2 feet, we can substitute W = 2 into the equation.\n\nNow we have two equations:\nL1 = (1/3) * L2\n(L1 * 2) = 12\n\nSimplifying the second equation, we get:\nL1 = 6\n\nSubstituting this value into the first equation, we get:\n6 = (1/3) * L2\n\nMultiplying both sides by 3, we get:\n18 = L2\n\nSo, the length of the original piece of cloth before cutting is L = L1 + L2 = 6 + 18 = 24 feet.\n\nThe answer: (C)"]
        accuracy, cover_ratio = calc_accuracy_AQuaRAT(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method, full_task_name = 'aquarat')

    if 'math_count' in test_task_name.lower():
        accuracy, cover_ratio = calc_accuracy_MATH(question_list, predict_list, groundtruth_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method, math_task_name = 'math_counting_and_probability')
    if test_task_name.lower() == 'api_bank':
        accuracy, cover_ratio, lv12_accuracy, lv3_accuracy = calc_accuracy_API_BANK(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method, prompt_style = prompt_style)
    if test_task_name.lower() == 'anli':
        accuracy, cover_ratio = calc_accuracy_ANLI(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method)
    if test_task_name.lower() == 'mnli':
        accuracy, cover_ratio = calc_accuracy_MNLI(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method)
    if test_task_name.lower() == 'esnli':
        accuracy, cover_ratio = calc_accuracy_ESNLI(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method)
    if test_task_name.lower() == 'scitail':
        accuracy, cover_ratio = calc_accuracy_SCITAIL(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method)
    if test_task_name.lower() == 'piqa':
        accuracy, cover_ratio = calc_accuracy_PIQA(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method)
    if test_task_name.lower() == 'boolq':
        accuracy, cover_ratio = calc_accuracy_BOOLQ(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method)
    if test_task_name.lower() == 'squad':
        accuracy, cover_ratio = calc_accuracy_SQUAD(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method)
    if test_task_name.lower() == 'winogrande':
        accuracy, cover_ratio = calc_accuracy_WINOGRANDE(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method)
    if test_task_name.lower() == 'plan_bench':
        accuracy, cover_ratio = calc_accuracy_PLAN_BENCH(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method)
    if test_task_name.lower() == 'mmlu':
        accuracy, cover_ratio = calc_accuracy_MMLU_AGI(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method, full_task_name = 'mmlu')
    if test_task_name.lower() == 'agieval':
        accuracy, cover_ratio = calc_accuracy_MMLU_AGI(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method, full_task_name = 'agieval')
    if test_task_name.lower() == 'ecqa':
        accuracy, cover_ratio = calc_accuracy_ECQA(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method)
    if test_task_name.lower() == 'code':
        accuracy, cover_ratio = calc_accuracy_CODE(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method, checkpoint_num = checkpoint_num)
    if test_task_name.lower() == 'apps':
        accuracy, cover_ratio = calc_accuracy_APPS(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method)
    if test_task_name.lower() == 'mbpp':
        accuracy, cover_ratio = calc_accuracy_MBPP(test_data_list, predict_list, output_folder_name, file_name, task_name = task_name, sampling_num = sampling_num, train_method = train_method)
    
    log_line = f'{task_name} Evaluation for ' + test_task_name
    write_log(file_name, output_folder_name, log_line)
    log_line = f'{task_name} Accuracy: ' + str(accuracy) + ', Cover Ratio: ' + str(cover_ratio) + ' for ' + test_task_name

    if test_task_name.lower() == 'api_bank':
        log_line += f'        lv12_accuracy: {lv12_accuracy}        lv3_accuracy: {lv3_accuracy}'
    write_log(file_name, output_folder_name, log_line)
    return accuracy, cover_ratio


def DO_SAMPLE(test_data_list, test_task_name, test_config, output_folder_name, file_name, check_point_folder_name = '', data_loader_config = {}, task_name = '', sampling_num = 1, minimum_change_list = [], gpt4_generated_list = [], train_data_list = [], train_math_category = '', LLAMA_FACTORY_DIRECTORY = ''):
    compacted_predict_list = []
    for i in test_data_list:
        compacted_predict_list.append([])
    for i in range(sampling_num):
        test_config['seed_num'] = i
        predict_list = []
        if test_task_name.lower() == 'xxx':
            a = 1
            for item in minimum_change_list:
                predict_list.append(item['previous_prediction'])
        else:
            predict_list = do_predict_llama_factory_unify(test_data_list, output_folder_name, test_config, file_name, check_point_folder_name = check_point_folder_name, data_loader_config = data_loader_config, sampling_num = 1, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY, data_name='do_sample')

        # predict_list = ['Based on the context of the question and the options provided, the most likely answer is (1): bloody mess. This option is the most obvious and directly related to the scene of a brutal killing. The other options (pleasure, being imprisoned, feeling of guilt, and cake) do not make logical sense in this context.\n\nFinal Answer: (1) bloody mess.', 'Inference: Setting a romantic atmosphere typically involves creating a cozy and intimate environment. Candles are often used to create a warm and inviting glow.\n\nFinal Answer: (1) dimly lit room. A dimly lit room with candles can help set a romantic atmosphere by creating a warm, inviting, and intimate environment. The other options are not typically associated with setting a romantic atmosphere. A synagogue is a place of worship, a birthday cake is a food item, and roses are a gift or decoration, but they do not directly create a romantic atmosphere on their own.']

        a = 1
        d_1 = str(len(compacted_predict_list))
        d_2 = str(len(compacted_predict_list[0]))
        len_predict = len(predict_list)
        # print('------------------')
        # print(i)
        # print(f"""compacted_predict_list size: {d_1}, {d_2}""")
        # print(f"""predict_list len: {len_predict}""")
        # print('------------------')

        predict_list_file_name = f"""{file_name}_predict_list_data_creation_{i}"""
        predict_list_data_path = f"{HOME_DIRECTORY}/dpo_data/intermediate_result/{predict_list_file_name}.json"
        with open(predict_list_data_path, 'w') as file:
            json.dump(predict_list, file, indent=4)

        for ii in range(len(predict_list)):
            compacted_predict_list[ii].append(predict_list[ii])
    
    # compacted_predict_list = [[" Sure, I'd be happy to help! Here's my inference and final answer:\n\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.\n\nIn April, Natalia sold 48 clips.\n\nIn May, Natalia sold half as many clips as she sold in April, which is 48 / 2 = 24 clips.\n\nTherefore, in total, Natalia sold 48 + 24 = 72 clips in April and May.\n\nFinal Answer: 72 ", " Sure, I'd be happy to help! Here's my inference and final answer:\n\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.\n\nIn April, Natalia sold 48 clips.\n\nIn May, Natalia sold half as many clips as she sold in April, which is 48 / 2 = 24 clips.\n\nTherefore, in total, Natalia sold 48 + 24 = 72 clips in April and May.\n\nFinal Answer: 72 "], [" Sure! I'd be happy to help.\n\nWeng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting.\n\nTo calculate how much Weng earned, we need to first convert the time she spent babysitting from minutes to hours. There are 60 minutes in 1 hour, so:\n\n50 minutes / 60 minutes per hour = 0.833 hours (or 50/60)\n\nNext, we can multiply the number of hours Weng spent babysitting by her hourly rate to find out how much she earned:\n\n0.833 hours x $12 per hour = $9.99\n\nFinal Answer: $9.99 ", " Sure! I'd be happy to help.\n\nWeng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting.\n\nTo calculate how much Weng earned, we need to first convert the time she spent babysitting from minutes to hours. There are 60 minutes in 1 hour, so:\n\n50 minutes / 60 minutes per hour = 0.833 hours (or 50/60)\n\nNext, we can multiply the number of hours Weng spent babysitting by her hourly rate to find out how much she earned:\n\n0.833 hours x $12 per hour = $9.99\n\nFinal Answer: $9.99 "]]
        
    temp_file_name = f"""{file_name}_compacted_predict_list"""
    test_data_path = f"{HOME_DIRECTORY}/dpo_data/intermediate_result/{temp_file_name}.json"
    with open(test_data_path, 'w') as file:
        json.dump(compacted_predict_list, file, indent=4)
    
    with open(test_data_path, 'r') as file:
        compacted_predict_list = json.load(file)
    
    self_generated_dpo_preferred_list = []
    dpo_preferred_list = []
    dpo_reject_list = []
    minimum_change_dpo_data_list = []
    
    question_list = []
    groundtruth_list = []
    for i in range(len(test_data_list)):
        question = test_data_list[i]['question']
        question_list.append(question)
        groundtruth_list.append(test_data_list[i]['answer'])

    if test_task_name.lower() == 'gsm8k' or 'math' in test_task_name.lower():
        for i in range(len(compacted_predict_list)):
            if i % 50 ==0:
                print(i)
            if i == 300:
                a = 1
            item_temp = {}
            final_answer_correct_final = ''
            final_answer_incorrect_final = ''
            extracted_incorrect_final_answer = ''

            correct_answer_found = False
            wrong_answer_found = False
            for predict_item in compacted_predict_list[i]:
                extracted_final_answer = extract_last_number(predict_item)
                final_answer = extracted_final_answer

                temp = groundtruth_list[i]
                # extracted_groundtruth = extract_last_number(temp)
                if 'math' in test_task_name.lower():
                    # if 'boxed{120}' in temp:
                    #     a = 1
                    extracted_groundtruth = extract_last_number(temp)
                elif test_task_name.lower() == 'gsm8k':
                    extracted_groundtruth = test_data_list[i]['numerical_final_answer']
                groundtruth_num = extracted_groundtruth

                if final_answer == groundtruth_num:
                    final_answer_correct_final = predict_item
                    correct_answer_found = True
                else:
                    final_answer_incorrect_final = predict_item
                    extracted_incorrect_final_answer = extracted_final_answer
                    wrong_answer_found = True

            if not correct_answer_found:
                final_answer_correct_final = groundtruth_list[i]
            if not wrong_answer_found:
                final_answer_incorrect_final = predict_item + ' The final answer: -99999'
# 'The question might be wrong. We do not know how to solve it.'
                extracted_incorrect_final_answer = -99999

            question = question_list[i]

            question_item_temp = question.replace(f"""

Please put the final digital answer at the end after you finish inference in this format FINAL ANSWER: final neumerical answer

Format:
SOME_INFERENCE

FINAL ANSWER: """, '')
            question_item_temp += """

Please provide the final answer (a number) at the end, after 'Final Answer:'
"""
                
            minimum_change_answer = minimum_change_list[i]['answer']
            minimum_change_question = minimum_change_list[i]['question']
            gpt4_generated_answer = gpt4_generated_list[i]['answer']

            if correct_answer_found:
                item_temp = {}
                item_temp['question'] = minimum_change_question
                item_temp['groundtruth number'] = groundtruth_num
                item_temp['answer by groundtruth'] = groundtruth_list[i]
                item_temp['correct_answer'] = final_answer_correct_final
                item_temp['incorrect_answer'] = final_answer_incorrect_final
                item_temp['minimum_change'] = minimum_change_answer
                item_temp['gpt4_generated_answer'] = gpt4_generated_answer
                item_temp['wrong_answer_found'] = wrong_answer_found
                item_temp['correct_answer_found'] = correct_answer_found
                item_temp['i'] = i
                self_generated_dpo_preferred_list.append(item_temp)
            
            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['groundtruth number'] = groundtruth_num
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['incorrect_answer'] = final_answer_incorrect_final
            item_temp['minimum_change'] = minimum_change_answer
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            item_temp['i'] = i
            minimum_change_dpo_data_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['groundtruth number'] = groundtruth_num
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_correct_final
            item_temp['correctness'] = 'Correct'
            item_temp['correct_answer_found'] = correct_answer_found
            item_temp['wrong_answer_found'] = wrong_answer_found
            dpo_preferred_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['groundtruth number'] = groundtruth_num
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_incorrect_final
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            item_temp['correctness'] = 'Incorrect'
            dpo_reject_list.append(item_temp)
    if test_task_name.lower() == 'ecqa':
        for i in range(len(compacted_predict_list)):
            if i % 50 ==0:
                print(i)
            if i == 300:
                a = 1
            item_temp = {}
            final_answer_correct_final = ''
            final_answer_incorrect_final = ''
            extracted_incorrect_final_answer = ''

            correct_answer_found = False
            wrong_answer_found = False
            
            for predict_item in compacted_predict_list[i]:
                gold_label = str(test_data_list[i]['gold_label'])
                gold_label_content = groundtruth_list[i]
                gold_label_content = gold_label_content.lower()
                option1 = test_data_list[i]['1']
                option2 = test_data_list[i]['2']
                option3 = test_data_list[i]['3']
                option4 = test_data_list[i]['4']
                option5 = test_data_list[i]['5']

                gold_label_content = gold_label_content.strip().lower().rstrip('.')
                option1 = option1.strip().lower().rstrip('.')
                option2 = option2.strip().lower().rstrip('.')
                option3 = option3.strip().lower().rstrip('.')
                option4 = option4.strip().lower().rstrip('.')
                option5 = option5.strip().lower().rstrip('.')

                final_answer = extract_option_1_to_5(predict_item)
                if gold_label.lower() == final_answer.lower():
                    final_answer_correct_final = predict_item
                    correct_answer_found = True
                else:
                    final_answer = extract_context_after_answer(predict_item)
                    if final_answer == option1.lower() or final_answer == option2.lower() or final_answer == option3.lower() or final_answer == option4.lower() or final_answer == option5.lower():
                        final_answer_correct_final = predict_item
                        correct_answer_found = True
                    else:
                        final_answer_incorrect_final = predict_item
                        wrong_answer_found = True

            if not correct_answer_found:
                final_answer_correct_final = groundtruth_list[i]
            if not wrong_answer_found:
                final_answer_incorrect_final = 'The question might be wrong. We do not know how to solve it.'
                # final_answer_incorrect_final = predict_item + 'The question might be wrong. We do not know how to solve it.'

            question = question_list[i]
            minimum_change_answer = minimum_change_list[i]['answer']
            minimum_change_question = minimum_change_list[i]['question']
            gpt4_generated_answer = gpt4_generated_list[i]['answer']
            if correct_answer_found:
                item_temp = {}
                item_temp['question'] = minimum_change_question
                item_temp['answer by groundtruth'] = groundtruth_list[i]
                item_temp['correct_answer'] = final_answer_correct_final
                item_temp['incorrect_answer'] = final_answer_incorrect_final
                item_temp['minimum_change'] = minimum_change_answer
                item_temp['gpt4_generated_answer'] = gpt4_generated_answer
                item_temp['wrong_answer_found'] = wrong_answer_found
                item_temp['correct_answer_found'] = correct_answer_found
                item_temp['i'] = i
                self_generated_dpo_preferred_list.append(item_temp)
            
            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['incorrect_answer'] = final_answer_incorrect_final
            item_temp['minimum_change'] = minimum_change_answer
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            item_temp['i'] = i
            minimum_change_dpo_data_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_correct_final
            item_temp['correctness'] = 'Correct'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_preferred_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_incorrect_final
            item_temp['correctness'] = 'Incorrect'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_reject_list.append(item_temp)
    if test_task_name.lower() == 'esnli' or test_task_name.lower() == 'anli': 
        for i in range(len(compacted_predict_list)):
            if i % 50 ==0:
                print(i)
            if i == 300:
                a = 1
            item_temp = {}
            final_answer_correct_final = ''
            final_answer_incorrect_final = ''
            extracted_incorrect_final_answer = ''

            correct_answer_found = False
            wrong_answer_found = False
            for predict_item in compacted_predict_list[i]:     
                extracted_final_answer = extract_nli_answer(predict_item)
                extracted_final_answer = extracted_final_answer.lower()
                if extracted_final_answer == groundtruth_list[i]:
                    final_answer_correct_final = predict_item
                    correct_answer_found = True
                else:
                    final_answer_incorrect_final = predict_item
                    wrong_answer_found = True

            if not correct_answer_found:
                final_answer_correct_final = groundtruth_list[i]
            if not wrong_answer_found:
                if groundtruth_list[i] == 'neutral':
                    final_answer_incorrect_final = random.choice(['entailment', 'contradiction'])
                if groundtruth_list[i] == 'entailment':
                    final_answer_incorrect_final = random.choice(['neutral', 'contradiction'])
                if groundtruth_list[i] == 'contradiction':
                    final_answer_incorrect_final = random.choice(['entailment', 'neutral'])

            question = question_list[i]
            minimum_change_answer = minimum_change_list[i]['answer']
            minimum_change_question = minimum_change_list[i]['question']
            gpt4_generated_answer = gpt4_generated_list[i]['answer']
            if correct_answer_found:
                item_temp = {}
                item_temp['question'] = minimum_change_question
                item_temp['answer by groundtruth'] = groundtruth_list[i]
                item_temp['correct_answer'] = final_answer_correct_final
                item_temp['incorrect_answer'] = final_answer_incorrect_final
                item_temp['minimum_change'] = minimum_change_answer
                item_temp['gpt4_generated_answer'] = gpt4_generated_answer
                item_temp['wrong_answer_found'] = wrong_answer_found
                item_temp['correct_answer_found'] = correct_answer_found
                item_temp['i'] = i
                self_generated_dpo_preferred_list.append(item_temp)
            
            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['incorrect_answer'] = final_answer_incorrect_final
            item_temp['minimum_change'] = minimum_change_answer
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            item_temp['i'] = i
            minimum_change_dpo_data_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_correct_final
            item_temp['correctness'] = 'Correct'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_preferred_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_incorrect_final
            item_temp['correctness'] = 'Incorrect'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_reject_list.append(item_temp)
    elif test_task_name.lower() == 'anli':
        for i in range(len(compacted_predict_list)):
            if i % 50 ==0:
                print(i)
            item_temp = {}
            final_answer_correct_final = ''
            final_answer_incorrect_final = ''
            extracted_incorrect_final_answer = ''

            correct_answer_found = False
            wrong_answer_found = False
            for predict_item in compacted_predict_list[i]:     
                extracted_final_answer = extract_nli_answer(predict_item)
                extracted_final_answer = extracted_final_answer.lower()
                if extracted_final_answer == groundtruth_list[i]:
                    final_answer_correct_final = predict_item
                    correct_answer_found = True
                else:
                    final_answer_incorrect_final = predict_item
                    wrong_answer_found = True

            if not correct_answer_found:
                final_answer_correct_final = groundtruth_list[i]
            if not wrong_answer_found:
                if groundtruth_list[i] == 'neutral':
                    final_answer_incorrect_final = random.choice(['entailment', 'contradiction'])
                if groundtruth_list[i] == 'entailment':
                    final_answer_incorrect_final = random.choice(['neutral', 'contradiction'])
                if groundtruth_list[i] == 'contradiction':
                    final_answer_incorrect_final = random.choice(['entailment', 'neutral'])

            if correct_answer_found:# and wrong_answer_found:
                question = question_list[i]
                minimum_change_answer = False
                gpt4_generated_answer = False
                # for item in minimum_change_list:
                #     if item['premise'] in question:
                minimum_change_answer = minimum_change_list[i]['answer']
                # for item in gpt4_generated_list:
                #     if item['question'] in question:
                gpt4_generated_answer = gpt4_generated_list[i]['answer']
                if gpt4_generated_answer and minimum_change_answer:
                    item_temp = {}
                    item_temp['question'] = question
                    item_temp['groundtruth'] = groundtruth_list[i]
                    item_temp['correct_answer'] = final_answer_correct_final
                    item_temp['incorrect_answer'] = final_answer_incorrect_final
                    item_temp['minimum_change'] = minimum_change_answer
                    item_temp['gpt4_generated_answer'] = gpt4_generated_answer
                    item_temp['wrong_answer_found'] = wrong_answer_found
                    item_temp['correct_answer_found'] = correct_answer_found
                    item_temp['i'] = i
                    self_generated_dpo_preferred_list.append(item_temp)
                
            item_temp = {}
            item_temp['question'] = question
            item_temp['groundtruth'] = groundtruth_list[i]
            item_temp['incorrect_answer'] = final_answer_incorrect_final
            item_temp['minimum_change'] = minimum_change_answer
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            item_temp['i'] = i
            minimum_change_dpo_data_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = question_list[i]
            item_temp['groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_correct_final
            item_temp['correctness'] = 'Correct'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_preferred_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = question_list[i]
            item_temp['groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_incorrect_final
            item_temp['correctness'] = 'Incorrect'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_reject_list.append(item_temp)

    elif test_task_name.lower() == 'mbpp':
        for i in range(len(compacted_predict_list)):
            if i % 50 ==0:
                print(i)
            if i == 300:
                a = 1
            item_temp = {}
            final_answer_correct_final = ''
            final_answer_incorrect_final = ''

            correct_answer_found = False
            wrong_answer_found = False
            for predict_item in compacted_predict_list[i]:
                completion = predict_item.replace("```python", '')
                completion = completion.replace("```", '')
                original_completion = predict_item

                # Modified pattern to be case-insensitive
                pattern = r"ANSWER:(.*)"

                # Using re.IGNORECASE to make the search case-insensitive
                match = re.search(pattern, completion, re.DOTALL | re.IGNORECASE)

                # Extract the matched portion
                if match:
                    completion = match.group(1).lstrip()
                else:
                    completion = completion.lstrip()
                passed = True
                # for test_item in item['test_list']:

                test_list = train_data_list[i]['test_list']
                challenge_test_list = train_data_list[i]['challenge_test_list']
                if challenge_test_list != []:
                    test_list += challenge_test_list
                passed = run_dynamic_test_with_timeout(test_list, completion)
                if not passed:
                    passed = False
                if passed:
                    final_answer_correct_final = predict_item
                    correct_answer_found = True
                else:
                    final_answer_incorrect_final = predict_item
                    wrong_answer_found = True

            if not correct_answer_found:
                final_answer_correct_final = groundtruth_list[i]
            if not wrong_answer_found:
                final_answer_incorrect_final = 'The question might be wrong. We do not know how to solve it.'

            question = question_list[i]

            minimum_change_answer = minimum_change_list[i]['answer']
            minimum_change_question = minimum_change_list[i]['question']
            gpt4_generated_answer = gpt4_generated_list[i]['answer']
            if correct_answer_found:
                item_temp = {}
                item_temp['question'] = minimum_change_question
                item_temp['answer by groundtruth'] = groundtruth_list[i]
                item_temp['correct_answer'] = final_answer_correct_final
                item_temp['incorrect_answer'] = final_answer_incorrect_final
                item_temp['minimum_change'] = minimum_change_answer
                item_temp['gpt4_generated_answer'] = gpt4_generated_answer
                item_temp['wrong_answer_found'] = wrong_answer_found
                item_temp['correct_answer_found'] = correct_answer_found
                item_temp['i'] = i
                self_generated_dpo_preferred_list.append(item_temp)
            
            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['incorrect_answer'] = final_answer_incorrect_final
            item_temp['minimum_change'] = minimum_change_answer
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            item_temp['i'] = i
            minimum_change_dpo_data_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_correct_final
            item_temp['correctness'] = 'Correct'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_preferred_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_incorrect_final
            item_temp['correctness'] = 'Incorrect'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_reject_list.append(item_temp)

    elif test_task_name.lower() == 'code':
        for i in range(len(compacted_predict_list)):
            if i % 50 ==0:
                print(i)
            if i == 300:
                a = 1
            item_temp = {}
            final_answer_correct_final = ''
            final_answer_incorrect_final = ''

            correct_answer_found = False
            wrong_answer_found = False
            for predict_item in compacted_predict_list[i]:
                pattern = r"ANSWER:(.*)"
                predict_item = predict_item.replace("```python", '')
                predict_item = predict_item.replace("```", '')
                # Using regex to find the matched part with re.IGNORECASE flag for case-insensitive matching
                # re.DOTALL flag allows '.' to match newline characters as well
                match = re.search(pattern, predict_item, re.DOTALL | re.IGNORECASE)

                # Extract the matched portion
                if match:
                    completion = match.group(1) 
                else:
                    completion = completion

                 # Path for saving the file
                task_name = task_name[:3]
                sample_path = f'{HOME_DIRECTORY}/code_eval/intermediate_file/HumanEval_{task_name}_{file_name}.jsonl' 
                problem_path = f'{HOME_DIRECTORY}/code_eval/intermediate_file/HumanEval_{task_name}_{file_name}.jsonl'

                temp = {}
                temp['question'] = train_data_list[i]['question']
                temp['prompt'] = train_data_list[i]['prompt']
                temp['task_id'] = f'test/{i}'
                temp['input'] = ''
                temp['test'] = train_data_list[i]['test']
                temp['entry_point'] = train_data_list[i]['entry_point']
                temp['completion'] = completion
                temp['original_prediction'] = predict_list[i]


                with open(problem_path, 'w') as file:
                    json_line = json.dumps(temp)
                    file.write(json_line + '\n')

                # Writing the code snippet to a JSONL file
                with open(sample_path, 'w') as file:
                    json_line = json.dumps(temp)
                    file.write(json_line + '\n')
                
                cmd = [
                    "evaluate_functional_correctness",
                    f"{sample_path}",
                    "--problem_file", f"{problem_path}"
                ]

                subprocess.run(" ".join(cmd), shell=True, cwd=f'{HOME_DIRECTORY}/code_eval')
                # try:
                log_file_path = sample_path.replace('jsonl', 'jsonl_results.jsonl')
                with open(f'{log_file_path}', 'r') as file:
                    # Read the contents of the file
                    content = file.read()
                pass_count = content.count('"passed": true')
      


                if pass_count > 0:
                    final_answer_correct_final = predict_item
                    correct_answer_found = True
                else:
                    final_answer_incorrect_final = predict_item
                    wrong_answer_found = True

            if not correct_answer_found:
                final_answer_correct_final = groundtruth_list[i]
            if not wrong_answer_found:
                final_answer_incorrect_final = 'The question might be wrong. We do not know how to solve it.'

            question = question_list[i]

            minimum_change_answer = minimum_change_list[i]['answer']
            minimum_change_question = minimum_change_list[i]['question']
            gpt4_generated_answer = gpt4_generated_list[i]['answer']
            if correct_answer_found:
                item_temp = {}
                item_temp['question'] = minimum_change_question
                item_temp['answer by groundtruth'] = groundtruth_list[i]
                item_temp['correct_answer'] = final_answer_correct_final
                item_temp['incorrect_answer'] = final_answer_incorrect_final
                item_temp['minimum_change'] = minimum_change_answer
                item_temp['gpt4_generated_answer'] = gpt4_generated_answer
                item_temp['wrong_answer_found'] = wrong_answer_found
                item_temp['correct_answer_found'] = correct_answer_found
                item_temp['i'] = i
                self_generated_dpo_preferred_list.append(item_temp)
            
            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['incorrect_answer'] = final_answer_incorrect_final
            item_temp['minimum_change'] = minimum_change_answer
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            item_temp['i'] = i
            minimum_change_dpo_data_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_correct_final
            item_temp['correctness'] = 'Correct'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_preferred_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_incorrect_final
            item_temp['correctness'] = 'Incorrect'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_reject_list.append(item_temp)

    elif test_task_name.lower() == 'mmlu' or test_task_name.lower() == 'agieval':
        for i in range(len(compacted_predict_list)):
            if i % 50 ==0:
                print(i)
            if i == 300:
                a = 1
            item_temp = {}
            final_answer_correct_final = ''
            final_answer_incorrect_final = ''

            correct_answer_found = False
            wrong_answer_found = False
            for predict_item in compacted_predict_list[i]:
                gold_label = str(train_data_list[i]['gold_label'])
                abcd_content = train_data_list[i][gold_label]
                gold_label = gold_label.lower()
                a_content = train_data_list[i]['A']
                b_content = train_data_list[i]['B']
                c_content = train_data_list[i]['C']
                d_content = train_data_list[i]['D']

                abcd_content = abcd_content.strip().lower().rstrip('.')
                a_content = a_content.strip().lower().rstrip('.')
                b_content = b_content.strip().lower().rstrip('.')
                c_content = c_content.strip().lower().rstrip('.')
                d_content = d_content.strip().lower().rstrip('.')

                final_answer = extract_option_mmlu_agi(predict_item)
                test_data_item = train_data_list[i]
                passed = False
                if gold_label == final_answer.lower():
                    passed = True
                else:
                    final_answer = extract_context_after_answer(predict_item)
                    if final_answer == a_content.lower() or final_answer == b_content.lower() or final_answer == c_content.lower() or final_answer == d_content.lower():
                        if final_answer == a_content.lower():
                            test_data_item['extracted_answer'] = 'A'
                        if final_answer == b_content.lower():
                            test_data_item['extracted_answer'] = 'B'
                        if final_answer == c_content.lower():
                            test_data_item['extracted_answer'] = 'C'
                        if final_answer == d_content.lower():
                            test_data_item['extracted_answer'] = 'D'
                        if final_answer == abcd_content:
                            passed = True
                        else:
                            passed = False
                    else:
                        passed = False

                if not passed:
                    passed = False
                if passed:
                    final_answer_correct_final = predict_item
                    correct_answer_found = True
                else:
                    final_answer_incorrect_final = predict_item
                    wrong_answer_found = True

            if not correct_answer_found:
                final_answer_correct_final = groundtruth_list[i]
            if not wrong_answer_found:
                final_answer_incorrect_final = 'The question might be wrong. We do not know how to solve it.'

            question = question_list[i]

            minimum_change_answer = minimum_change_list[i]['answer']
            minimum_change_question = minimum_change_list[i]['question']
            gpt4_generated_answer = gpt4_generated_list[i]['answer']
            if correct_answer_found:
                item_temp = {}
                item_temp['question'] = minimum_change_question
                item_temp['answer by groundtruth'] = groundtruth_list[i]
                item_temp['correct_answer'] = final_answer_correct_final
                item_temp['incorrect_answer'] = final_answer_incorrect_final
                item_temp['minimum_change'] = minimum_change_answer
                item_temp['gpt4_generated_answer'] = gpt4_generated_answer
                item_temp['wrong_answer_found'] = wrong_answer_found
                item_temp['correct_answer_found'] = correct_answer_found
                item_temp['i'] = i
                self_generated_dpo_preferred_list.append(item_temp)
            
            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['incorrect_answer'] = final_answer_incorrect_final
            item_temp['minimum_change'] = minimum_change_answer
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            item_temp['i'] = i
            minimum_change_dpo_data_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_correct_final
            item_temp['correctness'] = 'Correct'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_preferred_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_incorrect_final
            item_temp['correctness'] = 'Incorrect'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_reject_list.append(item_temp)
    elif test_task_name.lower() == 'piqa':
        for i in range(len(compacted_predict_list)):
            if i % 50 ==0:
                print(i)
            if i == 300:
                a = 1
            item_temp = {}
            final_answer_correct_final = ''
            final_answer_incorrect_final = ''

            correct_answer_found = False
            wrong_answer_found = False

            gold_label = str(train_data_list[i]['gold_label'])
            sol12_content = train_data_list[i]['sol'+gold_label]
            gold_label = gold_label.lower()
            option1 = train_data_list[i]['sol1']
            option2 = train_data_list[i]['sol2']

            sol12_content = sol12_content.strip().lower().rstrip('.')
            option1 = option1.strip().lower().rstrip('.')
            option2 = option2.strip().lower().rstrip('.')
            for predict_item in compacted_predict_list[i]:
                final_answer = extract_option(predict_item)
                if gold_label.lower() == final_answer.lower():
                    passed = True
                else:
                    final_answer = extract_context_after_answer(predict_item)
                    if final_answer == option1.lower() or final_answer == option2.lower():
                        if final_answer == sol12_content:
                            passed = True
                        else:
                            passed = False
                    else:
                        passed = False

                if not passed:
                    passed = False
                if passed:
                    final_answer_correct_final = predict_item
                    correct_answer_found = True
                else:
                    final_answer_incorrect_final = predict_item
                    wrong_answer_found = True

            if not correct_answer_found:
                final_answer_correct_final = groundtruth_list[i]
            if not wrong_answer_found:
                final_answer_incorrect_final = 'The question might be wrong. We do not know how to solve it.'

            question = question_list[i]

            minimum_change_answer = minimum_change_list[i]['answer']
            minimum_change_question = minimum_change_list[i]['question']
            gpt4_generated_answer = gpt4_generated_list[i]['answer']
            if correct_answer_found:
                item_temp = {}
                item_temp['question'] = minimum_change_question
                item_temp['answer by groundtruth'] = groundtruth_list[i]
                item_temp['correct_answer'] = final_answer_correct_final
                item_temp['incorrect_answer'] = final_answer_incorrect_final
                item_temp['minimum_change'] = minimum_change_answer
                item_temp['gpt4_generated_answer'] = gpt4_generated_answer
                item_temp['wrong_answer_found'] = wrong_answer_found
                item_temp['correct_answer_found'] = correct_answer_found
                item_temp['i'] = i
                self_generated_dpo_preferred_list.append(item_temp)
            
            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['incorrect_answer'] = final_answer_incorrect_final
            item_temp['minimum_change'] = minimum_change_answer
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            item_temp['i'] = i
            minimum_change_dpo_data_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_correct_final
            item_temp['correctness'] = 'Correct'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_preferred_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_incorrect_final
            item_temp['correctness'] = 'Incorrect'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_reject_list.append(item_temp)
    
    elif test_task_name.lower() == 'boolq':
        for i in range(len(compacted_predict_list)):
            if i % 50 ==0:
                print(i)
            if i == 300:
                a = 1
            item_temp = {}
            final_answer_correct_final = ''
            final_answer_incorrect_final = ''

            correct_answer_found = False
            wrong_answer_found = False

            gold_label = str(train_data_list[i]['gold_label'])

            for predict_item in compacted_predict_list[i]:
                final_answer = extract_bool(predict_item)
                if gold_label.lower() == final_answer.lower():
                    passed = True
                else:
                    passed = False

                if not passed:
                    passed = False
                if passed:
                    final_answer_correct_final = predict_item
                    correct_answer_found = True
                else:
                    final_answer_incorrect_final = predict_item
                    wrong_answer_found = True

            if not correct_answer_found:
                final_answer_correct_final = groundtruth_list[i]
            if not wrong_answer_found:
                fake_label = 'false'
                if final_answer.lower() == 'false':
                    fake_label == 'true'
                final_answer_incorrect_final = predict_item + f' Final Answer: {fake_label}'

            question = question_list[i]

            minimum_change_answer = minimum_change_list[i]['answer']
            minimum_change_question = minimum_change_list[i]['question']
            gpt4_generated_answer = gpt4_generated_list[i]['answer']
            if correct_answer_found:
                item_temp = {}
                item_temp['question'] = minimum_change_question
                item_temp['answer by groundtruth'] = groundtruth_list[i]
                item_temp['correct_answer'] = final_answer_correct_final
                item_temp['incorrect_answer'] = final_answer_incorrect_final
                item_temp['minimum_change'] = minimum_change_answer
                item_temp['gpt4_generated_answer'] = gpt4_generated_answer
                item_temp['wrong_answer_found'] = wrong_answer_found
                item_temp['correct_answer_found'] = correct_answer_found
                item_temp['i'] = i
                self_generated_dpo_preferred_list.append(item_temp)
            
            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['incorrect_answer'] = final_answer_incorrect_final
            item_temp['minimum_change'] = minimum_change_answer
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            item_temp['i'] = i
            minimum_change_dpo_data_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_correct_final
            item_temp['correctness'] = 'Correct'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_preferred_list.append(item_temp)

            item_temp = {}
            item_temp['question'] = minimum_change_question
            item_temp['answer by groundtruth'] = groundtruth_list[i]
            item_temp['answer'] = final_answer_incorrect_final
            item_temp['correctness'] = 'Incorrect'
            item_temp['wrong_answer_found'] = wrong_answer_found
            item_temp['correct_answer_found'] = correct_answer_found
            dpo_reject_list.append(item_temp)
    
    return dpo_preferred_list, dpo_reject_list, self_generated_dpo_preferred_list, minimum_change_dpo_data_list



def do_predict_llama_factory(test_data_path, output_folder_name, test_task_name, test_config, check_point_folder_name = '', LLAMA_FACTORY_DIRECTORY = ''):
    import os
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'  # Sets timeout to 2 seconds
    sys.path.append(f'{LLAMA_FACTORY_DIRECTORY}')
    from src import train_bash
    
    sys.path.append(parent_dir)
    seed = test_config['seed_num']
    output_folder_name = f'{HOME_DIRECTORY}/output/{output_folder_name}/{seed}'
    model_path = f"{MODEL_DIRECTORY}/{test_config['model_name']}"

    if not os.path.exists(f"{output_folder_name}"):
        os.makedirs(f"{output_folder_name}")
    
    cmd = [
        "accelerate launch",
        # "CUDA_VISIBLE_DEVICES=0,1 python",
        f"{LLAMA_FACTORY_DIRECTORY}/src/train_bash.py",
        "--stage", "sft",
        "--model_name_or_path", model_path,
        "--do_predict", str(True),
        "--dataset", test_data_path,
        # "--template", test_config['template'],
        "--finetuning_type", test_config['finetuning_type'],
        # "--max_length", str(test_config['max_length']s),
        "--output_dir", f'{output_folder_name}',
        "--per_device_eval_batch_size", str(test_config['per_device_eval_batch_size']),
        "--max_length", str(test_config['max_length']),
        "--predict_with_generate", str(True),
        "--fp16"
    ]

    if check_point_folder_name != '':
        cmd += ['--adapter_name_or_path', check_point_folder_name]

    if 'seed' in test_config:
        cmd += ['--seed', test_config['seed_num']]

    subprocess.run(" ".join(cmd), shell=True, cwd=LLAMA_FACTORY_DIRECTORY)

    source_file = f"{output_folder_name}/generated_predictions.jsonl"
    destination_file = f"{output_folder_name}/{test_task_name}.jsonl"
    shutil.copy(source_file, destination_file)

    predict_list = []
    with open(destination_file, 'r') as file:
        json_list = list(file)
        for line in json_list:
            data = json.loads(line)
            prediction = data.get("predict", "No label key found.")
            predict_list.append(prediction)

    return predict_list


def do_predict_llama_factory_unify(data_list, output_folder_name, test_config, file_name, check_point_folder_name = '',data_loader_config = {}, sampling_num = 1, merged_base_model_dir = '', data_name = '', LLAMA_FACTORY_DIRECTORY = '', SFT_for_dpo_lora_dir = ''):
    import os
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'  # Sets timeout to 2 seconds
    sys.path.append(f'{LLAMA_FACTORY_DIRECTORY}')
    # from src import train_bash

    file_name = file_name.replace('_log', '')
    put_json_list_to_data_info(data_list, data_name, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY)
    
    sys.path.append(parent_dir)
    output_folder_name = f'{HOME_DIRECTORY}/output/{file_name}'
    if merged_base_model_dir == '':
        model_path = f"{MODEL_DIRECTORY}/{test_config['model_name']}"
    else:
        model_path = merged_base_model_dir

    if not os.path.exists(f"{output_folder_name}"):
        os.makedirs(f"{output_folder_name}")
    
    if test_config['device_num'] > 1:
        start_line = 'accelerate launch'
    else:
        start_line = 'python'
    cmd = [
        # "accelerate launch",
        start_line,
        f"{LLAMA_FACTORY_DIRECTORY}/src/train_bash.py",
        "--stage", "sft",
        "--model_name_or_path", model_path,
        "--do_predict", str(True),
        "--dataset", data_name,
        "--template", test_config['template'],
        "--finetuning_type", test_config['finetuning_type'],
        "--max_length", str(test_config['max_length']),
        "--cutoff_len", str(test_config['max_input_length']),
        "--output_dir", f'{output_folder_name}',
        "--per_device_eval_batch_size", str(test_config['per_device_eval_batch_size']),
        "--max_new_tokens", str(test_config['max_new_tokens']),
        "--predict_with_generate", str(True),
        "--overwrite_cache",
        "--fp16"
    ]

    if check_point_folder_name != '':
        if SFT_for_dpo_lora_dir != '':
            combined_path = f"{SFT_for_dpo_lora_dir},{check_point_folder_name}"
            cmd += ["--adapter_name_or_path", combined_path]
        else:
            cmd += ['--adapter_name_or_path', check_point_folder_name]
        # cmd += ['--adapter_name_or_path', check_point_folder_name]

    if 'seed_num' in test_config:
        cmd += ['--seed', str(test_config['seed_num'])]
    
    if 'do_sample' in test_config:
        cmd += ['--do_sample', str(test_config['do_sample'])]

    if 'temperature' in test_config:
        cmd += ['--temperature', str(test_config['temperature'])]

    subprocess.run(" ".join(cmd), shell=True, cwd=LLAMA_FACTORY_DIRECTORY)

    source_file = f"{output_folder_name}/generated_predictions.jsonl"
# /gpfs/users/a1796450/ACL_2024/model/MetaMath-Llemma-7B/pytorch_model-00003-of-00003.bin
# /gpfs/users/a1796450/ACL_2024/model/MetaMath-Llemma-7B/pytorch_model-00003-of-00003.bin
    predict_list = []
    with open(source_file, 'r') as file:
        json_list = list(file)
        for line in json_list:
            data = json.loads(line)
            prediction = data.get("predict", "No label key found.")
            predict_list.append(prediction)

    return predict_list

def do_predict_llama_alpaca(data_list, output_folder_name, test_config, check_point_folder_name = '', data_loader_config = {}, sampling_num = 1):
    import os
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'  # Sets timeout to 2 seconds

    sys.path.append(parent_dir)
    seed = test_config['seed_num']
    output_folder_name = f'{HOME_DIRECTORY}/output/{output_folder_name}/{seed}'
    model_path = f"{MODEL_DIRECTORY}/{test_config['model_name']}"

    if sampling_num == 1:
        predict_list = alpaca_predict(data_list, 
                    base_model = model_path, 
                    lora_weights = check_point_folder_name, 
                    prompt_template = '', 
                    input = None, 
                    temperature = 1.0, 
                    top_p = 1.0, 
                    top_k = 50, 
                    num_beams = test_config['num_beams'], 
                    max_new_tokens = test_config['max_length'], 
                    batch_size = test_config['per_device_eval_batch_size'], 
                    max_input_length = test_config['max_input_length'],
                    num_workers = data_loader_config['num_workers'],
                    pin_memory = data_loader_config['pin_memory'],
                    #    load_8bit = test_config['load_8bit']
                    )
    else:
        predict_list = alpaca_predict(data_list, 
                   base_model = model_path, 
                   lora_weights = check_point_folder_name, 
                   prompt_template = '', 
                   input = None, 
                   temperature = 1.0, 
                   top_p = 1.0, 
                   top_k = 50, 
                   num_beams = test_config['num_beams'], 
                   max_new_tokens = test_config['max_length'], 
                   batch_size = test_config['per_device_eval_batch_size'], 
                   max_input_length = test_config['max_input_length'],
                   num_workers = data_loader_config['num_workers'],
                   pin_memory = data_loader_config['pin_memory'],
                   num_return_sequences = sampling_num
                #    load_8bit = test_config['load_8bit']
                   )

    for i in range(len(predict_list)):
        # Split the string at "\n\n### Response"
        parts = predict_list[i].split('\n\n### Response')

        # Check if there are at least two parts after splitting
        if len(parts) >= 2:
            # Take the part after "\n\n### Response" and do the replace operation
            predict_list[i] = parts[1].replace('\n\n### Response:', '')

    return predict_list
