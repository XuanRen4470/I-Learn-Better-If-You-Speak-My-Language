import sys
import os
import json
# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from config.config import HOME_DIRECTORY
import copy

from peft import PeftModel
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


import json
from utils.function import HOME_DIRECTORY
from utils.data_loader import load_GSM8K, load_MATH, load_ESNLI, load_CODE_code_only, load_MBPP_code_only, load_PIQA, load_BOOLQ, load_MMLU, load_AGIEVAL, load_ECQA, load_ANLI, load_SQUAD, load_API_BANK_optimized, load_API_BANK_aug_2, eval_MATH_correctness, extract_after_last_occurrence, load_WINOGRANDE
from utils.data_loader_in_context import in_context_learning_examples, random_select_in_context_learning_examples
from config.modify_config_on_current_job import set_config
import random
import os
import math


def process_data_list(data_list, prompt_style = '', test_task_name = '',sub_samples_num_list = '', end_template = '', similarity_compare_to_in_context_mc = '', use_in_context_learning = False, disable_incontext_learn = False, n_train = 1, test_idx = -1):#, plot_record_mispredicted_samples = False):
    for i, item in enumerate(data_list):
        prompt_style = prompt_style
        question_item = item['question']
        # original_question = item['question']
        if 'original_question' in item:
            original_question = item['original_question']
        else:
            original_question = item['question']

        if use_in_context_learning:
            if not disable_incontext_learn:
                # formated_question = in_context_learning_examples(question_item, original_question, prompt_style = prompt_style, task = test_task_name.lower(), enforce_prompt_style = '')
                formated_question = random_select_in_context_learning_examples(question_item, original_question, prompt_style = prompt_style, task = test_task_name.lower(), sub_samples_num_list = sub_samples_num_list[i], end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc)
            else:
                formated_question = question_item
            data_list[i]['question'] = formated_question
        data_list[i]['original_question'] = original_question
    data_list = data_list[:n_train]

    if test_idx != -1:
        data_list = [data_list[test_idx]]
    return data_list

    # if prompt_style == 'minimum_change':
    #     for i, item in enumerate(data_list):
    #         prompt_style = 'minimum_change'
    #         question_item = item['question']
    #         original_question = item['question']
    #         # formated_question = in_context_learning_examples(question_item, original_question, prompt_style = prompt_style, task = test_task_name.lower(), enforce_prompt_style = '')

    #         correct = True

    #         if plot_record_mispredicted_samples:
    #             answer_temp = item['answer']
    #             previous_prediction = item['previous_prediction']
    #             answer_filtered = extract_after_last_occurrence(answer_temp, "Groundtruth:")
    #             answer_filtered = answer_filtered.strip('\'"')
    #             correct = eval_MATH_correctness(previous_prediction, answer_filtered)
        
    #         if use_in_context_learning:
    #             if not disable_incontext_learn:
    #                 formated_question = random_select_in_context_learning_examples(question_item, original_question, prompt_style = prompt_style, task = test_task_name.lower(), sub_samples_num_list = sub_samples_num_list[i], end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc)
    #             else:
    #                 formated_question = question_item
    #             data_list[i]['question'] = formated_question
    #         data_list[i]['original_question'] = original_question

    #         if not correct:
    #             item = data_list[i].copy()
    #             minimum_change_train_data_list_filtered.append(item)
    #             item_1 = data_list[i].copy()
    #             item_1['answer'] = previous_prediction
    #             minimum_change_train_data_list_filtered.append(item_1)
    #     data_list = data_list[:n_train]
    



# def perplexity_calculation_in_context_data_loader(train_task_name, n_train, use_in_context_learning, plot_record_mispredicted_samples, similarity_compare_to_in_context_mc, disable_incontext_learn, test_idx, end_template):
def perplexity_calculation_in_context_data_loader(train_task_name, n_train, use_in_context_learning, similarity_compare_to_in_context_mc, disable_incontext_learn, test_idx, end_template):
    if 'GSM8K' in train_task_name.upper() or 'MATH' in train_task_name.upper() or 'API_BANK' in train_task_name.upper() or 'ANLI' in train_task_name.upper() or 'CODE' in train_task_name.upper() or 'APPS' in train_task_name.upper() or 'MBPP' in train_task_name.upper() or 'MNLI' in train_task_name.upper() or 'ESNLI' in train_task_name.upper() or 'SCITAIL' in train_task_name.upper() or 'BOOLQ' in train_task_name.upper() or 'WINOGRANDE' in train_task_name.upper() or 'PIQA' in train_task_name.upper() or 'TRIVIAQA' in train_task_name.upper() or 'MMLU' in train_task_name.upper() or 'AGIEVAL' in train_task_name.upper() or 'ECQA' in train_task_name.upper() or 'SQUAD' in train_task_name.upper() or 'AQUARAT' in train_task_name.upper() or 'PLAN_BENCH' in train_task_name.upper():
        test_task_name_list = [train_task_name.lower()]

    for test_task_name in test_task_name_list:
        minimum_change_train_data_list = []
        if test_task_name.lower() == 'gsm8k':
            train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/train_filtered.json'
            train_data_list = load_GSM8K(train_path, n_train, zeroshot = True)
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/mistral_minimum_change_1000_clean.json'
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/gpt4_generated_gsm8k_False_999999_march_27.json'
            anthropic_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name.upper()}/anthropic_gpt4_generated_gsm8k_False_1000_r1.json'
            mini_gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name.upper()}/varient/openai_mini_gpt4.json'
            step_by_step_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/varient/gpt4_generated_step_by_step_1000.json'
            redundant_data_train_path = f'{HOME_DIRECTORY}/dataset/GSM8K/varient/openai_gpt4_generated_gsm8k_redundant_1000.json'

        if 'math_algebra' in test_task_name.lower():
            train_path = f'{HOME_DIRECTORY}/dataset/MATH/train_algebra_total_filtered.json'
            train_data_list = load_MATH(train_path, n_train, zeroshot = True)
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MATH/math_algebra_total_mistral_minimum_change_1000.json'
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MATH/gpt4_generated_math_algebra_False_1000.json'
            anthropic_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MATH/anthropic_gpt4_generated_math_algebra_False_1000_r1.json'
            mini_gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MATH/varient/openai_mini_gpt4.json'
            step_by_step_data_train_path = f'{HOME_DIRECTORY}/dataset/MATH/varient/gpt4_generated_step_by_step_1000.json'

        if test_task_name.lower() == 'esnli':
            train_path = f'{HOME_DIRECTORY}/dataset/ESNLI/train.json'
            train_data_list = load_ESNLI(train_path, n_train, finetune = True, meta_math_template = False)
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/esnli_mistral_minimum_change_1000.json'
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/gpt4_generated_esnli_False_1000.json'
            anthropic_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name.upper()}/anthropic_gpt4_generated_esnli_False_1000_r1.json'
            mini_gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name.upper()}/varient/openai_mini_gpt4.json'
            mini_gpt4_style_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name.upper()}/varient/write_in_gpt4_style.json'
            step_by_step_data_train_path = f'{HOME_DIRECTORY}/dataset/ESNLI/varient/gpt4_generated_step_by_step_1000.json'
            redundant_data_train_path = f'{HOME_DIRECTORY}/dataset/ESNLI/varient/openai_gpt4_generated_esnli_redundant_1000.json'

        if test_task_name.lower() == 'boolq':
            train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/train.json'
            train_data_list = load_BOOLQ(train_path, n_train)
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ//boolq_mistral_minimum_change_1000_march_27.json'
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/gpt4_generated_boolq_False_1000.json'
            anthropic_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name.upper()}/anthropic_gpt4_generated_boolq_False_1000_r1.json'
            mini_gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name.upper()}/varient/openai_mini_gpt4.json'
            mini_gpt4_style_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name.upper()}/varient/write_in_gpt4_style.json'
            step_by_step_data_train_path = f'{HOME_DIRECTORY}/dataset/BOOLQ/varient/gpt4_generated_step_by_step_1000.json'

        if test_task_name.lower() == 'ecqa':
            train_path = f'{HOME_DIRECTORY}/dataset/ECQA/train.json'
            train_data_list = load_ECQA(train_path, n_train, finetune = True, use_gt_rationale = True, meta_math_template = False)
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/ECQA/ecqa_mistral_minimum_change_1000_march_27.json'
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/ECQA/gpt4_generated_ecqa_False_1000_march_27.json'
            anthropic_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name.upper()}/anthropic_gpt4_generated_ecqa_False_1000_r1.json'
            mini_gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name.upper()}/varient/openai_mini_gpt4.json'
            step_by_step_data_train_path = f'{HOME_DIRECTORY}/dataset/ECQA/varient/gpt4_generated_step_by_step_1000.json'
            redundant_data_train_path = f'{HOME_DIRECTORY}/dataset/ECQA/varient/openai_gpt4_generated_ecqa_redundant_1000.json'
        
        if test_task_name.lower() == 'mmlu':
            train_path = f'{HOME_DIRECTORY}/dataset/MMLU/mmlu_train.json'
            train_data_list = load_MMLU(train_path, n_train, finetune = True)
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/MMLU/mmlu_mistral_minimum_change_1000_sep_19.json'
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MMLU/mmlu_gpt-4o-2024-08-06_1000.json'
            anthropic_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MMLU/anthropic_gpt4_generated_mmlu_False_1000.json'
            mini_gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/MMLU/varient/openai_mini_gpt4.json'
            step_by_step_data_train_path = f'{HOME_DIRECTORY}/dataset/MMLU/varient/gpt4_generated_step_by_step_1000.json'
        
        if test_task_name.lower() == 'winogrande':
            train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/train.json'
            train_data_list = load_WINOGRANDE(train_path, n_train, finetune = True)
            minimum_change_train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/winogrande_mistral_minimum_change_1000_sep_9.json'
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/openai_gpt4_generated_winogrande_False_1000_r1.json'
            anthropic_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/anthropic_gpt4_generated_winogrande_False_1000_answer_without_groundtruth_False_enable_mini_gpt4_False.json'
            mini_gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/varient/openai_mini_gpt4.json'
            step_by_step_data_train_path = f'{HOME_DIRECTORY}/dataset/WINOGRANDE/varient/gpt4_generated_step_by_step_1000.json'
            
        if test_task_name.lower() == 'api_bank':
            train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/train/training-data_lv3-api-train.json'
            train_data_list = load_API_BANK_aug_2(train_path, n_train)
            human_written_examples_data_train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/varient/openai_human_written_examples.json'
            with open(human_written_examples_data_train_path, 'r') as file:
                human_written_examples_data_list = json.load(file)
            human_written_examples_data_list = human_written_examples_data_list[:n_train]
            original_human_written_examples_data_list = copy.deepcopy(human_written_examples_data_list)
            step_by_step_data_train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/varient/gpt4_generated_step_by_step_1000.json'
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/openai_gpt4_generated_api_bank_False_1000_r1.json'
            anthropic_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/anthropic_gpt4_generated_api_bank_1000.json'
            mini_gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/varient/openai_mini_gpt4.json'
            provide_gpt4_style_example_data_train_path = f'{HOME_DIRECTORY}/dataset/API_BANK/varient/openai_gpt4_generated_new_rewrite_strategy.json'
            with open(provide_gpt4_style_example_data_train_path, 'r') as file:
                provide_gpt4_style_example_data_list = json.load(file)
            provide_gpt4_style_example_data_list = provide_gpt4_style_example_data_list[:n_train]
            original_provide_gpt4_style_example_data_list = copy.deepcopy(provide_gpt4_style_example_data_list)
        
        if test_task_name.lower() == 'plan_bench':
            train_path = f'{HOME_DIRECTORY}/dataset/PLAN_BENCH/train_plan_generation.json'
            with open(train_path, 'r') as file:
                train_data_list = json.load(file)
            train_data_list = train_data_list[:n_train]
            gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/PLAN_BENCH/gpt4_generated_plan_bench_False_1000.json'
            anthropic_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/PLAN_BENCH/anthropic_generated_plan_bench_False_1000.json'
            mini_gpt4_generated_data_train_path = f'{HOME_DIRECTORY}/dataset/PLAN_BENCH/varient/openai_mini_gpt4.json'
            step_by_step_data_train_path = f'{HOME_DIRECTORY}/dataset/PLAN_BENCH/varient/gpt4_generated_step_by_step_1000.json'
            human_written_examples_data_train_path = f'{HOME_DIRECTORY}/dataset/PLAN_BENCH/varient/openai_human_written_examples.json'
            with open(human_written_examples_data_train_path, 'r') as file:
                human_written_examples_data_list = json.load(file)
            human_written_examples_data_list = human_written_examples_data_list[:n_train]
            original_human_written_examples_data_list = copy.deepcopy(human_written_examples_data_list)

            provide_gpt4_style_example_data_train_path = f'{HOME_DIRECTORY}/dataset/PLAN_BENCH/varient/write_in_gpt4_style.json'
            with open(provide_gpt4_style_example_data_train_path, 'r') as file:
                provide_gpt4_style_example_data_list = json.load(file)
            provide_gpt4_style_example_data_list = provide_gpt4_style_example_data_list[:n_train]
            original_provide_gpt4_style_example_data_list = copy.deepcopy(provide_gpt4_style_example_data_list)
        
        if test_task_name.lower() == 'ecqa' or test_task_name.lower() == 'esnli':
            with open(redundant_data_train_path, 'r') as file:
                redundant_data_train_data_list = json.load(file)
            redundant_data_train_data_list = redundant_data_train_data_list[:n_train]
            original_redundant_data_train_data_list = copy.deepcopy(redundant_data_train_data_list)
        
        if test_task_name.lower() == 'math_algebra':
            test_task_name_item = 'MATH'
        else:
            test_task_name_item = test_task_name.upper()
        if test_task_name.lower() == 'api_bank' or test_task_name.lower() == 'plan_bench' or test_task_name.lower() == 'gsm8k' or test_task_name.lower() == 'math_algebra' or test_task_name.lower() == 'ecqa' or test_task_name.lower() == 'esnli':
            if test_task_name.lower() != 'api_bank' and test_task_name.lower() != 'plan_bench':
                anthropic_step_by_step_data_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name_item}/varient/anthropic_gpt4_generated_step_by_step_1000.json'
                with open(anthropic_step_by_step_data_train_path, 'r') as file:
                    anthropic_step_by_step_data_list = json.load(file)
                anthropic_step_by_step_data_list = anthropic_step_by_step_data_list[:n_train]

        if test_task_name.lower() != 'api_bank' and test_task_name.lower() != 'plan_bench':
            with open(minimum_change_train_path, 'r') as file:
                minimum_change_train_data_list = json.load(file)
            minimum_change_train_data_list = minimum_change_train_data_list[:n_train]
            original_minimum_change_train_data_list = copy.deepcopy(minimum_change_train_data_list)

        with open(gpt4_generated_data_train_path, 'r') as file:
            gpt4_generated_train_data_list = json.load(file)
        gpt4_generated_train_data_list = gpt4_generated_train_data_list[:n_train]

        with open(anthropic_generated_data_train_path, 'r') as file:
            anthropic_data_list = json.load(file)
        anthropic_data_list = anthropic_data_list[:n_train]

        with open(mini_gpt4_generated_data_train_path, 'r') as file:
            mini_gpt4_data_list = json.load(file)
        mini_gpt4_data_list = mini_gpt4_data_list[:n_train]

        with open(step_by_step_data_train_path, 'r') as file:
            step_by_step_data_list = json.load(file)
        step_by_step_data_list = step_by_step_data_list[:n_train]
        


        if 'plan_bench' in test_task_name.lower():
            test_task_name_item = test_task_name.upper()
            in_own_words_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name_item}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
            with open(in_own_words_train_path, 'r') as file:
                in_own_words_data_list = json.load(file)
            in_own_words_data_list = in_own_words_data_list[:n_train]
            original_in_own_words_data_list = copy.deepcopy(in_own_words_data_list)

        if 'boolq' not in test_task_name.lower() and 'api_bank' not in test_task_name.lower() and 'mmlu' not in test_task_name.lower() and 'winogrande' not in test_task_name.lower() and 'plan_bench' not in test_task_name.lower():
            in_own_words_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name_item}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
            with open(in_own_words_train_path, 'r') as file:
                in_own_words_data_list = json.load(file)
            in_own_words_data_list = in_own_words_data_list[:n_train]
            original_in_own_words_data_list = copy.deepcopy(in_own_words_data_list)

            anthropic_in_own_words_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name_item}/varient/anthropic_gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
            with open(anthropic_in_own_words_train_path, 'r') as file:
                anthropic_in_own_words_data_list = json.load(file)
            anthropic_in_own_words_data_list = anthropic_in_own_words_data_list[:n_train]


            step_by_step_gt_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name_item}/varient/step_by_step_while_keep_original_distribution.json'
            with open(step_by_step_gt_train_path, 'r') as file:
                step_by_step_gt_data_list = json.load(file)
            step_by_step_gt_data_list = step_by_step_gt_data_list[:n_train]
            original_step_by_step_gt_data_list = copy.deepcopy(step_by_step_gt_data_list)

            anthropic_step_by_step_gt_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name_item}/varient/anthropic_step_by_step_while_keep_original_distribution.json'
            with open(anthropic_step_by_step_gt_train_path, 'r') as file:
                anthropic_step_by_step_gt_data_list = json.load(file)
            anthropic_step_by_step_gt_data_list = anthropic_step_by_step_gt_data_list[:n_train]

            detailed_step_by_step_gt_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name_item}/varient/detailed_step_by_step_while_keep_original_distribution.json'
            with open(detailed_step_by_step_gt_train_path, 'r') as file:
                detailed_step_by_step_gt_data_list = json.load(file)
            detailed_step_by_step_gt_data_list = detailed_step_by_step_gt_data_list[:n_train]
            original_detailed_step_by_step_gt_data_list = copy.deepcopy(detailed_step_by_step_gt_data_list)

            anthropic_detailed_step_by_step_gt_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name_item}/varient/anthropic_detailed_step_by_step_while_keep_original_distribution.json'
            with open(anthropic_detailed_step_by_step_gt_train_path, 'r') as file:
                anthropic_detailed_step_by_step_gt_data_list = json.load(file)
            anthropic_detailed_step_by_step_gt_data_list = anthropic_detailed_step_by_step_gt_data_list[:n_train]


            gpt4_generated_direct_paraphrase_gpt4_answer_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name_item}/varient/gpt4_generated_direct_paraphrase_gpt4_answer.json'
            with open(gpt4_generated_direct_paraphrase_gpt4_answer_train_path, 'r') as file:
                gpt4_generated_direct_paraphrase_gpt4_answer_data_list = json.load(file)
            gpt4_generated_direct_paraphrase_gpt4_answer_data_list = gpt4_generated_direct_paraphrase_gpt4_answer_data_list[:n_train]
            original_gpt4_generated_direct_paraphrase_gpt4_answer_data_list = copy.deepcopy(gpt4_generated_direct_paraphrase_gpt4_answer_data_list)

            # anthropic_gpt4_generated_direct_paraphrase_gpt4_answer_train_path = f'{HOME_DIRECTORY}/dataset/{test_task_name_item}/varient/anthropic_gpt4_generated_direct_paraphrase_gpt4_answer.json'
            # with open(anthropic_gpt4_generated_direct_paraphrase_gpt4_answer_train_path, 'r') as file:
            #     anthropic_gpt4_generated_direct_paraphrase_gpt4_answer_data_list = json.load(file)
            # anthropic_gpt4_generated_direct_paraphrase_gpt4_answer_data_list = anthropic_gpt4_generated_direct_paraphrase_gpt4_answer_data_list[:n_train]

        if 'boolq' in test_task_name.lower() or 'esnli' in test_task_name.lower() and 'mmlu' not in test_task_name.lower() and 'winogrande' not in test_task_name.lower():
            with open(mini_gpt4_style_train_path, 'r') as file:
                mini_gpt4_style_data_list = json.load(file)
            mini_gpt4_style_data_list = mini_gpt4_style_data_list[:n_train]
            original_mini_gpt4_style_data_list = copy.deepcopy(mini_gpt4_style_data_list)
        
        original_train_data_list = copy.deepcopy(train_data_list)
        original_mini_gpt4_data_list = copy.deepcopy(mini_gpt4_data_list)
        original_anthropic_data_list = copy.deepcopy(anthropic_data_list)
        original_gpt4_generated_train_data_list = copy.deepcopy(gpt4_generated_train_data_list)
        original_step_by_step_data_list = copy.deepcopy(step_by_step_data_list)
        
        
        
        def generate_random_lists(num_lists, sample_num=3, sub_sample_num=10):
            random_lists = []
            for i in range(num_lists):
                # Create a value range excluding the current index 'i'
                value_range = [x for x in range(sub_sample_num) if x != i]
                random_list = random.sample(value_range, sample_num)
                random_lists.append(random_list)
            return random_lists
        
        if 'BOOLQ' in test_task_name.upper():
            sample_num = 6
        elif 'PLAN_BENCH' in test_task_name.upper() or 'API_BANK' in test_task_name.upper():
            sample_num = 1
        else:
            sample_num = 3
        sub_sample_num = 10
        
        sub_samples_num_list = generate_random_lists(n_train, sample_num = sample_num, sub_sample_num = sub_sample_num)
        
        # minimum_change_train_data_list_filtered = []
        if test_task_name.lower() != 'api_bank' and test_task_name.lower() != 'plan_bench':
            # for i, item in enumerate(minimum_change_train_data_list):
            #     prompt_style = 'minimum_change'
            #     question_item = item['question']
            #     original_question = item['question']
            #     # formated_question = in_context_learning_examples(question_item, original_question, prompt_style = prompt_style, task = test_task_name.lower(), enforce_prompt_style = '')

            #     correct = True

            #     if plot_record_mispredicted_samples:
            #         answer_temp = item['answer']
            #         previous_prediction = item['previous_prediction']
            #         answer_filtered = extract_after_last_occurrence(answer_temp, "Groundtruth:")
            #         answer_filtered = answer_filtered.strip('\'"')
            #         correct = eval_MATH_correctness(previous_prediction, answer_filtered)
            
            #     if use_in_context_learning:
            #         if not disable_incontext_learn:
            #             formated_question = random_select_in_context_learning_examples(question_item, original_question, prompt_style = prompt_style, task = test_task_name.lower(), sub_samples_num_list = sub_samples_num_list[i], end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc)
            #         else:
            #             formated_question = question_item
            #         minimum_change_train_data_list[i]['question'] = formated_question
            #     minimum_change_train_data_list[i]['original_question'] = original_question

            #     if not correct:
            #         item = minimum_change_train_data_list[i].copy()
            #         minimum_change_train_data_list_filtered.append(item)
            #         item_1 = minimum_change_train_data_list[i].copy()
            #         item_1['answer'] = previous_prediction
            #         minimum_change_train_data_list_filtered.append(item_1)
            # minimum_change_train_data_list = minimum_change_train_data_list[:n_train]


            minimum_change_train_data_list = process_data_list(minimum_change_train_data_list, prompt_style = 'minimum_change', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)

            # if plot_record_mispredicted_samples:
            #     minimum_change_train_data_list = minimum_change_train_data_list_filtered[:n_train]
            # else:
            #     minimum_change_train_data_list = minimum_change_train_data_list[:n_train]

        elif test_task_name.lower() == 'api_bank' or test_task_name.lower() == 'plan_bench':
            human_written_examples_data_list = process_data_list(human_written_examples_data_list, prompt_style = 'human_written_examples', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)

            provide_gpt4_style_example_data_list = process_data_list(provide_gpt4_style_example_data_list, prompt_style = 'provide_gpt4_style_example', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)
        else:
            a = 1
        

        if test_task_name.lower() != 'code' and test_task_name.lower() != 'mbpp':
            step_by_step_data_list = process_data_list(step_by_step_data_list, prompt_style = 'step_by_step', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)


        train_data_list = process_data_list(train_data_list, prompt_style = 'gt_style', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)
        
        gpt4_generated_train_data_list = process_data_list(gpt4_generated_train_data_list, prompt_style = 'gpt4', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)

        anthropic_data_list = process_data_list(anthropic_data_list, prompt_style = 'anthropic', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)

        mini_gpt4_data_list = process_data_list(mini_gpt4_data_list, prompt_style = 'mini_gpt4', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)

        if test_task_name.lower() == 'esnli' or test_task_name.lower() == 'boolq':
            mini_gpt4_style_data_list = process_data_list(mini_gpt4_style_data_list, prompt_style = 'mini_gpt4_style', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)

        if 'plan_bench' in test_task_name.lower():
            in_own_words_data_list = process_data_list(in_own_words_data_list, prompt_style = 'in_own_words', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)
            
        if 'boolq' not in test_task_name.lower() and 'api_bank' not in test_task_name.lower() and 'mmlu' not in test_task_name.lower() and 'winogrande' not in test_task_name.lower() and 'plan_bench' not in test_task_name.lower():

            in_own_words_data_list = process_data_list(in_own_words_data_list, prompt_style = 'in_own_words', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)

            gpt4_generated_direct_paraphrase_gpt4_answer_data_list = process_data_list(gpt4_generated_direct_paraphrase_gpt4_answer_data_list, prompt_style = 'paraphrase', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)

            anthropic_in_own_words_data_list = process_data_list(anthropic_in_own_words_data_list, prompt_style = 'anthropic_in_own_words', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)

            anthropic_step_by_step_data_list = process_data_list(anthropic_step_by_step_data_list, prompt_style = 'anthropic_step_by_step', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)

            anthropic_step_by_step_gt_data_list = process_data_list(anthropic_step_by_step_gt_data_list, prompt_style = 'anthropic_step_by_step_gt', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)

            anthropic_detailed_step_by_step_gt_data_list = process_data_list(anthropic_detailed_step_by_step_gt_data_list, prompt_style = 'anthropic_detailed_step_by_step_gt', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)

            # anthropic_gpt4_generated_direct_paraphrase_gpt4_answer_data_list = process_data_list(anthropic_gpt4_generated_direct_paraphrase_gpt4_answer_data_list, prompt_style = 'anthropic_paraphrase', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)

        if 'esnli' in test_task_name.lower() or 'ecqa' in test_task_name.lower():

            redundant_data_train_data_list = process_data_list(redundant_data_train_data_list, prompt_style = 'redundant', test_task_name = test_task_name.lower(),sub_samples_num_list = sub_samples_num_list, end_template = end_template, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc, use_in_context_learning = use_in_context_learning, disable_incontext_learn = disable_incontext_learn, n_train = n_train, test_idx = test_idx)

        


        current_task_name = test_task_name.lower()
        train_config, test_config, data_loader_config = set_config(current_task_name, False, 0, 0, model_name = 'mistral')
        if 'api_bank' in test_task_name.lower() or 'plan_bench' in test_task_name.lower():
            per_device_train_batch_size = 2
            gradient_accumulation_steps = 16
        else:
            per_device_train_batch_size = train_config['per_device_train_batch_size']
            per_device_train_batch_size = math.floor(per_device_train_batch_size / 2)
            gradient_accumulation_steps = train_config['gradient_accumulation_steps']
            gradient_accumulation_steps *= 2

        train_config['per_device_train_batch_size'] = per_device_train_batch_size
        train_config['gradient_accumulation_steps'] = gradient_accumulation_steps
        
        test_config['max_length'] = 4024
        test_config['max_input_length'] = 3000
        test_config['max_new_tokens'] = 1024
        test_config['per_device_eval_batch_size'] = 2

    if 'gsm8k' in test_task_name.lower():
        # dataset_list = [['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''], ['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['minimum change', minimum_change_train_data_list, minimum_change_train_path, original_minimum_change_train_data_list, ''], ['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''], ['in_own_words', in_own_words_data_list, in_own_words_train_path, original_in_own_words_data_list, ''], ['paraphrase', gpt4_generated_direct_paraphrase_gpt4_answer_data_list, gpt4_generated_direct_paraphrase_gpt4_answer_train_path, original_gpt4_generated_direct_paraphrase_gpt4_answer_data_list, ''],['anthropic', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, '']]


        dataset_list = [
    ['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''],
    ['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''],
    ['minimum change', minimum_change_train_data_list, minimum_change_train_path, original_minimum_change_train_data_list, ''],
    ['groundtruth', train_data_list, train_path, original_train_data_list, ''],
    ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''],
    ['in_own_words', in_own_words_data_list, in_own_words_train_path, original_in_own_words_data_list, ''],
    ['paraphrase', gpt4_generated_direct_paraphrase_gpt4_answer_data_list, gpt4_generated_direct_paraphrase_gpt4_answer_train_path, original_gpt4_generated_direct_paraphrase_gpt4_answer_data_list, ''],
    ['anthropic', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, '']]

        # ,['anthropic_in_own_words', anthropic_in_own_words_data_list, anthropic_in_own_words_train_path, original_in_own_words_data_list, ''], ['anthropic_step_by_step', anthropic_step_by_step_data_list, anthropic_step_by_step_data_train_path, original_step_by_step_data_list, ''], ['anthropic_step_by_step_gt', anthropic_step_by_step_gt_data_list, anthropic_step_by_step_gt_train_path, original_step_by_step_gt_data_list, ''],
        # ['anthropic_detailed_step_by_step_gt', anthropic_detailed_step_by_step_gt_data_list, anthropic_detailed_step_by_step_gt_train_path, original_detailed_step_by_step_gt_data_list, ''],
        # ['anthropic_paraphrase', anthropic_gpt4_generated_direct_paraphrase_gpt4_answer_data_list, anthropic_gpt4_generated_direct_paraphrase_gpt4_answer_train_path, original_gpt4_generated_direct_paraphrase_gpt4_answer_data_list, '']]
            
    if 'math_algebra' in test_task_name.lower():
        dataset_list = [['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''],['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['minimum change', minimum_change_train_data_list, minimum_change_train_path, original_in_own_words_data_list, ''], ['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''],  ['in_own_words', in_own_words_data_list, in_own_words_train_path, original_in_own_words_data_list, ''], ['paraphrase', gpt4_generated_direct_paraphrase_gpt4_answer_data_list, gpt4_generated_direct_paraphrase_gpt4_answer_train_path, original_gpt4_generated_direct_paraphrase_gpt4_answer_data_list, ''], ['anthropic', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, '']]#,

        # ['anthropic_in_own_words', anthropic_in_own_words_data_list, anthropic_in_own_words_train_path, original_in_own_words_data_list, ''], ['anthropic_step_by_step', anthropic_step_by_step_data_list, anthropic_step_by_step_data_train_path, original_step_by_step_data_list, ''], ['anthropic_step_by_step_gt', anthropic_step_by_step_gt_data_list, anthropic_step_by_step_gt_train_path, original_step_by_step_gt_data_list, ''],
        # ['anthropic_detailed_step_by_step_gt', anthropic_detailed_step_by_step_gt_data_list, anthropic_detailed_step_by_step_gt_train_path, original_detailed_step_by_step_gt_data_list, ''],
        # ['anthropic_paraphrase', anthropic_gpt4_generated_direct_paraphrase_gpt4_answer_data_list, anthropic_gpt4_generated_direct_paraphrase_gpt4_answer_train_path, original_gpt4_generated_direct_paraphrase_gpt4_answer_data_list, '']]

    if 'ecqa' in test_task_name.lower():
        dataset_list = [['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''], ['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''],['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['minimum change', minimum_change_train_data_list, minimum_change_train_path, original_minimum_change_train_data_list, ''], ['in_own_words', in_own_words_data_list, in_own_words_train_path, original_in_own_words_data_list, ''], ['paraphrase', gpt4_generated_direct_paraphrase_gpt4_answer_data_list, gpt4_generated_direct_paraphrase_gpt4_answer_train_path, original_gpt4_generated_direct_paraphrase_gpt4_answer_data_list, ''], ['anthropic', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, ''], ['redundant', redundant_data_train_data_list, redundant_data_train_path, original_redundant_data_train_data_list, '']]#,
        # ['anthropic_in_own_words', anthropic_in_own_words_data_list, anthropic_in_own_words_train_path, original_in_own_words_data_list, ''], ['anthropic_step_by_step', anthropic_step_by_step_data_list, anthropic_step_by_step_data_train_path, original_step_by_step_data_list, ''], ['anthropic_step_by_step_gt', anthropic_step_by_step_gt_data_list, anthropic_step_by_step_gt_train_path, original_step_by_step_gt_data_list, ''],
        # ['anthropic_detailed_step_by_step_gt', anthropic_detailed_step_by_step_gt_data_list, anthropic_detailed_step_by_step_gt_train_path, original_detailed_step_by_step_gt_data_list, ''],
        # ['anthropic_paraphrase', anthropic_gpt4_generated_direct_paraphrase_gpt4_answer_data_list, anthropic_gpt4_generated_direct_paraphrase_gpt4_answer_train_path, original_gpt4_generated_direct_paraphrase_gpt4_answer_data_list, '']]

    if 'boolq' in test_task_name.lower():
        dataset_list = [['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''],['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['minimum change', minimum_change_train_data_list, minimum_change_train_path, original_minimum_change_train_data_list, ''], ['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''], ['anthropic', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, ''], ['mini_gpt4_style', mini_gpt4_style_data_list, mini_gpt4_style_train_path, original_mini_gpt4_style_data_list, '']]

    if 'winogrande' in test_task_name.lower():
        dataset_list = [['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''], ['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['minimum change', minimum_change_train_data_list, minimum_change_train_path, original_minimum_change_train_data_list, ''], ['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''], ['anthropic', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, '']]
    
    if 'mmlu' in test_task_name.lower():
        dataset_list = [['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''], ['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['minimum change', minimum_change_train_data_list, minimum_change_train_path, original_minimum_change_train_data_list, ''], ['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''], ['anthropic', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, '']]
        
        # dataset_list = [['groundtruth', train_data_list, train_path, original_train_data_list, '']]
    if 'esnli' in test_task_name.lower():
        dataset_list = [['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''],['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['minimum change', minimum_change_train_data_list, minimum_change_train_path, original_minimum_change_train_data_list, ''], ['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''], ['anthropic', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, ''], ['in_own_words', in_own_words_data_list, in_own_words_train_path, original_in_own_words_data_list, ''], ['mini_gpt4_style', mini_gpt4_style_data_list, mini_gpt4_style_train_path, original_mini_gpt4_style_data_list, ''], ['redundant', redundant_data_train_data_list, redundant_data_train_path, original_redundant_data_train_data_list, '']]
        
    if 'api_bank' in test_task_name.lower():
        dataset_list = [['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''], ['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['provide_gpt4_style_example', provide_gpt4_style_example_data_list, provide_gpt4_style_example_data_train_path, original_provide_gpt4_style_example_data_list, ''], ['human_written_examples', human_written_examples_data_list, human_written_examples_data_train_path, original_human_written_examples_data_list, ''], ['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''], ['anthropic', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, '']]

    if 'mbpp' in test_task_name.lower():
        dataset_list = [['anthropic', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, '']]

    if 'code' in test_task_name.lower():
        dataset_list = [['anthropic', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, '']]
    
    if 'plan_bench' in test_task_name.lower():
        dataset_list = [['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, ''], ['groundtruth', train_data_list, train_path, original_train_data_list, ''], ['mini_gpt4', mini_gpt4_data_list, mini_gpt4_generated_data_train_path, original_mini_gpt4_data_list, ''], ['anthropic', anthropic_data_list, anthropic_generated_data_train_path, original_anthropic_data_list, ''], ['step_by_step', step_by_step_data_list, step_by_step_data_train_path, original_step_by_step_data_list, ''], ['provide_gpt4_style_example', provide_gpt4_style_example_data_list, provide_gpt4_style_example_data_train_path, original_provide_gpt4_style_example_data_list, ''], ['human_written_examples', human_written_examples_data_list, human_written_examples_data_train_path, original_human_written_examples_data_list, ''], ['in_own_words', in_own_words_data_list, in_own_words_train_path, original_in_own_words_data_list, '']]

    # dataset_list = [['gpt4', gpt4_generated_train_data_list, gpt4_generated_data_train_path, original_gpt4_generated_train_data_list, '']]
    return dataset_list, train_config, test_config, test_task_name, minimum_change_train_data_list


