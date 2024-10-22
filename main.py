import os
import torch
import shutil
from utils.__init__ import *
from config.config import *
from config.modify_config_on_current_job import set_config
import argparse
import random
import gc
from utils.train import merge_lora_llama_factory
import matplotlib.pyplot as plt
import time

print(torch.__version__)

parser = argparse.ArgumentParser(description='train and evaluate')

# Add arguments
parser.add_argument('--file_suffix', type=str, required=True, help='Training method')
parser.add_argument('--train_task_name', type=str, required=True, help='Training task name')
parser.add_argument('--n_train', type=int, required=True, help='Number of training examples')
parser.add_argument('--n_eval', type=int, required=True, help='Number of evaluation examples')
parser.add_argument('--n_validation', type=int, required=True, help='Number of validation examples')
parser.add_argument('--seed_num', type=int, required=True, help='Seed number')
parser.add_argument('--zero_shot_evaluation', type=lambda x: (str(x).lower() == 'true'), default=False, help='Enable zero-shot evaluation')
parser.add_argument('--enable_sft', type=lambda x: (str(x).lower() == 'true'), default=False, help='only test on sft')
parser.add_argument('--enable_minimum_change', type=lambda x: (str(x).lower() == 'true'), default=False, help='only test on minimum change')
parser.add_argument('--enable_gpt4_gt', type=lambda x: (str(x).lower() == 'true'), default=False, help='only test on sft')
parser.add_argument('--sft_epoch', nargs='+', type=int, default=[10], help='')
parser.add_argument('--sft_lr', type=float, default=5e-5, help='')
parser.add_argument('--device_num', type=int, default=1, help='')
parser.add_argument('--num_of_sft_checkpoints', type=int, default=50, help='')
parser.add_argument('--lora_rank', type=int, required=False, default = 8, help='')
parser.add_argument('--model_type', type=str, required=False, default='llama2-13b')
parser.add_argument('--debug_mode', type=lambda x: (str(x).lower() == 'true'), default=False, help='')
parser.add_argument('--model_company', type=str, required=False, default='', help='')
parser.add_argument('--disable_final_eval', type=lambda x: (str(x).lower() == 'true'), default=False, help='')
parser.add_argument('--data_name', type=str, required=True, help='Training method')


# Parse arguments
args = parser.parse_args()

model_company = args.model_company

file_suffix = args.file_suffix
train_task_name = args.train_task_name
n_train = args.n_train
n_eval = args.n_eval
n_validation = args.n_validation
device_num = args.device_num
seed_num = args.seed_num
zero_shot_evaluation = args.zero_shot_evaluation
sft_epoch_list = args.sft_epoch
sft_lr = args.sft_lr
num_of_sft_checkpoints = args.num_of_sft_checkpoints
disable_final_eval = args.disable_final_eval
enable_sft = args.enable_sft
enable_minimum_change = args.enable_minimum_change
enable_gpt4_gt = args.enable_gpt4_gt
lora_rank = args.lora_rank
model_type = args.model_type
debug_mode = args.debug_mode
data_name = args.data_name

model_type_name = 'llama'
if 'mistral' in model_type:
    model_type = '_mistral'
    model_type_name = 'mistral'
elif 'llama_3_instruct' in model_type:
    model_type = '_llama_3_instruct'
    model_type_name = 'llama_3_instruct'
else:
    model_type = ''
    model_type_name = 'llama'

front_text = ''
if enable_sft:
    front_text += 'sft'
if enable_minimum_change:
    front_text += 'mc'
if enable_gpt4_gt:
    front_text += 'gpt4'

shorter_train_task_name = train_task_name.replace('math_algebra', 'math_al')


if model_company and enable_gpt4_gt:
    model_company += '_'

output_folder_name = f'{shorter_train_task_name}{model_company}{model_type_name}_{front_text}_{data_name}_{lora_rank}_{seed_num}_{file_suffix}_{n_train}_{n_validation}_{sft_epoch_list[0]}_{sft_epoch_list[-1]}_{sft_lr}_{num_of_sft_checkpoints}'
file_name = f'{model_type_name}_{front_text}_{data_name}_{lora_rank}_{file_suffix}_{seed_num}_{n_train}_{n_validation}_{sft_epoch_list[0]}_{sft_epoch_list[-1]}_{sft_lr}_{num_of_sft_checkpoints}_log'

if debug_mode:
    LLAMA_FACTORY_DIRECTORY_new = f"{LLAMA_FACTORY_DIRECTORY}-debug"
else:
    LLAMA_FACTORY_DIRECTORY_new = f"{LLAMA_FACTORY_DIRECTORY}_{model_type_name}_{front_text}_{data_name}_{shorter_train_task_name}_{file_suffix}_{seed_num}_{n_train}_{sft_epoch_list[0]}_{sft_epoch_list[-1]}"

    # Check if the destination directory exists, and if so, remove it
    if os.path.exists(LLAMA_FACTORY_DIRECTORY_new):
        shutil.rmtree(LLAMA_FACTORY_DIRECTORY_new)
        print(f"Existing directory {LLAMA_FACTORY_DIRECTORY_new} removed")
        time.sleep(10)
    # Copy the directory
    try:
        shutil.copytree(LLAMA_FACTORY_DIRECTORY, LLAMA_FACTORY_DIRECTORY_new)
    except:
        time.sleep(10)
        shutil.copytree(LLAMA_FACTORY_DIRECTORY, LLAMA_FACTORY_DIRECTORY_new)
    print(f"Directory copied successfully to {LLAMA_FACTORY_DIRECTORY_new}")
    time.sleep(2)

GSM8K_test_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/test.json'
MATH_algebra_test_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/test.json'

API_BANK_test_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/test/test-data_level-3-batch-inf.json'
variation_suffix_code = ''

if '_1' in data_name:
    variation_suffix_code = '_1'
if '_2' in data_name:
    variation_suffix_code = '_2'

if 'math' in train_task_name.lower():
    train_task_name = 'MATH'
    

CODE_test_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/test{variation_suffix_code}.json'
MBPP_test_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/test.json'
ESNLI_test_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/test.json'
ECQA_test_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/test.json'
PLAN_BENCH_test_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/test_plan_generation.json'
BOOLQ_test_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/test.json'

ECQA_validation_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/validation.json'
ESNLI_validation_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/validation.json'
MBPP_validation_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/validation.json'
PLAN_BENCH_validation_path = f'{HOME_DIRECTORY}/dataset/{train_task_name.upper()}/validation_plan_generation.json'

if train_task_name.upper() == 'API_BANK':
    test_task_name_list = ['api_bank']
elif train_task_name.upper() == 'GSM8K':
    test_task_name_list = ['gsm8k']
elif train_task_name.upper() == 'CODE':
    test_task_name_list = ['code']
elif train_task_name.upper() == 'MATH_ALGEBRA':
    test_task_name_list = ['math_algebra']
elif train_task_name.upper() == 'MBPP':
    test_task_name_list = ['mbpp']
elif train_task_name.upper() == 'ESNLI':
    test_task_name_list = ['esnli']
elif train_task_name.upper() == 'BOOLQ':
    test_task_name_list = ['boolq']
elif train_task_name.upper() == 'ECQA':
    test_task_name_list = ['ecqa']
elif train_task_name.upper() == 'PLAN_BENCH':
    test_task_name_list = ['plan_bench']


print('------------------------------------------------')

print('file_name', file_name)

print('data_name', data_name)

print('file_suffix', file_suffix)

print('train_task_name', train_task_name)

print('n_train', n_train)

print('n_eval', n_eval)

print('seed_num', seed_num)

print('sft_epoch_list', sft_epoch_list)

print('sft_lr', sft_lr)

print('num_of_sft_checkpoints', num_of_sft_checkpoints)

print('disable_final_eval', disable_final_eval)

print('enable_sft', enable_sft)

print('enable_minimum_change', enable_minimum_change)

print('enable_gpt4_gt', enable_gpt4_gt)

print('------------------------------------------------')

initial_output_folder(output_folder_name, seed_num)

with open(f"{HOME_DIRECTORY}/log/{output_folder_name}/log.txt", 'w') as f:
    pass
with open(f"{HOME_DIRECTORY}/log/{output_folder_name}/{file_name}.txt", 'w') as f:
    pass

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


train_data_list = []
minimum_change_train_data_list = []
gpt4_generated_train_data_list = []
paraphrased_data_train_list = []

intermediate_finetune_file_path, intermediate_minimum_change_train_file_path, intermediate_gpt4_generated_train_file_path = load_dataset(HOME_DIRECTORY, train_task_name, data_name, n_train)

training_method_list = []
if enable_minimum_change:
    training_method_list.append(['minimum_change', intermediate_minimum_change_train_file_path])
if enable_sft:
    training_method_list.append(['finetune', intermediate_finetune_file_path])
if enable_gpt4_gt:
    training_method_list.append(['gpt4_train', intermediate_gpt4_generated_train_file_path])

validation_mc_sft_accuracy_list = []
validation_mc_sft_step_list = []

validation_sft_sft_accuracy_list = []
validation_sft_sft_step_list = []

validation_gpt4_gt_sft_accuracy_list = []
validation_gpt4_gt_sft_step_list = []

for train_method, intermediate_sft_file_path in training_method_list:
    zeroshot = False

    intermediate_sft_file_path_list = [intermediate_sft_file_path]
        
    Best_lora_dir = ''
    SFT_Best_lora_dir = ''

    for intermediate_sft_file_path in intermediate_sft_file_path_list:
        stage_list = ['SFT']
        
        for current_stage in stage_list:
            epoch_list = sft_epoch_list
            max_accuracy = 0
            best_train_num = 0
            max_learning_rate = 0
            test_task_name = train_task_name
            task_name = 'validation'
            write_log(file_name, output_folder_name, f"""--------------------------{current_stage} Stage: {train_method} {task_name}--------------------------""")

            for epoch_num in epoch_list:
                enable_full_set = False
                log_file_item_path = f"{HOME_DIRECTORY}/log/{output_folder_name}/{train_method}_{current_stage}_Stage_{file_name}.txt"
                if os.path.exists(log_file_item_path):
                    with open(log_file_item_path, 'w') as f:
                        pass
                if num_of_sft_checkpoints == 0:
                    enable_full_set = True
                learning_rate = sft_lr
                save_chekpoints_num = num_of_sft_checkpoints
                intermediate_train_file_path = intermediate_sft_file_path
                
                with open(intermediate_train_file_path, 'r') as file:
                    full_data_set = json.load(file)
                full_data_set_length = len(full_data_set)            
                
                current_task_name = test_task_name.lower()                
                train_config, test_config, data_loader_config = set_config(current_task_name, device_num, seed_num, model_name = model_type, data_n_train = full_data_set_length)
                
                save_steps = int(full_data_set_length * epoch_num / save_chekpoints_num/ (train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']))
                train_config['save_steps'] = save_steps
                train_config['num_train_epochs'] = epoch_num
                train_config['learning_rate'] = learning_rate
                train_config['r'] = lora_rank

                
                if 'detail' in data_name or 'step' in data_name:
                    test_config['max_new_tokens'] = 1024
                    xxxxx = test_config['model_name']
                    if 'llama' in xxxxx.lower():
                        test_config['per_device_train_batch_size'] = 2


                train_config_curriculum_learning = train_config.copy()
                warmup_steps = int(full_data_set_length * epoch_num * 0.1/ (train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']))
                train_config['warmup_steps'] = warmup_steps
                batch_size = device_num * train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']
                if full_data_set_length < batch_size:
                    batch_size = full_data_set_length

                data_name = train_method + '_' + str(n_train) + '_' + train_task_name + data_name + '_train'
                
                check_point_folder = train_llama_factory(intermediate_train_file_path, output_folder_name, train_config, file_name, data_name = data_name, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY_new)
                
                
                checkpoints = extract_checkpoint_names(check_point_folder)
                if enable_full_set:
                    checkpoints = ['full_set']
                else:
                    checkpoints.append('full_set')
                
                checkpoints_temp = []
                for checkpoint in checkpoints:
                    if checkpoint != 'full_set':
                        numbers = re.findall(r'\d+', checkpoint)
                        checkpoint_num = int(numbers[0])
                        checkpoints_temp.append(checkpoint)
                    else:
                        checkpoints_temp.append(checkpoint)
                checkpoints = checkpoints_temp
                # Print sorted checkpoint names
                checkpoint_iteration = 0
                for checkpoint in checkpoints:
                    checkpoint_iteration += 1
                    if checkpoint != 'full_set':
                        numbers = re.findall(r'\d+', checkpoint)
                        checkpoint_num = int(numbers[0])
                        train_num = checkpoint_num * batch_size
                    else:
                        train_num = epoch_num * full_data_set_length
                        checkpoint_num = 99999

                    check_point_folder_temp = ''
                    if checkpoint != 'full_set':
                        check_point_folder_temp = check_point_folder + '/' + checkpoint
                    else:
                        check_point_folder_temp = check_point_folder
                
                    torch.cuda.empty_cache()
                    gc.collect()
                    minimum_change = False
                    if 'gpt4_train' in train_method or 'finetune' in train_method or 'sample_10_train' in train_method or 'paraphrased_data' in train_method or 'paraphrased_question_data' in train_method or 'given_answer_data' in train_method or 'proof_read_data' in train_method or 'enable_mix_gpt4_mc_data' in train_method:
                        minimum_change_or_zero_shot = False
                        minimum_change = False
                    elif 'minimum_change' in train_method:
                        minimum_change_or_zero_shot = True
                        minimum_change = True

                    if test_task_name.lower() == 'gsm8k':
                        test_data_list = load_GSM8K(GSM8K_test_path, n_validation, zeroshot = zeroshot)

                    if 'math_algebra' == test_task_name.lower():
                        test_data_list = load_MATH(MATH_algebra_test_path, n_validation, zeroshot = zeroshot)

                    if test_task_name.lower() == 'api_bank':
                        if minimum_change:
                            test_data_list = load_API_BANK_march_8_step1(API_BANK_test_path, n_validation)
                        else:
                            test_data_list = load_API_BANK_optimized(API_BANK_test_path, n_validation, minimum_change_or_zero_shot = minimum_change_or_zero_shot)
                    if test_task_name.lower() == 'esnli':
                        if 'finetune' in train_method:
                            test_data_list = load_ESNLI(ESNLI_validation_path, n_validation, finetune = True) 
                        else:
                            test_data_list = load_ESNLI(ESNLI_validation_path, n_validation) 
                        test_data_list = test_data_list[:1000]
                    if test_task_name.lower() == 'boolq':
                        if 'finetune' in train_method:
                            test_data_list = load_BOOLQ(BOOLQ_test_path, n_validation, finetune = True) 
                        else:
                            test_data_list = load_BOOLQ(BOOLQ_test_path, n_validation) 
                        test_data_list = test_data_list[:1000]
                    if test_task_name.lower() == 'ecqa':
                        if 'finetune' in train_method:
                            test_data_list = load_ECQA(ECQA_validation_path, n_validation, finetune = True, use_gt_rationale = True)
                        else:
                            test_data_list = load_ECQA(ECQA_validation_path, n_validation) 
                        test_data_list = test_data_list[:1000]
                    if test_task_name.lower() == 'code':
                        test_data_list = load_CODE_code_only(CODE_test_path, n_validation)
                        
                    if test_task_name.lower() == 'mbpp':
                        if train_task_name.lower() == 'mbpp':
                            if 'finetune' in train_method:
                                test_data_list = load_MBPP_code_only(MBPP_validation_path, n_validation)
                            else:
                                test_data_list = load_MBPP_code_only(MBPP_validation_path, n_validation)     
                        else:
                            test_data_list = load_MBPP_code_only(MBPP_validation_path, n_validation)

                    if train_task_name.lower() == 'plan_bench':
                        with open(PLAN_BENCH_validation_path, 'r') as f:
                            test_data_list = json.load(f)
                        test_data_list = test_data_list[:n_validation]
                        test_data_list = test_data_list[:100]
                    data_name = train_method + '_' + str(n_validation) + '_' + test_task_name.lower() +'_validation'

                    accuracy, cover_ratio = EVALUATION_LLAMA_FACTORY(test_data_list, test_task_name, test_config, output_folder_name, file_name, check_point_folder_name = check_point_folder_temp, data_loader_config = data_loader_config, task_name = task_name, train_method = train_method, checkpoint_num = checkpoint_num, data_name = data_name, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY_new, SFT_Best_lora_dir = SFT_Best_lora_dir)
                        
                    num_train_epochs, learning_rate = train_config['num_train_epochs'], train_config['learning_rate']
                    if checkpoint_num == 99999:
                        log_line = f'{task_name} train_num: {train_num} total_epoch_num: {num_train_epochs}, {task_name} learning_rate: {learning_rate}'
                    else:
                        log_line = f'{task_name} train_num: {train_num} checkpoint_iteration: {checkpoint_iteration} total_epoch_num: {num_train_epochs}, {task_name} learning_rate: {learning_rate}'
                    write_log(file_name, output_folder_name, log_line)

                    log_line = f'{accuracy}'
                    if accuracy > max_accuracy or max_accuracy == 0:
                        max_accuracy = accuracy
                        best_train_num = train_num
                        max_learning_rate = learning_rate
                        best_model_dir = f"{MODEL_DIRECTORY}/output/{output_folder_name}/{current_stage}_Stage_{train_config['seed_num']}_{train_method}_bestmodel"
                        if os.path.exists(best_model_dir):
                            shutil.rmtree(best_model_dir)
                            time.sleep(1)
                        shutil.copytree(check_point_folder_temp, best_model_dir)

                    torch.cuda.empty_cache()
                    gc.collect()
                        

            Best_lora_dir = f"{MODEL_DIRECTORY}/output/{output_folder_name}/{current_stage}_Stage_{train_config['seed_num']}_{train_method}_bestmodel"
            write_log(file_name, output_folder_name, f'{current_stage} Stage: Best validation best_train_num: {best_train_num} Best validation learning_rate: {max_learning_rate} Best validation accuracy: {max_accuracy}')

            if not disable_final_eval:
                write_log(file_name, output_folder_name, f"""

# --------------------------{current_stage} Stage: {train_method} Final Evaluation--------------------------""")
                zeroshot = False
                if 'finetune' in train_method:
                    minimum_change_or_zero_shot = False
                    minimum_change = False
                elif 'gpt4_train' in train_method:
                    minimum_change_or_zero_shot = False
                    minimum_change = False
                elif 'minimum_change' in train_method:
                    minimum_change_or_zero_shot = True
                    minimum_change = True
                elif 'paraphrased_data' in train_method:  
                    minimum_change_or_zero_shot = True
                    minimum_change = True
    
                task_name = f'best_model_evaluation_{current_stage}'
                for test_task_name in test_task_name_list:
                    if test_task_name.lower() == 'gsm8k':
                        test_data_list = load_GSM8K(GSM8K_test_path, n_eval, zeroshot = zeroshot)
                    if  test_task_name.lower() == 'math_algebra':
                        test_data_list = load_MATH(MATH_algebra_test_path, n_eval, zeroshot = zeroshot)
                    if test_task_name.lower() == 'plan_bench':
                        with open(PLAN_BENCH_test_path, 'r') as f:
                            test_data_list = json.load(f)
                        test_data_list = test_data_list[:200]
                    if test_task_name.lower() == 'api_bank':
                        if minimum_change:
                            test_data_list = load_API_BANK_march_8_step1(API_BANK_test_path, n_eval)
                        else:
                            test_data_list = load_API_BANK_optimized(API_BANK_test_path, n_eval, minimum_change_or_zero_shot = minimum_change_or_zero_shot)
                    if test_task_name.lower() == 'esnli':
                        if train_task_name.lower() == 'esnli':
                            if 'finetune' in train_method:
                                test_data_list = load_ESNLI(ESNLI_test_path, 1000, finetune = True)  
                            else:
                                test_data_list = load_ESNLI(ESNLI_test_path, 1000)        
                        else:
                            test_data_list = load_ESNLI(ESNLI_test_path, 1000) 
                    if test_task_name.lower() == 'boolq':
                        if train_task_name.lower() == 'boolq':
                            if 'finetune' in train_method:
                                test_data_list = load_BOOLQ(BOOLQ_test_path, 1000, finetune = True)  
                            else:
                                test_data_list = load_BOOLQ(BOOLQ_test_path, 1000)        
                        else:
                            test_data_list = load_BOOLQ(BOOLQ_test_path, 1000) 
                    if test_task_name.lower() == 'ecqa':
                        if train_task_name.lower() == 'ecqa':
                            if 'finetune' in train_method:
                                test_data_list = load_ECQA(ECQA_test_path, 1000, finetune = True, use_gt_rationale = True)
                            else:
                                test_data_list = load_ECQA(ECQA_test_path, 1000)        
                        else:
                            test_data_list = load_ECQA(ECQA_test_path, 1000) 
                    if test_task_name.lower() == 'code':
                        test_data_list = load_CODE_code_only(CODE_test_path, n_eval)
                    if test_task_name.lower() == 'mbpp':
                        if train_task_name.lower() == 'mbpp':
                            if 'finetune' in train_method:
                                test_data_list = load_MBPP_code_only(MBPP_test_path, n_eval)
                            else:
                                test_data_list = load_MBPP_code_only(MBPP_test_path, n_eval)
                        else:
                            test_data_list = load_MBPP_code_only(MBPP_test_path, n_eval)
                    if  test_task_name.lower() == 'math_algebra':
                        test_data_list = load_MATH(MATH_algebra_test_path, n_eval, zeroshot = zeroshot)
                    test_data_list = test_data_list[:1000]
                    current_task_name = test_task_name.lower()
                    
                    train_config, test_config, data_loader_config = set_config(current_task_name, False, device_num, seed_num, model_name = model_type)
                    data_name = current_task_name + '_full_' + train_method 

                    if 'detail' in data_name or 'step' in data_name:
                        test_config['max_new_tokens'] = 1024
                        xxxxx = test_config['model_name']
                        if 'llama' in xxxxx.lower():
                            test_config['per_device_train_batch_size'] = 2
                    accuracy, cover_ratio = EVALUATION_LLAMA_FACTORY(test_data_list, test_task_name, test_config, output_folder_name, file_name, check_point_folder_name = Best_lora_dir, data_loader_config = data_loader_config, task_name = task_name, data_name = data_name, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY_new, SFT_Best_lora_dir = SFT_Best_lora_dir)
            write_log(file_name, output_folder_name, f"""






""", accuracy = accuracy)

a = 1
