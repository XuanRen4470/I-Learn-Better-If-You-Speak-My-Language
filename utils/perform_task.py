import sys
import os
import re
import torch
import gc

import torch
from utils.data_loader import *

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from config.config import MODEL_DIRECTORY
from evaluation.eval import *
from utils.log_writter import *
from utils.train import finetune, finetune_trainner, train_llama_factory, train_llama_alpaca
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



def ZERO_SHOT_EVALUATION(task_name, test_dataloader, test_list, output_folder_name, test_config, file_name):
    accuracy, cover_ratio = Evaluation(task_name, test_dataloader, test_list, test_config, output_folder_name, lora_path = '')
    gc.collect()
    torch.cuda.empty_cache()
    log_line = 'Zeroshot Evaluation for ' + task_name
    write_log(file_name, output_folder_name, log_line)
    log_line = 'Accuracy: ' + str(accuracy) + ' Cover Ratio: ' + str(cover_ratio) + ' for ' + task_name
    write_log(file_name, output_folder_name, log_line)
    return accuracy, cover_ratio




    # model = AutoModelForCausalLM.from_pretrained(
    #     f"{MODEL_DIRECTORY}/{test_config['model_name']}",
    #     torch_dtype=torch.float16,
    #     device_map = 'auto'
    # )
    # groundtruth_list = []
    # answer_list = []
    # for i in range(len(data_list)):
    #     question_list = data_list[i]['question']
    #     answer_list.append(data_list[i]['answer'])
    #     groundtruth_list.append(data_list[i]['numerical_final_answer'])
        
    # batch = tokenizer(question_list, padding=True, truncation=True, return_tensors='pt')

    # output_list = []
    # devices_list = range(test_config['device_num'])
    # with torch.cuda.amp.autocast():
    #     for step, batch in enumerate(test_dataloader):
    #         input_ids, data_too_long = batch
    #         if not data_too_long:
    #             input_ids = move_to_device(input_ids, devices_list = devices_list)

    #             with torch.cuda.amp.autocast():
    #                 outputs = model(input_ids=input_ids, max_new_tokens=test_config['max_length'])
    #                 output = outputs[0]
    #             if step % print_interval == 0:
    #                 print(f'Step {step_counter}/{num_training_steps}, Epoch: {epoch}, Loss: {loss.item()}')
    

    # with torch.no_grad():
    #     devices_list = range(test_config['device_num'])
    #     for step, batch in enumerate(test_dataloader):
    #         with torch.cuda.amp.autocast():
    #             output = model.generate(**batch, max_new_tokens=test_config['max_length'])
    #             for output_item in output:
    #                 output = tokenizer.decode(output_item, skip_special_tokens=True)
    #                 output_list.append(output)

    #         accuracy, cover_ratio = calc_accuracy_GSM8K(question_list, output_list, groundtruth_list, output_folder_name, 'finetune')
    #         log_line = 'finetune Evaluation for ' + task_name
    #         write_log('finetune_log', output_folder_name, log_line)
    #         log_line = 'Accuracy: ' + str(accuracy) + ' Cover Ratio: ' + str(cover_ratio) + ' for ' + task_name
    #         write_log('finetune_log', output_folder_name, log_line)

    #         if step % print_interval == 0:
    #             print(f'testing process {step_counter}/{num_testing_steps}')





    #     output = model.generate(**batch, max_new_tokens=test_config['max_length'])
    #     output = tokenizer.decode(output[0], skip_special_tokens=True)
    #     output_list.append(output)

    # accuracy, cover_ratio = calc_accuracy_GSM8K(question_list, output_list, groundtruth_list, output_folder_name, 'zeroshot')
    # log_line = 'Zeroshot Evaluation for ' + task_name
    # write_log('zeroshot_log', output_folder_name, log_line)
    # log_line = 'Accuracy: ' + str(accuracy) + ' Cover Ratio: ' + str(cover_ratio) + ' for ' + task_name
    # write_log('zeroshot_log', output_folder_name, log_line)
    # return accuracy

def FINE_TUNING_EVALUATION(task_name, train_data, test_data, output_folder_name, train_config, test_config, test_data_list, file_name, use_trainner = ''):
    torch.cuda.empty_cache()
    gc.collect()
    if use_trainner:
        model = finetune_trainner(train_data, output_folder_name, train_config)
    else:
        model = finetune(train_data, output_folder_name, train_config)
    seed_num = train_config['seed_num']
    lora_dir = f"{MODEL_DIRECTORY}/output/{output_folder_name}/{seed_num}"
    accuracy, cover_ratio = Evaluation(task_name, test_data, test_data_list, test_config, output_folder_name, lora_path = lora_dir, model = model)
    # print(torch.cuda.memory_summary())
    log_line = 'Finetune Evaluation for ' + task_name
    write_log(file_name, output_folder_name, log_line)
    log_line = 'Accuracy: ' + str(accuracy) + ' Cover Ratio: ' + str(cover_ratio) + ' for ' + task_name
    write_log(file_name, output_folder_name, log_line)
    return accuracy, model

def FINE_TUNING_EVALUATION_LLAMA_FACTORY(intermediate_train_file_name, intermediate_test_file_name_suffix, task_name, output_folder_name, train_config, test_config, test_data_list, file_name, test_task_name_list):
    torch.cuda.empty_cache()
    gc.collect()
    check_point_folder_name = train_llama_factory(intermediate_train_file_name, output_folder_name, train_config)
    test_config['seed_num'] = train_config['seed_num']
    for test_task_name in test_task_name_list:
        test_file_name = f'{test_task_name.upper()}{intermediate_test_file_name_suffix}'
        predict_list = do_predict_llama_factory(test_file_name, output_folder_name, test_task_name, test_config, check_point_folder_name = check_point_folder_name)
    
    question_list = []
    groundtruth_list = []
    for i in range(len(test_data_list)):
        question_list.append(test_data_list[i]['question'])
        groundtruth_list.append(test_data_list[i]['answer'])
    if task_name.lower() == 'gsm8k':
        accuracy, cover_ratio = calc_accuracy_GSM8K(question_list, predict_list, groundtruth_list, output_folder_name, 'finetune')
    if task_name.lower() == 'api_bank':
        accuracy, cover_ratio = calc_accuracy_API_BANK(question_list, predict_list, groundtruth_list, output_folder_name, 'finetune')
    log_line = 'Finetune Evaluation for ' + task_name
    write_log(file_name, output_folder_name, log_line)
    log_line = 'Accuracy: ' + str(accuracy) + ' Cover Ratio: ' + str(cover_ratio) + ' for ' + task_name
    write_log(file_name, output_folder_name, log_line)
    return accuracy


# def FINE_TUNING_EVALUATION_ALPACA(intermediate_train_file_name, intermediate_test_file_name_suffix, output_folder_name, train_config, test_config, GSM8K_test_data_list, API_BANK_test_data_list, file_name, test_task_name_list, data_loader_config = {}, task_name = ''):
#     torch.cuda.empty_cache()
#     gc.collect()
#     check_point_folder_name = train_llama_alpaca(intermediate_train_file_name, output_folder_name, train_config)
#     test_config['seed_num'] = train_config['seed_num']

#     for test_task_name in test_task_name_list:
#         if test_task_name.lower() == 'gsm8k':
#             test_data_list = GSM8K_test_data_list
#         if test_task_name.lower() == 'api_bank':
#             test_data_list = API_BANK_test_data_list
#         predict_list = do_predict_llama_alpaca(test_data_list, output_folder_name, test_task_name, test_config, check_point_folder_name = check_point_folder_name, data_loader_config = data_loader_config)

#         question_list = []
#         groundtruth_list = []
#         for i in range(len(test_data_list)):
#             question = test_data_list[i]['question']
#             question_list.append(question)
#             groundtruth_list.append(test_data_list[i]['answer'])
    
#         if test_task_name.lower() == 'gsm8k':
#             accuracy, cover_ratio = calc_accuracy_GSM8K(question_list, predict_list, groundtruth_list, output_folder_name, file_name, task_name = task_name)
#         if test_task_name.lower() == 'api_bank':
#             accuracy, cover_ratio = calc_accuracy_API_BANK(test_data_list, predict_list, output_folder_name, file_name)
#         log_line = 'Finetune Validation Evaluation for ' + test_task_name
#         write_log(file_name, output_folder_name, log_line)
#         num_train_epochs, learning_rate = train_config['num_train_epochs'], train_config['learning_rate']
#         log_line = f'Validation epoch_num: {num_train_epochs} Validation learning_rate: {learning_rate} Validation Accuracy: ' + str(accuracy) + ' Cover Ratio: ' + str(cover_ratio) + ' for ' + test_task_name
#         write_log(file_name, output_folder_name, log_line)
#     return accuracy, check_point_folder_name

def FINE_TUNING_ALPACA(intermediate_train_file_path, output_folder_name, train_config):
    check_point_folder = train_llama_alpaca(intermediate_train_file_path, output_folder_name, train_config)
    return check_point_folder

