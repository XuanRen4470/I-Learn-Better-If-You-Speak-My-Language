import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config.config import HOME_DIRECTORY, OUTPUT_RECORD_DIRECTORY, MODEL_DIRECTORY, LLAMA_FACTORY_DIRECTORY

def initial_output_folder(output_folder_name, seed_num):
    if not os.path.exists(f"{HOME_DIRECTORY}/log/{output_folder_name}"):
        os.makedirs(f"{HOME_DIRECTORY}/log/{output_folder_name}")
    if not os.path.exists(f"{MODEL_DIRECTORY}/output/{output_folder_name}"):
        os.makedirs(f"{MODEL_DIRECTORY}/output/{output_folder_name}")
    if not os.path.exists(f"{HOME_DIRECTORY}/output/{output_folder_name}"):
        os.makedirs(f"{HOME_DIRECTORY}/output/{output_folder_name}")
    if not os.path.exists(f"{LLAMA_FACTORY_DIRECTORY}/intermediate_file"):
        os.makedirs(f"{LLAMA_FACTORY_DIRECTORY}/intermediate_file")
    if not os.path.exists(f'{MODEL_DIRECTORY}/output/{output_folder_name}/{seed_num}'):
        os.makedirs(f'{MODEL_DIRECTORY}/output/{output_folder_name}/{seed_num}')
    if not os.path.exists(f'{HOME_DIRECTORY}/output/{output_folder_name}/{seed_num}'):
        os.makedirs(f'{HOME_DIRECTORY}/output/{output_folder_name}/{seed_num}')
    if not os.path.exists(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results"):
        os.makedirs(f"{HOME_DIRECTORY}/output/{output_folder_name}/intermediate_results")
    
    
    
    if not os.path.exists(f"{OUTPUT_RECORD_DIRECTORY}/output/{output_folder_name}/intermediate_results"):
        os.makedirs(f"{OUTPUT_RECORD_DIRECTORY}/output/{output_folder_name}/intermediate_results")

    if not os.path.exists(f"{HOME_DIRECTORY}/code_eval/intermediate_file"):
        os.makedirs(f"{HOME_DIRECTORY}/code_eval/intermediate_file")
    
    if not os.path.exists(f"{HOME_DIRECTORY}/apps_eval/intermediate_file"):
        os.makedirs(f"{HOME_DIRECTORY}/apps_eval/intermediate_file")
    


os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'