import sys
import os
from transformers import AutoTokenizer

model_name = 'Llama-2-13b-chat-hf'
finetune_type = 'lora'
template = 'llama2_modified'
n_data_creation = 3

input_length = 1024
output_length = 1024

load_8bit = True

train_max_length = input_length + output_length
test_max_length = input_length + output_length


# per_device_eval_batch_size = 3
# per_device_train_batch_size = 6
# gradient_accumulation_steps = 2

per_device_eval_batch_size = 4
per_device_train_batch_size = 4#5 #6
gradient_accumulation_steps = 8#2
num_train_epochs = 5
lora_alpha = 8
warmup_steps = 10
r = 8
num_beams = 1
enable_sampling = True
drop_last = True
shuffle = False
use_trainner = False
use_llama_factory = False
use_alpaca = True

train_config = {}
train_config['model_name'] = model_name
train_config['finetune_type'] = finetune_type
train_config['max_length'] = train_max_length
# train_config['lr_scheduler_type'] = 'linear'
train_config['lr_scheduler_type'] = 'cosine'
train_config['warmup_steps'] = warmup_steps
train_config['per_device_eval_batch_size'] = per_device_eval_batch_size
train_config['per_device_train_batch_size'] = per_device_train_batch_size
train_config['gradient_accumulation_steps'] = gradient_accumulation_steps
train_config['num_train_epochs'] = num_train_epochs
train_config['lora_alpha'] = lora_alpha
train_config['template'] = template
train_config['r'] = r

test_config = {}
test_config['model_name'] = model_name
test_config['max_length'] = test_max_length
test_config['max_input_length'] = input_length
test_config['per_device_eval_batch_size'] = per_device_eval_batch_size
test_config['do_sample'] = enable_sampling
# test_config['top_p'] = None
# test_config['temperature'] = 0
test_config['template'] = template
test_config['finetuning_type'] = 'lora'
test_config['num_beams'] = num_beams
test_config['load_8bit'] = load_8bit


data_loader_config = {}
data_loader_config['batch_size'] = per_device_eval_batch_size
data_loader_config['shuffle'] = shuffle
data_loader_config['num_workers'] = 12
data_loader_config['pin_memory'] = True
data_loader_config['drop_last'] = drop_last
data_loader_config['input_length'] = input_length
data_loader_config['output_length'] = output_length




parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
MODEL_DIRECTORY = os.path.dirname(parent_dir) + '/model'
OUTPUT_RECORD_DIRECTORY = os.path.dirname(parent_dir) + '/output_record'
# PARENT_DIRECTORY = os.path.dirname(parent_dir)
LLAMA_FACTORY_DIRECTORY = os.path.dirname(parent_dir) + '/LLaMA-Factory-ACL-2025'
LLAMA_FACTORY_ALPACA = os.path.dirname(parent_dir) + '/alpaca-lora'
HOME_DIRECTORY = parent_dir
YOUR_API_KEY = os.getenv('GPT_API')
GPT_API = YOUR_API_KEY
# MODEL_ENGINE = 'gpt-4o-mini'#'gpt-4-1106-preview'
# MODEL_ENGINE = 'gpt-4-1106-preview'
# MODEL_ENGINE = 'gpt-4-turbo-2024-04-09'
# MINI_MODEL_ENGINE = 'gpt-4o-mini'


MINI_MODEL_ENGINE = 'gpt-4o-mini-2024-07-18'
MODEL_ENGINE = 'gpt-4o-2024-08-06'


IGNORE_INDEX = -100

tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_DIRECTORY}/{test_config['model_name']}", trust_remote_code=True, truncation_side = 'left')
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# tokenizer.pad_token='[PAD]'
# tokenizer.pad_token_id = -100

# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.pad_token_id = -100



tokenizer.padding_side = "left"  # Fix for fp16

