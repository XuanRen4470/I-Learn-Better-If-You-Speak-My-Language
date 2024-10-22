import sys
import os
import re
import gc
import torch
import hashlib
import json
import shutil
import contextlib
import subprocess
from trl import SFTTrainer

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.llama_factory_data_file_processor import put_file_path_to_data_info
from config.config import MODEL_DIRECTORY, HOME_DIRECTORY, IGNORE_INDEX, LLAMA_FACTORY_ALPACA, tokenizer, train_max_length
from transformers import (
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)

from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, PeftModel
from torch.optim import AdamW
from utils.function import move_to_device, get_gpu_with_most_memory
# from alpaca.alpaca_train import alpaca_train


def finetune(train_dataloader, output_folder_name, train_config):
    clip_value = 1.0
    model_path = f"{MODEL_DIRECTORY}/{train_config['model_name']}"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map = 'auto'
    )
    
    # LoRA Config
    peft_config = LoraConfig(
        # lora_alpha=train_config['lora_alpha'],
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        lora_dropout=0.1,
        r=train_config['r'],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    
    # Setting up the optimizer
    # optimizer = AdamW(model.parameters(), lr=train_config['learning_rate'])
    optimizer = AdamW(model.parameters(), lr=0)

    # Total number of training steps
    num_training_steps_before_accumulation = len(train_dataloader) * train_config['num_train_epochs']
    num_training_steps_after_accumulation = num_training_steps_before_accumulation // train_config['gradient_accumulation_steps']

    # Set up the linear scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,  # Number of warmup steps
        num_training_steps=num_training_steps_after_accumulation
    )

    print_interval = num_training_steps_before_accumulation // 5
    model.train()  # Set the model to training mode
    model.print_trainable_parameters()
    # devices_list = range(train_config['device_num'])
    # devices_list = [train_config['device_num']]


    # Get the GPU with the most available memory
    gpu_with_most_memory = get_gpu_with_most_memory()
    device = torch.device(f'cuda:{gpu_with_most_memory}' if torch.cuda.is_available() else 'cpu')
    step = 1
    running_loss = 0.0
    for epoch in range(train_config['num_train_epochs']):
        for batch in train_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            data_too_long = batch['data_too_long']
            if not data_too_long:
                # input_ids = move_to_device(input_ids, devices_list = devices_list)
                # attention_mask = move_to_device(attention_mask, devices_list = devices_list)
                # labels = move_to_device(labels, devices_list = devices_list)

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                # with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss         
                running_loss += loss.item()       

                # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                optimizer.step()
                running_loss += loss.item()
                
                # # Backward pass and optimization
                # # loss = loss.to(labels.device)
                # loss.backward()

                # # Perform optimization step after accumulating gradients
                # # if (step) % train_config['gradient_accumulation_steps'] == 0 and step > 1:
                # optimizer.step()  # Update model parameters
                # scheduler.step()  # Update learning rate
                # optimizer.zero_grad()
                # if step == 0:
                #     print(f'Step {step}/{num_training_steps_before_accumulation}, Epoch: {epoch}, Loss: {loss.item()}')
                if step % print_interval == 0 and step > 1:
                    scheduler.step()
                    running_loss /= print_interval
                    print(f'Step {step}/{num_training_steps_before_accumulation}, Epoch: {epoch}, Loss: {running_loss}')
                    running_loss = 0.0
            step += 1
    model.save_pretrained(f"{MODEL_DIRECTORY}/output/{output_folder_name}/{train_config['seed_num']}")
    # del model
    # model = 0
    return model






def finetune_trainner(train_data, output_folder_name, train_config):
    model_path = f"{MODEL_DIRECTORY}/{train_config['model_name']}"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map = 'auto'
    )
    
    # LoRA Config
    peft_config = LoraConfig(
        # lora_alpha=train_config['lora_alpha'],
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        lora_dropout=0.1,
        r=train_config['r'],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.train()  # Set the model to training mode
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX
    )

    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=train_data,
    #     peft_config=peft_config,
    #     dataset_text_field="text",
    #     max_seq_length=1024,
    #     tokenizer=tokenizer,
    #     args=TrainingArguments(
    #         per_device_train_batch_size=train_config['per_device_train_batch_size'], 
    #         gradient_accumulation_steps=1,#train_config['gradient_accumulation_steps'],
    #         warmup_steps=0, 
    #         num_train_epochs=train_config['num_train_epochs'], 
    #         learning_rate=train_config['learning_rate'],
    #         fp16=True,
    #         logging_steps=1, 
    #         output_dir=f"{MODEL_DIRECTORY}/output/{output_folder_name}/{train_config['seed_num']}",
    #         lr_scheduler_type="linear"
    #     ),
    #     data_collator=data_collator
    #     # packing=script_args.packing,
    # )


    trainer = Trainer(
        model=model, 
        train_dataset=train_data,
        args=TrainingArguments(
            per_device_train_batch_size=train_config['per_device_train_batch_size'], 
            gradient_accumulation_steps=1,#train_config['gradient_accumulation_steps'],
            warmup_steps=0, 
            num_train_epochs=train_config['num_train_epochs'], 
            learning_rate=train_config['learning_rate'],
            # fp16=True,
            logging_steps=1, 
            output_dir=f"{MODEL_DIRECTORY}/output/{output_folder_name}/{train_config['seed_num']}",
            lr_scheduler_type="linear"
        ),
        data_collator=data_collator
    )
    trainer.train()

    # model.save_pretrained(f"{MODEL_DIRECTORY}/output/{output_folder_name}/{train_config['seed_num']}")
    return model


# the code below is related to llama_factory 

def temporary_working_directory(path):
        """
        Temporarily change the working directory.
        """
        original_path = os.getcwd()
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(original_path)


def train_llama_factory(train_data_path, output_folder_name, train_config, file_name, dpo_enable = False, merged_base_model_dir = '', data_name = '', LLAMA_FACTORY_DIRECTORY = '', check_point_folder_name = '', enable_perplexity_curriculum_learning_initialization = False):
    import os
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'  # Sets timeout to 2 seconds
    file_name = file_name.replace('_log', '')
    put_file_path_to_data_info(data_name, train_data_path, dpo_enable = dpo_enable, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY)
    
    seed = train_config['seed_num']

    output_folder_name = f'{MODEL_DIRECTORY}/output/{output_folder_name}/{seed}'
    model_path = f"{MODEL_DIRECTORY}/{train_config['model_name']}"
    # sys.path.append(f'{LLAMA_FACTORY_DIRECTORY}')
    # from src import train_bash
    
    # Construct the command
    if merged_base_model_dir == '':
        model_path = f"{MODEL_DIRECTORY}/{train_config['model_name']}"
    else:
        model_path = f"{merged_base_model_dir}"
    if not dpo_enable:
        stage = "sft"
        per_device_train_batch_size = train_config['per_device_train_batch_size']
        gradient_accumulation_steps = train_config['gradient_accumulation_steps'] 
    else:
        stage = "dpo"
        per_device_train_batch_size = train_config['per_device_train_batch_size'] // 2
        output_folder_name = output_folder_name + '/dpo'

        if not os.path.exists(f"{output_folder_name}"):
            os.makedirs(f"{output_folder_name}")

        gradient_accumulation_steps = train_config['gradient_accumulation_steps'] * 2
    
    # Check if the folder exists
    if os.path.exists(output_folder_name):
        # Remove all files in the folder
        for filename in os.listdir(output_folder_name):
            file_path = os.path.join(output_folder_name, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        # Create the folder if it does not exist
        os.makedirs(output_folder_name)

    if 'save_steps' in train_config: 
        save_steps = train_config['save_steps']
    else:
        save_steps = 50
    
    if train_config['device_num'] > 1:
        start_line = 'accelerate launch'
    else:
        start_line = 'python'

    cmd = [
        start_line,
        f"{LLAMA_FACTORY_DIRECTORY}/src/train_bash.py",
        "--do_train",
        "--stage", stage,
        "--model_name_or_path", model_path,
        "--dataset", data_name,
        "--template", train_config['template'],
        # "--r", train_config['r'],
        "--finetuning_type", train_config['finetune_type'],
        "--lora_target", "q_proj,v_proj",
        "--output_dir", output_folder_name,
        # "--overwrite_cache",
        "--max_length", str(train_config['max_length']),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--lr_scheduler_type", train_config['lr_scheduler_type'],
        "--logging_steps",  str(100),
        "--save_steps", str(save_steps),
        "--learning_rate", str(train_config['learning_rate']),
        "--num_train_epochs", str(train_config['num_train_epochs']),
        "--lora_rank", str(train_config['r']),
        "--overwrite_output_dir", str(True),
        "--plot_loss",
        "--fp16"
    ]

    if 'checkpoint_dir' in train_config:
        cmd += ["--checkpoint_dir", f"{LLAMA_FACTORY_DIRECTORY}/{train_config['checkpoint_dir']}"]
    if dpo_enable:
        if 'DPO_BETA' in train_config:
            cmd += ["--dpo_beta", str(train_config['DPO_BETA'])]
    
    if 'seed' in train_config:
        cmd += ["--seed", str(train_config['seed_num'])]
    if dpo_enable:
        cmd += ['--create_new_adapter']
    # else: 
    #     cmd += ['--overwrite_cache']
    cmd += ['--overwrite_cache']
    if check_point_folder_name != '':
        cmd += ['--adapter_name_or_path', check_point_folder_name]


    

    if enable_perplexity_curriculum_learning_initialization:
        cmd += ['--streaming']
        cmd += ['--buffer_size', str(1)]

    subprocess.run(" ".join(cmd), shell=True, cwd=LLAMA_FACTORY_DIRECTORY)
    check_point_folder_name = output_folder_name
    return check_point_folder_name


def pre_train_llama_factory(train_data_path, output_folder_name, train_config, file_name, dpo_enable = False, merged_base_model_dir = '', data_name = '', LLAMA_FACTORY_DIRECTORY = '', check_point_folder_name = '', enable_perplexity_curriculum_learning_initialization = False):
    import os
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'
    file_name = file_name.replace('_log', '')
    put_file_path_to_data_info(data_name, train_data_path, dpo_enable = dpo_enable, LLAMA_FACTORY_DIRECTORY = LLAMA_FACTORY_DIRECTORY, pre_train_stage = True)
    
    seed = train_config['seed_num']

    output_folder_name = f'{MODEL_DIRECTORY}/output/{output_folder_name}/{seed}'
    model_path = f"{MODEL_DIRECTORY}/{train_config['model_name']}"
    if merged_base_model_dir == '':
        model_path = f"{MODEL_DIRECTORY}/{train_config['model_name']}"
    else:
        model_path = f"{merged_base_model_dir}"
    stage = "sft"
    per_device_train_batch_size = train_config['per_device_train_batch_size']
    gradient_accumulation_steps = train_config['gradient_accumulation_steps'] 
    
    if os.path.exists(output_folder_name):
        for filename in os.listdir(output_folder_name):
            file_path = os.path.join(output_folder_name, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(output_folder_name)

    if 'save_steps' in train_config: 
        save_steps = train_config['save_steps']
    else:
        save_steps = 50
    
    if train_config['device_num'] > 1:
        start_line = 'accelerate launch'
    else:
        start_line = 'python'

    cmd = [
        start_line,
        f"{LLAMA_FACTORY_DIRECTORY}/src/train_bash.py",
        "--do_train",
        "--stage", 'pt',
        "--model_name_or_path", model_path,
        "--dataset", data_name,
        "--cutoff_len", str(train_config['max_length']),
        "--template", train_config['template'],
        "--finetuning_type", train_config['finetune_type'],
        "--lora_target", "q_proj,v_proj",
        "--output_dir", output_folder_name,
        "--max_length", str(train_config['max_length']),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--lr_scheduler_type", train_config['lr_scheduler_type'],
        "--logging_steps",  str(100),
        "--save_steps", str(save_steps),
        "--learning_rate", str(train_config['learning_rate']),
        "--num_train_epochs", str(train_config['num_train_epochs']),
        "--lora_rank", str(train_config['r']),
        "--overwrite_output_dir", str(True),
        "--plot_loss",
        "--fp16"
    ]

    if 'checkpoint_dir' in train_config:
        cmd += ["--checkpoint_dir", f"{LLAMA_FACTORY_DIRECTORY}/{train_config['checkpoint_dir']}"]
    if dpo_enable:
        if 'DPO_BETA' in train_config:
            cmd += ["--dpo_beta", str(train_config['DPO_BETA'])]
    
    if 'seed' in train_config:
        cmd += ["--seed", str(train_config['seed_num'])]
    if dpo_enable:
        cmd += ['--create_new_adapter']
    cmd += ['--overwrite_cache']
    if check_point_folder_name != '':
        cmd += ['--adapter_name_or_path', check_point_folder_name]

    if enable_perplexity_curriculum_learning_initialization:
        cmd += ['--streaming']
        cmd += ['--buffer_size', str(1)]

    subprocess.run(" ".join(cmd), shell=True, cwd=LLAMA_FACTORY_DIRECTORY)
    check_point_folder_name = output_folder_name
    return check_point_folder_name

def merge_lora_llama_factory(adapter_name_or_path, export_dir, train_config, LLAMA_FACTORY_DIRECTORY = '', merged_base_model_dir = ''):
    import os
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'  # Sets timeout to 2 seconds
    if merged_base_model_dir == '':
        model_path = f"{MODEL_DIRECTORY}/{train_config['model_name']}"
    else:
        model_path = f"{merged_base_model_dir}"
    
    cmd = [
        "python",
        f"{LLAMA_FACTORY_DIRECTORY}/src/export_model.py",
        "--model_name_or_path", model_path,
        "--template", train_config['template'],
        "--finetuning_type", train_config['finetune_type'],
        "--adapter_name_or_path", adapter_name_or_path,
        "--export_dir", export_dir,
        "--overwrite_cache",
        "--export_size", str(2),
        "--export_legacy_format", str(False)
    ]

    subprocess.run(" ".join(cmd), shell=True, cwd=LLAMA_FACTORY_DIRECTORY)
    return export_dir

def train_llama_alpaca(train_data_path, output_folder_name, train_config):
    import os
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2'  # Sets timeout to 2 seconds
    seed = train_config['seed_num']

    check_point_folder = f'{MODEL_DIRECTORY}/output/{output_folder_name}/{seed}'
    model_path = f"{MODEL_DIRECTORY}/{train_config['model_name']}"

    batch_size = train_config['per_device_train_batch_size'] * int(train_config['gradient_accumulation_steps'])
    alpaca_train(base_model = model_path, 
                 output_dir = check_point_folder, 
                 data_path = train_data_path, 
                #  lr_scheduler_type = train_config['lr_scheduler_type'],
                 warmup_steps = train_config['warmup_steps'],
                 batch_size = int(batch_size), 
                 micro_batch_size = int(train_config['per_device_train_batch_size']), 
                 num_epochs = int(train_config['num_train_epochs']), 
                 learning_rate = float(train_config['learning_rate']), 
                 cutoff_len = int(train_config['max_length']), 
                 val_set_size = 0, 
                 lora_r = int(train_config['r']), 
                 lora_alpha = 16, 
                 lora_dropout = 0.05, 
                 lora_target_modules = ['q_proj','v_proj'], 
                 train_on_inputs = True, 
                 group_by_length = True)

    return check_point_folder



