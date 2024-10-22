from config.config import train_config, test_config, data_loader_config


original_per_device_train_batch_size = train_config['per_device_train_batch_size']
original_gradient_accumulation_steps = train_config['gradient_accumulation_steps']

def set_config(current_task_name, device_num, seed_num, model_name = '', data_n_train = 1000000):

    if 'mistral' in model_name:
        multiplier = 2
    elif 'llama_3' in  model_name:
        multiplier = 1
    else:
        multiplier = 1

    gsm8k_max_input_length = 512
    gsm8k_max_output_length = 1024
    gsm8k_max_length = gsm8k_max_input_length + gsm8k_max_output_length
    gsm8k_per_device_eval_batch_size = 2 * multiplier

    math_max_input_length = 512
    math_max_output_length = 1024
    math_max_length = math_max_input_length + math_max_output_length
    math_per_device_eval_batch_size = 2 * multiplier

    esnli_max_input_length = 512
    esnli_max_output_length = 1024
    esnli_max_length = esnli_max_input_length + esnli_max_output_length
    esnli_per_device_eval_batch_size = 2 * multiplier

    ecqa_max_input_length = 512
    ecqa_max_output_length = 1024
    ecqa_max_length = ecqa_max_input_length + ecqa_max_output_length
    ecqa_per_device_eval_batch_size = 2 * multiplier

    api_bank_max_input_length = 1536
    api_bank_max_output_length = 1024
    api_bank_max_length = api_bank_max_input_length + api_bank_max_output_length
    api_bank_per_device_eval_batch_size = 2 * multiplier

    code_max_input_length = 512
    code_max_output_length = 1024
    code_max_length = code_max_input_length + code_max_output_length
    code_per_device_eval_batch_size = 2 * multiplier

    mbpp_max_input_length = 768
    mbpp_max_output_length = 1024
    mbpp_max_length = mbpp_max_input_length + mbpp_max_output_length
    mbpp_per_device_eval_batch_size = 2 * multiplier

    boolq_max_input_length = 512
    boolq_max_output_length = 1024
    boolq_max_length = boolq_max_input_length + boolq_max_output_length
    boolq_per_device_eval_batch_size = 2 * multiplier

    plan_bench_max_input_length = 2048
    plan_bench_max_output_length = 1024
    plan_bench_max_length = plan_bench_max_input_length + plan_bench_max_output_length
    plan_bench_per_device_eval_batch_size = 1 * multiplier

    train_config['device_num'] = device_num
    test_config['device_num'] = device_num
    train_config['seed_num'] = seed_num
    test_config['seed_num'] = seed_num

    if 'API_BANK' in current_task_name.upper():
        # test_config['max_length'] = api_bank_max_length
        test_config['max_new_tokens'] = api_bank_max_output_length
        test_config['max_input_length'] = api_bank_max_input_length
        test_config['per_device_eval_batch_size'] = api_bank_per_device_eval_batch_size

        data_loader_config['input_length'] = api_bank_max_input_length
        data_loader_config['output_length'] = api_bank_max_output_length

        train_config['max_length'] = api_bank_max_length
        train_config['per_device_eval_batch_size'] = api_bank_per_device_eval_batch_size
    if 'PLAN_BENCH' in current_task_name.upper():
        test_config['max_new_tokens'] = plan_bench_max_output_length
        test_config['max_input_length'] = plan_bench_max_input_length
        test_config['per_device_eval_batch_size'] = plan_bench_per_device_eval_batch_size

        data_loader_config['input_length'] = plan_bench_max_input_length
        data_loader_config['output_length'] = plan_bench_max_output_length

        train_config['max_length'] = plan_bench_max_length
        train_config['per_device_eval_batch_size'] = plan_bench_per_device_eval_batch_size
    elif 'GSM8K' in current_task_name.upper():
        # test_config['max_length'] = gsm8k_max_length
        test_config['max_new_tokens'] = gsm8k_max_output_length
        test_config['max_input_length'] = gsm8k_max_input_length
        test_config['per_device_eval_batch_size'] = gsm8k_per_device_eval_batch_size

        data_loader_config['input_length'] = gsm8k_max_input_length
        data_loader_config['output_length'] = gsm8k_max_output_length

        train_config['max_length'] = gsm8k_max_length
        train_config['per_device_eval_batch_size'] = gsm8k_per_device_eval_batch_size
   
    elif 'MATH' in current_task_name.upper():
        # test_config['max_length'] = math_max_length
        test_config['max_new_tokens'] = math_max_output_length
        test_config['max_input_length'] = math_max_input_length
        test_config['per_device_eval_batch_size'] = math_per_device_eval_batch_size

        data_loader_config['input_length'] = math_max_input_length
        data_loader_config['output_length'] = math_max_output_length

        train_config['max_length'] = math_max_length
        train_config['per_device_eval_batch_size'] = math_per_device_eval_batch_size
    elif 'ESNLI' in current_task_name.upper():
        test_config['max_new_tokens'] = esnli_max_output_length
        test_config['max_input_length'] = esnli_max_input_length
        test_config['per_device_eval_batch_size'] = esnli_per_device_eval_batch_size

        data_loader_config['input_length'] = esnli_max_input_length
        data_loader_config['output_length'] = esnli_max_output_length

        train_config['max_length'] = esnli_max_length
        train_config['per_device_eval_batch_size'] = esnli_per_device_eval_batch_size
  
    elif 'CODE' in current_task_name.upper():
        # test_config['max_length'] = code_max_length
        test_config['max_new_tokens'] = code_max_output_length
        test_config['max_input_length'] = code_max_input_length
        test_config['per_device_eval_batch_size'] = code_per_device_eval_batch_size

        data_loader_config['input_length'] = code_max_input_length
        data_loader_config['output_length'] = code_max_output_length

        train_config['max_length'] = code_max_length
        train_config['per_device_eval_batch_size'] = code_per_device_eval_batch_size
 
    elif 'MBPP' in current_task_name.upper():
        test_config['max_new_tokens'] = mbpp_max_output_length
        test_config['max_input_length'] = mbpp_max_input_length
        test_config['per_device_eval_batch_size'] = mbpp_per_device_eval_batch_size

        data_loader_config['input_length'] = mbpp_max_input_length
        data_loader_config['output_length'] = mbpp_max_output_length

        train_config['max_length'] = mbpp_max_length
        train_config['per_device_eval_batch_size'] = mbpp_per_device_eval_batch_size

    elif 'BOOLQ' in current_task_name.upper():
        test_config['max_new_tokens'] = boolq_max_output_length
        test_config['max_input_length'] = boolq_max_input_length
        test_config['per_device_eval_batch_size'] = boolq_per_device_eval_batch_size

        data_loader_config['input_length'] = boolq_max_input_length
        data_loader_config['output_length'] = boolq_max_output_length

        train_config['max_length'] = boolq_max_length
        train_config['per_device_eval_batch_size'] = boolq_per_device_eval_batch_size
  
    elif 'ECQA' in current_task_name.upper():
        test_config['max_new_tokens'] = ecqa_max_output_length
        test_config['max_input_length'] = ecqa_max_input_length
        test_config['per_device_eval_batch_size'] = ecqa_per_device_eval_batch_size

        data_loader_config['input_length'] = ecqa_max_input_length
        data_loader_config['output_length'] = ecqa_max_output_length

        train_config['max_length'] = ecqa_max_length
        train_config['per_device_eval_batch_size'] = ecqa_per_device_eval_batch_size

    if 'mistral' in model_name:
        train_config['model_name'] = 'Mistral-7b-Instruct-v2'
        # per_device_train_batch_size = train_config['per_device_train_batch_size']
        # gradient_accumulation_steps = train_config['gradient_accumulation_steps']
        per_device_train_batch_size_temp = original_per_device_train_batch_size
        gradient_accumulation_steps_temp = original_gradient_accumulation_steps
        per_device_train_batch_size_temp *= 2
        gradient_accumulation_steps_temp /= 2
        train_config['per_device_train_batch_size'] = int(per_device_train_batch_size_temp)
        train_config['gradient_accumulation_steps'] = int(gradient_accumulation_steps_temp)
        

        if 'CODE' in current_task_name.upper() or 'MBPP' in current_task_name.upper() or data_n_train < 301:
            train_config['per_device_train_batch_size'] = int(5)
            train_config['gradient_accumulation_steps'] = int(2)
        if current_task_name.upper() == 'BOOLQ':
            train_config['per_device_train_batch_size'] = int(4)
            train_config['gradient_accumulation_steps'] = int(8)

        train_config['template'] = 'mistral'

        test_config['model_name'] = 'Mistral-7b-Instruct-v2'
        test_config['template'] = 'mistral'

    if 'llama_3' in model_name:
        train_config['model_name'] = 'Meta-Llama-3-8B'
        # per_device_train_batch_size = train_config['per_device_train_batch_size']
        # gradient_accumulation_steps = train_config['gradient_accumulation_steps']
        per_device_train_batch_size_temp = original_per_device_train_batch_size
        gradient_accumulation_steps_temp = original_gradient_accumulation_steps
        per_device_train_batch_size_temp *= 2
        gradient_accumulation_steps_temp /= 2
        train_config['per_device_train_batch_size'] = int(per_device_train_batch_size_temp)
        train_config['gradient_accumulation_steps'] = int(gradient_accumulation_steps_temp)
        

        if 'CODE' in current_task_name.upper() or 'MBPP' in current_task_name.upper() or data_n_train < 301:
            train_config['per_device_train_batch_size'] = int(5)
            train_config['gradient_accumulation_steps'] = int(2)

        train_config['template'] = 'default'

        test_config['model_name'] = 'Meta-Llama-3-8B'
        test_config['template'] = 'default'
    
    if 'llama_3_instruct' in model_name:
        # train_config['model_name'] = 'Meta-Llama-3-8B-Instruct'
        train_config['model_name'] = 'Meta-Llama-3-8B-Instruct'
        # per_device_train_batch_size = train_config['per_device_train_batch_size']
        # gradient_accumulation_steps = train_config['gradient_accumulation_steps']
        per_device_train_batch_size_temp = original_per_device_train_batch_size
        gradient_accumulation_steps_temp = original_gradient_accumulation_steps
        per_device_train_batch_size_temp *= 2
        gradient_accumulation_steps_temp /= 2
        train_config['per_device_train_batch_size'] = int(per_device_train_batch_size_temp)
        train_config['gradient_accumulation_steps'] = int(gradient_accumulation_steps_temp)
        

        if 'CODE' in current_task_name.upper() or 'MBPP' in current_task_name.upper() or data_n_train < 301:
            train_config['per_device_train_batch_size'] = int(5)
            train_config['gradient_accumulation_steps'] = int(2)

        train_config['template'] = 'llama3'

        # test_config['model_name'] = 'Meta-Llama-3-8B-Instruct'
        test_config['model_name'] = 'Meta-Llama-3-8B-Instruct'
        test_config['template'] = 'llama3'

    train_config['per_device_train_batch_size'] = 3
    train_config['gradient_accumulation_steps'] = 10

    if 'plan_bench' in current_task_name.lower():
        train_config['per_device_train_batch_size'] = 2
        train_config['gradient_accumulation_steps'] = 15
    test_config['use_cache'] = True
    return train_config, test_config, data_loader_config