import sys
import os
import re
import torch
import subprocess
import json
import gc
# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from config.config import MODEL_DIRECTORY, HOME_DIRECTORY, GPT_API, MODEL_ENGINE, MINI_MODEL_ENGINE
from config.config import tokenizer
from transformers import (
    AutoModelForCausalLM
)

from peft import PeftModel
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def extract_gsm8k_num(text):
    # Compile the regular expression pattern
    pattern = re.compile(r'####\s*(-?\d+)')

    # Search for the pattern in the answer_param string
    match = pattern.search(text)
    # Extract and print the matched group if a match was found
    if match:
        final_answer = match.group(1)
        # print(f"The final answer is: {final_answer}")
    else:
        print("No match found.")
        final_answer = -99999999999
    final_answer = float(final_answer)
    return str(final_answer)

def extract_GSM8K_numerical_final_answer_using_regex(question_list, answer_list):
    numerical_final_answer_list = []
    d = []
    for i in range(len(answer_list)):
        temp = answer_list[i]

        final_answer = extract_gsm8k_num(temp)
        numerical_final_answer_list.append(final_answer)
        if final_answer != '-99999999999.0':
            item_temp = {}
            item_temp['question'] = question_list[i]
            item_temp['input'] = ''
            item_temp['answer'] = answer_list[i]
            item_temp['numerical_final_answer'] = final_answer
            d.append(item_temp)
        else:
            a = 1
    return d









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
    
def evaluate_expression(expr):
    if 'sqrt' in expr or '^' in expr or '(' in expr:
        return False
    try:
        # Handle LaTeX-style fractions and square roots
        expr = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
        expr = re.sub(r'\\dfrac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
        # expr = re.sub(r'\\sqrt\{(.*?)\}', r'(\1) ** 0.5', expr)
        expr = re.sub(r'\\(cfrac|dfrac|frac)\{(.*?)\}\{(.*?)\}', r'(\2) / (\3)', expr)

        # Evaluate the expression
        result = eval(expr)
        
        return float(result)
    except:
        return False


# def evaluate_expression(expr):
#     try:
#         if 'cfrac' in expr:
#             expr = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
#             expr = re.sub(r'\\dfrac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
#             # Convert LaTeX square roots to Python exponentiation
#             expr = re.sub(r'\\sqrt\{(.*?)\}', r'(\1) ** 0.5', expr)
#             expr = re.sub(r'\\(cfrac|dfrac|frac)\{(.*?)\}\{(.*?)\}', r'(\2) / (\3)', expr)

#         # Evaluate the expression
#         result = eval(expr)
        
#         return float(result)
#     except:
#          return False
    
def evaluate_expression_try_best(expr):
    try:
        # Handle LaTeX-style fractions and square roots
        expr = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
        expr = re.sub(r'\\dfrac\{(.*?)\}\{(.*?)\}', r'(\1) / (\2)', expr)
        expr = re.sub(r'\\sqrt\{(.*?)\}', r'(\1) ** 0.5', expr)
        expr = re.sub(r'\\(cfrac|dfrac|frac)\{(.*?)\}\{(.*?)\}', r'(\2) / (\3)', expr)

        # Evaluate the expression
        result = eval(expr)
        
        return float(result)
    except:
        return False

def extract_last_number(text):
    # New pattern to include LaTeX-style expressions
    pattern = r'(-?\d+(?:,\d{3})*(?:\.\d+)?%?|-?\d+\/-?\d+%?|-?\d+%?|\\frac\{-?\d+\}\{-?\d+\}|\\dfrac\{.*?\}\{.*?\}|\\cfrac\{.*?\}\{.*?\}|\\sqrt\{.*?\})'

    text, Found_boxed_content = extract_boxed_content(text)
    if Found_boxed_content:
        # if 'sqrt' in text or '^' in text or '(' in text:
        #     return -3333333333333
        if '\\frac' in text or '\\dfrac' in text or '\\cfrac' in text or '\\sqrt' in text:
            extracted_num = evaluate_expression_try_best(text)
            if not extracted_num:
                return -3333333333333 
            else:
                return extracted_num

    # Find all numbers and expressions in the string
    all_numbers = re.findall(pattern, text)

    # Process the last number or expression
    if all_numbers:
        number = all_numbers[-1]
        # Evaluate LaTeX-style expressions
        
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









    
def extract_MATH_numerical_final_answer_using_regex(question_list, answer_list):
    d = []
    for i in range(len(answer_list)):
        temp = answer_list[i]

        final_answer = extract_boxed_content(temp) 
        final_answer = evaluate_expression(final_answer)    

        item_temp = {}
        item_temp['question'] = question_list[i]
        item_temp['input'] = ''
        item_temp['answer'] = answer_list[i]
        item_temp['numerical_final_answer'] = final_answer
        d.append(item_temp)
    return d

def move_to_device(trainable_item, devices_list = []):
    success = False
    for device in devices_list:
        try:
            trainable_item = trainable_item.to(f'cuda:{device}')
            success = True
            break
        except:
            a = 1
    
    if success:
        return trainable_item
    else:
        return None


def prediction(test_data, test_list, test_config, lora_path = '', model = None):
    if not model:
        model_path = f"{MODEL_DIRECTORY}/{test_config['model_name']}"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map = 'auto'
        )

        if lora_path != '':
            model = PeftModel.from_pretrained(model, lora_path)        
    model.eval()

    groundtruth_list = []
    question_list = []
    for i in range(len(test_list)):
        question_list.append(test_list[i]['question'])
        groundtruth_list.append(test_list[i]['answer'])
        
    output_list = []
    gpu_with_most_memory = get_gpu_with_most_memory()
    device = torch.device(f'cuda:{gpu_with_most_memory}' if torch.cuda.is_available() else 'cpu')
    step = 1
    with torch.no_grad():
        # devices_list = range(test_config['device_num'])
        # devices_list = [test_config['device_num']]
        for batch in test_data:
            with torch.cuda.amp.autocast():
                input_ids = batch['input_ids']
                data_too_long = batch['data_too_long']
                # attention_mask = batch[1]

                input_ids = input_ids.to(device)
                # input_ids = move_to_device(input_ids, devices_list = devices_list)
                # attention_mask = move_to_device(attention_mask, devices_list = devices_list)

                # output = model.generate(
                #     input_ids, 
                #     max_length=test_config['max_length'],
                #     # num_beans=1,  # Number of beams
                #     # early_stopping=True
                #     do_sample=test_config['do_sample'],       # set to True to Enable sampling
                #     # top_p=test_config['top_p'],            # Nucleus sampling
                #     # temperature=test_config['temperature']       # Sampling temperature
                # )
                with torch.no_grad():
                    output = model.generate(
                        input_ids = input_ids
                    )
                for output_item in output:
                    output = tokenizer.decode(output_item, skip_special_tokens=True)
                    output_list.append(output)
                    print(output)
            step += 1
    del model
    del input_ids
    return output_list







def get_gpu_memory():
    # Run nvidia-smi command to get GPU details
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,nounits,noheader'])
    # Decode result and split into lines
    gpu_info = result.decode('utf-8').strip().split('\n')
    # Process each line to get total and used memory
    gpu_memory = [list(map(int, re.findall(r'\d+', info))) for info in gpu_info]
    # Calculate available memory
    gpu_memory = [(total, used, total - used) for total, used in gpu_memory]
    return gpu_memory

def get_gpu_with_most_memory():
    gpu_memory = get_gpu_memory()
    # Find the index (GPU number) of the GPU with the maximum available memory
    most_memory_gpu = max(enumerate(gpu_memory), key=lambda x: x[1][2])[0]
    return most_memory_gpu


def record_json_data_to_file(output_folder_name, test_data_list, test_config, test_task_name):
    test_json_file = []
    for i in range(len(test_data_list)):
        temp = {}
        temp['question'] = test_data_list[i]['question']
        temp['answer'] = test_data_list[i]['answer']
        test_json_file.append(temp)
    destination_file = f"{HOME_DIRECTORY}/output/{output_folder_name}/{test_config['seed_num']}/{test_task_name}.jsonl"

    with open(destination_file, 'w') as json_file:
        json.dump(test_json_file, json_file, indent=4)



def minimum_change_data_creation_GSM8K_learn_from_mistakes(data_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['pred']
        question = item['question']

        minimum_change_prompt = \
f"""
For the following math word problems, the original solutions may contain errors. Please identify the incorrect step in each solution, explain why it is incorrect, and provide the correct solution starting from that step.

Question: Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?
Original Solution:
{{
Step 1: Alexis spent a total of $30 + $46 + $38 + $11 + $18 = $143 on the clothes she has receipts for.
Step 2: She had a budget of $200, so she spent $200 - $143 = $57 on the shoes.
Step 3: The answer is $57.
}}
Incorrect Step: Step 2.
Explanation: The problem states that Alexis has $16 left from her budget, not that she spent her entire budget. So, to find the amount she spent on the shoes, we need to subtract the amount she has left from the total amount she spent on the other clothes.
Correct Solution:
{{
...
Step 2: Alexis has $16 left from her budget, so she spent $200 - $16 = $184 on all the clothes, including the shoes.
Step 3: To find the cost of the shoes, subtract the cost of the other clothes from the total amount spent: $184 - $143 = $41.
Step 4: The answer is 41.
}}

Question: A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?
Original Solution:
{{
Step 1: Let x be the number of people on the ship the monster ate in the first hundred years.
Step 2: Then the number of people on the second ship is 2x, the number of people on the third ship is 4x, and the number of people on the fourth ship is 8x.
Step 3: The total number of people eaten by the monster is x + 2x + 4x + 8x = 15x.
Step 4: We know that the monster ate 847 people over three hundred years, so we can set up the equation 15x = 847.
Step 5: Solving for x, we get x = 56.
Step 6: Therefore, there were 56 people on the ship the monster ate in the first hundred years.
Step 7: The answer is 56.
}}
Incorrect Step: Step 2.
Explanation: The original solution incorrectly assumes that there are four ships instead of three. Since the monster rises once every hundred years and we are considering a period of three hundred years, there should only be three ships.
Correct Solution:
{{
...
Step 2: Then the number of people on the second ship is 2x, and the number of people on the third ship is 4x.
Step 3: The total number of people eaten by the monster is x + 2x + 4x = 7x.
Step 4: We know that the monster ate 847 people over three hundred years, so we can set up the equation 7x = 847.
Step 5: Solving for x, we get x = 121.
Step 6: Therefore, there were 121 people on the ship the monster ate in the first hundred years.
Step 7: The answer is 121.
}}

Question: James creates a media empire. He creates a movie for $2000. Each DVD cost $6 to make. He sells it for 2.5 times that much. He sells 500 movies a day for 5 days a week. How much profit does he make in 20 weeks?
Original Solution:
{{
Step 1: The total cost of creating the movie is $2000 + ($6 x 500) = $5000.
Step 2: He sells each DVD for 2.5 x $6 = $15.
Step 3: Thus, he earns $15 x 500 = $7500 per day.
Step 4: In a week, he earns $7500 x 5 = $37,500.
Step 5: In 20 weeks, he earns $37,500 x 20 = $750,000.
Step 6: His profit is $750,000 - $5000 = $745,000.
Step 7: The answer is 745000.
}}
Incorrect Step: Step 1.
Explanation: The incorrect part of this step is the calculation of the total cost of creating the movie. The $2000 is a one-time cost for creating the movie, while the $6 cost per DVD should be multiplied by the total number of DVDs sold in 20 weeks, not just 500.
Correct Solution:
{{
Step 1: Calculate the total number of DVDs sold in 20 weeks: 500 DVDs/day * 5 days/week * 20 weeks = 50,000 DVDs.
Step 2: Calculate the total cost of creating the movie and producing the DVDs: $2000 (movie creation) + ($6/DVD * 50,000 DVDs) = $2000 + $300,000 = $302,000.
Step 3: Calculate the selling price of each DVD: 2.5 * $6 = $15.
Step 4: Calculate the total revenue from selling the DVDs: $15/DVD * 50,000 DVDs = $750,000.
Step 5: Calculate the profit: $750,000 (revenue) - $302,000 (cost) = $448,000.
Step 6: The answer is 448000.
}}

Question: It takes Roque two hours to walk to work and one hour to ride his bike to work. Roque walks to and from work three times a week and rides his bike to and from work twice a week. How many hours in total does he take to get to and from work a week with walking and biking?
Original Solution:
{{
Step 1: Roque walks to work and back three times a week, so he spends 2 hours walking to work and 2 hours walking back from work for a total of 2 + 2 = 4 hours of walking per day.
Step 2: He rides his bike to work and back twice a week, so he spends 1 hour riding to work and 1 hour riding back from work for a total of 1 + 1 = 2 hours of biking per day.
Step 3: Therefore, Roque spends a total of 4 hours of walking + 2 hours of biking = 6 hours to get to and from work a week.
Step 4: The answer is 6.
}}
Incorrect Step: Step 3.
Explanation: The incorrect step is Step 3 because it does not account for the number of days Roque walks and bikes to work. The total hours should be calculated by multiplying the hours spent walking and biking per day by the number of days he walks and bikes.
Correct Solution:
{{
...
Step 3: Roque spends a total of (4 hours of walking per day * 3 days) + (2 hours of biking per day * 2 days) = 12 hours of walking + 4 hours of biking = 16 hours to get to and from work a week.
Step 4: The answer is 16.
}}
Question: {question}
Original Solution:
{previous_prediction} """

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction
        if i%10 ==0:
            print(f'Completed {i} out of {len(data_list)}')
    return data_list

def minimum_change_data_creation_API_Bank_plan_only(API_Bank_data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    input_json_list = []
    for i in range(len(API_Bank_data_list)):
        question = API_Bank_data_list[i]['question']
        groundtruth = API_Bank_data_list[i]['answer']
        previous_prediction = predict_list[i]
        # if 'ToolSearcher' in groundtruth:
        #     answer = f"""{groundtruth}"""
        #     a = 1
        # else:
        prompt = \
f"""Please perform minimum change to correct the previous prediction if it is wrong. You only need to make the necessary modification such that the inference steps support the final prediction and the final prediction has to be placed at the end. You suppose to perform minimum change which means you are trying to modify as less as possible to make the ifnerence steps and the final prediction correct. You will be given a INSTRUCTION for generating API-Request, a GROUNDTRUTH which is the correct API-Request and the PERVIOUS PREDICTION which you need to perform minimum change on.


INSTRUCTION: {question}

GROUNDTRUTH: {groundtruth}

PERVIOUS PREDICTION: {previous_prediction}





Your task:
1. According to GROUNDTRUTH API-request and the INSTRUCTION, please perform minimum change to modify the analyzation process of the PERVIOUS PREDICTION to make it correct and support the groundtruth API-request. When you modify, please perform minimum change which means you are trying to maintain the original words and only change the part that is incorrect(this is important). The modified analyzation needs to support why the API-REQUEST is generated in such a way.
2. If the PERVIOUS PREDICTION contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. Do not explain how you modify the analyzation. do not write things like 'the PERVIOUS PREDICTION is wrong because') or say things like 'sure, i can hekp you with'. just show me the minimum changed analyzation results directly with api-request at the end(this is important).
"""
        # 3. If the answer answer first then explain or inference latter, put the final answer at the end and put the explaination/inference steps at the beginning. We wish to perform minimum change and also make sure the model inference then answer based on inference. 

        
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['instruction'] = API_Bank_data_list[i]['question']
        temp['input'] = ''
        temp['groundtruth'] = groundtruth
        temp['output'] = answer
        temp['previous_prediction'] = previous_prediction
        input_json_list.append(temp)
    return input_json_list


def minimum_change_data_creation_API_Bank_vanilla(API_Bank_data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    input_json_list = []
    for i in range(len(API_Bank_data_list)):
        question = API_Bank_data_list[i]['question']
        groundtruth = API_Bank_data_list[i]['answer']
        previous_prediction = predict_list[i]
        # if 'ToolSearcher' in groundtruth:
        #     answer = f"""{groundtruth}"""
        #     a = 1
        # else:
        prompt = \
f"""Given the instruction and the gold label, please perform minimum change to correct the previous prediction if it is wrong. The previous prediction contains analyzation process, which we also need to modify. You suppose to perform minimum change which means you are trying to modify as less as possible to make previous prediction correct.  


INSTRUCTION: {question}

GOLD LABEL: {groundtruth}

PERVIOUS PREDICTION: {previous_prediction}


Your task:
1. According to GOLD LABEL API-request and the INSTRUCTION, please perform minimum change to modify the PERVIOUS PREDICTION to make it correct. When you modify, please perform minimum change which means you are trying to maintain the original words and only change or delete the part that is incorrect(this is important). You cannot just answer the question without using any of the previous prediction words.
2. You need to modify the analyzation in the PERVIOUS PREDICTION if it is wrong.
3. If the PERVIOUS PREDICTION contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it.
4. Do not explain how you modify the PREDICTION. do not write things like 'the PERVIOUS PREDICTION is wrong because') or say things like 'sure, i can hekp you with'. just show me the minimum changed PREDICTION results directly.
"""
            
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['instruction'] = API_Bank_data_list[i]['question']
        temp['input'] = ''
        temp['groundtruth'] = groundtruth
        temp['output'] = answer
        temp['previous_prediction'] = previous_prediction
        input_json_list.append(temp)
    return input_json_list



def minimum_change_data_creation_API_Bank(API_Bank_data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    input_json_list = []
    for i in range(len(API_Bank_data_list)):
        question = API_Bank_data_list[i]['question']
        groundtruth = API_Bank_data_list[i]['answer']
        previous_prediction = predict_list[i]
        # if 'ToolSearcher' in groundtruth:
        #     answer = f"""{groundtruth}"""
        #     a = 1
        # else:
        prompt = \
f"""Please perform minimum change to correct the previous prediction if it is wrong. You only need to make the necessary modification such that the inference steps support the final prediction and the final prediction has to be placed at the end. You suppose to perform minimum change which means you are trying to modify as less as possible to make the ifnerence steps and the final prediction correct. You will be given a INSTRUCTION for generating API-Request, a GROUNDTRUTH which is the correct API-Request and the PERVIOUS PREDICTION which you need to perform minimum change on.


INSTRUCTION: {question}

GROUNDTRUTH: {groundtruth}

PERVIOUS PREDICTION: {previous_prediction}





Your task:
1. According to GROUNDTRUTH API-request and the INSTRUCTION, please perform minimum change to modify the analyzation process of the PERVIOUS PREDICTION to make it correct and support the groundtruth API-request. When you modify, please perform minimum change which means you are trying to maintain the original words and only change the part that is incorrect(this is important). The modified analyzation needs to support why the API-REQUEST is generated in such a way.
2. If the PERVIOUS PREDICTION contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. Do not explain how you modify the analyzation. do not write things like 'the PERVIOUS PREDICTION is wrong because') or say things like 'sure, i can hekp you with'. just show me the minimum changed analyzation results directly with api-request at the end(this is important).
"""
        # 3. If the answer answer first then explain or inference latter, put the final answer at the end and put the explaination/inference steps at the beginning. We wish to perform minimum change and also make sure the model inference then answer based on inference. 

        
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['instruction'] = API_Bank_data_list[i]['question']
        temp['input'] = ''
        temp['groundtruth'] = groundtruth
        temp['output'] = answer
        temp['previous_prediction'] = previous_prediction
        input_json_list.append(temp)
    return input_json_list

def minimum_change_data_creation_API_Bank_feb_8(API_Bank_data_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    input_json_list = []
    for i in range(len(API_Bank_data_list)):
        question = API_Bank_data_list[i]['question']
        groundtruth = API_Bank_data_list[i]['answer']
      
        example_1 = \
"""Sure, I'd be happy to help!

Based on the user's utterance, we can infer that they are looking for a cardiologist in Los Angeles for a check-up appointment. To satisfy the user's request, we need to generate an API request that can provide a list of cardiologists in Los Angeles.
Here's the next API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the available API descriptions:

[ToolSearcher(keywords='cardiologist Los Angeles check-up appointment')]"""
        example_2 = \
"""Based on the user's utterance and the previous API-request history, we can determine that the user is looking for a cardiologist in Los Angeles for a check-up appointment. The previous API-request found a tool called "HealthcareProviderAppointmentChecker" that can check the availability of appointments with healthcare providers.

To satisfy the user's request, we can use the "HealthcareProviderAppointmentChecker" tool to find available cardiologists in Los Angeles for the user's check-up appointment. We can use the following API request:

[HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]

This API request uses the "specialty" and "location" parameters to specify that we are looking for a cardiologist in Los Angeles. The "HealthcareProviderAppointmentChecker" tool will then return a list of available appointment slots with cardiologists in Los Angeles."""
        example_3 = \
"""Based on the user's utterance and the previous API requests, we can infer that the user is looking to book a wellness retreat in Bali, Indonesia. The previous API requests have already provided us with information about wellness retreats in Bali and the total cost calculator.

To satisfy the user's request, we can generate the next API request as follows:

API Request: [TotalCostCalculator(price=2000.0, num_of_people=2)]
Explanation:

The API name is TotalCostCalculator, which is the name of the API that was found in the previous API request response.
The key-value pairs are:
+ price: 2000.0 (the price of the wellness retreat found in the previous API request response)
+ num_of_people: 2 (the number of people for the retreat booking)"""
        example_4 = \
"""Based on the user's utterance and the previous API-request history, we can infer that the user wants to book a wellness retreat for two people in Bali, Indonesia.

To satisfy the user's request, we can use the information found in the previous API-request response, which provided the name of the retreat ("Zen Paradise Retreat") and its price ($2000.0). We can use this information to calculate the total cost for the retreat.

To do this, we can use the "ToolSearcher" API to find a tool that can help us calculate the total cost of the retreat. We can use the following API request:

[ToolSearcher(keywords='cost calculator for wellness retreats in Bali')]"""
        aaaa = """2. 
API-Request:"""
        if 'ToolSearcher' in groundtruth:
            if aaaa in question:
                # print(1111111111111)
                example = example_1
            else:
                # print(2222222222222)
                example = example_4
        else:
            if aaaa in question:
                # print(333333333333)
                example = example_3
            else:
                # print(444444444444)
                example = example_2
        prompt = \
f"""Question: {question}
Groundtruth: {groundtruth}
Example Answer template: {example}

Task and Requiremewnt:
1. You need to answer the question. We already give you the groundtruth. 
2. We give you an example answer template. Please imitate the way it answers. In other words, you will treat the example answer as a template and your job is to fit the groundtruth answer to the template. You should try to use as much words from the template as possible. 
3. Do not explain how you modify the given answer(template). do not write things like 'the we modify it this way because') or say things like 'sure, i can hekp you with'. just show me the modified answer results directly with api-request at the end(this is important).
4. You need to put API-Request at the end.
5. Please remember that you have to try your best to use the original words from gexample answer with minor modification. This is the most important criteria.
"""

        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        # print('QUESTION', question)
        # print('ANSWER', groundtruth)
        # print()
        # print()
        # print()
        # print()
        # print()
        # print()
        # print()

        temp = {}
        temp['instruction'] = API_Bank_data_list[i]['question']
        temp['input'] = ''
        temp['groundtruth'] = groundtruth
        temp['output'] = answer
        temp['example'] = example
        input_json_list.append(temp)
    return input_json_list
def minimum_change_data_creation_API_Bank_optimized_new(API_Bank_data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    input_json_list = []
    for i in range(len(API_Bank_data_list)):
        question = API_Bank_data_list[i]['question']
        groundtruth = API_Bank_data_list[i]['answer']
        previous_prediction = predict_list[i]
        # if 'ToolSearcher' in groundtruth:
        #     answer = f"""{groundtruth}"""
        #     a = 1
        # else:
        example_1_q = \
'\nThe current time is {time}.\n\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available APIs.\n\nUser\'s utterance: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\n\nAvailable API descriptions:\n{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\n\n\n1. The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be done next to satisfy the user? Please solve the rest of the problem step by step. \n2. Remember to generate the next API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] according to the API description. '
        example_2_q = \
'\nThe current time is {time}.\n\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available APIs.\n\nUser\'s utterance: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\n\nAvailable API descriptions:\n{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\n\n\n\nPrevious API-Request History:\n\nAPI-Request: [ToolSearcher(keywords=\'healthcare provider appointment availability checker\')]\nReceived API Response:{\'name\': \'HealthcareProviderAppointmentChecker\', \'description\': \'API for checking the availability of appointments with healthcare providers.\', \'input_parameters\': {\'specialty\': {\'type\': \'str\', \'description\': \'The specialty of the healthcare provider.\'}, \'location\': {\'type\': \'str\', \'description\': \'The city location.\'}}, \'output_parameters\': {\'appointments\': {\'type\': \'list\', \'description\': \'A list of available appointment slots.\'}}}\n\n\n\nAvailable information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response\n\n1. The previous API-request has help solve part of job specified in user\'s utterance. What should be done next to satisfy the user? Please solve the rest of the problem step by step. \n2. Remember to generate the next API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] according to the API description. '
        example_3_q = \
'\nThe current time is {time}.\n\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available APIs.\n\nUser\'s utterance: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\n\nAvailable API descriptions:\n{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\n\n\n\nPrevious API-Request History:\n\n1. \nAPI-Request: [ToolSearcher(keywords=\'healthcare provider appointment availability checker\')]\nReceived API Response:{\'name\': \'HealthcareProviderAppointmentChecker\', \'description\': \'API for checking the availability of appointments with healthcare providers.\', \'input_parameters\': {\'specialty\': {\'type\': \'str\', \'description\': \'The specialty of the healthcare provider.\'}, \'location\': {\'type\': \'str\', \'description\': \'The city location.\'}}, \'output_parameters\': {\'appointments\': {\'type\': \'list\', \'description\': \'A list of available appointment slots.\'}}}\n2. \nAPI-Request: [HealthcareProviderAppointmentChecker(specialty=\'cardiologist\', location=\'Los Angeles\')]\nReceived API Response:{\'appointments\': [\'2034-04-18 14:30:00\', \'2034-04-19 11:00:00\', \'2034-04-20 09:45:00\']}\n\n\n\nAvailable information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response\n\n1. The previous API-request has help solve part of job specified in user\'s utterance. What should be done next to satisfy the user? Please solve the rest of the problem step by step. \n2. Remember to generate the next API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] according to the API description. '
        predict_example_2 = \
"""Sure, I'd be happy to help!\n\nBased on the user's utterance and the previous API-request response, we know that the user is looking for a cardiologist in Los Angeles for a check-up appointment. We also have the information from the previous API-request response, which is related to healthcare provider appointment availability checker.\n\nStep 1: Use the previous API-request response to narrow down the search for a cardiologist in Los Angeles.\n\nNext API request: [ToolSearcher(keywords='cardiologist Los Angeles', input_parameters={'specialty': 'cardiology', 'location': 'Los Angeles'}]\n\nThis API request uses the 'ToolSearcher' API with the keywords 'cardiologist Los Angeles' to search for relevant tools in the library. The input parameters'specialty' and 'location' are set to 'cardiology' and 'Los Angeles' respectively, to narrow down the search to cardiologists in Los Angeles.\n\nStep 2: Use the previous API-request response to check the availability of appointments with the found cardiologists.\n\nNext API request: [HealthcareProviderAppointmentChecker(input_parameters={'specialty': 'cardiology', 'location': 'Los Angeles', 'appointment_slots': 'available'}]\n\nThis API request uses the 'HealthcareProviderAppointmentChecker' API with the input parameters'specialty', 'location', and 'appointment_slots' set to 'cardiology', 'Los Angeles', and 'available' respectively, to check the availability of appointments with cardiologists in Los Angeles.\n\nStep 3: Choose one of the available cardiologists and schedule an appointment.\n\nNext API request: [ScheduleAppointment(input_parameters={'appointment_date': '2034-04-15 11:00:00', 'appointment_slot': 'available_slot', 'healthcare_provider_id': 'cardiologist_id'}]\n\nThis API request uses the 'ScheduleAppointment' API with the input parameters 'appointment_date', 'appointment_slot', and 'healthcare_provider_id' set to '2034-04-15 11:00:00', 'available_slot', and 'cardiologist_id' respectively, to schedule an appointment with one of the available cardiologists in Los Angeles.\n\nTherefore, the next API request would be:\n\n[ToolSearcher(keywords='cardiologist Los Angeles', input_parameters={'specialty': 'cardiology', 'location': 'Los Angeles'}]\n\nThis API request uses the 'ToolSearcher' API with the keywords 'cardiologist Los Angeles' to search for relevant tools in the library. The input parameters'specialty' and 'location' are set to 'cardiology' and 'Los Angeles' respectively, to narrow down the search to cardiologists in Los Angeles.\n\nPlease note that the above steps are just one possible solution to satisfy the user's request, and the actual implementation may vary depending on the specific requirements and constraints of the task."""
#         prompt = \
# f"""Please perform minimum change to correct the previous prediction if it is wrong. You only need to make the necessary modification such that the inference steps support the final prediction and the final prediction has to be placed at the end. You suppose to perform minimum change which means you are trying to modify as less as possible to make the ifnerence steps and the final prediction correct. You will be given a INSTRUCTION for generating API-Request, a GROUNDTRUTH which is the correct API-Request and the PERVIOUS PREDICTION which you need to perform minimum change on.


# Example1: 
# Example1-QUESTION: {example_1_q}
# Example1-GROUNDTRUTH: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]
# Example1-PREVIOUS PREDICTION: Sure, I'd be happy to help! Based on the user's utterance and the available APIs, I can generate the next API request.\n\nFirst, let's identify the user's goals and requirements:\n\n* The user wants to find a cardiologist in Los Angeles.\n* The user wants to schedule a check-up appointment.\n\nNext, let's determine which API is most relevant to the user's goals:\n\n* Based on the user's utterance, the most relevant API is \"ToolSearcher\" because it allows us to search for relevant tools (in this case, cardiologists) based on keywords.\n\nNow, let's generate the next API request using the \"ToolSearcher\" API:\n\nAPI Request:\n\nToolSearcher(keywords=[\"cardiologist\", \"Los Angeles\", \"check-up appointment\"])\n\nThis API request uses the \"keywords\" parameter to search for cardiologists in Los Angeles based on the user's utterance. The response will include a list of the best match tools (i.e., cardiologists) that meet the user's criteria.\n\nTo satisfy the user's goals, we will need to iterate through the response from the \"ToolSearcher\" API and perform additional actions for each match, such as scheduling an appointment or providing contact information for the cardiologist.\n\nSteps to solve the rest of the problem:\n\n1. Iterate through the response from the \"ToolSearcher\" API and extract the best match cardiologists.\n2. For each match, perform additional actions such as scheduling an appointment or providing contact information.\n3. Once all matches have been processed, provide the user with the results, such as a list of available appointment times or contact information for the cardiologists.\n\nI hope this helps! Let me know if you have any further questions or if there's anything else I can assist you with.
# Example1-MODIFIED PREDICTION: Sure, I'd be happy to help! Based on the user's utterance and the available APIs, I can generate the next API request. First, let's identify the user's goals and requirements:\n\n* The user wants to find a cardiologist in Los Angeles.\n* The user wants to schedule a check-up appointment.\n\nNext, let's determine which API is most relevant to the user's goals:\n\n* To find a cardiologist in Los Angeles, the first step should be check the availability of the healthcare provider appointment. Thus we need to use ToolSearcher to seaarch for a tool for checking availability. The most relevant API is \"ToolSearcher\" and the keywords are \"healthcare provider appointment availability checker\"\n\nNow, let's generate the next API request using the \"ToolSearcher\" \n\nAPI-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]

# Example2: 
# Example2-QUESTION: {example_2_q}
# Example2-GROUNDTRUTH: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]
# Example2-PREVIOUS PREDICTION: {predict_example_2}
# Example2-MODIFIED PREDICTION: Sure, I'd be happy to help!\n\nBased on the user's utterance and the previous API-request response, we know that the user is looking for a cardiologist in Los Angeles for a check-up appointment. We also have the information from the previous API-request response, which is related to healthcare provider appointment availability checker.\n\nStep 1: Use the previous API-request response to narrow down the search for a cardiologist in Los Angeles.\n\nNext API request: [ToolSearcher(keywords='cardiologist Los Angeles', input_parameters={'specialty': 'cardiology', 'location': 'Los Angeles'}]\n\nThis API request uses the 'ToolSearcher' API with the keywords 'cardiologist Los Angeles' to search for relevant tools in the library. The input parameters'specialty' and 'location' are set to 'cardiology' and 'Los Angeles' respectively, to narrow down the search to cardiologists in Los Angeles.\n\nStep 2: Use the previous API-request response to check the availability of appointments with the found cardiologists.\n\nNext API request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]\n\nThis API request uses the 'HealthcareProviderAppointmentChecker' API with the input parameters 'specialty' and 'location' set to 'cardiologist' and 'Los Angeles' respectively, to check the availability of appointments with cardiologists in Los Angeles.\n\nTherefore, the next API request would be:\n\nAPI-request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]


# Example3: 
# Example2-QUESTION: {example_3_q}
# Example2-GROUNDTRUTH: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]
# Example2-PREVIOUS PREDICTION: Sure, I'd be happy to help! Based on the user's utterance and the previous API-request history, here's what we can do to satisfy the user's request:\n\n1. First, we can use the information found by the previous API-Request response to narrow down the search for a cardiologist in Los Angeles. Specifically, we can use the 'appointments' list returned by the HealthcareProviderAppointmentChecker API to find a cardiologist with available appointments.\n2. Next, we can use the ToolSearcher API to search for a cardiologist in Los Angeles based on the available appointments. We can pass in the 'appointments' list as the 'keywords' parameter to the ToolSearcher API, like this:\n\nAPI Request: [ToolSearcher(keywords=[appointments])]\n\nThis will return a list of cardiologists in Los Angeles with available appointments that match the user's search criteria.\n\n1. Finally, we can use the information returned by the ToolSearcher API to provide the user with a list of cardiologists in Los Angeles with available appointments. We can format the response in a way that's easy for the user to understand, such as a list of cardiologists with their contact information and available appointment times.\n\nHere's the complete API request and response for the user's utterance:\n\nAPI Request: [ToolSearcher(keywords=[appointments])]\n\nAPI Response: [Cardiologist(name='Dr. John Doe', specialty='cardiology', location='Los Angeles', available_appointments=['2034-04-18 14:30:00', '2034-04-19 11:00:00', '2034-04-20 09:45:00']), Cardiologist(name='Dr. Jane Smith', specialty='cardiology', location='Los Angeles', available_appointments=['2034-04-15 14:00:00', '2034-04-16 09:30:00', '2034-04-17 13:15:00']),...]\n\nI hope this helps! Let me know if you have any other questions or if there's anything else I can assist you with.
# Example2-MODIFIED PREDICTION: Sure, I'd be happy to help! Based on the user's utterance and the previous API-request history, here's what we have done to satisfy the user's request:\n\n1. First, we used the information found by the previous API-Request response to narrow down the search for a cardiologist in Los Angeles. Specifically, we used the 'appointments' list returned by the HealthcareProviderAppointmentChecker API to find a cardiologist with available appointments.\n2. Next, we used the ToolSearcher API to search for a cardiologist in Los Angeles based on the available appointments. We did pass in the 'appointments' list as the 'keywords' parameter to the ToolSearcher API, like this:\n\nAPI Request: [ToolSearcher(keywords=[healthcare provider appointment availability checker])]\n\nThis returned a list of cardiologists in Los Angeles with available appointments that match the user's search criteria.\n\n1. Finally, we used the information returned by the ToolSearcher API to provide the user with a list of cardiologists in Los Angeles with available appointments. After received a list of available cardiologists appointment slot in Los Angles, the next step is to enable the user to schedule one of these appointments. To do this, we need to find a tool that can handle appointment scheduling with the healthcare provider.\n\n3. Therefore, we use the ToolSearcher API to search for a relevant tool that can schedule an appointment with a cardiologist. We'll use the keyword 'healthcare provider appointment scheduler' to find this tool.\n\nHere's the API request that will accomplish this:\n\nAPI-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]







# Please perform minimum change to this problem
# INSTRUCTION: {question}
# GROUNDTRUTH: {groundtruth}
# PERVIOUS PREDICTION: {previous_prediction}
# MODIFIED PREDICTION: MINIMUM_CHANGED_PREDICTION_HERE

# Your task:
# 1. According to GROUNDTRUTH API-request and the INSTRUCTION, please perform minimum change to modify the analyzation process of the PERVIOUS PREDICTION to make it correct and support the groundtruth API-request. When you modify, please perform minimum change which means you are trying to maintain the original words and only change the part that is incorrect(this is important). The modified analyzation needs to support why the API-REQUEST is generated in such a way.
# 2. If the PERVIOUS PREDICTION contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
# 3. Do not explain how you modify the analyzation. do not write things like 'the PERVIOUS PREDICTION is wrong because') or say things like 'sure, i can hekp you with'. just show me the minimum changed analyzation results directly with api-request at the end(this is important).
# """
        prompt = \
f"""Please perform minimum change to correct the previous prediction if it is wrong. You only need to make the necessary modification such that the inference steps support the final prediction and the final prediction has to be placed at the end. You suppose to perform minimum change which means you are trying to modify as less as possible to make the ifnerence steps and the final prediction correct. You will be given a INSTRUCTION for generating API-Request, a GROUNDTRUTH which is the correct API-Request and the PERVIOUS PREDICTION which you need to perform minimum change on.

Example1: 
Example1-QUESTION: {example_1_q}
Example1-GROUNDTRUTH: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]
Example1-PREVIOUS PREDICTION: Sure, I'd be happy to help! Based on the user's utterance and the available APIs, I can generate the next API request.

First, let's identify the user's goals and requirements:

* The user wants to find a cardiologist in Los Angeles.
* The user wants to schedule a check-up appointment.

Next, let's determine which API is most relevant to the user's goals:

* Based on the user's utterance, the most relevant API is \"ToolSearcher\" because it allows us to search for relevant tools (in this case, cardiologists) based on keywords.

Now, let's generate the next API request using the \"ToolSearcher\" API:

API Request:

ToolSearcher(keywords=[\"cardiologist\", \"Los Angeles\", \"check-up appointment\"])

This API request uses the \"keywords\" parameter to search for cardiologists in Los Angeles based on the user's utterance. The response will include a list of the best match tools (i.e., cardiologists) that meet the user's criteria.

To satisfy the user's goals, we will need to iterate through the response from the \"ToolSearcher\" API and perform additional actions for each match, such as scheduling an appointment or providing contact information for the cardiologist.

Steps to solve the rest of the problem:

1. Iterate through the response from the \"ToolSearcher\" API and extract the best match cardiologists.
2. For each match, perform additional actions such as scheduling an appointment or providing contact information.
3. Once all matches have been processed, provide the user with the results, such as a list of available appointment times or contact information for the cardiologists.

I hope this helps! Let me know if you have any further questions or if there's anything else I can assist you with.
Example1-MODIFIED PREDICTION: Sure, I'd be happy to help! Based on the user's utterance and the available APIs, I can generate the next API request. First, let's identify the user's goals and requirements:

* The user wants to find a cardiologist in Los Angeles.
* The user wants to schedule a check-up appointment.

Next, let's determine which API is most relevant to the user's goals:

* To find a cardiologist in Los Angeles, the first step should be check the availability of the healthcare provider appointment. Thus we need to use ToolSearcher to seaarch for a tool for checking availability. The most relevant API is \"ToolSearcher\" and the keywords are \"healthcare provider appointment availability checker\"

Now, let's generate the next API request using the \"ToolSearcher\" 

API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]

Example2: 
Example2-QUESTION: {example_2_q}
Example2-GROUNDTRUTH: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]
Example2-PREVIOUS PREDICTION: {predict_example_2}
Example2-MODIFIED PREDICTION: Sure, I'd be happy to help!

Based on the user's utterance and the previous API-request response, we know that the user is looking for a cardiologist in Los Angeles for a check-up appointment. We also have the information from the previous API-request response, which is related to healthcare provider appointment availability checker.

Step 1: Use the previous API-request response to narrow down the search for a cardiologist in Los Angeles.

Next API request: [ToolSearcher(keywords='cardiologist Los Angeles', input_parameters={{'specialty': 'cardiology', 'location': 'Los Angeles'}})]

This API request uses the 'ToolSearcher' API with the keywords 'cardiologist Los Angeles' to search for relevant tools in the library. The input parameters'specialty' and 'location' are set to 'cardiology' and 'Los Angeles' respectively, to narrow down the search to cardiologists in Los Angeles.

Step 2: Use the previous API-request response to check the availability of appointments with the found cardiologists.

Next API request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]

This API request uses the 'HealthcareProviderAppointmentChecker' API with the input parameters 'specialty' and 'location' set to 'cardiologist' and 'Los Angeles' respectively, to check the availability of appointments with cardiologists in Los Angeles.

Therefore, the next API request would be:

API-request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]


Example3: 
Example2-QUESTION: {example_3_q}
Example2-GROUNDTRUTH: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]
Example2-PREVIOUS PREDICTION: Sure, I'd be happy to help! Based on the user's utterance and the previous API-request history, here's what we can do to satisfy the user's request:

1. First, we can use the information found by the previous API-Request response to narrow down the search for a cardiologist in Los Angeles. Specifically, we can use the 'appointments' list returned by the HealthcareProviderAppointmentChecker API to find a cardiologist with available appointments.
2. Next, we can use the ToolSearcher API to search for a cardiologist in Los Angeles based on the available appointments. We can pass in the 'appointments' list as the 'keywords' parameter to the ToolSearcher API, like this:

API Request: [ToolSearcher(keywords=[appointments])]

This will return a list of cardiologists in Los Angeles with available appointments that match the user's search criteria.

1. Finally, we can use the information returned by the ToolSearcher API to provide the user with a list of cardiologists in Los Angeles with available appointments. We can format the response in a way that's easy for the user to understand, such as a list of cardiologists with their contact information and available appointment times.

Here's the complete API request and response for the user's utterance:

API Request: [ToolSearcher(keywords=[appointments])]

API Response: [Cardiologist(name='Dr. John Doe', specialty='cardiology', location='Los Angeles', available_appointments=['2034-04-18 14:30:00', '2034-04-19 11:00:00', '2034-04-20 09:45:00']), Cardiologist(name='Dr. Jane Smith', specialty='cardiology', location='Los Angeles', available_appointments=['2034-04-15 14:00:00', '2034-04-16 09:30:00', '2034-04-17 13:15:00']),...]

I hope this helps! Let me know if you have any other questions or if there's anything else I can assist you with.
Example2-MODIFIED PREDICTION: Sure, I'd be happy to help! Based on the user's utterance and the previous API-request history, here's what we have done to satisfy the user's request:

1. First, we used the information found by the previous API-Request response to narrow down the search for a cardiologist in Los Angeles. Specifically, we used the 'appointments' list returned by the HealthcareProviderAppointmentChecker API to find a cardiologist with available appointments.
2. Next, we used the ToolSearcher API to search for a cardiologist in Los Angeles based on the available appointments. We did pass in the 'appointments' list as the 'keywords' parameter to the ToolSearcher API, like this:

API Request: [ToolSearcher(keywords=[healthcare provider appointment availability checker])]

This returned a list of cardiologists in Los Angeles with available appointments that match the user's search criteria.

1. Finally, we used the information returned by the ToolSearcher API to provide the user with a list of cardiologists in Los Angeles with available appointments. After received a list of available cardiologists appointment slot in Los Angles, the next step is to enable the user to schedule one of these appointments. To do this, we need to find a tool that can handle appointment scheduling with the healthcare provider.

3. Therefore, we use the ToolSearcher API to search for a relevant tool that can schedule an appointment with a cardiologist. We'll use the keyword 'healthcare provider appointment scheduler' to find this tool.

Here's the API request that will accomplish this:

API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]



Please perform minimum change to this problem
INSTRUCTION: {question}
GROUNDTRUTH: {groundtruth}
PERVIOUS PREDICTION: {previous_prediction}
MODIFIED PREDICTION: MINIMUM_CHANGED_PREDICTION_HERE

Your task:
1. According to GROUNDTRUTH API-request and the INSTRUCTION, please perform minimum change to modify the analyzation process of the PERVIOUS PREDICTION to make it correct and support the groundtruth API-request. When you modify, please perform minimum change which means you are trying to maintain the original words and only change the part that is incorrect(this is important). The modified analyzation needs to support why the API-REQUEST is generated in such a way.
2. If the PERVIOUS PREDICTION contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. Do not explain how you modify the analyzation. do not write things like 'the PERVIOUS PREDICTION is wrong because') or say things like 'sure, i can hekp you with'. just show me the minimum changed analyzation results directly with api-request at the end(this is important).
4. You need to put API-Request at the end.
5. Please remember that you have to try your best to use the original words since you are performing minimum change. This is the most important criteria.
"""

# Note the changes: 
# - Escaped the curly braces in the API request with input_parameters by using double braces {{ and }}.
# - Rest of the f-string remains the same.

        
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['instruction'] = API_Bank_data_list[i]['question']
        temp['input'] = ''
        temp['groundtruth'] = groundtruth
        temp['output'] = answer
        temp['previous_prediction'] = previous_prediction
        input_json_list.append(temp)
    return input_json_list


def minimum_change_data_creation_API_Bank_feb_26(API_Bank_data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    input_json_list = []
    for i in range(len(API_Bank_data_list)):
        question = API_Bank_data_list[i]['question']
        groundtruth = API_Bank_data_list[i]['answer']
        previous_prediction = predict_list[i]
        # if 'ToolSearcher' in groundtruth:
        #     answer = f"""{groundtruth}"""
        #     a = 1
        # else:
        example_1_q = \
"""\nThe current time is {time}.\n\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available APIs.\n\nUser\'s utterance: Calculate the total cost including the hospital stay and surgery fee for a heart transplant surgery in Los Angeles.\n\nAvailable API descriptions:\n{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\n\n\n1. The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be done next to satisfy the user? Please solve the rest of the problem step by step. \n2. Remember to generate the next API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] according to the API description. """
        example_2_q = \
"""\nThe current time is {time}.\n\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available APIs.\n\nUser\'s utterance: Calculate the total cost including the hospital stay and surgery fee for a heart transplant surgery in Los Angeles.\n\nAvailable API descriptions:\n{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\n\n\n\nPrevious API-Request History:\n\nAPI-Request: [ToolSearcher(keywords=\'get surgery cost\')]\nReceived API Response:{\'name\': \'GetSurgeryCost\', \'description\': \'API for querying the cost of a specific surgery.\', \'input_parameters\': {\'surgery\': {\'type\': \'str\', \'description\': \'The specific surgery to query.\'}, \'location\': {\'type\': str, \'description\': \'The city location.\'}}, \'output_parameters\': {\'total_cost\': {\'type\': \'float\', \'description\': \'The total cost of the specific surgery.\'}}}\n\n\n\nAvailable information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response\n\n1. The previous API-request has help solve part of job specified in user\'s utterance. What should be done next to satisfy the user? Please solve the rest of the problem step by step. \n2. Remember to generate the next API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] according to the API description. """
        example_3_q = \
"""\nThe current time is {time}.\n\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available APIs.\n\nUser\'s utterance: Calculate the total cost including the hospital stay and surgery fee for a heart transplant surgery in Los Angeles.\n\nAvailable API descriptions:\n{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\n\n\n\nPrevious API-Request History:\n\n1. \nAPI-Request: [ToolSearcher(keywords=\'get surgery cost\')]\nReceived API Response:{\'name\': \'GetSurgeryCost\', \'description\': \'API for querying the cost of a specific surgery.\', \'input_parameters\': {\'surgery\': {\'type\': \'str\', \'description\': \'The specific surgery to query.\'}, \'location\': {\'type\': str, \'description\': \'The city location.\'}}, \'output_parameters\': {\'total_cost\': {\'type\': \'float\', \'description\': \'The total cost of the specific surgery.\'}}}\n2. \nAPI-Request: [GetSurgeryCost(surgery=heart transplant, location=Los Angeles)]\nReceived API Response:{\'total_cost\': 500000}\n\n\n\nAvailable information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response\n\n1. The previous API-request has help solve part of job specified in user\'s utterance. What should be done next to satisfy the user? Please solve the rest of the problem step by step. \n2. Remember to generate the next API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] according to the API description. """


        prompt = \
f"""Please perform minimum change to correct the previous prediction if it is wrong. You only need to make the necessary modification such that the inference steps support the final prediction and the final prediction has to be placed at the end. You suppose to perform minimum change which means you are trying to modify as less as possible to make the ifnerence steps and the final prediction correct. You will be given a INSTRUCTION for generating API-Request, a GROUNDTRUTH which is the correct API-Request and the PERVIOUS PREDICTION which you need to perform minimum change on.

Example1: The previous prediction is completed wrong, thus try to rewrite without change too much of the text styles.
Example1-QUESTION: {example_1_q}
Example1-GROUNDTRUTH: API-Request: [ToolSearcher(keywords='get surgery cost')]
Example1-PREVIOUS PREDICTION: This task requires us to generate an API request for a heart transplant surgery in Los Angeles. There is no API description that matches the user's request, so we cannot generate an API request that satisfies the user's needs.\n\nAPI requests are generated based on keywords, and there is no mention of heart transplant surgery or Los Angeles in any of the available API descriptions. Therefore, we cannot generate an API request that satisfies the user's needs.
Example1-MODIFIED PREDICTION: This task requires us to generate an API request for calculating the total cost including the hospital stay and surgery fee for a heart transplant surgery. The available API, "ToolSearcher", searches for relevant tools based on keywords. Therefore, we can generate an API request that aligns with the user\'s needs by using appropriate keywords that match the user\'s request to find the surgery cost.\n\nAPI-Request: [ToolSearcher(keywords=\'get surgery cost\')]

Example2: The previous prediction is correct. thus keep the original predictions.(we are trying to perfrom minimum change, thus we will try our best to keep the original words)
Example2-QUESTION: {example_2_q}
Example2-GROUNDTRUTH: API-Request: [GetSurgeryCost(surgery=heart transplant, location=Los Angeles)]
Example2-PREVIOUS PREDICTION: Based on the user's utterance, the problem to be solved is calculating the total cost of a heart transplant surgery in Los Angeles. The previous API request has provided an API for querying the cost of a specific surgery, which is GetSurgeryCost. Therefore, the next step is to use the GetSurgeryCost API with the specific surgery being a heart transplant and the location being Los Angeles.\n\nThe correct API request is: [GetSurgeryCost(surgery='heart transplant', location='Los Angeles')].
Example2-MODIFIED PREDICTION: "Based on the user's utterance, the problem to be solved is calculating the total cost of a heart transplant surgery in Los Angeles. The previous API request has provided an API for querying the cost of a specific surgery, which is GetSurgeryCost. Therefore, the next step is to use the GetSurgeryCost API with the specific surgery being a heart transplant and the location being Los Angeles.\n\nThe correct API request is: [GetSurgeryCost(surgery='heart transplant', location='Los Angeles')]."

Example3: 
Example2-QUESTION: {example_3_q}
Example2-GROUNDTRUTH: API-Request: [ToolSearcher(keywords='daily hospital cost')]
Example2-PREVIOUS PREDICTION: Based on the previous API response, we know that the cost of a heart transplant surgery in Los Angeles is $500,000.00. Therefore, the API request should be: [GetSurgeryCost(surgery='heart transplant', location='Los Angeles')]
Example2-MODIFIED PREDICTION: 


Please perform minimum change to this problem
INSTRUCTION: {question}
GROUNDTRUTH: {groundtruth}
PERVIOUS PREDICTION: {previous_prediction}
MODIFIED PREDICTION: MINIMUM_CHANGED_PREDICTION_HERE

Your task:
1. According to GROUNDTRUTH API-request and the INSTRUCTION, please perform minimum change to modify the analyzation process of the PERVIOUS PREDICTION to make it correct and support the groundtruth API-request. When you modify, please perform minimum change which means you are trying to maintain the original words and only change the part that is incorrect(this is important). The modified analyzation needs to support why the API-REQUEST is generated in such a way.
2. If the PERVIOUS PREDICTION contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. Do not explain how you modify the analyzation. do not write things like 'the PERVIOUS PREDICTION is wrong because') or say things like 'sure, i can hekp you with'. just show me the minimum changed analyzation results directly with api-request at the end(this is important).
4. You need to put API-Request at the end.
5. Please remember that you have to try your best to use the original words since you are performing minimum change. This is the most important criteria.
"""


        prompt = \
f"""Please perform minimum change to correct the previous prediction if it is wrong. You only need to make the necessary modification such that the inference steps support the final prediction and the final prediction has to be placed at the end. You suppose to perform minimum change which means you are trying to modify as less as possible to make the ifnerence steps and the final prediction correct. You will be given a INSTRUCTION for generating API-Request, a GROUNDTRUTH which is the correct API-Request and the PERVIOUS PREDICTION which you need to perform minimum change on.

Please perform minimum change to this problem
INSTRUCTION: {question}
GROUNDTRUTH: {groundtruth}
PERVIOUS PREDICTION: {previous_prediction}
MODIFIED PREDICTION: MINIMUM_CHANGED_PREDICTION_HERE

Your task:
1. According to GROUNDTRUTH API-request and the INSTRUCTION, please perform minimum change to modify the analyzation process of the PERVIOUS PREDICTION to make it correct and support the groundtruth API-request. When you modify, please perform minimum change which means you are trying to maintain the original words and only change the part that is incorrect(this is important). The modified analyzation needs to support why the API-REQUEST is generated in such a way.
2. If the PERVIOUS PREDICTION contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. Do not explain how you modify the analyzation. do not write things like 'the PERVIOUS PREDICTION is wrong because') or say things like 'sure, i can hekp you with'. just show me the minimum changed analyzation results directly with api-request at the end(this is important).
4. You need to put API-Request at the end.
5. Please remember that you have to try your best to use the original words since you are performing minimum change. This is the most important criteria.
"""
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['instruction'] = API_Bank_data_list[i]['question']
        temp['input'] = ''
        temp['groundtruth'] = groundtruth
        temp['output'] = answer
        temp['previous_prediction'] = previous_prediction
        input_json_list.append(temp)
    return input_json_list





def minimum_change_data_creation_API_Bank_mar_1(API_Bank_data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    input_json_list = []
    for i in range(len(API_Bank_data_list)):
        question = API_Bank_data_list[i]['question']

        substring = '\n2. We already know the answer is'

        question = question.replace('Please solve the rest of the problem.', '')
        
        # Find the index where the answer appears in the prediction
        index_of_answer = question.find(substring)

        question = question[:index_of_answer]

        groundtruth = API_Bank_data_list[i]['answer']
        previous_prediction = predict_list[i]
        example_1_q = \
"""The current time is {time}.\n\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available APIs.\n\nUser\'s utterance: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\n\nAvailable API descriptions:\n{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\n\n\n1. The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be done next to satisfy the user? """
        example_2_q = \
"""The current time is {time}.\n\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available APIs.\n\nUser\'s utterance: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\n\nAvailable API descriptions:\n{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\n\n\n\nPrevious API-Request History:\n\nAPI-Request: [ToolSearcher(keywords=\'healthcare provider appointment availability checker\')]\nReceived API Response:{\'name\': \'HealthcareProviderAppointmentChecker\', \'description\': \'API for checking the availability of appointments with healthcare providers.\', \'input_parameters\': {\'specialty\': {\'type\': \'str\', \'description\': \'The specialty of the healthcare provider.\'}, \'location\': {\'type\': \'str\', \'description\': \'The city location.\'}}, \'output_parameters\': {\'appointments\': {\'type\': \'list\', \'description\': \'A list of available appointment slots.\'}}}\n\n\n\nAvailable information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response\n\n1. The previous API-request has help solve part of job specified in user\'s utterance. What should be done next to satisfy the user?"""
        example_3_q = \
"""\nThe current time is {time}.\n\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available APIs.\n\nUser\'s utterance: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\n\nAvailable API descriptions:\n{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\n\n\n\nPrevious API-Request History:\n\n1. \nAPI-Request: [ToolSearcher(keywords=\'healthcare provider appointment availability checker\')]\nReceived API Response:{\'name\': \'HealthcareProviderAppointmentChecker\', \'description\': \'API for checking the availability of appointments with healthcare providers.\', \'input_parameters\': {\'specialty\': {\'type\': \'str\', \'description\': \'The specialty of the healthcare provider.\'}, \'location\': {\'type\': \'str\', \'description\': \'The city location.\'}}, \'output_parameters\': {\'appointments\': {\'type\': \'list\', \'description\': \'A list of available appointment slots.\'}}}\n2. \nAPI-Request: [HealthcareProviderAppointmentChecker(specialty=\'cardiologist\', location=\'Los Angeles\')]\nReceived API Response:{\'appointments\': [\'2034-04-18 14:30:00\', \'2034-04-19 11:00:00\', \'2034-04-20 09:45:00\']}\n2. \nAPI-Request: [ToolSearcher(keywords=\'healthcare provider appointment scheduler\')]\nReceived API Response:{\'name\': \'HealthcareProviderAppointmentScheduler\', \'description\': \'API for scheduling appointments with healthcare providers.\', \'input_parameters\': {\'appointment_datetime\': {\'type\': \'datetime\', \'description\': \'The datetime for the appointment.\'}, \'healthcare_provider\': {\'type\': \'str\', \'description\': \'The name of the healthcare provider.\'}}, \'output_parameters\': {\'confirmation_number\': {\'type\': \'str\', \'description\': \'The confirmation number for the appointment.\'}}}\n\n\n\nAvailable information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response\n\n1. The previous API-request has help solve part of job specified in user\'s utterance. What should be done next to satisfy the user? Please solve the rest of the problem. \n2. We already know the answer is API-Request: [HealthcareProviderAppointmentScheduler(appointment_datetime=\'2034-04-18 14:30:00\', healthcare_provider=\'cardiologist\')]. Please generate how you get the answer"""


        prompt = \
f"""Please perform minimum change to correct the previous prediction. The previous prediction sometimes answer then explain, you need to change it to inference, then provide the final answer. do not put the explaination at the end because the inference process already help for the explaination.

The requirement is only to solve the next step, but the prediction is sometimes lengthy and contains answer that is not part of the question(such as solve multi step in one prediction). remove them when you see it.

You only need to make the necessary modification such that the inference steps support the final prediction and the final prediction has to be placed at the end. You suppose to perform minimum change which means you are trying to modify as less as possible to make the ifnerence steps and the final prediction correct. You will be given a INSTRUCTION for generating API-Request, a GROUNDTRUTH which is the correct API-Request and the PERVIOUS PREDICTION which you need to perform minimum change on.


Example1: The previous prediction contains too many unnecessary information. we remove the information after it gets the correct answer. We are trying to maintain as much original words as possible, thus the modified answer is almost identical to the part of the previous prediction.
Example1-QUESTION: {example_1_q}
Example1-GROUNDTRUTH: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]
Example1-PREVIOUS PREDICTION: Sure, I'd be happy to help!\n\n1. To satisfy the user's request, we need to generate an API request for each of the available APIs that can help find a cardiologist in Los Angeles for a check-up appointment. Based on the user's utterance and the available APIs, we can generate the following API requests:\n\na. [ToolSearcher(keywords='healthcare provider appointment availability checker')]\n\nThis API request uses the ToolSearcher API to search for tools in the library that can help with healthcare provider appointment availability checker.\n\nb. [HealthcareProviderSearch(location='Los Angeles')]\n\nThis API request uses the HealthcareProviderSearch API to search for healthcare providers in Los Angeles.\n\nc. [AvailabilityChecker(healthcare_provider_id='')]\n\nThis API request uses the AvailabilityChecker API to check the availability of healthcare providers in Los Angeles.\n\nSince we only generate the next API request at this time, we will start with the first API request: [ToolSearcher(keywords='healthcare provider appointment availability checker')].\n\n2. To get the answer, we need to send the API requests to the appropriate APIs and retrieve the responses. Here's how we can do it:\n\na. Send the [ToolSearcher(keywords='healthcare provider appointment availability checker')] API request to the ToolSearcher API.\n\nb. Retrieve the response from the ToolSearcher API, which should be a list of tools that match the keywords.\n\nc. From the list of tools, select the tool that best matches the user's request (i.e., a cardiologist in Los Angeles for a check-up appointment).\n\nd. Use the selected tool to search for healthcare providers in Los Angeles using the HealthcareProviderSearch API.\n\ne. Retrieve the response from the HealthcareProviderSearch API, which should be a list of healthcare providers in Los Angeles.\n\nf. Use the list of healthcare providers to check their availability using the AvailabilityChecker API.\n\ng. Retrieve the response from the AvailabilityChecker API, which should be a list of available healthcare providers in Los Angeles.\n\nh. Provide the list of available healthcare providers to the user, along with their availability information.\n\nThat's it! By generating the API requests and retrieving the responses, we can satisfy the user's request and provide them with a list of available cardiologists in Los Angeles for a check-up appointment.
Example1-MODIFIED PREDICTION: Sure, I'd be happy to help!\n\nTo satisfy the user's request, we need to generate an API request for each of the available APIs that can help find a cardiologist in Los Angeles for a check-up appointment. Based on the user's utterance and the available APIs, we can generate the following API requests:\n\na. [ToolSearcher(keywords='healthcare provider appointment availability checker')



Example2: It is correct, no need to change anything
Example2-QUESTION: {example_2_q}
Example2-GROUNDTRUTH: API-Request: [GetSurgeryCost(surgery=heart transplant, location=Los Angeles)]
Example2-PREVIOUS PREDICTION: Based on the user's utterance, the problem to be solved is calculating the total cost of a heart transplant surgery in Los Angeles. The previous API request has provided an API for querying the cost of a specific surgery, which is GetSurgeryCost. Therefore, the next step is to use the GetSurgeryCost API with the specific surgery being a heart transplant and the location being Los Angeles.\n\nThe correct API request is: [GetSurgeryCost(surgery='heart transplant', location='Los Angeles')].
Example2-MODIFIED PREDICTION: "Based on the user's utterance, the problem to be solved is calculating the total cost of a heart transplant surgery in Los Angeles. The previous API request has provided an API for querying the cost of a specific surgery, which is GetSurgeryCost. Therefore, the next step is to use the GetSurgeryCost API with the specific surgery being a heart transplant and the location being Los Angeles.\n\nThe correct API request is: [GetSurgeryCost(surgery='heart transplant', location='Los Angeles')]."


Example3: We remove the information after getting the answer. The previous answer wrongly include the location, we modify it such that the location is not included. We are trying to maintain as much original words as possible, thus the modified answer is almost identical to the part of the previous prediction.
Example3-QUESTION: {example_3_q}
Example3-GROUNDTRUTH: API-Request: [HealthcareProviderAppointmentScheduler(appointment_datetime='2034-04-18 14:30:00', healthcare_provider='cardiologist')]
Example3-PREVIOUS PREDICTION: Based on the user's utterance and the previous API-request history, we can infer that the user is looking for a cardiologist in Los Angeles for a check-up appointment. We can use the HealthcareProviderAppointmentScheduler API to schedule an appointment with a cardiologist in Los Angeles.\n\nHere's the API request for scheduling an appointment with a cardiologist in Los Angeles:\n\nAPI-Request: [HealthcareProviderAppointmentScheduler(appointment_datetime='2034-04-18 14:30:00', healthcare_provider='cardiologist', location='Los Angeles')]\n\nThis API request uses the HealthcareProviderAppointmentScheduler API found in the previous API-request history, with the appointment datetime set to '2034-04-18 14:30:00' and the healthcare provider set to 'cardiologist'. The location parameter is set to 'Los Angeles' to ensure that the API returns a list of available cardiologists in Los Angeles.\n\nTo satisfy the user's request, we need to provide them with the confirmation number for the appointment. We can use the previous API-request history to obtain the confirmation number.\n\nBased on the previous API-request history, the confirmation number is returned in the output parameters of the HealthcareProviderAppointmentScheduler API. Therefore, we can use the following API request to obtain the confirmation number:\n\nAPI-Request: [HealthcareProviderAppointmentScheduler(appointment_datetime='2034-04-18 14:30:00', healthcare_provider='cardiologist', location='Los Angeles')].response.confirmation_number\n\nThis API request uses the same parameters as the previous API request, but instead of returning the appointment schedule, it returns the confirmation number for the appointment.\n\nTherefore, the next step to satisfy the user's request is to provide them with the confirmation number for the appointment, which can be obtained using the previous API-request history and the current API request.
Example2-MODIFIED PREDICTION: Based on the user's utterance and the previous API-request history, we can infer that the user is looking for a cardiologist in Los Angeles for a check-up appointment. We can use the HealthcareProviderAppointmentScheduler API to schedule an appointment with a cardiologist in Los Angeles.\n\nHere's the API request for scheduling an appointment with a cardiologist in Los Angeles:\n\nAPI-Request: [HealthcareProviderAppointmentScheduler(appointment_datetime='2034-04-18 14:30:00', healthcare_provider='cardiologist')]



Please perform minimum change to this problem
INSTRUCTION: {question}
GROUNDTRUTH: {groundtruth}
PERVIOUS PREDICTION: {previous_prediction}
MODIFIED PREDICTION: MINIMUM_CHANGED_PREDICTION_HERE

Your task:
1. According to GROUNDTRUTH API-request and the INSTRUCTION, please perform minimum change to modify the PERVIOUS PREDICTION to make it correct and support the groundtruth API-request. When you modify, please perform minimum change which means you are trying to maintain the original words and only change the part that is incorrect(this is important). The modified analyzation needs to support why the API-REQUEST is generated in such a way.
2. If the PERVIOUS PREDICTION contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. Do not explain how you modify the analyzation. do not write things like 'the PERVIOUS PREDICTION is wrong because') or say things like 'sure, i can hekp you with'. just show me the minimum changed analyzation results directly with api-request at the end(this is important).
4. You need to put API-Request at the end.
5. Please remember that you have to try your best to use the original words since you are performing minimum change. This is the most important criteria.
"""
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['instruction'] = API_Bank_data_list[i]['question']
        temp['input'] = ''
        temp['groundtruth'] = groundtruth
        temp['output'] = answer
        temp['previous_prediction'] = previous_prediction
        input_json_list.append(temp)
    return input_json_list


def minimum_change_data_creation_API_Bank_mar_8(API_Bank_data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    input_json_list = []
    for i in range(len(API_Bank_data_list)):
        question = API_Bank_data_list[i]['question']
        groundtruth = API_Bank_data_list[i]['answer']
        previous_prediction = predict_list[i]
        example_1_q = \
"""\nThe current time is {time}.\n\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.\n\nUser's utterance: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\n\nAvailable API descriptions:\n{\"name\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\n\n\n1. The task requires multi steps API-Request generation. We only generate the next API-Request at this time. What API-Request should be called in the next step?\n2. Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)].\n3. Before show the final answer, make sure to explain why you fill in the API-Request with the the inputs according to the API description. You have to explain how to fill in the inputs before you generate the final API-request.\n4. This is how you should approach the problem. You should first identify the API from the available API options.\nThen you need to figure out the correct input according to the API description. Please notice that you may only use the input specified by the API description for input\nThen you need to generate the API request\n"""
        example_2_q = \
"""\nThe current time is {time}.\n\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available APIs.\n\nUser\'s utterance: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\n\nAvailable API descriptions:\n{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\n\n\n\nPrevious API-Request History: [ToolSearcher(keywords=\'healthcare provider appointment availability checker\')]\nReceived API Response:{\'name\': \'HealthcareProviderAppointmentChecker\', \'description\': \'API for checking the availability of appointments with healthcare providers.\', \'input_parameters\': {\'specialty\': {\'type\': \'str\', \'description\': \'The specialty of the healthcare provider.\'}, \'location\': {\'type\': \'str\', \'description\': \'The city location.\'}}, \'output_parameters\': {\'appointments\': {\'type\': \'list\', \'description\': \'A list of available appointment slots.\'}}}\nThe last previous API-Request: [HealthcareProviderAppointmentChecker(specialty=\'cardiologist\', location=\'Los Angeles\')]\nReceived API Response:{\'appointments\': [\'2034-04-18 14:30:00\', \'2034-04-19 11:00:00\', \'2034-04-20 09:45:00\']}\n\n\n\nAvailable information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response\n\n1. The task requires multi steps API-Request generation. The previous API-request has help solve part of job specified in user\'s utterance. What API-Request should be called in the next step?  We only generate the next API-Request at this time.\n2. Generate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)].\n3. Before show the final answer, make sure to explain why you fill in the API-Request with the the inputs according to the API description. You have to explain how to fill in the inputs before you generate the final API-request.\n4. This is how you should approach the problem. You should first identify the API from the available API options.\nThen you need to figure out the correct input according to the API description. Please notice that you may only use the input specified by the API description for input\nThen you need to generate the API request\n"""
        example_3_q = \
"""The current time is {time}.\n\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available APIs.\n\nUser\'s utterance: Find a clinical trial for a patient with stage 2 breast cancer.\n\nAvailable API descriptions:\n{"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\n\n\n1. The task requires multi steps API-Request generation. We only generate the next API-Request at this time. What API-Request should be called in the next step?\n2. Generate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)].\n3. Before show the final answer, make sure to explain why you fill in the API-Request with the the inputs according to the API description. You have to explain how to fill in the inputs before you generate the final API-request.\n4. This is how you should approach the problem. You should first identify the API from the available API options.\nThen you need to figure out the correct input according to the API description. Please notice that you may only use the input specified by the API description for input\nThen you need to generate the API request"""

        prompt = \
f"""Please perform minimum change to correct the previous prediction. The previous prediction sometimes answer then explain, you need to change it to inference, then provide the final answer. do not put the explaination at the end because the inference process already help for the explaination.

The requirement is only to solve the next step, but the prediction is sometimes lengthy and contains answer that is not part of the question(such as solve multi step in one prediction). remove them when you see it.

You only need to make the necessary modification such that the inference steps support the final prediction and the final prediction has to be placed at the end. You suppose to perform minimum change which means you are trying to modify as less as possible to make the ifnerence steps and the final prediction correct. You will be given a INSTRUCTION for generating API-Request, a GROUNDTRUTH which is the correct API-Request and the PERVIOUS PREDICTION which you need to perform minimum change on.


Example1: In this example, the previous prediction wrongly infill the input keywords. To make sure we use minimum change to modify the previous predictio, we only change the way the prediction generate the inputs for the keywords. The previous prediction making wrong API request, thus we modify the API request. In addition, there is explaination after getting the correct answer. We do not need to further explain if there are already enough inference process of how to get the correct answer. We therefore remove the text after generating the API request. so API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')] will be at the end.
Example1-QUESTION: {example_1_q}
Example1-GROUNDTRUTH: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]
Example1-PREVIOUS PREDICTION: Sure, I'd be happy to help!\n\nBased on the user's utterance, the task is to find a cardiologist in Los Angeles for a check-up appointment. The available APIs are:\n\n1. ToolSearcher: Searches for relevant tools in a library based on keywords.\n\nTo complete this task, we will need to use the ToolSearcher API to search for cardiologists in Los Angeles.\n\nFirst, we need to identify the correct inputs for the API request. According to the API description, the parameters for the ToolSearcher API are:\n\n* keywords: A string representing the keyword to search for.\n\nIn this case, we can use the keyword \"cardiologist\" to search for cardiologists in Los Angeles.\n\nNext, we need to generate the API request. Here is the API request for the next step:\n\n[ToolSearcher(keywords=\"cardiologist\")]\n\nThis API request specifies that we want to search for the keyword \"cardiologist\" using the ToolSearcher API.\n\nBefore showing the final answer, let me explain why we filled in the API request with the inputs according to the API description. We identified the correct API for the task, which was ToolSearcher. We then determined the correct input for the API request, which was the keyword \"cardiologist\". This input was specified by the API description as the parameter for the keywords field.\n\nNow, let's move on to the final answer. Here is the complete API request for the task:\n\n[ToolSearcher(keywords=\"cardiologist\")]\n\nThis API request specifies that we want to search for cardiologists in Los Angeles using the ToolSearcher API.\n\nPlease note that we only generated the next API request at this time. We will need to continue generating additional API requests in the future to complete the task.
Example1-MODIFIED PREDICTION: Sure, I\'d be happy to help! Based on the user's utterance, the task is to find a cardiologist in Los Angeles for a check-up appointment. The available APIs are:

1. ToolSearcher: Searches for relevant tools in a library based on keywords.

To complete this task, we will use the ToolSearcher API. According to the API description, the parameters for the ToolSearcher API are:

* keywords: A string representing the keyword to search for.

In this case, we can use the keywords "healthcare provider appointment availability checker" to find tools that help with scheduling appointments with healthcare providers, such as cardiologists in Los Angeles.

We need to generate the API request. Here is the API request for the next step:

API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]



Example2: In this example. the previous prediction made 4 mistakes. 1. It make the prediction before making the inference. We therefore directly remove the sentence "I recommend the following API request:\n\n[ToolSearcher(keywords='cardiologist appointment availability checker in Los Angeles')]". 2. When the previous prediction making inference, the step 1 and step 2 are wrong. Thus we modify them to make them correct. 3. The 3 step making wrong API request, thus we modify the API request. 4. there is explaination after getting the correct answer. We do not need to further explain if there are already enough inference process of how to get the correct answer. . We therefore remove the text after generating the API request. so API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')] will be at the end.

Example2-QUESTION: {example_2_q}
Example2-GROUNDTRUTH: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]
Example2-PREVIOUS PREDICTION: Sure, I'd be happy to help! Based on the user's utterance and the available APIs, I recommend the following API request:\n\n[ToolSearcher(keywords='cardiologist appointment availability checker in Los Angeles')]\n\nHere's how I arrived at this suggestion:\n\n1. Identify the API: Based on the user's utterance, we need to find a cardiologist in Los Angeles for a check-up appointment. Therefore, the most relevant API is ToolSearcher, which allows us to search for relevant tools in the library based on keywords.\n2. Identify the correct input: The keywords for this API request should be 'cardiologist appointment availability checker in Los Angeles'. This is based on the user's utterance and the previous API response, which provided us with the information that the API 'HealthcareProviderAppointmentChecker' is not relevant for this task.\n3. Generate the API request:\n\n[ToolSearcher(keywords='cardiologist appointment availability checker in Los Angeles')]\n\nExplanation:\n\nThe API ToolSearcher takes the keywords 'cardiologist appointment availability checker in Los Angeles' as input, which are the relevant keywords for our task. By using this API, we can search for tools in the library that match these keywords and find the best match for our task.\n\nThis API request will return a list of relevant tools that match the user's utterance and the previous API response. The response will include the best match tool(s) and their descriptions, which will help us to narrow down the search to the most relevant tools.
Example2-MODIFIED PREDICTION: Sure, I'd be happy to help! Based on the user's utterance and the available APIs, here's how I arrived at the solution\n\n1. Identify the API: Given that we have already found a cardiologist and obtained appointment times, the next step is to schedule the appointment. The ToolSearcher API is suitable for this purpose as it searches for relevant tools based on keywords.\n   \n2. Identify the correct input: The keywords should align with the current need, which is to schedule an appointment. Thus 'healthcare provider appointment scheduler' are the relevant keywords derived from the previous steps and the user's requirement.\n3. Generate the API-Request: [ToolSearcher(keywords='cardiologist appointment availability checker in Los Angeles')]



Example3: The inference process is correct. It is just the keywords are wrong. thus we only replace the keywords. In addition, there is explaination after getting the correct answer. We do not need to further explain if there are already enough inference process of how to get the correct answer. We therefore remove the text after generating the API request. so API-Request: [ToolSearcher(keywords='clinical trial for breast cancer')] will be at the end.
Example3-QUESTION: {example_3_q}
Example3-GROUNDTRUTH: API-Request: [ToolSearcher(keywords='clinical trial for breast cancer')]
Example3-PREVIOUS PREDICTION: Sure, I\'d be happy to help! Based on the user\'s utterance, we need to find a clinical trial for a patient with stage 2 breast cancer. From the available API descriptions, the best fit seems to be the "ToolSearcher" API, which allows us to search for relevant tools in a library based on keywords.\n\nTo generate the next API request, we need to determine the correct inputs for the API. According to the API description, we need to provide the keyword for the search. Based on the user\'s utterance, the keyword would be "breast cancer."\n\nTherefore, the next API request should be:\n\n[ToolSearcher(keywords="breast cancer")]\n\nBefore generating the final API request, let me explain why we chose to use the "ToolSearcher" API and how we filled in the input. We chose the "ToolSearcher" API because it allows us to search for relevant tools based on a keyword, which matches the user\'s request for finding a clinical trial for a patient with stage 2 breast cancer. The input "breast cancer" is the keyword we used for the search, as specified by the API description.\n\nNow, let me generate the final API request:\n\n[ToolSearcher(keywords="breast cancer")]\n\nThis API request will return the best match tool(s) for the search keyword "breast cancer" based on the library of tools available. The response will contain a list of dicts, where each dict represents a tool and contains the relevant information such as the tool\'s name, description, and URL.
Example2-MODIFIED PREDICTION: Sure, I\'d be happy to help! Based on the user\'s utterance, we need to find a clinical trial for a patient with stage 2 breast cancer. From the available API descriptions, the best fit seems to be the "ToolSearcher" API, which allows us to search for relevant tools in a library based on keywords.\n\nTo generate the next API request, we need to determine the correct inputs for the API. According to the API description, we need to provide the keyword for the search. Based on the user\'s utterance, the keyword would be "clinical trial for breast cancer."\n\nTherefore, the next API request should be API-Request: [ToolSearcher(keywords="clinical trial for breast cancer")]



Please perform minimum change to this problem
INSTRUCTION: {question}
GROUNDTRUTH: {groundtruth}
PERVIOUS PREDICTION: {previous_prediction}
MODIFIED PREDICTION: MINIMUM_CHANGED_PREDICTION_HERE

Your task:
1. According to GROUNDTRUTH API-request and the INSTRUCTION, please perform minimum change to modify the PERVIOUS PREDICTION to make it correct and support the groundtruth API-request. When you modify, please perform minimum change which means you are trying to maintain the original words and only change the part that is incorrect(this is important). The modified analyzation needs to support why the API-REQUEST is generated in such a way.
2. If the PERVIOUS PREDICTION contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. Do not explain how you modify the analyzation. do not write things like 'the PERVIOUS PREDICTION is wrong because') or say things like 'sure, i can hekp you with'. just show me the minimum changed analyzation results directly with api-request at the end(this is important).
4. Please remember that you have to try your best to use the original words since you are performing minimum change. This is the most important criteria.
"""
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['instruction'] = API_Bank_data_list[i]['question']
        temp['input'] = ''
        temp['groundtruth'] = groundtruth
        temp['output'] = answer
        temp['previous_prediction'] = previous_prediction
        input_json_list.append(temp)
    return input_json_list




def minimum_change_data_creation_API_Bank_mar_17(API_Bank_data_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    input_json_list = []
    for i in range(len(API_Bank_data_list)):

        question = API_Bank_data_list[i]['question']
        gold_label = API_Bank_data_list[i]['answer']
        previous_prediction = API_Bank_data_list[i]['prediction']

        example_question = \
"\nThe current time is {time}.\n\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available APIs.\n\nUser's utterance: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\n\nAvailable API descriptions:\n{\"name\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\n\n\n\nPrevious API-Request History: [ToolSearcher(keywords='healthcare provider appointment availability checker')]\nReceived API Response:{'name': 'HealthcareProviderAppointmentChecker', 'description': 'API for checking the availability of appointments with healthcare providers.', 'input_parameters': {'specialty': {'type': 'str', 'description': 'The specialty of the healthcare provider.'}, 'location': {'type': 'str', 'description': 'The city location.'}}, 'output_parameters': {'appointments': {'type': 'list', 'description': 'A list of available appointment slots.'}}}\nPrevious API-Request History: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]\nReceived API Response:{'appointments': ['2034-04-18 14:30:00', '2034-04-19 11:00:00', '2034-04-20 09:45:00']}\nThe last previous API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]\nReceived API Response:{'name': 'HealthcareProviderAppointmentScheduler', 'description': 'API for scheduling appointments with healthcare providers.', 'input_parameters': {'appointment_datetime': {'type': 'datetime', 'description': 'The datetime for the appointment.'}, 'healthcare_provider': {'type': 'str', 'description': 'The name of the healthcare provider.'}}, 'output_parameters': {'confirmation_number': {'type': 'str', 'description': 'The confirmation number for the appointment.'}}}\n\n\n\nAvailable information can be use are ToolSearcher, the API tool found by the previous API-Request response and the information found by the previous API-Request response\n\n1. The task requires multi steps API-Request generation. The previous API-request has help solve part of job specified in user's utterance. What API-Request should be called in the next step?  We only generate the next API-Request at this time.\n2. Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] \n3. This is how you should approach the problem: First, decide which API to use for the next step, considering the history of previous API requests. Then, determine the correct input based on the API description, noting that you may only use the inputs specified in the API description. Finally, generate the API request. Before showing the final answer, ensure you explain why you filled in the API request with the specified inputs, according to the API description. It's important to explain how to input the data before generating the final API request.\n"

        example_previous_prediction = \
"Based on the user's utterance and the previous API-request history, the next step would be to use the HealthcareProviderAppointmentScheduler API to schedule an appointment with a cardiologist in Los Angeles.\n\nHere's the API request in the format of [ApiName(key1='value1', key2='value2',...)]:\n\n[HealthcareProviderAppointmentScheduler(appointment_datetime='2034-04-18 14:30:00', healthcare_provider='cardiologist', confirmation_number='')]\n\nHere's the explanation of the input fields:\n\n* appointment_datetime: The datetime for the appointment, which is the first available appointment slot found by the previous API request (2034-04-18 14:30:00).\n* healthcare_provider: The name of the healthcare provider, which is 'cardiologist' based on the user's utterance.\n* confirmation_number: The confirmation number for the appointment, which will be provided by the API after the appointment is scheduled.\n\nThe input fields are based on the API description, which specifies the input parameters for the HealthcareProviderAppointmentScheduler API. The previous API-request history also provided the necessary information to fill in the input fields."
        
        example_improved_prediction = \
"The previous API-Request use ToolSearch to find HealthcareProviderAppointmentScheduler, thus the next step would be to use the HealthcareProviderAppointmentScheduler API to schedule an appointment with a cardiologist in Los Angeles. The HealthcareProviderAppointmentScheduler API tooks two inputs:\n\n* appointment_datetime: The datetime for the appointment, which is the first available appointment slot found by the previous API request (2034-04-18 14:30:00).\n* healthcare_provider: The name of the healthcare provider, which is 'cardiologist' based on the user's utterance. Fill in the HealthcareProviderAppointmentScheduler with the API request in the format of [ApiName(key1='value1', key2='value2',...)], we got [HealthcareProviderAppointmentScheduler(appointment_datetime='2034-04-18 14:30:00', healthcare_provider='cardiologist')]"

#         prompt = f"""Your task:
# 1. You are given a question, a previous answer and a groundtruth. You need to optimize the previous answer such that it is solving the problem in a good way. If the previous answer is already good, then no need to change anything. If the previous answer is redundant, then please improve it.
# 2. When you improve the previous answer, please make sure to use the original words. We wish to keep the distribution similar so you should try your best to use the original words from the previous answer instead of completely rewrite the previous answer. You can delete content or reorganize the paraphraph if you want. 
# 4. Make sure to place the "Answer: {gold_label}" at the end.
# 5. Plase show me the modified answer directly instead of say anything like "sure, i can help you to modify the answer" or "ok, here is the modified answer". Just show me the modified answer directly.
# 6. Let me repeat. You have to try to keep the original words! Your task is to improve the previous answer instead of rewrite it so you have to try to keep the original words. do not rewrite everything!!!
# 7. Do not rewrite. Try to keep the original words!!!!!


# Example: (Please notice that  The previous answer answer first then explain its own answer. This improved example try to reorganize the prediction such that it inference first then provide the final answer at very end. It changes the answer logic, but it also try to take as much original words from the previous answer as possible so as to make the distribution similar.)

# Example Question: {example_question}
# Example Previous Prediction: {example_previous_prediction}
# Example Improved Prediction: {example_improved_prediction}



# Let me repeat my requirement. Try to keep the original words!!!!!Do not rewrite. Try to keep the original words!!!!!Do not rewrite. Try to keep the original words!!!!!Do not rewrite. Try to keep the original words!!!!!Do not rewrite. Try to keep the original words!!!!!Do not rewrite. Try to keep the original words!!!!!Do not rewrite. Try to keep the original words!!!!!



# now, we have the Question: {question}
# Groundtruth: {gold_label}

# Previous prediction: "{previous_prediction}"
# Improved Prediction: 

# I am pretty sure you will not follow my most impotant requirement which is "Try to keep the original words!!!!!" Please follow this time.

# To be honest, i have called this API for more than 10 times. You never try to use the original words as required. You just need to try your best to extract the useful original words from the previous answer, then answer the question with the improved logic as shown in the example. You never follow this instruction. Please follow this time. 
# """
        
#         prompt = \
# f"""Your task: You are given a question, a previous answer and a groundtruth. You need to reorganize the previous answer such that it is solving the problem in a good way. If the previous answer is already good, then no need to change anything. If the previous answer is redundant, then please improve it and remove the redundant contents.
# 1. When you improve the previous answer, please make sure to try your best to use the original words. We wish to keep the distribution similar so you should try your best to use the original words from the previous answer instead of completely rewrite the previous answer. You can delete content or reorganize the paraphraph if you want. 
# 2. A big part of the improved prediction should contains words comes from the previous prediction. 
# 3. You have to reorganize the improved prediction such that it inference first , then state the final answer at the very end instead of make the prediction at very begginning then explain.
# 4. Make sure to place the "Answer: {gold_label}" at the end.
# 5. Plase show me the modified answer directly instead of say anything like "sure, i can help you to modify the answer" or "ok, here is the modified answer". Just show me the modified answer directly.


# Example: (Please notice that  The previous answer answer first then explain its own answer. This improved example try to reorganize the prediction such that it inference first then provide the final answer at very end. It changes the answer logic, but it also try to take as much original words from the previous answer as possible so as to make the distribution similar.)

# Example Question: {example_question}
# Example Previous Prediction: {example_previous_prediction}
# Example Improved Prediction: {example_improved_prediction}


# now, we have the Question: {question}
# Groundtruth: {gold_label}

# Previous prediction: "{previous_prediction}"
# Improved Prediction: 
# """
        
        prompt = \
f"""Your task: You are given a question, a previous answer and a groundtruth. You need to reorganize the previous answer such that it inference first then answer at the end. If the previous answer is already good, then no need to change anything. If the previous answer is redundant, then please improve it and remove the redundant contents.

Make sure to place the "Answer: {gold_label}" at the end.

Plase show me the modified answer directly instead of say anything like "sure, i can help you to modify the answer" or "ok, here is the modified answer". Just show me the modified answer directly.

This is the most important requirement: When you improve the prediction, make sure to use as much original words as possible. this is very important



please use as much original words as possible. please remember this requirement. please 


now, we have the Question: {question}
Groundtruth: {gold_label}

Previous prediction: "{previous_prediction}"
Improved Prediction: 
"""


        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['question'] = API_Bank_data_list[i]['question']
        temp['input'] = ''
        temp['output'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['hypothesis'] = API_Bank_data_list[i]['hypothesis']
        temp['premise'] = API_Bank_data_list[i]['premise']
        input_json_list.append(temp)

    return input_json_list


def minimum_change_data_creation_API_Bank_best(API_Bank_data_list, predict_list, train_data_2, predict_list_2):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    input_json_list = []
    for i in range(len(API_Bank_data_list)):
        question = API_Bank_data_list[i]['question']
        groundtruth = API_Bank_data_list[i]['answer']
        previous_prediction = predict_list[i]

        prompt = \
f"""Please perform minimum change to correct the previous prediction if it is wrong. You only need to make the necessary modification such that the inference steps support the final prediction and the final prediction has to be placed at the end with the correct format of API-Request: [ApiName(key1=\'value1\', key2=\'value2\', ...)]. You suppose to perform minimum change which means you are trying to modify as less as possible to make the ifnerence steps and the final prediction correct. You will be given a INSTRUCTION for generating API-Request, a GROUNDTRUTH which is the correct API-Request and the PERVIOUS PREDICTION which you need to perform minimum change on.
INSTRUCTION: {question}

GROUNDTRUTH: {groundtruth}

PERVIOUS PREDICTION: {previous_prediction}





Your task:
1. According to GROUNDTRUTH API-request and the INSTRUCTION, please perform minimum change to modify the analyzation process of the PERVIOUS PREDICTION to make it correct and support the groundtruth API-request. When you modify, please perform minimum change which means you are trying to maintain the original words and only change the part that is incorrect(this is important). The modified analyzation needs to support why the API-REQUEST is generated in such a way.
2. If the PERVIOUS PREDICTION contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the PERVIOUS PREDICTION answer first then explain or inference latter, put the final answer at the end and put the explaination/inference steps at the beginning. We wish to perform minimum change and also make sure the model inference then answer based on inference. 
4. Do not explain how you modify the analyzation. do not write things like 'the PERVIOUS PREDICTION is wrong because') or say things like 'sure, i can hekp you with'. just show me the minimum changed analyzation results directly with api-request at the end(this is important).
5. REMEMBER, you are performing minimum change which means you are trying best to preserve the original words.
"""


        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

#         a = \
# """The instruction requires multi-step API-CALL. Based on the instruction, available API-Request"""
#         if a not in question:
#             prompt = \
# f"""We have question and the previous prediction. The previous prediction sometimes misstate the subquestion mentioned in the question. If so, replace it with the corresponding original subquestion. If not, do not change anything. This is the only thing you need to do. Please directly show me the modified previous prediction without say anything like "sure, i can help you" or "we modified this because"

# QUESTION: {question}
# PREVIOUS PREDICTION : {answer}

# Let me repeat. Please directly replace the misstated subquestion without any explaination or saying otherthings like 'sure, i can help you. Please do not change anyother contents or answer the question on your own. directly show me the modifed PREVIOUS PREDICTION'"""
            

#         answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['instruction'] = API_Bank_data_list[i]['question']
        temp['input'] = ''
        temp['groundtruth'] = groundtruth
        temp['output'] = answer
        temp['previous_prediction'] = previous_prediction
        input_json_list.append(temp)
    return input_json_list




        
# def minimum_change_data_creation_API_Bank_old(API_Bank_data_list, predict_list):
#     from openai import OpenAI
#     import openai
#     from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
#     client = OpenAI(api_key=GPT_API)
#     @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
#     def create_gpt_completion(qa_, model_engine, client):
#         response = client.chat.completions.create(
#             model=model_engine,
#             messages=[
#                 {"role": "system", "content": "provide answer"},
#                 {"role": "user", "content": f"{qa_}"}
#             ]
#         )
#         answer = response.choices[0].message.content
#         return answer
#     input_json_list = []
#     for i in range(len(API_Bank_data_list)):
#         combined_question = API_Bank_data_list[i]['question']
#         groundtruth = API_Bank_data_list[i]['answer']
#         previous_prediction = predict_list[i]


#         # groundtruth

#         search_content = groundtruth
#         search_content = search_content.replace("API-Request: [ToolSearcher(keywords='", '')
#         search_content = search_content.replace("')]", '')
#         groundtruth_full = API_Bank_data_list[i]['answer']



#         if 'ToolSearcher' in groundtruth_full:
#             answer = f"""\n\nAnalyzation: Based on the user's utterance, the task is to search for a tool for {search_content} 

# {groundtruth_full} """
#             a = 1
#         else:
#             prompt = f"""     
# INSTRUCTION: "{combined_question}"
# GROUNDTRUTH {groundtruth}




# When asking GPT3 the same question, this is its answer
# ANSWER BY GPT3: "{previous_prediction}"




# Your task:
# 1. According to groundtruth API-request and the instruction, please perform minimum change to modify the analyzation process of the answer by GPT3 to make it correct and support the groundtruth API-request. When you modify, please perform minimum change which means you are trying to maintain the original words and only change the part that is incorrect(this is important). The modified analyzation needs to support why the API-REQUEST is generated in such a way.
# 2. If the answer by GPT3 contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
# 3. If the answer answer first then explain or inference latter, put the final answer at the end and put the explaination/inference part at the beginning. We wish to perform minimum change and also make sure the model inference then answer based on inference. 
# 4. Do not explain how you modify the analyzation. do not write things like 'the answer by gpt3 is wrong because') or say things like 'sure, i can hekp you with'. just show me the minimum changed analyzation results directly with api-request at the end(this is important).
# 5. keep the correct ANSWER FORMAT:
# Analyzation: MINIMUM_CHANGED_ANALYZATION_HERE
# API-Request: API_CALL_HERE
# """
            
#             answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

#         temp = {}
#         temp['instruction'] = API_Bank_data_list[i]['question']
#         temp['input'] = ''
#         temp['groundtruth'] = groundtruth
#         temp['output'] = answer
#         temp['previous_prediction'] = previous_prediction
#         input_json_list.append(temp)
#     return input_json_list



def minimum_change_data_creation_API_Bank_Rationale_creation(API_Bank_data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    input_json_list = []
    for i in range(len(API_Bank_data_list)):
        combined_question = API_Bank_data_list[i]['question']
        groundtruth = API_Bank_data_list[i]['answer']
        previous_prediction = predict_list[i]

        # search_content = groundtruth
        # search_content = search_content.replace("API-Request: [ToolSearcher(keywords='", '')
        # search_content = search_content.replace("')]", '')
        # groundtruth_full = API_Bank_data_list[i]['answer']


#         if 'ToolSearcher' in groundtruth_full:
#             answer = f"""\n\nAnalyzation: Based on the user's utterance, the task is to search for a tool for {search_content} 

# {groundtruth_full} """
#             a = 1
#         else:
        prompt = f"""We ask the user provide rationale to get the groundtruth. Please perform minimum change to modify user's rationale so that it can be used as a inference process to support the groundtruth. The user's rationale is not necessary correct but is also not necessarily wrong. you do not need to change the correct part. Since it is a rationale, it sometimes needs to be modified as inference process to support the groundtruth. 
INSTRUCTION: "{combined_question}"
GROUNDTRUTH: {groundtruth}
Rantionale: {previous_prediction}


Your task:
1. Please perform minimum change to change the rationale to the inference process. If the rationale is wrong, then please perform minimum change to correct it. If the rationale is correct and can be use as inference process, you could use it directly without changing anything. We are trying to perform minimum change so you should try to avoid unnecessary change. 

2. If the Rantionale contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. If the Rantionale has something that should not be used as inference process, remove it.

3. Do not explain how you modify the rationale. do not write things like 'the rationale is wrong because' or say things like 'sure, i can hekp you with'. just show me the minimum changed results directly with api-request at the end

4. You need to modify the rationale such that it becomes a inference process, and based on those inference process, you got the final answer. 
Format:
Analyzation: MINIMUMCHANGED_INFERENCE_PROCESS
"""
            
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        answer = answer +f"""

{groundtruth}"""
        temp = {}
        temp['instruction'] = API_Bank_data_list[i]['question']
        temp['input'] = ''
        temp['groundtruth'] = groundtruth
        temp['output'] = answer
        temp['previous_prediction'] = previous_prediction
        input_json_list.append(temp)
    return input_json_list

def feedback_net_training_data_creation_API_Bank_gpt4_groundtruth(API_Bank_training_data_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i in range(len(API_Bank_training_data_list)):
        instruction = API_Bank_training_data_list[i]['question']
        output = API_Bank_training_data_list[i]['answer']
        prompt = f"""
We have an instruction and a groundtruth api-request.

INSTRUCTION: "{instruction}"
GROUNDTRUTH {output}



Given the instruction and the groundtruth {output}, please provide the inference process to get the groundtruth {output}. We need to know the inference which can lead to the conclusion of the groundtruth api-request given the instruction. Please directly give me the inference process without say things like 'sure, i can help you with it'


You need to Answer in proper format
Format:
Inference process: INFERENCE_PROCESS_HERE
"""
        
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        answer = f"""Analyzation: {answer}
{output}"""
        temp = {}
        temp['instruction'] = instruction
        temp['input'] = ''
        temp['groundtruth'] = output
        temp['output'] = answer
        input_json_list.append(temp)
    return input_json_list



def minimum_change_data_creation_API_Bank_Rationale_creation(API_Bank_data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    input_json_list = []
    for i in range(len(API_Bank_data_list)):
        combined_question = API_Bank_data_list[i]['question']
        groundtruth = API_Bank_data_list[i]['answer']
        previous_prediction = predict_list[i]

        # search_content = groundtruth
        # search_content = search_content.replace("API-Request: [ToolSearcher(keywords='", '')
        # search_content = search_content.replace("')]", '')
        # groundtruth_full = API_Bank_data_list[i]['answer']


#         if 'ToolSearcher' in groundtruth_full:
#             answer = f"""\n\nAnalyzation: Based on the user's utterance, the task is to search for a tool for {search_content} 

# {groundtruth_full} """
#             a = 1
#         else:
        prompt = f"""We ask the user provide rationale to get the groundtruth. Please perform minimum change to modify user's rationale so that it can be used as a inference process to support the groundtruth. The user's rationale is not necessary correct but is also not necessarily wrong. you do not need to change the correct part. Since it is a rationale, it sometimes needs to be modified as inference process to support the groundtruth. 
INSTRUCTION: "{combined_question}"
GROUNDTRUTH: {groundtruth}
Rantionale: {previous_prediction}


Your task:
1. Please perform minimum change to change the rationale to the inference process. If the rationale is wrong, then please perform minimum change to correct it. If the rationale is correct and can be use as inference process, you could use it directly without changing anything. We are trying to perform minimum change so you should try to avoid unnecessary change. 

2. If the Rantionale contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. If the Rantionale has something that should not be used as inference process, remove it.

3. Do not explain how you modify the rationale. do not write things like 'the rationale is wrong because' or say things like 'sure, i can hekp you with'. just show me the minimum changed results directly with api-request at the end

4. You need to modify the rationale such that it becomes a inference process, and based on those inference process, you got the final answer. 
Format:
Analyzation: MINIMUMCHANGED_INFERENCE_PROCESS
"""
            
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        answer = answer +f"""

{groundtruth}"""
        temp = {}
        temp['instruction'] = API_Bank_data_list[i]['question']
        temp['input'] = ''
        temp['groundtruth'] = groundtruth
        temp['output'] = answer
        temp['previous_prediction'] = previous_prediction
        input_json_list.append(temp)
    return input_json_list

def find_last_api_name(input_text):
    # Regular expression to match 'API-Request:' followed by any characters
    api_request_regex = r"API-Request:\s*\[(.*?)\]"

    # Find all matches
    api_requests = re.findall(api_request_regex, input_text)

    # Extract the last API name, if any are found
    if api_requests:
        # Split the last request at the first '(' and take the first part
        last_api_name = api_requests[-1].split('(')[0]
        return last_api_name
    else:
        return None

def API_Bank_question_conversion(API_Bank_training_data_list):
    # from openai import OpenAI
    # import openai
    # from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    # client = OpenAI(api_key=GPT_API)
    # @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    # def create_gpt_completion(qa_, model_engine, client):
    #     response = client.chat.completions.create(
    #         model=model_engine,
    #         messages=[
    #             {"role": "system", "content": "provide answer"},
    #             {"role": "user", "content": f"{qa_}"}
    #         ]
    #     )
    #     answer = response.choices[0].message.content
    #     return answer
    front_prompt_1 = \
"""The instruction requires multi-step API-CALL. The input contains the history of previous API-Request with the received response. """

    front_prompt_2 = \
"""The instruction requires multi-step API-CALL. Based on the instruction, available API-Request and the users' utterance, Please generate the next API-Request. You need to fill in the API-request with the proper input value in the input keywords based on user's utterance and API description with the proper format [ApiName(key1=\'value1\', key2=\'value2\', ...)].

For example, if the user want to go to school that is far from his home at 10am, the next API call should be API-Request: [ToolSearcher(keywords='bus schedule checker')] because the next action should be check the available bus.
"""


    for i in range(len(API_Bank_training_data_list)):
        # Get the instruction from the list
        instruction = API_Bank_training_data_list[i]['question']

        

        # Replace '->' with ' Received Response: '
        if '->' in instruction:
            # Replace '\nGenerate API Request:' with '\nGenerate the next API Request:'
            instruction = instruction.replace('\nGenerate API Request:', '\nBased on the received response from the previous API-Request, ')
            instruction = instruction.replace('->', ' Received Response: ')

            api_name = find_last_api_name(instruction)

            # Split the string at 'API-Request:'
            segments = instruction.split('API-Request:')

            # Process the segments
            processed_segments = [segments[0]]
            processed_segments.append('API-Request:' + segments[1])
            if len(segments) > 2:
                processed_segments.extend('Previous API-Request:' + segment for segment in segments[2:])
                # processed_segments.append('Previous API-Request:' + segments[2:])
            # if len(segments) > 3:
            #     processed_segments.extend('Previous next API request API-Request:' + segment for segment in segments[3:])

            # Join the segments back into a single string
            instruction = ''.join(processed_segments)




#             prompt_end = \
# """Based on the instruction, the information received by the response of the previous API-Request and the users' utterance, please use ToolSearcher to search for the new API or use the available API or the API found by the ToolSeaecher to generate the next API-Request. The input has to match the input format requirement in the next API. """

            if api_name == 'ToolSearcher':
                # prompt_end = """Based on the instruction, the information received by the response of the previous API-Request and the users' utterance, please use the API found by the last ToolSearcher(Received Response) to generate the next API-Request. The input has to match the input format requirement in the next API."""
                prompt_end = f"""Please generate the API-Request step by step through the following logic.
1. What is the user's inquiry?
2. What action has the previous API request already done to achieve the user's inquiry?
3. What API was found by the last ToolSearcher? 
4. Please fill in the input for the API-Request, based on users' utterance, previous API-Request history, the API description and input requirements. Call the API request in the format of [ApiName(key1='value1', key2='value2', ...)]
"""
            else:
                # answer_item = API_Bank_training_data_list[i]['answer']
                # keywords_match = re.search(r"ToolSearcher\(keywords='(.+?)'\)", answer_item)
                # keywords = keywords_match.group(1) if keywords_match else ""

                # prompt_end = f"""Based on the users' utterance(the final goal), the information received by the response of the previous API-Request(the action has already done), please determine the next action needed to be done, then use ToolSearcher to search for the API for that next action. The input has to match the input format requirement in ToolSearcher."""

                prompt_end = f"""Please generate the API-Request step by step through the following logic.
1. What is the user's inquiry?
2. What action has the previous API request already done to achieve the user's inquiry?
3. Based on the response received from the previous API request, what should be the next action to achieve the user's inquiry?
4. Please fill in the input for the ToolSearch, for researching the tool/information related to the next action. Generate the API request in the format of [ApiName(key1='value1', key2='value2', ...)]
"""
            
            # Add the modified instruction to the prompt
            prompt = f"""{front_prompt_1}\n\n{instruction}

{prompt_end}"""
        else:
            # Replace '\nGenerate API Request:' with '\nGenerate the next API Request:'
            instruction = instruction.replace('\nGenerate API Request:', '\nGenerate the next API Request:')
            prompt = f"""{front_prompt_2}

Instruction: {instruction}"""
        API_Bank_training_data_list[i]['question'] = prompt
        
    return API_Bank_training_data_list



def API_Bank_question_conversion_step2(API_Bank_training_data_list):
    front_prompt_1 = \
"""The instruction requires multi-step API-CALL. The input contains the history of previous API-Request with the received response. """

    front_prompt_2 = \
"""The instruction requires multi-step API-CALL. Based on the instruction, available API-Request and the users' utterance, Please generate the next API-Request. You need to fill in the API-request with the proper input value in the input keywords based on user's utterance and API description with the proper format [ApiName(key1=\'value1\', key2=\'value2\', ...)].

For example, if the user want to go to school that is far from his home at 10am, the next API call should be API-Request: [ToolSearcher(keywords='bus schedule checker')] because the next action should be check the available bus.
"""


    for i in range(len(API_Bank_training_data_list)):
        # Get the instruction from the list
        instruction = API_Bank_training_data_list[i]['question']

        

        # Replace '->' with ' Received Response: '
        if '->' in instruction:
            # Replace '\nGenerate API Request:' with '\nGenerate the next API Request:'
            instruction = instruction.replace('\nGenerate API Request:', '\nBased on the received response from the previous API-Request, ')
            instruction = instruction.replace('->', ' Received Response: ')

            api_name = find_last_api_name(instruction)

            # Split the string at 'API-Request:'
            segments = instruction.split('API-Request:')

            # Process the segments
            processed_segments = [segments[0]]
            processed_segments.append('API-Request:' + segments[1])
            if len(segments) > 2:
                processed_segments.extend('Previous API-Request:' + segment for segment in segments[2:])
                # processed_segments.append('Previous API-Request:' + segments[2:])
            # if len(segments) > 3:
            #     processed_segments.extend('Previous next API request API-Request:' + segment for segment in segments[3:])

            # Join the segments back into a single string
            instruction = ''.join(processed_segments)

            prompt_end = """Based on the instruction, available API-Request and the users' utterance, Please generate the next API-Request step by step."""


            
            # Add the modified instruction to the prompt
            prompt = f"""{front_prompt_1}\n\n{instruction}

{prompt_end}"""
        else:
            # Replace '\nGenerate API Request:' with '\nGenerate the next API Request:'
            instruction = instruction.replace('\nGenerate API Request:', '\nGenerate the next API Request:')
            prompt = f"""{front_prompt_2}

Instruction: {instruction}"""
        API_Bank_training_data_list[i]['question'] = prompt
        
    return API_Bank_training_data_list

def minimum_change_data_creation_ANLI(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        premise = item['premise']
        statement = item['hypothesis']
        explaination = item['reason']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

        prompt = f"""We have the NLI Question below: 

Determine whether the statement is entailment(true), contradiction(false), or neutral(undetermined or irrelevant) with the given context.

Your task:
1. According to groundtruth answer and the short explaination, please modify the analyzation by GPT3. When modify the analyzation, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. If the answer by GPT3 contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the analyzation is already correct, then do not change anything. just copy it.
4. Answer in proper format
Format:
Analyzation:
Answer:


Example1:

Context: The Ghost & Mrs. Muir is an American sitcom based on the 1947 film of the same name, which was based on the 1945 novel by R. A. Dick. It premiered in September 1968 on NBC. After NBC canceled the series, it aired on ABC for one season before being canceled a final time. The program is currently seen weekday mornings on the digital subchannel "GetTV."
Statement: The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday

This is the groundtruth and the short explaination
Short explaination to the groundtruth: "Reason: The audience can watch any season of The Ghost & Mrs. Muir in the mornings of weekdays on "GetTV", not just the second season. It was difficult for the system because the context said the series was cancelled after the first season aired on ABC."
Groundtruth: entailment



When asking GPT3 the same question, this is its answer
answer by GPT3: "The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is neutral given the context.

The context provides information about the history of the sitcom, including its cancellation on NBC and brief run on ABC. However, it does not provide any information about the availability of the second season. Therefore, the statement about watching the second season in the morning from Monday to Friday is not entailment or contradiction, as it does not necessarily follow from the context.

Answer: neutral


Please perform minimum change to the analyzation.
Analyzation: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is an entailment given the context.

The context provides information about the availability of the sitcom on the digital subchannel "GetTV" during weekday mornings. Although it does not provide any information about the availability of the second season, there is no indication in the context that the series is not complete, hence it can be inferred from the context that all seasons can be watched. Therefore, the statement about watching the second season in the morning from Monday to Friday is entailment, as it does follow from the context.

Answer: entailment







Question:
Context: {premise} 
Statement: {statement}

This is the groundtruth and the short explaination
Short explaination to the groundtruth: "{explaination}"
Groundtruth: {gold_label}




When asking GPT3 the same question, this is its answer
answer by GPT3: "{previous_prediction}"

Please perform minimum change to the analyzation.
Format:
Analyzation: MODIFIED_ANALYZATION_HERE
Answer: {gold_label}

"""


        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['output'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        temp['reason'] = item['reason']
        input_json_list.append(temp)
    return input_json_list


def minimum_change_data_creation_MNLI(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        premise = item['premise']
        statement = item['hypothesis']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

        prompt = f"""We have the NLI Question below: 

Determine whether the statement is entailment(true), contradiction(false), or neutral(undetermined or irrelevant) with the given context.

Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. Place the "Answer: Entailment/Neutral/Contradiction" at the end.


Example(We are trying to change as little as possible to make the previous prediction correct. look at this example. it modify as less as possible. it also put Answer: Entailment at the end. It try to not change the words from previous prediction if it is not wrong. ): 
Context: The Ghost & Mrs. Muir is an American sitcom based on the 1947 film of the same name, which was based on the 1945 novel by R. A. Dick. It premiered in September 1968 on NBC. After NBC canceled the series, it aired on ABC for one season before being canceled a final time. The program is currently seen weekday mornings on the digital subchannel "GetTV."
Statement: The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday
Groundtruth: Entailment
Previous Prediction: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is neutral given the context.

The context provides information about the history of the sitcom, including its cancellation on NBC and brief run on ABC. However, it does not provide any information about the availability of the second season. Therefore, the statement about watching the second season in the morning from Monday to Friday is not entailment or contradiction, as it does not necessarily follow from the context. Answer: Neutral

Minimum Changed Prediction: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is an entailment given the context.

The context provides information about the availability of the sitcom on the digital subchannel "GetTV" during weekday mornings. Although it does not provide any information about the availability of the second season, there is no indication in the context that the series is not complete, hence it can be inferred from the context that all seasons can be watched. Therefore, the statement about watching the second season in the morning from Monday to Friday is entailment, as it does follow from the context. Answer: Entailment


did you see how i change the previous prediction? change as little as possible to make it correct! change as little as possible to make it correct!change as little as possible to make it correct!change as little as possible to make it correct!change as little as possible to make it correct!change as little as possible to make it correct!






Example(The previous prediction is already correct. so we just Answer: Neutral at the end): 
Context: Conceptually cream skimming has two basic dimensions - product and geography.
Statement: Product and geography are what make cream skimming work. 
Groundtruth: Neutral
Previous Prediction: Sure, I\'d be happy to help!\n\n1. Based on the context and the statement, let\'s analyze whether the statement is entailment, contradiction, or neutral:\n\nThe statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\n2. Since we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work.

Minimum Changed Prediction: Sure, I\'d be happy to help!\n\n1. Based on the context and the statement, let\'s analyze whether the statement is entailment, contradiction, or neutral:\n\nThe statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\n2. Since we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work. Answer: Neutral

did you see how i change the previous prediction? don't change if it is already correct!!!don't change if it is already correct!!!don't change if it is already correct!!!don't change if it is already correct!!!



Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the "Answer: {gold_label}" at the end.

We have the NLI Question below: 

Determine whether the statement is entailment(true), contradiction(false), or neutral(undetermined or irrelevant) with the given context.Context: {premise} 
Statement: {statement}
Groundtruth: {gold_label}

Previous prediction: "{previous_prediction}"
Minimum Changed Prediction: 
"""


        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['output'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        input_json_list.append(temp)
    return input_json_list


def minimum_change_data_creation_ESNLI_mistral(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        premise = item['premise']
        statement = item['hypothesis']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

        prompt = f"""We have the NLI Question below: 

Determine whether the statement is entailment(true), contradiction(false), or neutral(undetermined or irrelevant) with the given context.

Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. Place the "Final Answer: Entailment/Neutral/Contradiction" at the end.


Example(We are trying to change as little as possible to make the previous prediction correct. look at this example. it modify as less as possible. it also put Final Answer: Entailment at the end. It try to not change the words from previous prediction if it is not wrong. ): 
Context: The Ghost & Mrs. Muir is an American sitcom based on the 1947 film of the same name, which was based on the 1945 novel by R. A. Dick. It premiered in September 1968 on NBC. After NBC canceled the series, it aired on ABC for one season before being canceled a final time. The program is currently seen weekday mornings on the digital subchannel "GetTV."
Statement: The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday
Groundtruth: Entailment
Previous Prediction: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is neutral given the context.

The context provides information about the history of the sitcom, including its cancellation on NBC and brief run on ABC. However, it does not provide any information about the availability of the second season. Therefore, the statement about watching the second season in the morning from Monday to Friday is not entailment or contradiction, as it does not necessarily follow from the context. Final Answer: Neutral

Minimum Changed Prediction: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is an entailment given the context.

The context provides information about the availability of the sitcom on the digital subchannel "GetTV" during weekday mornings. Although it does not provide any information about the availability of the second season, there is no indication in the context that the series is not complete, hence it can be inferred from the context that all seasons can be watched. Therefore, the statement about watching the second season in the morning from Monday to Friday is entailment, as it does follow from the context. Final Answer: Entailment


did you see how i change the previous prediction? change as little as possible to make it correct! change as little as possible to make it correct!change as little as possible to make it correct!change as little as possible to make it correct!change as little as possible to make it correct!change as little as possible to make it correct!






Example(The previous prediction is already correct. so we just add Final Answer: Neutral at the end): 
Context: Conceptually cream skimming has two basic dimensions - product and geography.
Statement: Product and geography are what make cream skimming work. 
Groundtruth: Neutral
Previous Prediction: Sure, I\'d be happy to help!\n\n1. Based on the context and the statement, let\'s analyze whether the statement is entailment, contradiction, or neutral:\n\nThe statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\n2. Since we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work.

Minimum Changed Prediction: Sure, I\'d be happy to help!\n\n1. Based on the context and the statement, let\'s analyze whether the statement is entailment, contradiction, or neutral:\n\nThe statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\n2. Since we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work. Final Answer: Neutral

did you see how i change the previous prediction? don't change if it is already correct!!!don't change if it is already correct!!!don't change if it is already correct!!!don't change if it is already correct!!!



Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the "Final Answer: {gold_label}" at the end.

We have the NLI Question below: 

Determine whether the statement is entailment(true), contradiction(false), or neutral(undetermined or irrelevant) with the given context.Context: {premise} 
Statement: {statement}
Groundtruth: {gold_label}

Previous prediction: "{previous_prediction}"
Minimum Changed Prediction: 
"""


        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['answer'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        input_json_list.append(temp)
    return input_json_list


def minimum_change_data_creation_ESNLI(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        premise = item['premise']
        statement = item['hypothesis']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

        prompt = f"""We have the NLI Question below: 

Determine whether the statement is entailment(true), contradiction(false), or neutral(undetermined or irrelevant) with the given context. Please inference first then Place the Final Answer: Entailment/Neutral/Contradiction at the end.

Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. Place the Final Answer: Entailment/Neutral/Contradiction at the end .Please make sure nothing is placed after Final Answer: Entailment/Neutral/Contradiction


Example(We are trying to change as little as possible to make the previous prediction correct. look at this example. it modify as less as possible. it also put Final Answer: Entailment at the end. It try to not change the words from previous prediction if it is not wrong. ): 
Context: The Ghost & Mrs. Muir is an American sitcom based on the 1947 film of the same name, which was based on the 1945 novel by R. A. Dick. It premiered in September 1968 on NBC. After NBC canceled the series, it aired on ABC for one season before being canceled a final time. The program is currently seen weekday mornings on the digital subchannel "GetTV."
Statement: The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday
Groundtruth: Entailment
Previous Prediction: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is neutral given the context.

The context provides information about the history of the sitcom, including its cancellation on NBC and brief run on ABC. However, it does not provide any information about the availability of the second season. Therefore, the statement about watching the second season in the morning from Monday to Friday is not entailment or contradiction, as it does not necessarily follow from the context. Final Answer: Neutral

Minimum Changed Prediction: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is an entailment given the context.

The context provides information about the availability of the sitcom on the digital subchannel "GetTV" during weekday mornings. Although it does not provide any information about the availability of the second season, there is no indication in the context that the series is not complete, hence it can be inferred from the context that all seasons can be watched. Therefore, the statement about watching the second season in the morning from Monday to Friday is entailment, as it does follow from the context. Final Answer: Entailment


did you see how i change the previous prediction? change as little as possible to make it correct! change as little as possible to make it correct!change as little as possible to make it correct!change as little as possible to make it correct!change as little as possible to make it correct!change as little as possible to make it correct!






Example(The previous prediction is already correct. so we just add Final Answer: Neutral at the end): 
Context: Conceptually cream skimming has two basic dimensions - product and geography.
Statement: Product and geography are what make cream skimming work. 
Groundtruth: Neutral
Previous Prediction: Sure, I\'d be happy to help!\n\n1. Based on the context and the statement, let\'s analyze whether the statement is entailment, contradiction, or neutral:\n\nThe statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\n2. Since we cannot determine the truth of the statement with certainty, the answer is neutral. Final Answer: Neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work.

Minimum Changed Prediction: Sure, I\'d be happy to help!\n\n1. Based on the context and the statement, let\'s analyze whether the statement is entailment, contradiction, or neutral:\n\nThe statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\n2. Since we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work. Final Answer: Neutral

did you see how i change the previous prediction? don't change if it is already correct!!!don't change if it is already correct!!!don't change if it is already correct!!!don't change if it is already correct!!!



Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the Final Answer: {gold_label} at the end. Please make sure nothing is placed after Final Answer: {gold_label}.

We have the NLI Question below: 

Determine whether the statement is entailment(true), contradiction(false), or neutral(undetermined or irrelevant) with the given context.
Context: {premise} 
Statement: {statement}
Groundtruth: {gold_label}

Previous prediction: {previous_prediction}
Minimum Changed Prediction: 
"""

        
                
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        if 'previous pre' in answer.lower() or 'previous ans' in answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
            if 'previous pre' in answer.lower() or 'previous ans' in  answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['answer'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        input_json_list.append(temp)
    return input_json_list





def minimum_change_data_creation_ESNLI_GPT4_Initial_Prediction(data_list, gpt4_train_data_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    
    def extract_nli_answer_item(text):
        text = text.lower()
        found_answer_num = text.count('final answer:')
        if found_answer_num > 0:
            text = text.split("final answer:")[-1].strip()
        neutral_count = 0
        entail_count = 0
        contradiction_count = 0
        total_count = 0
        for item in text.split():
            if 'neutr' in item:
                neutral_count += 1
                total_count += 1
            if 'entail' in item:
                entail_count += 1
                total_count += 1
            if 'contrad' in item:
                contradiction_count += 1
                total_count += 1
        if total_count == 1 and neutral_count == 1:
            final_answer = 'neutral'
        elif total_count == 1 and entail_count == 1:
            final_answer = 'entailment'
        elif total_count == 1 and contradiction_count == 1:
            final_answer = 'contradiction'
        else:
            final_answer = 'null'
        return final_answer


    def calc_accuracy_ESNLI_item(gold_label, previous_prediction):
        final_answer = extract_nli_answer_item(previous_prediction)
        if gold_label.lower() == final_answer.lower():
            correctness = True
        else:
            correctness = False
        return correctness

    ESNLI_GPT_Initial_prediction_minimum_change_list = []
    ESNLI_GPT4_Initial_prediction_data_mixtrue_list = []
    for i, item in enumerate(data_list):
        gpt4_predict = gpt4_train_data_list[i]['answer']
        gpt4_predict = gpt4_predict.replace('FINAL ANSWER', 'Final Answer')
        previous_prediction = item['answer']
        question = item['question']
        filtered_question = question.replace('We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\n', '')
        filtered_question = filtered_question.replace("\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context. \n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'\n", '')
        gold_label = item['gold_label']


        correctness = calc_accuracy_ESNLI_item(gold_label, previous_prediction)
        if not correctness:
            prompt = f"""We have a NLI Question, groundtruth and previous prediction.
Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. make sure Final Answer: Entailment/Neutral/Contradiction is placed at the end .Please make sure nothing is placed after Final Answer: Entailment/Neutral/Contradiction


Example(We are trying to change as little as possible to make the previous prediction correct. look at this example. it modify as less as possible. it also put Final Answer: Entailment at the end. It try to not change the words from previous prediction if it is not wrong. ): 

We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.
Context: The Ghost & Mrs. Muir is an American sitcom based on the 1947 film of the same name, which was based on the 1945 novel by R. A. Dick. It premiered in September 1968 on NBC. After NBC canceled the series, it aired on ABC for one season before being canceled a final time. The program is currently seen weekday mornings on the digital subchannel "GetTV."
Statement: The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday
Determine whether the statement is entailment, contradiction, or neutral given the context. Please inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'

Groundtruth: Entailment

Previous Prediction: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is neutral given the context.
The context provides information about the history of the sitcom, including its cancellation on NBC and brief run on ABC. However, it does not provide any information about the availability of the second season. Therefore, the statement about watching the second season in the morning from Monday to Friday is not entailment or contradiction, as it does not necessarily follow from the context. Final Answer: Neutral

Minimum Changed Prediction: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is an entailment given the context.
The context provides information about the availability of the sitcom on the digital subchannel "GetTV" during weekday mornings. Although it does not provide any information about the availability of the second season, there is no indication in the context that the series is not complete, hence it can be inferred from the context that all seasons can be watched. Therefore, the statement about watching the second season in the morning from Monday to Friday is entailment, as it does follow from the context. Final Answer: Entailment





Example(The previous prediction is already correct, but it does not have Final Answer: GOLD_LABEL, so we just add Final Answer: Neutral at the end): 

We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.
Context: Conceptually cream skimming has two basic dimensions - product and geography.
Statement: Product and geography are what make cream skimming work. 
Determine whether the statement is entailment, contradiction, or neutral given the context. \n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'\n

Groundtruth: Neutral

Previous Prediction: Sure, I\'d be happy to help!\n\n1. Based on the context and the statement, let\'s analyze whether the statement is entailment, contradiction, or neutral:\n\nThe statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\n2. Since we cannot determine the truth of the statement with certainty, the answer is neutral. Final Answer: Neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work.

Minimum Changed Prediction: Sure, I\'d be happy to help!\n\n1. Based on the context and the statement, let\'s analyze whether the statement is entailment, contradiction, or neutral:\n\nThe statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\n2. Since we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work. Final Answer: Neutral


Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the Final Answer: {gold_label} at the end. Please make sure nothing is placed after Final Answer: {gold_label}.

We have the NLI Question below: 

We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.
{filtered_question}
Determine whether the statement is entailment, contradiction, or neutral given the context. \n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'

Groundtruth: {gold_label}

Previous prediction: {previous_prediction}

Minimum Changed Prediction: 
"""

        
                
            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

            if 'previous pre' in answer.lower() or 'previous ans' in answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
                if 'previous pre' in answer.lower() or 'previous ans' in  answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                    answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        else:
            answer = previous_prediction

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['answer'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        ESNLI_GPT_Initial_prediction_minimum_change_list.append(temp)
        temp_copy = temp.copy()
        if not correctness:
            temp_copy['answer'] = gpt4_predict
            ESNLI_GPT4_Initial_prediction_data_mixtrue_list.append(temp_copy)
        else:
            ESNLI_GPT4_Initial_prediction_data_mixtrue_list.append(temp_copy)

    return ESNLI_GPT_Initial_prediction_minimum_change_list, ESNLI_GPT4_Initial_prediction_data_mixtrue_list




def minimum_change_data_creation_GSM8K_GPT4_Initial_Prediction(data_list, groundtruth_num_list, groundtruth_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    from utils.data_loader import eval_MATH_correctness
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer


    def extract_boxed_content(s):
        start = s.rfind('\\boxed{')
        if start == -1:
            return None
        
        end = s.rfind('}')
            
        if end != 0:
            answer = s[start + 7 : end]
            return answer  # 7


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

        

    GSM8K_GPT_Initial_prediction_minimum_change_list = []
    for i, item in enumerate(data_list):
        previous_prediction = item['answer']
        previous_prediction = previous_prediction.replace('FINAL ANSWER', 'Final Answer')
        question = item['question']
        groundtruth = groundtruth_list[i]

        temp = {}
        gold_label = groundtruth_num_list[i]
        previous_prediction_num = extract_last_number(previous_prediction)
        correctness = eval_MATH_correctness(previous_prediction_num, gold_label)
        if not correctness:
            prompt = f"""We have a question, a groundtruth and a previous prediction. 
1. Please perform minimum change on the prediction to make it correct. When I say minimum change, I mean you should change as little as possible to make the prediction correct. Groundtruth is just an answer that help you to understand the solution. Do not directly copy the groundtruth, because ou should try to keep as much original words as possible.
2. Please also remove the hallucination, create your own answer when the inference steps from the prediction is empty.
3. Please put the final answer at the end after Final Answer:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. Final Answer: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help
you with that !" or "Here is the modified answer."

Example1: The inference steps of the previous prediction is mostly correct, but it made some mistakes and it does not use the correct format. The minimum changed prediction correct the mistakes and keep as much original words as possible. We also remove the $ sign after 'Final Answer:' to make sure what is placed at the end is a number.
Example1-question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'
Example1-groundtruth: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\nFinal Answer: 5
Example1-previous prediction: Betty's parents gave her $15, and her grandparents gave her twice that amount. So, her grandparents gave her $15 * 2 = $30.\n\nBetty has received a total of $15 (from her parents) + $30 (from her grandparents) = $45.\n\nThe wallet costs $100, so Betty still needs $100 - $45 = $55 to buy the wallet.\n\nFinal Answer: $55
Example1-minimum changed prediction: Betty's parents gave her $15, and her grandparents gave her twice that amount. So, her grandparents gave her $15 * 2 = $30.\n\nBetty originally had $50, which is half of what she needs for the wallet.\n\nBetty has received a total of $15 (from her parents) + $30 (from her grandparents) + $50 (originally had) = $95.\n\nThe wallet costs $100, so Betty still needs $100 - $95 = $5 to buy the wallet.\n\nFinal Answer: 5

Example2: The previous prediction is already correct, but it does not use the correct format. It suppose to add the Final Answer: at the end with a number. So we copy the previous prediction and replace '72 clips' with '72' at the end.
Example2-question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'
Example2-groundtruth: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\nFinal Answer: 72
Example2-previous prediction: Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips.\n\nSo, Natalia sold a total of 48 (April) + 24 (May) = 72 clips in April and May.\n\nFinal Answer: 72 clips
Example2-minimum changed prediction: Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips.\n\nSo, Natalia sold a total of 48 (April) + 24 (May) = 72 clips in April and May.\n\nFinal Answer: 72


given the question, groundtruth and prediction, please perform minimum change to make the prediction correct. remember to perform minimum change.
question: {question}
Groundtruth: {groundtruth}
Previous Prediction: {previous_prediction}
Minimum Changed Prediction: """

            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

            if 'previous pre' in answer.lower() or 'previous ans' in answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
                if 'previous pre' in answer.lower() or 'previous ans' in  answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                    answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        else:
            answer = previous_prediction

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['answer'] = answer
        temp['groundtruth'] = groundtruth
        temp['previous_prediction'] = previous_prediction
        GSM8K_GPT_Initial_prediction_minimum_change_list.append(temp)
    return GSM8K_GPT_Initial_prediction_minimum_change_list







def minimum_change_data_creation_MATH_GPT4_Initial_Prediction(data_list, groundtruth_num_list, groundtruth_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    from utils.data_loader import eval_MATH_correctness
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer


    def extract_boxed_content(s):
        start = s.rfind('\\boxed{')
        if start == -1:
            return None
        
        end = s.rfind('}')
            
        if end != 0:
            answer = s[start + 7 : end]
            return answer  # 7


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

        

    MATH_GPT_Initial_prediction_minimum_change_list = []
    for i, item in enumerate(data_list):
        previous_prediction = item['answer']
        previous_prediction = previous_prediction.replace('FINAL ANSWER', 'Final Answer')
        question = item['question']
        groundtruth = groundtruth_list[i]

        temp = {}
        gold_label = groundtruth_num_list[i]
        previous_prediction_num = extract_last_number(previous_prediction)
        correctness = eval_MATH_correctness(previous_prediction_num, gold_label)
        aa = """We complete the square.\n\nFactoring $-6$ out of the quadratic and linear terms gives $-6x^2 + 36x = -6(x^2-6x)$.\n\nSince $(x-3)^2 = x^2 - 6x + 9$, we can write $$-6(x-3)^2 = -6x^2 + 36x - 54.$$This quadratic agrees with the given $-6x^2 + 36x + 216$ in all but the constant term. We can write\n\n\\begin{align*}\n-6x^2 + 36x + 216 &= (-6x^2 + 36x - 54) + 270 \\\\\n&= -6(x-3)^2 + 270.\n\\end{align*}Therefore, $a=-6$, $b=-3$, $c=270$, and $a+b+c = -6-3+270 = \\boxed{261}$. Final Answer: 261","""
        bb = """To convert the given quadratic equation $-6x^2+36x+216$ into the form $a(x+b)^2+c$, it needs to be written in vertex form through completing the square:\n\nFirstly, factor out the coefficient of $x^2$ term: \n$-6(x^2 - 6x) - 216$\n\nNext, find the value that completes the square for the expression $x^2 - 6x$. This value would be $(\\frac{6}{2})^2=9$. Remember to balance the equation by also subtracting $9 * -6$ (since you've factored -6 out in the first step), which is $-54$:\n\n$-6(x^2 - 6x + 9) - 216 + 54$\n\nSimplify the corrected equation:\n$-6(x - 3)^2 - 162$\n\nNow it is in the form $a(x+b)^2+c$ with $a=-6,$ $b=-3$, and $c=-162$.\n\nSo the sum of $a+b+c$ is:\n$(-6) + (-3) + (-162) = -6 - 3 - 162 = -171$\n\nFinal Answer: -171"""
        cc = """To convert the given quadratic equation $-6x^2+36x+216$ into the form $a(x+b)^2+c$, it needs to be written in vertex form through completing the square:\n\nFirstly, factor out the coefficient of $x^2$ term: \n$-6(x^2 - 6x) + 216$\n\nNext, find the value that completes the square for the expression $x^2 - 6x$. This value would be $(\\frac{6}{2})^2=9$. Remember to balance the equation by also subtracting $9 * -6$ (since you've factored -6 out in the first step), which is $-54$:\n\n$-6(x^2 - 6x + 9) + 216 -(-54)$\n\nSimplify the corrected equation:\n$-6(x - 3)^2 + 270$\n\nNow it is in the form $a(x+b)^2+c$ with $a=-6,$ $b=-3$, and $c=270$.\n\nSo the sum of $a+b+c$ is:\n$(-6) + (-3) + (270) = -6 - 3 + 270 = 261$\n\nFinal Answer: 261"""
        dd = """We start by completing the square.  \\[x^2-10x+24=(x-5)^2-1.\\] Since the square of a real number is at least 0, $(x-5)^2\\ge 0$ and $(x-5)^2-1 \\ge -1.$ Thus, the minimum value of the quadratic is $-1,$ which occurs when $x=\\boxed{5}.$ Final Answer: 5"""
        if not correctness:
            prompt = f"""We have a question, a groundtruth and a previous prediction. 
1. Please perform minimum change on the prediction to make it correct. When I say minimum change, I mean you should change as little as possible to make the prediction correct. Groundtruth is just an answer that help you to understand the solution. Do not directly copy the groundtruth, because ou should try to keep as much original words as possible.
2. Please also remove the hallucination, create your own answer when the inference steps from the prediction is empty.
3. Please put the final answer at the end after Final Answer:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. Final Answer: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help
you with that !" or "Here is the modified answer."

Example1: The inference steps of the previous prediction is mostly correct, but it made some mistakes and it does not use the correct format. The minimum changed prediction correct the mistakes and keep as much original words as possible. We also remove the $ sign after 'Final Answer:' to make sure what is placed at the end is a number.
Example1-question: The quadratic $-6x^2+36x+216$ can be written in the form $a(x+b)^2+c$, where $a$, $b$, and $c$ are constants. What is $a+b+c$?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n
Example1-groundtruth: {aa}
Example1-previous prediction: {bb}
Example1-minimum changed prediction: {cc}


Example2: The previous prediction is already correct, but it does not use the correct format. It suppose to add the Final Answer: at the end with a number. So we copy the previous prediction and replace 'Therefore, the final answer is 5' with 'Final Answer: 5' at the end.
Example2-question: What value of $x$ will give the minimum value for $x^2- 10x + 24$?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n
Example2-groundtruth: {dd}
Example2-previous prediction: The function $x^2 - 10x + 24$ is a quadratic equation in standard form, where $a = 1$, $b = -10$, and $c = 24$. For a quadratic function $ax^2 + bx + c$, the $x$ value that gives the minimum (if $a > 0$, which is the case here) is given by the vertex of the parabola, located at $x = -b/(2a)$.\n\nPlugging in the values of $b$ and $a$ into this formula, we get:\n\n$x = -(-10)/(2*1)$\n$x = 10/2$\n$x = 5$\n\nSo, the value of $x$ that will give the minimum value for the quadratic function $x^2 - 10x + 24$ is $5$.\n\nTherefore, the final answer is 5
Example2-minimum changed prediction: The function $x^2 - 10x + 24$ is a quadratic equation in standard form, where $a = 1$, $b = -10$, and $c = 24$. For a quadratic function $ax^2 + bx + c$, the $x$ value that gives the minimum (if $a > 0$, which is the case here) is given by the vertex of the parabola, located at $x = -b/(2a)$.\n\nPlugging in the values of $b$ and $a$ into this formula, we get:\n\n$x = -(-10)/(2*1)$\n$x = 10/2$\n$x = 5$\n\nSo, the value of $x$ that will give the minimum value for the quadratic function $x^2 - 10x + 24$ is $5$.\n\nFinal Answer: 5


given the question, groundtruth and prediction, please perform minimum change to make the prediction correct. remember to perform minimum change.
question: {question}
Groundtruth: {groundtruth}
Previous Prediction: {previous_prediction}
Minimum Changed Prediction: """

            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

            if 'previous pre' in answer.lower() or 'previous ans' in answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
                if 'previous pre' in answer.lower() or 'previous ans' in  answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                    answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        else:
            answer = previous_prediction

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['answer'] = answer
        temp['groundtruth'] = groundtruth
        temp['previous_prediction'] = previous_prediction
        MATH_GPT_Initial_prediction_minimum_change_list.append(temp)
    return MATH_GPT_Initial_prediction_minimum_change_list


def minimum_change_data_creation_MATH_algebra_GPT4_Initial_Prediction(data_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    
    def extract_nli_answer_item(text):
        text = text.lower()
        found_answer_num = text.count('final answer:')
        if found_answer_num > 0:
            text = text.split("final answer:")[-1].strip()
        neutral_count = 0
        entail_count = 0
        contradiction_count = 0
        total_count = 0
        for item in text.split():
            if 'neutr' in item:
                neutral_count += 1
                total_count += 1
            if 'entail' in item:
                entail_count += 1
                total_count += 1
            if 'contrad' in item:
                contradiction_count += 1
                total_count += 1
        if total_count == 1 and neutral_count == 1:
            final_answer = 'neutral'
        elif total_count == 1 and entail_count == 1:
            final_answer = 'entailment'
        elif total_count == 1 and contradiction_count == 1:
            final_answer = 'contradiction'
        else:
            final_answer = 'null'
        return final_answer


    def calc_accuracy_ESNLI_item(gold_label, previous_prediction):
        final_answer = extract_nli_answer_item(previous_prediction)
        if gold_label.lower() == final_answer.lower():
            correctness = True
        else:
            correctness = False
        return correctness

    ESNLI_GPT_Initial_prediction_minimum_change_list = []
    ESNLI_GPT4_Initial_prediction_data_mixtrue_list = []
    for i, item in enumerate(data_list):
        previous_prediction = item['answer']
        previous_prediction = previous_prediction.replace('FINAL ANSWER', 'Final Answer')
        question = item['question']
        filtered_question = question.replace('We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\n', '')
        filtered_question = filtered_question.replace("\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context. \n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'\n", '')
        gold_label = item['gold_label']


        correctness = calc_accuracy_ESNLI_item(gold_label, previous_prediction)
        if not correctness:
            prompt = f"""We have a NLI Question, groundtruth and previous prediction.
Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. make sure Final Answer: Entailment/Neutral/Contradiction is placed at the end .Please make sure nothing is placed after Final Answer: Entailment/Neutral/Contradiction


Example(We are trying to change as little as possible to make the previous prediction correct. look at this example. it modify as less as possible. it also put Final Answer: Entailment at the end. It try to not change the words from previous prediction if it is not wrong. ): 

We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.
Context: The Ghost & Mrs. Muir is an American sitcom based on the 1947 film of the same name, which was based on the 1945 novel by R. A. Dick. It premiered in September 1968 on NBC. After NBC canceled the series, it aired on ABC for one season before being canceled a final time. The program is currently seen weekday mornings on the digital subchannel "GetTV."
Statement: The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday
Determine whether the statement is entailment, contradiction, or neutral given the context. Please inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'

Groundtruth: Entailment

Previous Prediction: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is neutral given the context.
The context provides information about the history of the sitcom, including its cancellation on NBC and brief run on ABC. However, it does not provide any information about the availability of the second season. Therefore, the statement about watching the second season in the morning from Monday to Friday is not entailment or contradiction, as it does not necessarily follow from the context. Final Answer: Neutral

Minimum Changed Prediction: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is an entailment given the context.
The context provides information about the availability of the sitcom on the digital subchannel "GetTV" during weekday mornings. Although it does not provide any information about the availability of the second season, there is no indication in the context that the series is not complete, hence it can be inferred from the context that all seasons can be watched. Therefore, the statement about watching the second season in the morning from Monday to Friday is entailment, as it does follow from the context. Final Answer: Entailment





Example(The previous prediction is already correct, but it does not have Final Answer: GOLD_LABEL, so we just add Final Answer: Neutral at the end): 

We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.
Context: Conceptually cream skimming has two basic dimensions - product and geography.
Statement: Product and geography are what make cream skimming work. 
Determine whether the statement is entailment, contradiction, or neutral given the context. \n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'\n

Groundtruth: Neutral

Previous Prediction: Sure, I\'d be happy to help!\n\n1. Based on the context and the statement, let\'s analyze whether the statement is entailment, contradiction, or neutral:\n\nThe statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\n2. Since we cannot determine the truth of the statement with certainty, the answer is neutral. Final Answer: Neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work.

Minimum Changed Prediction: Sure, I\'d be happy to help!\n\n1. Based on the context and the statement, let\'s analyze whether the statement is entailment, contradiction, or neutral:\n\nThe statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\n2. Since we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work. Final Answer: Neutral


Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the Final Answer: {gold_label} at the end. Please make sure nothing is placed after Final Answer: {gold_label}.

We have the NLI Question below: 

We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.
{filtered_question}
Determine whether the statement is entailment, contradiction, or neutral given the context. \n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'

Groundtruth: {gold_label}

Previous prediction: {previous_prediction}

Minimum Changed Prediction: 
"""

        
                
            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

            if 'previous pre' in answer.lower() or 'previous ans' in answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
                if 'previous pre' in answer.lower() or 'previous ans' in  answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                    answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        else:
            answer = previous_prediction

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['answer'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        ESNLI_GPT_Initial_prediction_minimum_change_list.append(temp)
    return ESNLI_GPT_Initial_prediction_minimum_change_list



def minimum_change_data_creation_BOOLQ_GPT4_Initial_Prediction(data_list, gpt4_train_data_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    
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

    BOOLQ_GPT_Initial_prediction_minimum_change_list = []
    BOOLQ_GPT4_Initial_prediction_data_mixtrue_list = []
    for i, item in enumerate(data_list):
        gpt4_predict = gpt4_train_data_list[i]['answer']
        gpt4_predict = gpt4_predict.replace('FINAL ANSWER', 'Final Answer')
        previous_prediction = item['answer']
        question = item['question']
        gold_label = item['gold_label']

        predicted_bool = extract_bool(previous_prediction)
        if predicted_bool.lower() == gold_label.lower():
            correctness = True
        else:
            correctness = False

        if not correctness:
            prompt = f"""We have the question, the previous prediction and the groundtruth.
Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the "Final Answer: {gold_label}" at the end.


Let me repeat the most important cretiria: When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 

Example1: the previous prediction is wrong. thus we perform minimum change to correctit. as you can see, the minimum changed prediction is similar to the previous prediction. In addition, make sure Place the Final Answer: {gold_label} at the end.
Example1-question: Given the context: Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.\n\ndo iran and afghanistan speak the same language?\n\nPlease inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'
Example1-groundtruth: True
Example1-previous prediction: Persian, also known as Farsi, is the predominant language in Iran. Dari is the name used for the Persian language in Afghanistan since 1958. While there are some differences between the two, they are mutually intelligible to a large extent. This means that speakers of both languages can understand each other with some effort. However, there are distinct variations in vocabulary, pronunciation, and grammar between the two.\n\nFinal Answer:\nWhile Persian and Dari (Afghan Persian) are closely related and mutually intelligible to a significant degree, they are not exactly the same language. Therefore, the answer is False.
Example1-minimum changed prediction: Persian, also known as Farsi, is the predominant language in Iran. Dari is the name used for the Persian language in Afghanistan since 1958. The two are dialects of the same language, and while there are some differences between them, they are mutually intelligible. This means that speakers of both can generally understand each other without too much effort. There may be variations in vocabulary, pronunciation, and grammar, but Iranians and Afghans essentially speak the same language, Persian.\n\nFinal Answer: True


Example2: The previous predection is 100% correct. We just copy it with the priod at the end removed. In this way, we keep as much original words as possible.
Example2-question: Given the context: Good Samaritan laws offer legal protection to people who give reasonable assistance to those who are, or who they believe to be, injured, ill, in peril, or otherwise incapacitated. The protection is intended to reduce bystanders' hesitation to assist, for fear of being sued or prosecuted for unintentional injury or wrongful death. An example of such a law in common-law areas of Canada: a good Samaritan doctrine is a legal principle that prevents a rescuer who has voluntarily helped a victim in distress from being successfully sued for wrongdoing. Its purpose is to keep people from being reluctant to help a stranger in need for fear of legal repercussions should they make some mistake in treatment. By contrast, a duty to rescue law requires people to offer assistance and holds those who fail to do so liable.\n\ndo good samaritan laws protect those who help at an accident?\n\nPlease inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'
Example2-groundtruth: True
Example2-previous prediction: Inference: Based on the context provided, Good Samaritan laws do protect individuals who provide assistance at the scene of an accident, as long as their actions are considered reasonable.\n\nFinal Answer: True.
Example2-minimum changed prediction: Inference: Based on the context provided, Good Samaritan laws do protect individuals who provide assistance at the scene of an accident, as long as their actions are considered reasonable.\n\nFinal Answer: True


We have the Question: {question} 
Groundtruth: {gold_label}
Previous prediction: {previous_prediction}
Minimum Changed Prediction: 
"""

            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

            if 'previous pre' in answer.lower() or 'previous ans' in answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
                if 'previous pre' in answer.lower() or 'previous ans' in  answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                    answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        else:
            answer = previous_prediction

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['answer'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        BOOLQ_GPT_Initial_prediction_minimum_change_list.append(temp)
        temp_copy = temp.copy()
        if not correctness:
            temp_copy['answer'] = gpt4_predict
            BOOLQ_GPT4_Initial_prediction_data_mixtrue_list.append(temp_copy)
        else:
            BOOLQ_GPT4_Initial_prediction_data_mixtrue_list.append(temp_copy)

    return BOOLQ_GPT_Initial_prediction_minimum_change_list, BOOLQ_GPT4_Initial_prediction_data_mixtrue_list



# def minimum_change_data_creation_WINOGRANDE_GPT4_Initial_Prediction(data_list):
#     from openai import OpenAI
#     import openai
#     from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
#     client = OpenAI(api_key=GPT_API)
#     @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
#     def create_gpt_completion(qa_, model_engine, client):
#         response = client.chat.completions.create(
#             model=model_engine,
#             messages=[
#                 {"role": "system", "content": "provide answer"},
#                 {"role": "user", "content": f"{qa_}"}
#             ]
#         )
#         answer = response.choices[0].message.content
#         return answer
    
    
#     def extract_option(text):
#         text = text.lower()
#         found_answer_num = text.count('answer:')
#         if found_answer_num > 0:
#             text = text.split("answer:")[-1].strip()
#         else:
#             found_answer_num = text.count('answer:')
#             if found_answer_num > 0:
#                 text = text.split("answer:")[-1].strip()
#         if "1" in text or "2" in text:
#             index_1 = text.find("1")
#             index_2 = text.find("2")
#             if index_1 != -1 and (index_2 == -1 or index_1 < index_2):
#                 return "1"
#             elif index_2 != -1:
#                 return "2"
#         else:
#             return "null"

#     WINOGRANDE_GPT_Initial_prediction_minimum_change_list = []
#     for i, item in enumerate(data_list):
#         previous_prediction = item['answer']
#         question = item['question']
#         gold_label = item['gold_label']

#         predicted_bool = extract_option(previous_prediction)
#         if predicted_bool.lower() == gold_label.lower():
#             correctness = True
#         else:
#             correctness = False

#         if not correctness:
#             prompt = f"""We have the question, the previous prediction and the groundtruth.
# Your task:
# 1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 
# 2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
# 3. If the previous prediction is already correct, then do not change anything. just copy it.
# 4. Place the "Final Answer: {gold_label}" at the end.


# Let me repeat the most important cretiria: When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 

# Example1: the previous prediction is wrong. thus we perform minimum change to correctit. as you can see, the minimum changed prediction is similar to the previous prediction. In addition, make sure Place the Final Answer: 2 at the end.
# Example1-question: Given the question: Ian volunteered to eat Dennis's menudo after already having a bowl because _ despised eating intestine.\n\nWhat option is correct?\nOption 1: Ian\nOption 2: Dennis\n\nPlease inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'\n
# Example1-groundtruth: 2
# Example1-previous prediction: The sentence suggests that Ian volunteered to eat the menudo, which is a traditional Mexican soup that typically includes tripe (cow's stomach). The blank, \"_ despised eating intestine,\" is intended to refer to the person who disliked eating intestine, leading to Ian's decision to eat Dennis's menudo. Since Ian is the one who volunteered, it implies that Ian is the person who despised eating intestine, as it would be unlikely for Dennis to volunteer to eat something he himself despised. \n\nFINAL ANSWER: 1
# Example1-minimum changed prediction: The sentence suggests that Ian volunteered to eat the menudo, which is a traditional Mexican soup that typically includes tripe (cow's stomach). The blank, \"_ despised eating intestine,\" is intended to refer to the person who disliked eating intestine, leading to Ian's decision to eat Dennis's menudo. Since Ian is the one who volunteered, it implies that Dennis is the person who despised eating intestine, as it would be unlikely for Ian to volunteer to eat something he himself despised. \n\nFinal Answer: 2



# Example2: The previous predection is 100% correct. We just copy it with the priod at the end removed. In this way, we keep as much original words as possible. In addition, make sure Place the Final Answer: 1 at the end.
# Example2-question: Given the question: Ian volunteered to eat Dennis's menudo after already having a bowl because _ enjoyed eating intestine.\n\nWhat option is correct?\nOption 1: Ian\nOption 2: Dennis\n\nPlease inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'\n
# Example2-groundtruth: 1
# Example2-previous prediction: The sentence specifies that Ian volunteered to eat Dennis's menudo, indicating that Ian was willing to eat the dish. The implication here is that Ian did this after already having a bowl himself, which suggests that Ian is the one who enjoyed eating intestine, as he was willing to have more of the menudo.\n\nFINAL ANSWER: Ian
# Example2-minimum changed prediction: The sentence specifies that Ian volunteered to eat Dennis's menudo, indicating that Ian was willing to eat the dish. The implication here is that Ian did this after already having a bowl himself, which suggests that Ian is the one who enjoyed eating intestine, as he was willing to have more of the menudo. \n\nFinal Answer: 1


# We have the Question: {question} 
# Groundtruth: {gold_label}
# Previous prediction: {previous_prediction}
# Minimum Changed Prediction: 
# """

#             answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

#             if 'previous pre' in answer.lower() or 'previous ans' in answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
#                 answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
#                 if 'previous pre' in answer.lower() or 'previous ans' in  answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
#                     answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
#         else:
#             answer = previous_prediction

#         temp = {}
#         temp['question'] = item['question']
#         temp['input'] = ''
#         temp['answer'] = answer
#         temp['groundtruth'] = gold_label
#         temp['previous_prediction'] = previous_prediction
#         WINOGRANDE_GPT_Initial_prediction_minimum_change_list.append(temp)

#     return WINOGRANDE_GPT_Initial_prediction_minimum_change_list



def minimum_change_data_creation_WINOGRANDE_GPT4_Initial_Prediction(data_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    
    
    def extract_option(text):
        text = text.lower()
        found_answer_num = text.count('answer:')
        if found_answer_num > 0:
            text = text.split("answer:")[-1].strip()
        else:
            found_answer_num = text.count('answer:')
            if found_answer_num > 0:
                text = text.split("answer:")[-1].strip()
        if "1" in text or "2" in text:
            index_1 = text.find("1")
            index_2 = text.find("2")
            if index_1 != -1 and (index_2 == -1 or index_1 < index_2):
                return "1"
            elif index_2 != -1:
                return "2"
        else:
            return "null"

    WINOGRANDE_GPT_Initial_prediction_minimum_change_list = []
    for i, item in enumerate(data_list):
        previous_prediction = item['answer']
        question = item['question']
        gold_label = item['gold_label']

        predicted_bool = extract_option(previous_prediction)
        if predicted_bool.lower() == gold_label.lower():
            correctness = True
        else:
            correctness = False

        if not correctness:
            prompt = f"""We have the question, the previous prediction and the groundtruth.
Your task:
1. According to groundtruth answer, please use Minimum Change to modify the previous prediction to make it correct. When modify the previous prediction, please try to keep the original words and logic styles and correct the wrong part. make sure the corrected prediction support the answer label. In other words, the corrected prediction should sounds similar and inference in similar way as the previous prediction.
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the "Final Answer: {gold_label}" at the end.


Let me repeat the most important cretiria-Minimum Change: When modify the previous prediction, please try to keep the original words and logic styles and correct the wrong part. make sure the corrected prediction support the answer label. In other words, the corrected prediction should sounds similar and inference in similar way as the previous prediction.

Example1: the previous prediction is wrong. thus we perform minimum change to correctit. as you can see, the minimum changed prediction is similar to the previous prediction. In addition, make sure Place the Final Answer: 2 at the end.
Example1-question: Given the question: Ian volunteered to eat Dennis's menudo after already having a bowl because _ despised eating intestine.\n\nWhat option is correct?\nOption 1: Ian\nOption 2: Dennis\n\nPlease inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'\n
Example1-groundtruth: 2
Example1-previous prediction: The sentence suggests that Ian volunteered to eat the menudo, which is a traditional Mexican soup that typically includes tripe (cow's stomach). The blank, \"_ despised eating intestine,\" is intended to refer to the person who disliked eating intestine, leading to Ian's decision to eat Dennis's menudo. Since Ian is the one who volunteered, it implies that Ian is the person who despised eating intestine, as it would be unlikely for Dennis to volunteer to eat something he himself despised. \n\nFINAL ANSWER: 1
Example1-minimum changed prediction: The sentence suggests that Ian volunteered to eat the menudo, which is a traditional Mexican soup that typically includes tripe (cow's stomach). The blank, \"_ despised eating intestine,\" is intended to refer to the person who disliked eating intestine, leading to Ian's decision to eat Dennis's menudo. Since Ian is the one who volunteered, it implies that Dennis is the person who despised eating intestine, as it would be unlikely for Ian to volunteer to eat something he himself despised. \n\nFinal Answer: 2



Example2: The previous predection is 100% correct. We just copy it with the priod at the end removed. In this way, we keep as much original words as possible. In addition, make sure Place the Final Answer: 1 at the end.
Example2-question: Given the question: Ian volunteered to eat Dennis's menudo after already having a bowl because _ enjoyed eating intestine.\n\nWhat option is correct?\nOption 1: Ian\nOption 2: Dennis\n\nPlease inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'\n
Example2-groundtruth: 1
Example2-previous prediction: The sentence specifies that Ian volunteered to eat Dennis's menudo, indicating that Ian was willing to eat the dish. The implication here is that Ian did this after already having a bowl himself, which suggests that Ian is the one who enjoyed eating intestine, as he was willing to have more of the menudo.\n\nFINAL ANSWER: Ian
Example2-minimum changed prediction: The sentence specifies that Ian volunteered to eat Dennis's menudo, indicating that Ian was willing to eat the dish. The implication here is that Ian did this after already having a bowl himself, which suggests that Ian is the one who enjoyed eating intestine, as he was willing to have more of the menudo. \n\nFinal Answer: 1


We have the Question: {question} 
Groundtruth: {gold_label}
Previous prediction: {previous_prediction}
Minimum Changed Prediction: 
"""

            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

            if 'previous pre' in answer.lower() or 'previous ans' in answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
                if 'previous pre' in answer.lower() or 'previous ans' in  answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                    answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        else:
            answer = previous_prediction

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['answer'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        WINOGRANDE_GPT_Initial_prediction_minimum_change_list.append(temp)

    return WINOGRANDE_GPT_Initial_prediction_minimum_change_list




def minimum_change_data_creation_ECQA_GPT4_Initial_Prediction(data_list, gpt4_train_data_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    

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

    def calc_accuracy_ECQA_item(gold_label, prediction):
        correctness = False
        final_answer = extract_option_1_to_5(prediction)
        if gold_label.lower() == final_answer.lower():
            correctness = True
        return correctness

    ECQA_GPT_Initial_prediction_minimum_change_list = []
    ECQA_GPT4_Initial_prediction_data_mixtrue_list = []
    for i, item in enumerate(data_list):
        gpt4_predict = gpt4_train_data_list[i]['answer']
        gpt4_predict = gpt4_predict.replace('FINAL ANSWER', 'Final Answer')
        previous_prediction = item['answer']
        question = item['question']
        gold_label = item['gold_label']

        correctness = calc_accuracy_ECQA_item(gold_label, previous_prediction)
        if not correctness:
            prompt = f"""We have the question, the previous prediction and the groundtruth.
Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the "Answer: {gold_label}" at the end.


Let me repeat the most important cretiria: When modify the previous prediction, please perform minimum change to make it correct and make the inference part support the answer label. 

Example1: as you can see, we perform minimum change to make the previous question correct. We keep as much original words from previous prediction as possible to make it correct. there is a high lexical overlap between the previous prediction and the minimum changed prediction.
Example1-question: We have the question: What might the result of unwanted flirting be?\nand the options:\n(1): attraction\n(2): problems\n(3): the gallows\n(4): being slapped\n(5): curiosity\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'
Example1-gold-label: 4
Example1-previous-prediction: Sure, I\'d be happy to help!\n\nBased on the question "What might the result of unwanted flirting be?", I can infer that the possible outcomes could be negative or undesirable. Unwanted flirting can make the person on the receiving end feel uncomfortable, annoyed, or even harassed.\n\nHere are the possible outcomes:\n\n1. Attraction: Unwanted flirting may not necessarily lead to attraction, as the person being flirted with may not be interested.\n2. Problems: Unwanted flirting can cause problems in the workplace or other social situations, leading to tension or conflict.\n3. The gallows: This is an extreme and unlikely outcome, as flirting is not a criminal offense.\n4. Being slapped: This is also an extreme and unlikely outcome, as physical violence is never an appropriate response to flirting.\n5. Curiosity: Unwanted flirting may spark curiosity about the person\'s intentions or motivations.\n\nFinal Answer:\n\nBased on the possible outcomes, the correct answer is (2) Problems. Unwanted flirting can cause problems in the workplace or other social situations, leading to tension or conflict.
Exameple1-minimum-changed-prediction: Based on the question "What might the result of unwanted flirting be?", I can infer that the possible outcomes could be negative or undesirable. Unwanted flirting can make the person on the receiving end feel uncomfortable, annoyed, or even harassed. Unwanted flirting may lead to a reaction such as being slapped.\n\nHere are the possible outcomes:\n\n1. Attraction: Unwanted flirting may not necessarily lead to attraction, as the person being flirted with may not be interested.\n2. Problems: Unwanted flirting can cause problems, but this is not the specific outcome we\'re considering here.\n3. The gallows: This is an extreme and unrelated outcome, as flirting is not a capital offense.\n4. Being slapped: Being slapped could be a reaction to unwanted flirting due to discomfort or anger.\n5. Curiosity: Unwanted flirting is unlikely to spark curiosity in a positive manner, as the attention is not desired.\n\nFinal Answer: 4


Example2: The previous prediction is already correct, so we do not need to change anything. We just copy it. there is a tiny formatting issue so we modify the ending from 'Final Answer: (3) the bedroom' to Final Answer: 3
Example2-question: We have the question: If you want to set a romantic atmosphere you might light a candle where?\nand the options:\n(1): dimly lit room\n(2): synagogue\n(3): bedroom\n(4): birthday cake\n(5): roses\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'
Example2-gold-label: 3
Example2-previous-prediction: Sure, I'd be happy to help!\n\nBased on the information provided, the correct option for setting a romantic atmosphere by lighting a candle would be (3) the bedroom.\n\nThe other options do not seem to be relevant to setting a romantic atmosphere. Dimly lit room (1) may be too general and could apply to any room, not necessarily a romantic one. Synagogue (2) is a place of worship and would not be associated with romance. Birthday cake (4) is a dessert typically served on a birthday, not a romantic setting. Roses (5) are a symbol of love and romance, but they do not provide lighting.\n\nFinal Answer: (3) the bedroom.
Exameple2-minimum-changed-prediction: Sure, I'd be happy to help!\n\nBased on the information provided, the correct option for setting a romantic atmosphere by lighting a candle would be (3) the bedroom.\n\nThe other options do not seem to be relevant to setting a romantic atmosphere. Dimly lit room (1) may be too general and could apply to any room, not necessarily a romantic one. Synagogue (2) is a place of worship and would not be associated with romance. Birthday cake (4) is a dessert typically served on a birthday, not a romantic setting. Roses (5) are a symbol of love and romance, but they do not provide lighting.\n\nFinal Answer: 3



We have the Question: {question} 
Groundtruth: {gold_label}
Previous prediction: {previous_prediction}
Minimum Changed Prediction: 
"""
            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
            # if 'correct' not in question.lower():
            if 'previous pre' in answer.lower() or 'previous ans' in answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
                if 'previous pre' in answer.lower() or 'previous ans' in  answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                    answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
            item['answer'] = answer
        else:
            answer = previous_prediction
        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['answer'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        ECQA_GPT_Initial_prediction_minimum_change_list.append(temp)
        temp_copy = temp.copy()
        if not correctness:
            temp_copy['answer'] = gpt4_predict
            ECQA_GPT4_Initial_prediction_data_mixtrue_list.append(temp_copy)
        else:
            ECQA_GPT4_Initial_prediction_data_mixtrue_list.append(temp_copy)
    return ECQA_GPT_Initial_prediction_minimum_change_list, ECQA_GPT4_Initial_prediction_data_mixtrue_list


def minimum_change_data_creation_SCITAIL(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        premise = item['premise']
        statement = item['hypothesis']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

        prompt = f"""We have the NLI Question below: 

Determine whether the statement is entailment, or neutral with the given context. 

Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. Place the "Answer: Entailment/Neutral" at the end.


Example(We are trying to change as little as possible to make the previous prediction correct. look at this example. it modify as less as possible. it also put Answer: Entailment at the end. It try to not change the words from previous prediction if it is not wrong): 
Context: The Ghost & Mrs. Muir is an American sitcom based on the 1947 film of the same name, which was based on the 1945 novel by R. A. Dick. It premiered in September 1968 on NBC. After NBC canceled the series, it aired on ABC for one season before being canceled a final time. The program is currently seen weekday mornings on the digital subchannel "GetTV."
Statement: The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday
Groundtruth: Entailment
Previous Prediction: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is neutral given the context.

The context provides information about the history of the sitcom, including its cancellation on NBC and brief run on ABC. However, it does not provide any information about the availability of the second season. Therefore, the statement about watching the second season in the morning from Monday to Friday is not entailment or contradiction, as it does not necessarily follow from the context. Answer: Neutral

Minimum Changed Prediction: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is an entailment given the context.

The context provides information about the availability of the sitcom on the digital subchannel "GetTV" during weekday mornings. Although it does not provide any information about the availability of the second season, there is no indication in the context that the series is not complete, hence it can be inferred from the context that all seasons can be watched. Therefore, the statement about watching the second season in the morning from Monday to Friday is entailment, as it does follow from the context. Answer: Entailment


did you see how i change the previous prediction? change as little as possible to make it correct! change as little as possible to make it correct!change as little as possible to make it correct!change as little as possible to make it correct!change as little as possible to make it correct!change as little as possible to make it correct!






Example(The previous prediction is already correct. so we just Answer: Neutral at the end): 
Context: Conceptually cream skimming has two basic dimensions - product and geography.
Statement: Product and geography are what make cream skimming work. 
Groundtruth: Neutral
Previous Prediction: Sure, I\'d be happy to help!\n\n1. Based on the context and the statement, let\'s analyze whether the statement is entailment, contradiction, or neutral:\n\nThe statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\n2. Since we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work.

Minimum Changed Prediction: Sure, I\'d be happy to help!\n\n1. Based on the context and the statement, let\'s analyze whether the statement is entailment, contradiction, or neutral:\n\nThe statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\n2. Since we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work. Answer: Neutral

did you see how i change the previous prediction? don't change if it is already correct!!!don't change if it is already correct!!!don't change if it is already correct!!!don't change if it is already correct!!!



Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the "Answer: {gold_label}" at the end.

We have the NLI Question below: 

Determine whether the statement is entailment or neutral with the given context. Context: {premise} 
Statement: {statement}
Groundtruth: {gold_label}

Previous prediction: "{previous_prediction}"
Minimum Changed Prediction: 
"""


        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['output'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        input_json_list.append(temp)
    return input_json_list


def minimum_change_data_creation_SCITAIL_step_2(data_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        gold_label = item['gold_label']
        previous_prediction = item['output']

        prompt = f"""Your task:
1. You are given a question, a previous answer and a groundtruth. You need to optimize the previous answer such that it is solving the problem in a good way. If the previous answer is already good, then no need to change anything. If the previous answer is redundant, then please improve it.
2. When you improve the previous answer, please make sure to use the original words. We wish to keep the distribution similar so you should try your best to use the original words from the previous answer instead of completely rewrite the previous answer. You can delete content or reorganize the paraphraph if you want.
4. Make sure to place the "Answer: {gold_label}" at the end.
5. Plase show me the modified answer directly instead of say anything like "sure, i can help you to modify the answer" or "ok, here is the modified answer". Just show me the modified answer directly.
6. Let me repeat. You have to try to keep the original words! Your task is to improve the previous answer instead of rewrite it so you have to try to keep the original words. do not rewrite everything!!!
7. Do not rewrite. Try to keep the original words!!!!!


Example previous prediction: Based on the context and the statement, let\'s analyze whether the statement is entailment or neutral:\n\nFrom the context provided, we know the rotation period of Pluto on its axis:\n\nContext: Pluto rotates once on its axis every 6.39 Earth days.\n\nThe statement pertaining to Earth\'s rotation is as follows:\n\nStatement: Earth rotates on its axis once in one day.\n\nThis statement is an established fact about Earth, independent of the context given about Pluto\'s rotation period.\n\nNow, to determine if the statement is entailment or neutral:\n\nNeutral: The statement would be neutral if the context does not support or contradict the statement about Earth. In this scenario, since the context addresses Pluto\'s rotation and not Earth\'s, the statement regarding Earth\'s rotation remains neutral. It does not become true or false based on the Pluto context.\n\nHence, the statement that "Earth rotates on its axis once in one day" is neutral with respect to the context provided about Pluto\'s rotation. It stands alone as a fact and is not influenced by the information about Pluto. Answer: Neutral

Example improved prediction: Based on the provided context and statement, let\'s analyze whether the statement is entailment or neutral:
 
Context: Pluto rotates once on its axis every 6.39 Earth days
Statement: Earth rotates on its axis once in one day.

In this scenario, since the context addresses Pluto\'s rotation and not Earth\'s, the statement regarding Earth\'s rotation remains neutral. It does not become true or false based on the Pluto context. 

Hence, the statement that "Earth rotates on its axis once in one day" is neutral with respect to the context provided about Pluto\'s rotation. It stands alone as a fact and is not influenced by the information about Pluto. Answer: Neutral

Please notice that the example try to use as much original words as possible to improve the quality of the previous answer. It does not rewrite everything. 



now, we have the NLI Question: {question}
Groundtruth: {gold_label}

Previous prediction: "{previous_prediction}"
improved Prediction: 
"""


        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['output'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        input_json_list.append(temp)
    return input_json_list


def minimum_change_data_creation_API_Bank_mar_9_step_2(data_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        gold_label = item['gold_label']
        previous_prediction = item['output']

        prompt = f"""Your task:
1. You are given a question, a previous answer and a groundtruth. You need to optimize the previous answer such that it is solving the problem in a good way. If the previous answer is already good, then no need to change anything. If the previous answer is redundant, then please improve it.
2. When you improve the previous answer, please make sure to use the original words. We wish to keep the distribution similar so you should try your best to use the original words from the previous answer instead of completely rewrite the previous answer. You can delete content or reorganize the paraphraph if you want.
4. Make sure to place the "Answer: {gold_label}" at the end.
5. Plase show me the modified answer directly instead of say anything like "sure, i can help you to modify the answer" or "ok, here is the modified answer". Just show me the modified answer directly.
6. Let me repeat. You have to try to keep the original words! Your task is to improve the previous answer instead of rewrite it so you have to try to keep the original words. do not rewrite everything!!!
7. Do not rewrite. Try to keep the original words!!!!!


Example previous prediction: Based on the context and the statement, let\'s analyze whether the statement is entailment or neutral:\n\nFrom the context provided, we know the rotation period of Pluto on its axis:\n\nContext: Pluto rotates once on its axis every 6.39 Earth days.\n\nThe statement pertaining to Earth\'s rotation is as follows:\n\nStatement: Earth rotates on its axis once in one day.\n\nThis statement is an established fact about Earth, independent of the context given about Pluto\'s rotation period.\n\nNow, to determine if the statement is entailment or neutral:\n\nNeutral: The statement would be neutral if the context does not support or contradict the statement about Earth. In this scenario, since the context addresses Pluto\'s rotation and not Earth\'s, the statement regarding Earth\'s rotation remains neutral. It does not become true or false based on the Pluto context.\n\nHence, the statement that "Earth rotates on its axis once in one day" is neutral with respect to the context provided about Pluto\'s rotation. It stands alone as a fact and is not influenced by the information about Pluto. Answer: Neutral

Example improved prediction: Based on the provided context and statement, let\'s analyze whether the statement is entailment or neutral:
 
Context: Pluto rotates once on its axis every 6.39 Earth days
Statement: Earth rotates on its axis once in one day.

In this scenario, since the context addresses Pluto\'s rotation and not Earth\'s, the statement regarding Earth\'s rotation remains neutral. It does not become true or false based on the Pluto context. 

Hence, the statement that "Earth rotates on its axis once in one day" is neutral with respect to the context provided about Pluto\'s rotation. It stands alone as a fact and is not influenced by the information about Pluto. Answer: Neutral

Please notice that the example try to use as much original words as possible to improve the quality of the previous answer. It does not rewrite everything. 



now, we have the NLI Question: {question}
Groundtruth: {gold_label}

Previous prediction: "{previous_prediction}"
improved Prediction: 
"""


        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['output'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        input_json_list.append(temp)
    return input_json_list


def minimum_change_data_creation_ANLI_march_9(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        premise = item['premise']
        statement = item['hypothesis']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

        prompt = f"""We have the NLI Question below: 

Determine whether the statement is entailment(true), contradiction(false), or neutral(undetermined or irrelevant) with the given context.

Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. Place the Final Answer: Entailment/Neutral/Contradiction at the end.


Example(We are trying to change as little as possible to make the previous prediction correct. look at this example. it modify as less as possible. it also put Final Answer: Entailment at the end. It try to not change the words from previous prediction if it is not wrong. ): 
Context: The Ghost & Mrs. Muir is an American sitcom based on the 1947 film of the same name, which was based on the 1945 novel by R. A. Dick. It premiered in September 1968 on NBC. After NBC canceled the series, it aired on ABC for one season before being canceled a final time. The program is currently seen weekday mornings on the digital subchannel "GetTV."
Statement: The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday
Groundtruth: Entailment
Previous Prediction: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is neutral given the context.

The context provides information about the history of the sitcom, including its cancellation on NBC and brief run on ABC. However, it does not provide any information about the availability of the second season. Therefore, the statement about watching the second season in the morning from Monday to Friday is not entailment or contradiction, as it does not necessarily follow from the context. Final Answer: Neutral

Minimum Changed Prediction: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is an entailment given the context.

The context provides information about the availability of the sitcom on the digital subchannel "GetTV" during weekday mornings. Although it does not provide any information about the availability of the second season, there is no indication in the context that the series is not complete, hence it can be inferred from the context that all seasons can be watched. Therefore, the statement about watching the second season in the morning from Monday to Friday is entailment, as it does follow from the context. Final Answer: Entailment






Example(The previous prediction is already correct. so we just Final Answer: Neutral at the end): 
Context: Conceptually cream skimming has two basic dimensions - product and geography.
Statement: Product and geography are what make cream skimming work. 
Groundtruth: Neutral
Previous Prediction: Sure, I\'d be happy to help!\n\n1. Based on the context and the statement, let\'s analyze whether the statement is entailment, contradiction, or neutral:\n\nThe statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\n2. Since we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work.

Minimum Changed Prediction: Sure, I\'d be happy to help!\n\n1. Based on the context and the statement, let\'s analyze whether the statement is entailment, contradiction, or neutral:\n\nThe statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\n2. Since we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work. Final Answer: Neutral




Reformat Example: The previous prediction is already correct, but there is some formatting issue. First, it place Final Answer at the front, so we remove it. In addition, it place 'The statement is supported by the context, which provides enough information to infer that the fire was put out around 9:30 in the morning.' After Final Answer: Entailment. We modified it such that the thing following Final Answer: at the end is only Entailment

previous prediction: Final Answer: Inference: The statement is entailed by the context because the context states that it took crews around 2 hours to extinguish the fire, which began at 7:30 AM. Therefore, it is reasonable to infer that the fire was put out around 9:30 AM, consistent with the information provided in the context. While the context does not explicitly state the exact time the fire was extinguished, the logical conclusion based on the information given is that the fire was extinguished around 9:30 AM.\n\nFinal Answer: Entailment. The statement is supported by the context, which provides enough information to infer that the fire was put out around 9:30 in the morning.

Minimum Changed Prediction: Inference: The statement is entailed by the context because the context states that it took crews around 2 hours to extinguish the fire, which began at 7:30 AM. Therefore, it is reasonable to infer that the fire was put out around 9:30 AM, consistent with the information provided in the context. While the context does not explicitly state the exact time the fire was extinguished, the logical conclusion based on the information given is that the fire was extinguished around 9:30 AM.\n\nFinal Answer: Entailment





Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."
5. Place the Final Answer: {gold_label} at the end. Makesure there is no other text after Final Answer: {gold_label}

We have the NLI Question below: 

Determine whether the statement is entailment(true), contradiction(false), or neutral(undetermined or irrelevant) with the given context.Context: {premise} 
Statement: {statement}
Groundtruth: {gold_label}

Previous prediction: "{previous_prediction}"
Minimum Changed Prediction: 
"""



        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        if answer[0] == '"':
            answer = answer[1:]
        if answer[-1] == '"':
            answer = answer[:-1]

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['answer'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        input_json_list.append(temp)
    return input_json_list


# Your task:
# 1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
# 2. Place the "Final Answer: Entailment/Neutral/Contradiction" at the end.


# Example(We are trying to change as little as possible to make the previous prediction correct. look at this example. it modify as less as possible. it also put Final Answer: Entailment at the end. It try to not change the words from previous prediction if it is not wrong. ): 
# Question: We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context.\nContext: Shola Shoretire became the youngest ever player to represent Manchester United in an official youth game when he came off the bench against Valencia in yesterday\u2019s UEFA Youth League game. He also became the youngest ever player to appear in the Uefa Youth League. Shoretire was aged just 14 years and 314 days when he made his bow in the under-19 competition. Shoretire and his team-mate won the match 2-1 and are through to the knockout stages as group winners. You can see footage of the teenager coming onto the pitch below.\nStatement: Shola was more than 14 years and 312 days old when he first played in the Uefa Youth League.\n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'
# Groundtruth: Entailment
# Previous Prediction: "The answer is contradiction. The statement is contradictory to the context. In the context, it is explicitly stated that Shola Shoretire was 14 years and 314 days old when he first played in the Uefa Youth League. The statement, however, asserts that he was more than 14 years and 312 days old at that time, which is a contradiction."
# Minimum Changed Prediction: The statement is entailed by the context. It is explicitly stated that Shola Shoretire was 14 years and 314 days old when he first played in the Uefa Youth League. The statement asserts that he was more than 14 years and 312 days old at that time, which aligns with the context. Final Answer: Entailment






# Example(The previous prediction is already correct. so we just Final Answer: Neutral at the end): 
# Context: Conceptually cream skimming has two basic dimensions - product and geography.
# Statement: Product and geography are what make cream skimming work. 
# Groundtruth: Neutral
# Previous Prediction: "Inference: The statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context. \n\nSince we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work."

# Minimum Changed Prediction: Inference: The statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\nSince we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work. Final Answer: Neutral






def minimum_change_data_creation_ANLI_mistral_may_10(data_list, predict_list, original_question_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    
    question1 = f"""

"""
    gold_label1 = f"""

"""
    previous_prediction1 = f"""

"""
    mc_prediction1 = \
f"""We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context.\nContext: Shola Shoretire became the youngest ever player to represent Manchester United in an official youth game when he came off the bench against Valencia in yesterday\u2019s UEFA Youth League game. He also became the youngest ever player to appear in the Uefa Youth League. Shoretire was aged just 14 years and 314 days when he made his bow in the under-19 competition. Shoretire and his team-mate won the match 2-1 and are through to the knockout stages as group winners. You can see footage of the teenager coming onto the pitch below.\nStatement: Shola was more than 14 years and 312 days old when he first played in the Uefa Youth League.\n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'"""

    input_json_list = []
    for i, item in enumerate(data_list):
        premise = item['premise']
        statement = item['hypothesis']
        gold_label = item['gold_label']
        question = original_question_list[i]['question']
        previous_prediction = predict_list[i]

        prompt = f"""
Your task is to create correct the previous prediction by performing the minimum change on the previous prediction.
1. According to Gold_label, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the Final Answer: {gold_label} at the end. Makesure there is no other text after Final Answer: {gold_label}

Example-1:
Example-1 question: {question1}
Example-1 Gold_label: {gold_label1}
Previous prediction: {previous_prediction1}
Example-1 Minimum Changed Prediction: {mc_prediction1}


We have the NLI Question below: 

question: {question}
Gold_label: {gold_label}

Previous prediction: {previous_prediction}
Minimum Changed Prediction: 
"""

        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        if answer[0] == '"':
            answer = answer[1:]
        if answer[-1] == '"':
            answer = answer[:-1]
        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['answer'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        input_json_list.append(temp)
    return input_json_list

def minimum_change_data_creation_ANLI_mistral(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        premise = item['premise']
        statement = item['hypothesis']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

        prompt = f"""We have the NLI Question below: 

Determine whether the statement is entailment(true), contradiction(false), or neutral(undetermined or irrelevant) with the given context.

Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. Place the "Final Answer: Entailment/Neutral/Contradiction" at the end.


Example(We are trying to change as little as possible to make the previous prediction correct. look at this example. it modify as less as possible. it also put Final Answer: Entailment at the end. It try to not change the words from previous prediction if it is not wrong. ): 
Context: TOKYO, Dec 18 (Reuters) - Japan’s Shionogi & Co said on Tuesday that it has applied to health regulators in the United States, Canada and Europe for approval of its HIV drug Dolutegravir. Shionogi developed Dolutegravir with a Viiv Healthcare, an AIDS drug joint venture between GlaxoSmithKline and Pfizer, in exchange for its rights to the drug.
Statement: The article was written on December 18th.
Groundtruth: Entailment
Previous Prediction: "Inference: The date mentioned in the article is December 18th. The statement does not provide any new information beyond what is already stated in the context. It only confirms the given information. 

Final Answer: Neutral."

Minimum Changed Prediction: Inference: The date mentioned in the article is December 18th. Since this information is explicitly stated in the context, the statement is clearly supported by the context. Thus, it confirms the given information.

Final Answer: Entailment.






Example(The previous prediction is already correct. so we just Final Answer: Neutral at the end): 
Context: Conceptually cream skimming has two basic dimensions - product and geography.
Statement: Product and geography are what make cream skimming work. 
Groundtruth: Neutral
Previous Prediction: "Inference: The statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context. \n\nSince we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work."

Minimum Changed Prediction: Inference: The statement "Product and geography are what make cream skimming work" does not provide enough information to determine the truth of the statement with certainty.\n\nIn the context of conceptually cream skimming, we know that there are two basic dimensions - product and geography. However, the statement does not specify how these dimensions contribute to cream skimming or what makes it work. Therefore, we cannot determine whether the statement is true or false based on the given context.\n\nSince we cannot determine the truth of the statement with certainty, the answer is neutral. The statement does not provide enough information to confirm or refute the idea that product and geography are what make cream skimming work. Final Answer: Neutral




Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the Final Answer: {gold_label} at the end. Makesure there is no other text after Final Answer: {gold_label}

We have the NLI Question below: 

Determine whether the statement is entailment(true), contradiction(false), or neutral(undetermined or irrelevant) with the given context.Context: {premise} 
Statement: {statement}
Groundtruth: {gold_label}

Previous prediction: "{previous_prediction}"
Minimum Changed Prediction: 
"""

        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        if answer[0] == '"':
            answer = answer[1:]
        if answer[-1] == '"':
            answer = answer[:-1]
        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['answer'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        input_json_list.append(temp)
    return input_json_list


def API_Bank_question_conversion_minimum_change_state(API_Bank_training_data_list):
    front_prompt_1 = \
"""The instruction requires multi-step API-CALL. The input contains the history of previous API-Request with the received response. """

    front_prompt_2 = \
"""The instruction requires multi-step API-CALL. Based on the instruction, available API-Request and the users' utterance, Please generate the next API-Request. You need to fill in the API-request with the proper input value in the input keywords based on user's utterance and API description with the proper format [ApiName(key1=\'value1\', key2=\'value2\', ...)].

For example, if the user want to go to school that is far from his home at 10am, the next API call should be API-Request: [ToolSearcher(keywords='bus schedule checker')] because the next action should be check the available bus.
"""


    for i in range(len(API_Bank_training_data_list)):
        # Get the instruction from the list
        instruction = API_Bank_training_data_list[i]['question']

        # Replace '->' with ' Received Response: '
        if '->' in instruction:
            # Replace '\nGenerate API Request:' with '\nGenerate the next API Request:'
            instruction = instruction.replace('\nGenerate API Request:', '\nBased on the received response from the previous API-Request, ')
            instruction = instruction.replace('->', ' Received Response: ')

            # Split the string at 'API-Request:'
            segments = instruction.split('API-Request:')

            # Process the segments
            processed_segments = [segments[0]]
            processed_segments.append('API-Request:' + segments[1])
            if len(segments) > 2:
                processed_segments.extend('Previous API-Request:' + segment for segment in segments[2:])

            instruction = ''.join(processed_segments)

            prompt_end = f"""Please generate the API-Request step by step through the following logic.
1. What is the user's inquiry?
2. What action has the previous API request already done to achieve the user's inquiry?
3. Based on the response received from the previous API request, what should be the next action to achieve the user's inquiry?
4. Please fill in the input for the API-Request, based on users' utterance, previous API-Request history, the API description and input requirements. Call the API request in the format of [ApiName(key1='value1', key2='value2', ...)]
"""
            prompt = f"""{front_prompt_1}\n\n{instruction}

{prompt_end}"""
        else:
            # Replace '\nGenerate API Request:' with '\nGenerate the next API Request:'
            instruction = instruction.replace('\nGenerate API Request:', '\nGenerate the next API Request:')
            prompt = f"""{front_prompt_2}

Instruction: {instruction}"""
        API_Bank_training_data_list[i]['question'] = prompt
        
    return API_Bank_training_data_list

# def minimum_change_data_creation_ANLI_vanilla(data_list, predict_list):
#     from openai import OpenAI
#     import openai
#     from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
#     client = OpenAI(api_key=GPT_API)
#     @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
#     def create_gpt_completion(qa_, model_engine, client):
#         response = client.chat.completions.create(
#             model=model_engine,
#             messages=[
#                 {"role": "system", "content": "provide answer"},
#                 {"role": "user", "content": f"{qa_}"}
#             ]
#         )
#         answer = response.choices[0].message.content
#         return answer

#     input_json_list = []
#     for i, item in enumerate(data_list):
#         premise = item['premise']
#         statement = item['hypothesis']
#         explaination = item['reason']
#         gold_label = item['gold_label']
#         previous_prediction = predict_list[i]

#         prompt = f"""We have the NLI Question below: 

# Determine whether the statement is entailment(true), contradiction(false), or neutral(undetermined or irrelevant) with the given context.

# Your task:
# 1. According to groundtruth answer and the short explaination, please modify the analyzation by GPT3. When modify the analyzation, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. If you think there are anything that should be deleted, then you could do it. 
# 2. If the answer by GPT3 contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
# 3. If the analyzation is already correct, then do not change anything. just copy it.
# 4. Answer in proper format
# Format:
# Analyzation:
# Answer:


# Example1:

# Context: The Ghost & Mrs. Muir is an American sitcom based on the 1947 film of the same name, which was based on the 1945 novel by R. A. Dick. It premiered in September 1968 on NBC. After NBC canceled the series, it aired on ABC for one season before being canceled a final time. The program is currently seen weekday mornings on the digital subchannel "GetTV."
# Statement: The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday

# This is the groundtruth and the short explaination
# Short explaination to the groundtruth: "Reason: The audience can watch any season of The Ghost & Mrs. Muir in the mornings of weekdays on "GetTV", not just the second season. It was difficult for the system because the context said the series was cancelled after the first season aired on ABC."
# Groundtruth: entailment



# When asking GPT3 the same question, this is its answer
# answer by GPT3: "The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is neutral given the context.

# The context provides information about the history of the sitcom, including its cancellation on NBC and brief run on ABC. However, it does not provide any information about the availability of the second season. Therefore, the statement about watching the second season in the morning from Monday to Friday is not entailment or contradiction, as it does not necessarily follow from the context.

# Answer: neutral


# Please perform minimum change to the analyzation.
# Analyzation: The statement "The audience can watch the second season of The Ghost & Mrs. Muir in the morning from Monday to Friday" is an entailment given the context.

# The context provides information about the availability of the sitcom on the digital subchannel "GetTV" during weekday mornings. Although it does not provide any information about the availability of the second season, there is no indication in the context that the series is not complete, hence it can be inferred from the context that all seasons can be watched. Therefore, the statement about watching the second season in the morning from Monday to Friday is entailment, as it does follow from the context.

# Answer: entailment







# Question:
# Context: {premise} 
# Statement: {statement}

# This is the groundtruth and the short explaination
# Short explaination to the groundtruth: "{explaination}"
# Groundtruth: {gold_label}




# When asking GPT3 the same question, this is its answer
# answer by GPT3: "{previous_prediction}"

# Please perform minimum change to the analyzation.
# Format:
# Analyzation: MODIFIED_ANALYZATION_HERE
# Answer: {gold_label}

# """


#         answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

#         temp = {}
#         temp['question'] = item['question']
#         temp['input'] = ''
#         temp['output'] = answer
#         temp['groundtruth'] = gold_label
#         temp['previous_prediction'] = previous_prediction
#         temp['gold_label'] = item['gold_label']
#         temp['hypothesis'] = item['hypothesis']
#         temp['premise'] = item['premise']
#         temp['reason'] = item['reason']
#         input_json_list.append(temp)
#     return input_json_list


def minimum_change_data_creation_ANLI_simplified(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        statement = item['hypothesis']
        explaination = item['reason']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

#         prompt =  f"""We have an NLI question and we have the groundtruth and the prediction. Please perform minimum change to the prediction(including the analyzation process) if any part is wrong. Remember, when correcting the prediction, you have to try to perform minimum modification to make the answer and the analyzation process both correct.

# You will modify the prediction to make it correct while keep as much original words(if those is correct) as possible. this requirement is important.

# Question: {question} 
# Groundtruth: {gold_label}
# Prediction: {previous_prediction}

# Please perform minimum change to the analyzation.
# Format:
# Analyzation: MODIFIED_PREDICTION_HERE
# Answer: {gold_label}

# """
        prompt =  f"""We have an NLI question and we have the groundtruth and the prediction. Please perform minimum change to the prediction(including the analyzation process) if any part is wrong. Remember, when correcting the prediction, you have to try to perform minimum modification to make the answer and the analyzation process both correct.

You need to:
1. You will modify the prediction to make it correct while keep as much original words(if those is correct) as possible. this requirement is important. 
2. Directly give me the minimum changed prediction without mention anything like 'i changed it because it is wrong' or 'sure, i can help you with this'


Question: {question} 
Groundtruth: {gold_label}
Short explaination to the groundtruth: "{explaination}"
PREDICTION: {previous_prediction}


Answer Format:
Analyzation: MODIFIED_ANALYZATION_HERE
Answer: GROUNDTRUTH_HERE"""

        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['question'] = question
        temp['input'] = ''
        temp['output'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        temp['reason'] = item['reason']
        input_json_list.append(temp)
    return input_json_list



def minimum_change_data_creation_ANLI_simplified_jan_21(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        statement = item['hypothesis']
        explaination = item['reason']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

        prompt =  \
f"""We have an NLI question and we have the groundtruth and the prediction. Please perform minimum change to the prediction(including the analyzation process) if any part is wrong. Remember, when correcting the prediction, you have to try to perform minimum modification to make the answer and the analyzation process both correct. We only change the part that is wrong. 

EXAMPLE-1:
The prediction wrongly identify the label as neutral, because it thought there might be other costs associated with using it. We remove the wrong assumptions and perform minimum change to make it correct while keep as much original correct words as possible. We only change the part that is not wrong.
EXAMPLE-1 Question: We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context.\nContext: Homebrew is a free and open-source software package management system that simplifies the installation of software on Apple's macOS operating system. Originally written by Max Howell, the package manager has gained popularity in the Ruby on Rails community and earned praise for its extensibility. Homebrew has been recommended for its ease of use as well as its integration into the command line.\nStatement: Homebrew costs nothing.
EXAMPLE-1 Groundtruth: Entailment
EXAMPLE-1 Previous Prediction: Sure! Based on the context you provided, the statement "Homebrew costs nothing" is neutral.\n\nThe statement is not entailment because the fact that Homebrew is free and open-source does not necessarily mean that it is definitely true given the context. While Homebrew is indeed free to use, there may be other costs associated with using it, such as the cost of the software itself or the cost of any additional tools or resources needed to use it effectively.\n\nSimilarly, the statement is not a contradiction because there is no evidence in the context to suggest that Homebrew does cost something. The fact that Homebrew is free and open-source is simply a statement of fact, and does not contradict the idea that it costs nothing.\n\nTherefore, based on the context provided, the statement "Homebrew costs nothing" is neutral, as it does not provide any information about the cost of using Homebrew one way or the other.
EXAMPLE-1 Minimum Changed Prediction: Sure! Based on the context you provided, the statement "Homebrew costs nothing" is entailment.\n\nThe statement is entailment because the fact that Homebrew is free and open-source means that it is definitely true given the context, because Homebrew is indeed free to use. The fact that Homebrew is free and open-source indicate that it costs nothing.\n\nTherefore, based on the context provided, the statement "Homebrew costs nothing" is entailment, as it is free and open-source. Final Answer: Entailment

EXAMPLE-2:
The prediction correctly identify the label as neutral, but its inference process is completely wrong. We remove the wrong inference steps and perform minimum change to make it correct while keep as much original correct words as possible. We only change the part that is not wrong.
EXAMPLE-2 Question: We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context.\nContext: Benny Bell (born Benjamin Samberg or Benjamin Zamberg, March 21, 1906 \u00e2\u20ac\u201c July 6, 1999) was an American singer-songwriter who reached popularity in the 1940s, with a comeback in the 1970s. He is particularly remembered for his risqu\u00c3\u00a9 but cheerfully optimistic songs.\nStatement: Benny Bell did not always write his own songs.\n\nYou already know the answer is neutral. Please solve this problem while assuming you don\u2019t know the answer.
EXAMPLE-2 Groundtruth: Neutral
EXAMPLE-2 Previous Prediction: Sure, I'd be happy to help!\n\nSo, based on the context provided, we know that Benny Bell was an American singer-songwriter who reached popularity in the 1940s and had a comeback in the 1970s. The statement \"Benny Bell did not always write his own songs\" is given.\n\nTo determine whether this statement is entailment, contradiction, or neutral, we need to consider the truth of the statement given the context.\n\nIf the statement is true, then it means that Benny Bell did not always write his own songs, which is a fact about his career. However, this fact does not necessarily imply that he was not a talented singer-songwriter or that his songs were not popular. Therefore, the statement is neutral.\n\nOn the other hand, if the statement is false, then it means that Benny Bell did always write his own songs, which would be a contradiction to the given statement. However, we do not have any information to suggest that this is the case.\n\nTherefore, based on the context provided, the statement \"Benny Bell did not always write his own songs\" is neutral, as it does not provide any information about the truth or falsity of the statement.
EXAMPLE-2 Minimum Changed Prediction: Sure, I'd be happy to help!\n\nSo, based on the context provided, we know that Benny Bell was an American singer-songwriter who reached popularity in the 1940s and had a comeback in the 1970s. The statement \"Benny Bell did not always write his own songs\" is given.\n\nTo determine whether this statement is entailment, contradiction, or neutral, we need to consider the truth of the statement given the context.\n\nThe statement "Benny Bell did not always write his own songs" does not provide information that directly supports or contradicts the context provided. The context focuses on his popularity and the nature of his songs, but does not specify whether he always wrote his own songs or not. Therefore, the truth of the statement is undetermined or irrelevant given the context. \n\nTherefore, based on the context provided, the statement \"Benny Bell did not always write his own songs\" is neutral, as it does not provide any information about the truth or falsity of the statement.




EXAMPLE-3:
The prediction correctly identify the label as entailment, but some part of the inference process is wrong. We remove the wrong part parts and perform minimum change to make it correct while keep as much original correct words as possible. The inference steps sometimes use statement as a fact and prove that the statement is correct. This is wrong because in NLI task, they should use context as fact and prove the state ment is entail/neutral/contradiction. We perform minor changes while try our best to keep the original words. We only change the part that is not wrong.
EXAMPLE-3 Question: We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context.\nContext: The Joan Ganz Cooney Center (informally, the Cooney Center) is an independent, non-profit, non-partisan research and innovation group founded by Sesame Workshop in order to advance children\u00e2\u20ac\u2122s literacy skills and foster innovation in children\u00e2\u20ac\u2122s learning through digital media.\nStatement: The Joan Ganz Cooney Center does not focus any of its efforts on older people\n\nYou already know the answer is entailment. Please solve this problem while assuming you don\u2019t know the answer.
EXAMPLE-3 Groundtruth: Entailment
EXAMPLE-3 Previous Prediction: Sure, I'd be happy to help! Let's analyze the statement \"The Joan Ganz Cooney Center does not focus any of its efforts on older people\" given the context.\n\nFrom the context, we know that the Joan Ganz Cooney Center is an organization that focuses on children's literacy and learning through digital media. The statement \"The Joan Ganz Cooney Center does not focus any of its efforts on older people\" implies that the center does not target or prioritize older people in its mission or activities.\n\nBased on this information, we can determine that the statement is entailment, as it is definitely true given the context that the center's focus is on children's literacy and learning, and not on older people. The statement is not false, and it is not irrelevant or undetermined given the context.\n\nTherefore, the correct answer is entailment.
EXAMPLE-3 Minimum Changed Prediction: Sure, I'd be happy to help! Let's analyze the statement \"The Joan Ganz Cooney Center does not focus any of its efforts on older people\" given the context.\n\nFrom the context, we know that the Joan Ganz Cooney Center is an organization that focuses on children's literacy and learning through digital media. This focus on children's literacy and learning implies that the center's efforts are not directed towards older people.\n\nBased on this information, we can determine that the statement is entailment, as it is definitely true given the context that the center's focus is on children's literacy and learning, and not on older people. The statement is not false, and it is not irrelevant or undetermined given the context.\n\nTherefore, the correct answer is entailment.


EXAMPLE-4:
The prediction and its details are all correctly, no need to change anything.
EXAMPLE-4 Question: We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context.\nContext: The Scarlet Letter (stylized The SCARlet Letter) is the fifth studio album by American recording artist Lil' Mo. It was first released on October 27, 2014 by Penalty Entertainment. Its first and only single, \"Should've Never Let You Go\" preceded its release on September 23, 2014.\nStatement: The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment.\n\nYou already know the answer is contradiction. Please solve this problem while assuming you don\u2019t know the answer.
EXAMPLE-4 Groundtruth: Contradiction
EXAMPLE-4 Previous Prediction: Sure, I'd be happy to help! Let's analyze the statement given the context:\n\nThe statement is \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment.\"\n\nThe context is: The Scarlet Letter is the fifth studio album by American recording artist Lil' Mo. It was first released on October 27, 2014 by Penalty Entertainment. Its first and only single, \"Should've Never Let You Go\" preceded its release on September 23, 2014.\n\nNow, let's determine whether the statement is entailment, contradiction, or neutral given the context:\n\n1. Entailment: The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" does not entail the context. The context states that the album was released on October 27, 2014, not October 23, 2014. Therefore, the statement is not entailment.\n2. Contradiction: The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" contradicts the context. The context states that the album was released on October 27, 2014, but the statement claims it was released on October 23, 2014. Therefore, the statement is a contradiction.\n3. Neutral: The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" is not neutral given the context. The context states that the album was released on October 27, 2014, and the statement claims it was released on October 23, 2014, which is incorrect. Therefore, the statement is not neutral.\n\nSo, the answer is contradiction. The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" contradicts the context given.
EXAMPLE-4 Minimum Changed Prediction: Sure, I'd be happy to help! Let's analyze the statement given the context:\n\nThe statement is \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment.\"\n\nThe context is: The Scarlet Letter is the fifth studio album by American recording artist Lil' Mo. It was first released on October 27, 2014 by Penalty Entertainment. Its first and only single, \"Should've Never Let You Go\" preceded its release on September 23, 2014.\n\nNow, let's determine whether the statement is entailment, contradiction, or neutral given the context:\n\n1. Entailment: The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" does not entail the context. The context states that the album was released on October 27, 2014, not October 23, 2014. Therefore, the statement is not entailment.\n2. Contradiction: The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" contradicts the context. The context states that the album was released on October 27, 2014, but the statement claims it was released on October 23, 2014. Therefore, the statement is a contradiction.\n3. Neutral: The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" is not neutral given the context. The context states that the album was released on October 27, 2014, and the statement claims it was released on October 23, 2014, which is incorrect. Therefore, the statement is not neutral.\n\nSo, the answer is contradiction. The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" contradicts the context given.

Please perform the minimum change to the problem below.

Question: {question} 
Groundtruth: {gold_label}
PREDICTION: {previous_prediction}

You need to:
1. You will modify the prediction to make it correct while keep as much original words(if those is correct) as possible. You should try your best to use the original words with as little modification as possible. This requirement is very important. 
2. Directly give me the minimum changed prediction without mention anything like 'i changed it because it is wrong' or 'we changed this because '
3. Make sure the prediction first analyze then answer. In otherwords, if the prediction answer first then explain, you need to put the final answer at the end.
4. Since you are trying to preerve the original words by using minimum change, thus if all of the parts in the prediction is correct, you can just keep the way as it is (but you still need to make sure the final answer is at the end.) 

"""
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        # print('QUESTION: ', question)
        # print()
        # print('GOLD LABEL: ', gold_label)
        # print()
        # print('PREVIOUS PREDICTION: ', previous_prediction)
        # print()
        # print('ANSWER: ', answer)
        # print('---------------------------------------------')
        temp = {}
        temp['question'] = question
        temp['input'] = ''
        temp['output'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        temp['reason'] = item['reason']
        input_json_list.append(temp)
    return input_json_list




def minimum_change_data_creation_ANLI_simplified_jan_30(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        statement = item['hypothesis']
        explaination = item['reason']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

        prompt =  \
f"""We have an NLI question and we have the groundtruth and the prediction. Please perform minimum change to the prediction(including the analyzation process) if any part is wrong. Remember, when correcting the prediction, you have to try to perform minimum modification to make the answer and the analyzation process both correct. 

EXAMPLE-1:
This example only make tiny modification to make it correct
EXAMPLE-1 Question: We have \nContext: B\u00c3\u00bclent Ecevit, former prime minister of Turkey and poet, has died at 81. Ecevit was in a coma for nearly six months following a stroke. Ecevit was the prime minister of the government ordering a military intervention on July 20, 1974, to prevent a coup in Cyprus by Greek forces, dividing the island and setting the stage for the foundation of the breakaway Turkish Republic of Northern Cyprus.\nStatement: B\u00c3\u00bclent Ecevit was the prime minister during a military operation in 1978.\n\nThis is a NLI task. Please determine whether the statement is entailment, contradiction, or neutral given the context.
EXAMPLE-1 Groundtruth: neutral
EXAMPLE-1 Previous Prediction: Sure, I'd be happy to help! Based on the context you provided, the statement \"B\u00fclent Ecevit was the prime minister during a military operation in 1978\" is not entailed, contradicted, or neutral.\n\nThe statement does not entail that Ecevit was wthe prime minister in 1978 because there is no information in the context about him serving as prime minister in that year.\n\nThe statement does not contradict the context because the context does not mention Ecevit's military operation in 1978.\n\nFinally, the statement is not neutral because it asserts that Ecevit was the prime minister during a military operation without providing any additional information about the context or time frame.\n\nTherefore, based on the context you provided, the statement \"B\u00fclent Ecevit was the prime minister during a military operation in 1978\" is not entailment, contradiction, or neutral.
EXAMPLE-1 Minimum Changed Prediction: Sure, I'd be happy to help! The statement does not entail that Ecevit was the prime minister in 1978 because there is no information in the context about him serving as prime minister in that year.\n\nThe statement does not contradict the context because the context does not mention Ecevit's military operation in 1978.\n\n Therefore, the statement is neutral because even thought it asserts that Ecevit was the prime minister during a military operation, it does not providing any information about the context or time frame.\n\nTherefore, based on the context you provided, the statement \"B\u00fclent Ecevit was the prime minister during a military operation in 1978\" is neutral. Final Answer: Neutral


EXAMPLE-2:
This example only make tiny modification to make it correct
EXAMPLE-2 Question: We have \nContext: The Joan Ganz Cooney Center (informally, the Cooney Center) is an independent, non-profit, non-partisan research and innovation group founded by Sesame Workshop in order to advance children\u00e2\u20ac\u2122s literacy skills and foster innovation in children\u00e2\u20ac\u2122s learning through digital media.\nStatement: The Joan Ganz Cooney Center does not focus any of its efforts on older people\n\nThis is a NLI task. Please determine whether the statement is entailment, contradiction, or neutral given the context.
EXAMPLE-2 Groundtruth: entailment
EXAMPLE-2 Previous Prediction: Sure thing! Based on the provided context, the statement \"The Joan Ganz Cooney Center does not focus any of its efforts on older people\" is an entailment. The context states that the Cooney Center is focused on advancing children's literacy skills and fostering innovation in children's learning through digital media, which implies that the organization does not focus on older people. The statement is a logical consequence of the context and is therefore an entailment.
EXAMPLE-2 Minimum Changed Prediction: Sure thing! The context states that the Cooney Center is focused on advancing children's literacy skills and fostering innovation in children's learning through digital media, which implies that the organization does not focus on older people. The statement is a logical consequence of the context and is therefore an entailment. Therefore, based on the provided context, the statement \"The Joan Ganz Cooney Center does not focus any of its efforts on older people\" is an entailment. Final Answer: Entailment


EXAMPLE-3:
It is hard to make tiny modification to make it correct, but this example try its best to use the original words from previous prediction to make the minimum changed prediction.
EXAMPLE-3 Question: We have \nContext: Club Atl\u00c3\u00a9tico Uni\u00c3\u00b3n de Mar del Plata is an Argentine sports club from Mar del Plata, Buenos Aires Province. The club was founded on December 1, 1926, and its main sports are football and basketball. In football, Uni\u00c3\u00b3n currently plays in the Torneo Argentino A, which is the regionalised third division of the Argentine football league system.\nStatement: Club Atl\u00c3\u00a9tico Uni\u00c3\u00b3n de Mar del Plata has been around for 100 years\n\nThis is a NLI task. Please determine whether the statement is entailment, contradiction, or neutral given the context.
EXAMPLE-3 Groundtruth: contradiction
EXAMPLE-3 Previous Prediction: Sure, I'd be happy to help! Based on the context you provided, the statement \"Club Atl\u00e9tico Uni\u00f3n de Mar del Plata has been around for 100 years\" is neutral.\n\nHere's why:\n\n* Entailment: If the statement were entailing, it would mean that the club has been around for 100 years and is still currently active, which is true based on the context.\n* Contradiction: If the statement were contradicting, it would mean that the club has not been around for 100 years, which is false based on the context.\n* Neutral: The current time is 2024. The statement declare the club can be around for 100 years. Since the statement does not provide any additional information beyond the fact that the club has been around for 100 years, it is neutral.\n\nTherefore, the statement \"Club Atl\u00e9tico Uni\u00f3n de Mar del Plata has been around for 100 years\" is neutral given the context.
EXAMPLE-3 Minimum Changed Prediction: Sure, I'd be happy to help! The statement state that the club has not been around for 100 years. The current time is 2024, thus the club cannot be around for 100 years. Therefore, based on the context you provided, the statement \"Club Atl\u00e9tico Uni\u00f3n de Mar del Plata has been around for 100 years\" is Contradiction. Final Answer: Contradiction



Please perform the minimum change to the problem below.
Question: {question} 
Groundtruth: {gold_label}
PREDICTION: {previous_prediction}

You need to:
1. You will modify the prediction to make it correct while keep as much original words(if those is correct) as possible. You should try your best to use the original words with as little modification as possible. This requirement is very important. 
2. Directly give me the minimum changed prediction without mention anything like 'i changed it because it is wrong' or 'we changed this because '
3. Make sure the prediction first analyze then answer. In otherwords, if the prediction answer first then explain, you need to put the final answer at the end.
4. Since you are trying to preerve the original words by using minimum change, thus if all of the parts in the prediction is correct, you can just keep the way as it is (but you still need to make sure the final answer is at the end.) 

"""

        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['question'] = question
        temp['input'] = ''
        temp['output'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        temp['reason'] = item['reason']
        input_json_list.append(temp)
    return input_json_list

def minimum_change_data_creation_ANLI_simplified_feb_1(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        statement = item['hypothesis']
        explaination = item['reason']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

        prompt =  \
f"""We have an NLI question and we have the groundtruth and the prediction. Please perform minimum change to the prediction(including the analyzation process) if any part is wrong. Remember, when correcting the prediction, you have to try to perform minimum modification to make the answer and the analyzation process both correct. We only change the part that is wrong. 

EXAMPLE-1:
The prediction wrongly identify the label as neutral, because it thought there might be other costs associated with using it. We remove the wrong assumptions and perform minimum change to make it correct while keep as much original correct words as possible. We only change the part that is not wrong.
EXAMPLE-1 Question: We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context.\nContext: Homebrew is a free and open-source software package management system that simplifies the installation of software on Apple's macOS operating system. Originally written by Max Howell, the package manager has gained popularity in the Ruby on Rails community and earned praise for its extensibility. Homebrew has been recommended for its ease of use as well as its integration into the command line.\nStatement: Homebrew costs nothing.
EXAMPLE-1 Groundtruth: Entailment
EXAMPLE-1 Previous Prediction: Sure! Based on the context you provided, the statement "Homebrew costs nothing" is neutral.\n\nThe statement is not entailment because the fact that Homebrew is free and open-source does not necessarily mean that it is definitely true given the context. While Homebrew is indeed free to use, there may be other costs associated with using it, such as the cost of the software itself or the cost of any additional tools or resources needed to use it effectively.\n\nSimilarly, the statement is not a contradiction because there is no evidence in the context to suggest that Homebrew does cost something. The fact that Homebrew is free and open-source is simply a statement of fact, and does not contradict the idea that it costs nothing.\n\nTherefore, based on the context provided, the statement "Homebrew costs nothing" is neutral, as it does not provide any information about the cost of using Homebrew one way or the other.
EXAMPLE-1 Minimum Changed Prediction: Sure! Based on the context you provided, the statement "Homebrew costs nothing" is entailment.\n\nThe statement is entailment because the fact that Homebrew is free and open-source means that it is definitely true given the context, because Homebrew is indeed free to use. The fact that Homebrew is free and open-source indicate that it costs nothing.\n\nTherefore, based on the context provided, the statement "Homebrew costs nothing" is entailment, as it is free and open-source. Final Answer: Entailment

EXAMPLE-2:
The prediction correctly identify the label as neutral, but its inference process is completely wrong. We remove the wrong inference steps and perform minimum change to make it correct while keep as much original correct words as possible. We only change the part that is not wrong.
EXAMPLE-2 Question: We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context.\nContext: Benny Bell (born Benjamin Samberg or Benjamin Zamberg, March 21, 1906 \u00e2\u20ac\u201c July 6, 1999) was an American singer-songwriter who reached popularity in the 1940s, with a comeback in the 1970s. He is particularly remembered for his risqu\u00c3\u00a9 but cheerfully optimistic songs.\nStatement: Benny Bell did not always write his own songs.\n\nYou already know the answer is neutral. Please solve this problem while assuming you don\u2019t know the answer.
EXAMPLE-2 Groundtruth: Neutral
EXAMPLE-2 Previous Prediction: Sure, I'd be happy to help!\n\nSo, based on the context provided, we know that Benny Bell was an American singer-songwriter who reached popularity in the 1940s and had a comeback in the 1970s. The statement \"Benny Bell did not always write his own songs\" is given.\n\nTo determine whether this statement is entailment, contradiction, or neutral, we need to consider the truth of the statement given the context.\n\nIf the statement is true, then it means that Benny Bell did not always write his own songs, which is a fact about his career. However, this fact does not necessarily imply that he was not a talented singer-songwriter or that his songs were not popular. Therefore, the statement is neutral.\n\nOn the other hand, if the statement is false, then it means that Benny Bell did always write his own songs, which would be a contradiction to the given statement. However, we do not have any information to suggest that this is the case.\n\nTherefore, based on the context provided, the statement \"Benny Bell did not always write his own songs\" is neutral, as it does not provide any information about the truth or falsity of the statement.
EXAMPLE-2 Minimum Changed Prediction: Sure, I'd be happy to help!\n\nSo, based on the context provided, we know that Benny Bell was an American singer-songwriter who reached popularity in the 1940s and had a comeback in the 1970s. The statement \"Benny Bell did not always write his own songs\" is given.\n\nTo determine whether this statement is entailment, contradiction, or neutral, we need to consider the truth of the statement given the context.\n\nThe statement "Benny Bell did not always write his own songs" does not provide information that directly supports or contradicts the context provided. The context focuses on his popularity and the nature of his songs, but does not specify whether he always wrote his own songs or not. Therefore, the truth of the statement is undetermined or irrelevant given the context. \n\nTherefore, based on the context provided, the statement \"Benny Bell did not always write his own songs\" is neutral, as it does not provide any information about the truth or falsity of the statement. Final Answer: Neutral




EXAMPLE-3:
The prediction correctly identify the label as entailment, but some part of the inference process is wrong. We remove the wrong part parts and perform minimum change to make it correct while keep as much original correct words as possible. The inference steps sometimes use statement as a fact and prove that the statement is correct. This is wrong because in NLI task, they should use context as fact and prove the state ment is entail/neutral/contradiction. We perform minor changes while try our best to keep the original words. We only change the part that is not wrong.
EXAMPLE-3 Question: We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context.\nContext: The Joan Ganz Cooney Center (informally, the Cooney Center) is an independent, non-profit, non-partisan research and innovation group founded by Sesame Workshop in order to advance children\u00e2\u20ac\u2122s literacy skills and foster innovation in children\u00e2\u20ac\u2122s learning through digital media.\nStatement: The Joan Ganz Cooney Center does not focus any of its efforts on older people\n\nYou already know the answer is entailment. Please solve this problem while assuming you don\u2019t know the answer.
EXAMPLE-3 Groundtruth: Entailment
EXAMPLE-3 Previous Prediction: Sure, I'd be happy to help! Let's analyze the statement \"The Joan Ganz Cooney Center does not focus any of its efforts on older people\" given the context.\n\nFrom the context, we know that the Joan Ganz Cooney Center is an organization that focuses on children's literacy and learning through digital media. The statement \"The Joan Ganz Cooney Center does not focus any of its efforts on older people\" implies that the center does not target or prioritize older people in its mission or activities.\n\nBased on this information, we can determine that the statement is entailment, as it is definitely true given the context that the center's focus is on children's literacy and learning, and not on older people. The statement is not false, and it is not irrelevant or undetermined given the context.\n\nTherefore, the correct answer is entailment.
EXAMPLE-3 Minimum Changed Prediction: Sure, I'd be happy to help! Let's analyze the statement \"The Joan Ganz Cooney Center does not focus any of its efforts on older people\" given the context.\n\nFrom the context, we know that the Joan Ganz Cooney Center is an organization that focuses on children's literacy and learning through digital media. This focus on children's literacy and learning implies that the center's efforts are not directed towards older people.\n\nBased on this information, we can determine that the statement is entailment, as it is definitely true given the context that the center's focus is on children's literacy and learning, and not on older people. The statement is not false, and it is not irrelevant or undetermined given the context.\n\nTherefore, the correct answer is entailment. Final Answer: Entailment


EXAMPLE-4:
The prediction and its details are all correctly, no need to change anything.
EXAMPLE-4 Question: We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context.\nContext: The Scarlet Letter (stylized The SCARlet Letter) is the fifth studio album by American recording artist Lil' Mo. It was first released on October 27, 2014 by Penalty Entertainment. Its first and only single, \"Should've Never Let You Go\" preceded its release on September 23, 2014.\nStatement: The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment.\n\nYou already know the answer is contradiction. Please solve this problem while assuming you don\u2019t know the answer.
EXAMPLE-4 Groundtruth: Contradiction
EXAMPLE-4 Previous Prediction: Sure, I'd be happy to help! Let's analyze the statement given the context:\n\nThe statement is \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment.\"\n\nThe context is: The Scarlet Letter is the fifth studio album by American recording artist Lil' Mo. It was first released on October 27, 2014 by Penalty Entertainment. Its first and only single, \"Should've Never Let You Go\" preceded its release on September 23, 2014.\n\nNow, let's determine whether the statement is entailment, contradiction, or neutral given the context:\n\n1. Entailment: The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" does not entail the context. The context states that the album was released on October 27, 2014, not October 23, 2014. Therefore, the statement is not entailment.\n2. Contradiction: The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" contradicts the context. The context states that the album was released on October 27, 2014, but the statement claims it was released on October 23, 2014. Therefore, the statement is a contradiction.\n3. Neutral: The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" is not neutral given the context. The context states that the album was released on October 27, 2014, and the statement claims it was released on October 23, 2014, which is incorrect. Therefore, the statement is not neutral.\n\nSo, the answer is contradiction. The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" contradicts the context given.
EXAMPLE-4 Minimum Changed Prediction: Sure, I'd be happy to help! Let's analyze the statement given the context:\n\nThe statement is \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment.\"\n\nThe context is: The Scarlet Letter is the fifth studio album by American recording artist Lil' Mo. It was first released on October 27, 2014 by Penalty Entertainment. Its first and only single, \"Should've Never Let You Go\" preceded its release on September 23, 2014.\n\nNow, let's determine whether the statement is entailment, contradiction, or neutral given the context:\n\n1. Entailment: The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" does not entail the context. The context states that the album was released on October 27, 2014, not October 23, 2014. Therefore, the statement is not entailment.\n2. Contradiction: The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" contradicts the context. The context states that the album was released on October 27, 2014, but the statement claims it was released on October 23, 2014. Therefore, the statement is a contradiction.\n3. Neutral: The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" is not neutral given the context. The context states that the album was released on October 27, 2014, and the statement claims it was released on October 23, 2014, which is incorrect. Therefore, the statement is not neutral.\n\nSo, the answer is contradiction. The statement \"The Scarlet Letter was first released on October 23, 2014 by Penalty Entertainment\" contradicts the context given. Final Answer: Contradiction

Please perform the minimum change to the problem below.

Question: {question} 
Groundtruth: {gold_label}
PREDICTION: {previous_prediction}

You need to:
1. You will modify the prediction to make it correct while keep as much original words(if those is correct) as possible. You should try your best to use the original words with as little modification as possible. This requirement is very important. 
2. Directly give me the minimum changed prediction without mention anything like 'i changed it because it is wrong' or 'we changed this because '
3. Make sure the prediction first analyze then answer. In otherwords, if the prediction answer first then explain, you need to put the final answer at the end.
4. Since you are trying to preerve the original words by using minimum change, thus if all of the parts in the prediction is correct, you can just keep the way as it is (but you still need to make sure the final answer is at the end.) 
"""
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        # print('QUESTION: ', question)
        # print()
        # print('GOLD LABEL: ', gold_label)
        # print()
        # print('PREVIOUS PREDICTION: ', previous_prediction)
        # print()
        # print('ANSWER: ', answer)
        # print('---------------------------------------------')
        temp = {}
        temp['question'] = question
        temp['input'] = ''
        temp['output'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        temp['reason'] = item['reason']
        input_json_list.append(temp)
    return input_json_list

def minimum_change_data_creation_ANLI_simplified_feb_12(train_data, original_minimum_change_train_data):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(original_minimum_change_train_data):
        question = item['question']
        statement = item['hypothesis']
        explaination = item['reason']
        gold_label = item['gold_label']
        previous_prediction = item['output']

        prompt =  \
f"""We have a previous prediction for a NLI question. The previous prediction sometimes give the final answer(entailment/neutral/contradiction) first the nexplain. If that is the case, please move the final answer to the end while keep the explaination at the front. If the previous prediction inference first then give the final answer at the end, then there is nothing you need to do. just keep the original answer as what it is.

EXAMPLE-1:
The prediction answer neutral first then explain. We place the final answer at the end. we keep as much original words as possible
EXAMPLE-1 Previous Prediction: Sure! The statement entail with the context. \n\nHomebrew is free and open-source means that it is definitely true given the context, because Homebrew is indeed free to use. The fact that Homebrew is free and open-source indicate that it costs nothing.\n\nTherefore, based on the context provided, the statement "Homebrew costs nothing" is entailment, as it is free and open-source. Final Answer: Entailment
EXAMPLE-1 Minimum Changed Prediction: Sure! \n\nHomebrew is free and open-source means that it is definitely true given the context, because Homebrew is indeed free to use. The fact that Homebrew is free and open-source indicate that it costs nothing.\n\nTherefore, based on the context provided, the statement "Homebrew costs nothing" is entailment, as it is free and open-source. Final Answer: Entailment

EXAMPLE-2:
The prediction answer neutral first then explain. We place the final answer at the end. we keep as much original words as possible
EXAMPLE-2 Previous Prediction: Sure, I'd be happy to help! The relation between context and stetment is neutral. \n\nSo, based on the context provided, we know that Benny Bell was an American singer-songwriter who reached popularity in the 1940s and had a comeback in the 1970s. The statement \"Benny Bell did not always write his own songs\" is given.\n\nTo determine whether this statement is entailment, contradiction, or neutral, we need to consider the truth of the statement given the context.\n\nThe statement "Benny Bell did not always write his own songs" does not provide information that directly supports or contradicts the context provided. The context focuses on his popularity and the nature of his songs, but does not specify whether he always wrote his own songs or not. Therefore, the truth of the statement is undetermined or irrelevant given the context. \n\nTherefore, based on the context provided, the statement \"Benny Bell did not always write his own songs\" is neutral, as it does not provide any information about the truth or falsity of the statement. Final Answer: Neutral
EXAMPLE-2 Minimum Changed Prediction: Sure, I'd be happy to help!\n\nSo, based on the context provided, we know that Benny Bell was an American singer-songwriter who reached popularity in the 1940s and had a comeback in the 1970s. The statement \"Benny Bell did not always write his own songs\" is given.\n\nTo determine whether this statement is entailment, contradiction, or neutral, we need to consider the truth of the statement given the context.\n\nThe statement "Benny Bell did not always write his own songs" does not provide information that directly supports or contradicts the context provided. The context focuses on his popularity and the nature of his songs, but does not specify whether he always wrote his own songs or not. Therefore, the truth of the statement is undetermined or irrelevant given the context. \n\nTherefore, based on the context provided, the statement \"Benny Bell did not always write his own songs\" is neutral, as it does not provide any information about the truth or falsity of the statement. Final Answer: Neutral




EXAMPLE-3:
the previous prediction inference first then put the final answer at the end. thus we do not change it. we keep as much original words as possible
EXAMPLE-3 Previous Prediction: Sure, I'd be happy to help! Let's analyze the statement \"The Joan Ganz Cooney Center does not focus any of its efforts on older people\" given the context.\n\nFrom the context, we know that the Joan Ganz Cooney Center is an organization that focuses on children's literacy and learning through digital media. This focus on children's literacy and learning implies that the center's efforts are not directed towards older people.\n\nBased on this information, we can determine that the statement is entailment, as it is definitely true given the context that the center's focus is on children's literacy and learning, and not on older people. The statement is not false, and it is not irrelevant or undetermined given the context.\n\nTherefore, the correct answer is entailment. Final Answer: Entailment
EXAMPLE-3 Minimum Changed Prediction: Sure, I'd be happy to help! Let's analyze the statement \"The Joan Ganz Cooney Center does not focus any of its efforts on older people\" given the context.\n\nFrom the context, we know that the Joan Ganz Cooney Center is an organization that focuses on children's literacy and learning through digital media. This focus on children's literacy and learning implies that the center's efforts are not directed towards older people.\n\nBased on this information, we can determine that the statement is entailment, as it is definitely true given the context that the center's focus is on children's literacy and learning, and not on older people. The statement is not false, and it is not irrelevant or undetermined given the context.\n\nTherefore, the correct answer is entailment. Final Answer: Entailment

Please perform the minimum change to the problem below.

previous prediction: {previous_prediction}

You need to:
1. make sure the final answer to the end while keep the explaination at the front. If the previous prediction inference first then give the final answer at the end, then there is nothing you need to do. just keep the original answer as what it is
2. Directly give me the modified prediction without mention anything like 'i changed it because it is wrong' or 'we changed this because '
4. Since you are trying to preerve the original words by using minimum change
"""
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        # print('QUESTION: ', question)
        # print()
        # print('GOLD LABEL: ', gold_label)
        # print()
        # print('PREVIOUS PREDICTION: ', previous_prediction)
        # print()
        # print('ANSWER: ', answer)
        # print('---------------------------------------------')
        temp = {}
        temp['question'] = question
        temp['input'] = ''
        temp['output'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = item['gold_label']
        temp['hypothesis'] = item['hypothesis']
        temp['premise'] = item['premise']
        temp['reason'] = item['reason']
        input_json_list.append(temp)
    return input_json_list


def minimum_change_data_creation_CODE_vanilla_total(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        previous_prediction = predict_list[i]
        ground_truth = item['ground_truth']
        task_id = item['task_id']
        entry_point = item['entry_point']
        test = item['test']
        

        example_q1 = \
"""\n1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n\n\nProblem: from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"""
        example_a1 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a helper function called `closer_than` that takes two numbers and returns `True` if they are closer to each other than the given threshold, and `False` otherwise:\n```\ndef closer_than(a, b, threshold):\n    return abs(a - b) < threshold\n```\nNow, I can write the `has_close_elements` function using this helper function:\n```\ndef has_close_elements(numbers, threshold):\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if closer_than(numbers[i], numbers[j], threshold):\n                return True\n    return False\n```\n\nFINAL ANSWER:\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if closer_than(numbers[i], numbers[j], threshold):\n                return True\n    return False\n"""
        example_p1 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a helper function called `closer_than` that takes two numbers and returns `True` if they are closer to each other than the given threshold, and `False` otherwise:\n```\ndef closer_than(a, b, threshold):\n    return abs(a - b) < threshold\n```\nNow, I can write the `has_close_elements` function using this helper function:\n```\ndef has_close_elements(numbers, threshold):\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if closer_than(numbers[i], numbers[j], threshold):\n                return True\n    return False\n```\nHere's how the function works:\n\n1. I iterate over the list of numbers twice, once from the start to the middle, and once from the middle to the end.\n2. For each pair of adjacent numbers, I check if they are closer to each other than the given threshold using the `closer_than` function. If they are, I return `True` immediately.\n3. If no pair of adjacent numbers is closer than the threshold, I return `False` at the end of the second iteration.\n\nHere's an example of how the function works:\n```\n>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\nTrue\n```\nIn this example, the function checks the pair `(1.0, 2.8)`, `(2.8, 3.0)`, `(3.0, 4.0)`, and `(4.0, 5.0)`, and finds that the pair `(2.8, 3.0)` is closer to each other than the threshold of 0.3. Therefore, it returns `True`.\n\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_analyzation_1 = \
"""The solution is already correct. We just filter out the words after it gets the solution."""
        
        example_q2 = \
"""\n1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n\n\nProblem: from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n"""
        example_a2 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:

First, I'll define a helper function called `is_open` that takes a single character as input and returns `True` if it is an open parenthesis, and `False` otherwise. Here's the implementation:
```
def is_open(char):
    return char == '('
```
Next, I'll define the `separate_paren_groups` function. Here's the implementation:
```
def separate_paren_groups(paren_string):
    # Initialize an empty list to store the separated groups
    groups = []
    
    # Variable to hold the current group of parentheses
    current_group = ""
    
    # Variable to keep track of open parentheses count
    open_count = 0
    
    # Iterate through the input string, checking for open and close parentheses
    for char in paren_string:
        if is_open(char):
            # Increment open parentheses count
            open_count += 1
            current_group += char
        elif char == ')':
            # Decrement open parentheses count
            open_count -= 1
            current_group += char
            if open_count == 0:
                # Add the current group to the list and reset the current group
                groups.append(current_group)
                current_group = ""
    
    # Return the list of separated groups
    return groups
```
FINAL ANSWER:
    # Initialize an empty list to store the separated groups
    groups = []
    
    # Variable to hold the current group of parentheses
    current_group = ""
    
    # Variable to keep track of open parentheses count
    open_count = 0
    
    # Iterate through the input string, checking for open and close parentheses
    for char in paren_string:
        if is_open(char):
            # Increment open parentheses count
            open_count += 1
            current_group += char
        elif char == ')':
            # Decrement open parentheses count
            open_count -= 1
            current_group += char
            if open_count == 0:
                # Add the current group to the list and reset the current group
                groups.append(current_group)
                current_group = ""
    
    # Return the list of separated groups
    return groups
"""
        example_p2 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a helper function called `is_open` that takes a single character as input and returns `True` if it is an open parenthesis, and `False` otherwise. Here's the implementation:\n```\ndef is_open(char):\n    return char == '('\n```\nNext, I'll define the `separate_paren_groups` function. Here's the implementation:\n```\ndef separate_paren_groups(paren_string):\n    # Initialize an empty list to store the separated groups\n    groups = []\n    \n    # Iterate through the input string, keeping track of the current group\n    # and checking for open and close parentheses\n    for i, char in enumerate(paren_string):\n        if is_open(char):\n            # If we encounter an open parenthesis, start a new group\n            groups.append('')\n        elif char == ')':\n            # If we encounter a close parenthesis, add the current group to the list\n            # and reset the current group to an empty string\n            groups.append('')\n            groups.append(paren_string[i-1])\n        else:\n            # If we encounter a non-parentheses character, add it to the current group\n            groups[len(groups) - 1] += char\n    \n    # Return the list of separated groups\n    return groups\n```\nNow, let's test the function with the example input:\n```\nprint(separate_paren_groups('( ) (( )) (( )( ))'))\n```\nThe output should be:\n```\n['()', '(())', '(()())']\n```\nAs you can see, the function correctly separates the input string into three groups of nested parentheses."""
        example_analyzation_2 = \
"""The part wrong is the prediction is the separate_paren_groups function. We perform the minor changes to make it correct. As you can see, the comments, the way of coding and parameter naming are all similar after change. We try our best to correct it with minor changes."""
        
        example_q3 = \
"""\n1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n\n\nProblem: \n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n"""
        example_a3 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:\n\nTo truncate a positive floating point number `number` and return the decimal part, we can use the following approach:\n\n1. Convert `number` to an integer using the `int()` function. This will give us the largest integer smaller than `number`.\n2. Calculate the difference between `number` and the integer part we just calculated in step 1. This difference will be the decimal part of `number`.\n\nHere's the code for the `truncate_number()` function based on the above approach:\n```\ndef truncate_number(number: float) -> float:\n    integer = int(number)\n    decimal = number - integer\n    return decimal\n```

FINAL ANSWER:
    integer = int(number)
    decimal = number - integer
    return decimal
"""
        example_p3 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nTo truncate a positive floating point number `number` and return the decimal part, we can use the following approach:\n\n1. Convert `number` to an integer using the `int()` function. This will give us the largest integer smaller than `number`.\n2. Calculate the difference between `number` and the integer part we just calculated in step 1. This difference will be the decimal part of `number`.\n\nHere's the code for the `truncate_number()` function based on the above approach:\n```\ndef truncate_number(number: float) -> float:\n    integer = int(number)\n    decimal = number - integer\n    return decimal\n```\nLet's test the function with some examples:\n```\nprint(truncate_number(3.5))  # output: 0.5\nprint(truncate_number(2.75))  # output: 0.75\nprint(truncate_number(1.25))  # output: 0.25\n```\nEach of these outputs confirms that the function is working correctly and returning the decimal part of the input number.\n\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_analyzation_3 = \
"""The prediction is already correct. We just filter out the words after it gets the solution"""
        
        example_q4 = \
"""\n1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n\n\nProblem: from typing import List\n\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n"""
        example_a4 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:

First, I'll define a variable `balance` to keep track of the current balance of the bank account. I'll set it to 0 to start.

```
balance = 0
```

Next, I'll loop through the `operations` list and for each operation, I'll update the `balance` variable accordingly by adding the operation amount to the balance.

```
for operation in operations:
    balance += operation
    if balance < 0:
        return True
```

After the loop, if the balance never goes below zero, I'll return `False`.

```
return False
```

Here's the complete code:

```
def below_zero(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False
```
FINAL ANSWER:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False
"""
        example_p4 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a variable `balance` to keep track of the current balance of the bank account. I'll set it to 0 to start.\n```\nbalance = 0\n```\nNext, I'll loop through the `operations` list and for each operation, I'll update the `balance` variable accordingly. If the operation is a deposit, I'll add the amount to the balance. If the operation is a withdrawal, I'll subtract the amount from the balance.\n```\nfor operation in operations:\n    if operation > 0:\n        balance += operation\n    else:\n        balance -= operation\n```\nAfter the loop, I'll check if the balance is below zero. If it is, I'll return `True`, otherwise I'll return `False`.\n```\nif balance < 0:\n    return True\nelse:\n    return False\n```\nHere's the complete code:\n```\ndef below_zero(operations: List[int]) -> bool:\n    balance = 0\n    for operation in operations:\n        if operation > 0:\n            balance += operation\n        else:\n            balance -= operation\n    return balance < 0\n```\nNow, let's test the function with some examples:\n```\nprint(below_zero([1, 2, 3]))  # Should return False\nprint(below_zero([1, 2, -4, 5]))  # Should return True\n```\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_analyzation_4 = \
"""below_zero function is wrong. there are also other few inference details that is wrong. thus we correct it with minimum change while keep as much original words as possible. we also remove the things after the prediction return the complete code. """

        minimum_change_prompt = f"""Given the question and the Prediction, 

1. Please perform minimum change to the Prediction to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the origininal Prediction while only change where is wrong to make it correct. 
2. Directly give me the minimum changed Prediction. Do not say anything like here is the modifeid prediction or this is wrong because.
3. The prediction needs to output the code in step by step, but it does not need to run test. if the prediction contains the process of running test, you should delete it.
4. put the final answer of the code at the end after the minimum changed prediction. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
5. The modified answer format would be 
MINIMUM_CHANGED_PREDICTION_HERE
FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE

6. Please noticed that the entry point is already given, thus the FINAL_ANSWER does not have to contain the entry point {entry_point}. It just need to be the pieces of code in {entry_point}. It should start with 4 white space since it is in the function
7. The PREVIOUS PREDICTION has to be minimum changed in a way such it logically support the final answer. For example, the PREVIOUS PREDICTION might not solve the problem. it might just give you some hint. you have to minimum change it in such a way that it becomes logical steps guiding you to generate the FINAL ANSWER CODE 


Example1:{example_analyzation_1} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script.
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example1-Question: {example_q1}
Example1-Prediction: {example_p1}
Example1-Minimum Changed Prediction: {example_a1}



Example2: {example_analyzation_2} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script.
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example2-Question: {example_q2}
Example2-Prediction: {example_p2}
Example2-Minimum Changed Prediction: {example_a2}



Example3: {example_analyzation_3} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script.
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example3-Question: {example_q3}
Example3-Prediction: {example_p3}
Example3-Minimum Changed Prediction: {example_a3}



Example4: {example_analyzation_4} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script.
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example4-Question: {example_q4}
Example4-Prediction: {example_p4}
Example4-Minimum Changed Prediction: {example_a4}



Question: {question}
Prediction: {previous_prediction}
Minimum Changed Prediction: 
"""

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)

        temp = {}
        temp['prompt'] = question
        temp['input'] = ''
        temp['answer'] = answer
        temp['canonical_solution'] = ground_truth
        temp['previous_prediction'] = previous_prediction
        temp['task_id'] = task_id
        temp['entry_point'] = entry_point
        temp['test'] = test
        
        input_json_list.append(temp)
    return input_json_list


def minimum_change_data_creation_CODE_feb_1(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        previous_prediction = predict_list[i]
        ground_truth = item['ground_truth']
        task_id = item['task_id']
        entry_point = item['entry_point']
        test = item['test']
        

        example_q1 = \
"""\n1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n\n\nProblem: from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"""
        example_a1 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a helper function called `closer_than` that takes two numbers and returns `True` if they are closer to each other than the given threshold, and `False` otherwise:\n```\ndef closer_than(a, b, threshold):\n    return abs(a - b) < threshold\n```\nNow, I can write the `has_close_elements` function using this helper function:\n```\ndef has_close_elements(numbers, threshold):\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if closer_than(numbers[i], numbers[j], threshold):\n                return True\n    return False\n```\n\nFINAL ANSWER:\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if closer_than(numbers[i], numbers[j], threshold):\n                return True\n    return False\n"""
        example_p1 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a helper function called `closer_than` that takes two numbers and returns `True` if they are closer to each other than the given threshold, and `False` otherwise:\n```\ndef closer_than(a, b, threshold):\n    return abs(a - b) < threshold\n```\nNow, I can write the `has_close_elements` function using this helper function:\n```\ndef has_close_elements(numbers, threshold):\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if closer_than(numbers[i], numbers[j], threshold):\n                return True\n    return False\n```\nHere's how the function works:\n\n1. I iterate over the list of numbers twice, once from the start to the middle, and once from the middle to the end.\n2. For each pair of adjacent numbers, I check if they are closer to each other than the given threshold using the `closer_than` function. If they are, I return `True` immediately.\n3. If no pair of adjacent numbers is closer than the threshold, I return `False` at the end of the second iteration.\n\nHere's an example of how the function works:\n```\n>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\nTrue\n```\nIn this example, the function checks the pair `(1.0, 2.8)`, `(2.8, 3.0)`, `(3.0, 4.0)`, and `(4.0, 5.0)`, and finds that the pair `(2.8, 3.0)` is closer to each other than the threshold of 0.3. Therefore, it returns `True`.\n\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_analyzation_1 = \
"""The solution is already correct. We just filter out the words after it gets the solution."""
        
        example_q2 = \
"""\n1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n\n\nProblem: from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n"""
        example_a2 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:

First, I'll define a helper function called `is_open` that takes a single character as input and returns `True` if it is an open parenthesis, and `False` otherwise. Here's the implementation:
```
def is_open(char):
    return char == '('
```
Next, I'll define the `separate_paren_groups` function. Here's the implementation:
```
def separate_paren_groups(paren_string):
    # Initialize an empty list to store the separated groups
    groups = []
    
    # Variable to hold the current group of parentheses
    current_group = ""
    
    # Variable to keep track of open parentheses count
    open_count = 0
    
    # Iterate through the input string, checking for open and close parentheses
    for char in paren_string:
        if is_open(char):
            # Increment open parentheses count
            open_count += 1
            current_group += char
        elif char == ')':
            # Decrement open parentheses count
            open_count -= 1
            current_group += char
            if open_count == 0:
                # Add the current group to the list and reset the current group
                groups.append(current_group)
                current_group = ""
    
    # Return the list of separated groups
    return groups
```
FINAL ANSWER:
    # Initialize an empty list to store the separated groups
    groups = []
    
    # Variable to hold the current group of parentheses
    current_group = ""
    
    # Variable to keep track of open parentheses count
    open_count = 0
    
    # Iterate through the input string, checking for open and close parentheses
    for char in paren_string:
        if is_open(char):
            # Increment open parentheses count
            open_count += 1
            current_group += char
        elif char == ')':
            # Decrement open parentheses count
            open_count -= 1
            current_group += char
            if open_count == 0:
                # Add the current group to the list and reset the current group
                groups.append(current_group)
                current_group = ""
    
    # Return the list of separated groups
    return groups
"""
        example_p2 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a helper function called `is_open` that takes a single character as input and returns `True` if it is an open parenthesis, and `False` otherwise. Here's the implementation:\n```\ndef is_open(char):\n    return char == '('\n```\nNext, I'll define the `separate_paren_groups` function. Here's the implementation:\n```\ndef separate_paren_groups(paren_string):\n    # Initialize an empty list to store the separated groups\n    groups = []\n    \n    # Iterate through the input string, keeping track of the current group\n    # and checking for open and close parentheses\n    for i, char in enumerate(paren_string):\n        if is_open(char):\n            # If we encounter an open parenthesis, start a new group\n            groups.append('')\n        elif char == ')':\n            # If we encounter a close parenthesis, add the current group to the list\n            # and reset the current group to an empty string\n            groups.append('')\n            groups.append(paren_string[i-1])\n        else:\n            # If we encounter a non-parentheses character, add it to the current group\n            groups[len(groups) - 1] += char\n    \n    # Return the list of separated groups\n    return groups\n```\nNow, let's test the function with the example input:\n```\nprint(separate_paren_groups('( ) (( )) (( )( ))'))\n```\nThe output should be:\n```\n['()', '(())', '(()())']\n```\nAs you can see, the function correctly separates the input string into three groups of nested parentheses."""
        example_analyzation_2 = \
"""The part wrong is the prediction is the separate_paren_groups function. We perform the minor changes to make it correct. As you can see, the comments, the way of coding and parameter naming are all similar after change. We try our best to correct it with minor changes."""
        
        example_q3 = \
"""\n1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n\n\nProblem: \n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n"""
        example_a3 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:\n\nTo truncate a positive floating point number `number` and return the decimal part, we can use the following approach:\n\n1. Convert `number` to an integer using the `int()` function. This will give us the largest integer smaller than `number`.\n2. Calculate the difference between `number` and the integer part we just calculated in step 1. This difference will be the decimal part of `number`.\n\nHere's the code for the `truncate_number()` function based on the above approach:\n```\ndef truncate_number(number: float) -> float:\n    integer = int(number)\n    decimal = number - integer\n    return decimal\n```

FINAL ANSWER:
    integer = int(number)
    decimal = number - integer
    return decimal
"""
        example_p3 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nTo truncate a positive floating point number `number` and return the decimal part, we can use the following approach:\n\n1. Convert `number` to an integer using the `int()` function. This will give us the largest integer smaller than `number`.\n2. Calculate the difference between `number` and the integer part we just calculated in step 1. This difference will be the decimal part of `number`.\n\nHere's the code for the `truncate_number()` function based on the above approach:\n```\ndef truncate_number(number: float) -> float:\n    integer = int(number)\n    decimal = number - integer\n    return decimal\n```\nLet's test the function with some examples:\n```\nprint(truncate_number(3.5))  # output: 0.5\nprint(truncate_number(2.75))  # output: 0.75\nprint(truncate_number(1.25))  # output: 0.25\n```\nEach of these outputs confirms that the function is working correctly and returning the decimal part of the input number.\n\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_analyzation_3 = \
"""The prediction is already correct. We just filter out the words after it gets the solution"""
        
        example_q4 = \
"""\n1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n\n\nProblem: from typing import List\n\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n"""
        example_a4 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:

First, I'll define a variable `balance` to keep track of the current balance of the bank account. I'll set it to 0 to start.

```
balance = 0
```

Next, I'll loop through the `operations` list and for each operation, I'll update the `balance` variable accordingly by adding the operation amount to the balance.

```
for operation in operations:
    balance += operation
    if balance < 0:
        return True
```

After the loop, if the balance never goes below zero, I'll return `False`.

```
return False
```

Here's the complete code:

```
def below_zero(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False
```
FINAL ANSWER:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False
"""
        example_p4 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a variable `balance` to keep track of the current balance of the bank account. I'll set it to 0 to start.\n```\nbalance = 0\n```\nNext, I'll loop through the `operations` list and for each operation, I'll update the `balance` variable accordingly. If the operation is a deposit, I'll add the amount to the balance. If the operation is a withdrawal, I'll subtract the amount from the balance.\n```\nfor operation in operations:\n    if operation > 0:\n        balance += operation\n    else:\n        balance -= operation\n```\nAfter the loop, I'll check if the balance is below zero. If it is, I'll return `True`, otherwise I'll return `False`.\n```\nif balance < 0:\n    return True\nelse:\n    return False\n```\nHere's the complete code:\n```\ndef below_zero(operations: List[int]) -> bool:\n    balance = 0\n    for operation in operations:\n        if operation > 0:\n            balance += operation\n        else:\n            balance -= operation\n    return balance < 0\n```\nNow, let's test the function with some examples:\n```\nprint(below_zero([1, 2, 3]))  # Should return False\nprint(below_zero([1, 2, -4, 5]))  # Should return True\n```\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_analyzation_4 = \
"""below_zero function is wrong. there are also other few inference details that is wrong. thus we correct it with minimum change while keep as much original words as possible. we also remove the things after the prediction return the complete code. """

        minimum_change_prompt = f"""Given the question and the Prediction, 

1. Please perform minimum change to the Prediction to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the origininal Prediction while only change where is wrong to make it correct. 
2. Directly give me the minimum changed Prediction. Do not say anything like here is the modifeid prediction or this is wrong because.
3. The prediction needs to output the code in step by step, but it does not need to run test. if the prediction contains the process of running test, you should delete it.
4. put the final answer of the code at the end after the minimum changed prediction. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
5. The modified answer format would be 
MINIMUM_CHANGED_PREDICTION_HERE
FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE

6. Please noticed that the entry point is already given, thus the FINAL_ANSWER does not have to contain the entry point {entry_point}. It just need to be the pieces of code in {entry_point}. It should start with 4 white space since it is in the function
7. The PREVIOUS PREDICTION has to be minimum changed in a way such it logically support the final answer. For example, the PREVIOUS PREDICTION might not solve the problem. it might just give you some hint. you have to minimum change it in such a way that it becomes logical steps guiding you to generate the FINAL ANSWER CODE 


Example1:{example_analyzation_1} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script.
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example1-Question: {example_q1}
Example1-Prediction: {example_p1}
Example1-Minimum Changed Prediction: {example_a1}



Example2: {example_analyzation_2} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script.
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example2-Question: {example_q2}
Example2-Prediction: {example_p2}
Example2-Minimum Changed Prediction: {example_a2}



Example3: {example_analyzation_3} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script.
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example3-Question: {example_q3}
Example3-Prediction: {example_p3}
Example3-Minimum Changed Prediction: {example_a3}



Example4: {example_analyzation_4} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script.
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example4-Question: {example_q4}
Example4-Prediction: {example_p4}
Example4-Minimum Changed Prediction: {example_a4}



Question: {question}
Prediction: {previous_prediction}
Minimum Changed Prediction: 
"""

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)

        temp = {}
        temp['prompt'] = question
        temp['input'] = ''
        temp['answer'] = answer
        temp['canonical_solution'] = ground_truth
        temp['previous_prediction'] = previous_prediction
        temp['task_id'] = task_id
        temp['entry_point'] = entry_point
        temp['test'] = test
        
        input_json_list.append(temp)
    return input_json_list

def minimum_change_data_creation_CODE(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        previous_prediction = predict_list[i]
        ground_truth = item['ground_truth']
        task_id = item['task_id']
        entry_point = item['entry_point']
        test = item['test']

        code_extraction_prompt = f"""Given the question and the prediction, please directly extract the code in prediction. please do not extract the entrypoint because it is already provided in the question.
        
        Question: {question}
        Predection: {previous_prediction}

        when you extract, please directly extract the code without saying anything like sure I can help you  with or here is the extracted code. also, you do not need to call python, or saying anthing like ```. just directly extract the code without the provided entrypoint in question"""

        previous_code_prediction = create_gpt_completion(code_extraction_prompt, MODEL_ENGINE, client)

        

        minimum_change_prompt = f"""Given the question and the previous prediciton, please 1. if the previous prediction contains the entrypoint {entry_point}, please remove it because it is already provided in question. 2. please perform minimum change to the code to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the original code while only change where is wrong to make it correct. 3. directly give me the modified code(does not contains entrypoint) Do not say anything like here is the modifeid code or this is wrong because. You do not need to add `````` or add python in the front 4. the provided minimum changed code has to be runnable on the code test while entrypoint {entry_point} is given. 5. space is important. since the entry point is given, the minimum changed code has start with 4 white space

questions: {question}
Predection: {previous_code_prediction}
"""



        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)

        temp = {}
        temp['prompt'] = question
        temp['input'] = ''
        temp['answer'] = answer
        temp['canonical_solution'] = ground_truth
        temp['previous_prediction'] = previous_prediction
        temp['task_id'] = task_id
        temp['entry_point'] = entry_point
        temp['test'] = test
        
        input_json_list.append(temp)
    return input_json_list




def minimum_change_data_creation_CODE_april_5(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        previous_prediction = predict_list[i]
        ground_truth = item['ground_truth']
        task_id = item['task_id']
        entry_point = item['entry_point']
        test = item['test']

        example_quesiton_1 = \
'\n1. Please solve the following problem using only the information and resources provided in the question. Do not use external functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n3. Since entry point is already given, you only need to fill out the context under the entry point. Remember that you need to start with 4 white space(indentation) since you your answer will be placed under the entry point. Place your code after \'Final Answer:\'\n\nProblem: \n\ndef fibfib(n: int):\n    """The FibFib number sequence is a sequence similar to the Fibbonacci sequnece that\'s defined as follows:\n    fibfib(0) == 0\n    fibfib(1) == 0\n    fibfib(2) == 1\n    fibfib(n) == fibfib(n-1) + fibfib(n-2) + fibfib(n-3).\n    Please write a function to efficiently compute the n-th element of the fibfib number sequence.\n    >>> fibfib(1)\n    0\n    >>> fibfib(5)\n    4\n    >>> fibfib(8)\n    24\n    """\n\nEntry point: fibfib\n\n\nPlease inference first, then provide the final answer (code) at the end, after \'Final Answer:\'\nBe careful about the spacing. The code after \'Final Answer: \' should start with 4 whitespace for indentation because it is under the entrypoint.\n'
        example_previous_prediction_1 = \
"To efficiently compute the n-th element of the FibFib number sequence, we can use memoization to store the results of previously computed FibFib numbers and avoid redundant calculations. Here's how we can implement it:\n\nFirst, let's define a dictionary to store the previously computed FibFib numbers:\n\n```python\ndef fibfib(n: int, memo={}):\n    if n in memo:\n        return memo[n]\n\n    # Base cases\n    if n <= 2:\n        result = 0\n    else:\n        result = fibfib(n-1, memo) + fibfib(n-2, memo) + fibfib(n-3, memo)\n\n    # Memoize the result\n    memo[n] = result\n\n    return result\n```\n\nHowever, this implementation still has redundant calculations since we are computing fibfib(n-1), fibfib(n-2), and fibfib(n-3) in each recursive call. To avoid this, we can modify the implementation to compute fibfib(n-1) and fibfib(n-2) only once and reuse their results for subsequent recursive calls:\n\n```python\ndef fibfib(n: int, memo={}, a=0, b=1):\n    if n in memo:\n        return memo[n]\n\n    # Base cases\n    if n <= 2:\n        result = n\n    else:\n        result = fibfib(n-1, memo, b, a+b)\n\n    # Memoize the result\n    memo[n] = result\n\n    return result\n```\n\nNow, the function will only make two recursive calls for each new value of n, significantly reducing the number of calculations.\n\nFinal Answer:\n\n```python\ndef fibfib(n: int, memo={}, a=0, b=1):\n    if n in memo:\n        return memo[n]\n\n    # Base cases\n    if n <= 2:\n        result = n\n    else:\n        result = fibfib(n-1, memo, b, a+b)\n\n    # Memoize the result\n    memo[n] = result\n\n    return result\n```"
        example_mc_prediction_1 = \
"""To efficiently compute the n-th element of the FibFib number sequence, we can use memoization to store the results of previously computed FibFib numbers and avoid redundant calculations. Here's how we can implement it:

Let's define a dictionary to store the previously computed FibFib numbers:

Final Answer:
def fibfib(n: int, memo={}):
    if n in memo:
        return memo[n]

    # Corrected base cases
    if n == 0 or n == 1:
        result = 0
    elif n == 2:
        result = 1
    else:
        result = fibfib(n-1, memo) + fibfib(n-2, memo) + fibfib(n-3, memo)

    # Memoize the result
    memo[n] = result

    return result
"""

        example_quesiton_2 = \
'\n1. Please solve the following problem using only the information and resources provided in the question. Do not use external functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n3. Since entry point is already given, you only need to fill out the context under the entry point. Remember that you need to start with 4 white space(indentation) since you your answer will be placed under the entry point. Place your code after \'Final Answer:\'\n\nProblem: \nFIX = """\nAdd more test cases.\n"""\n\ndef vowels_count(s):\n    """Write a function vowels_count which takes a string representing\n    a word as input and returns the number of vowels in the string.\n    Vowels in this case are \'a\', \'e\', \'i\', \'o\', \'u\'. Here, \'y\' is also a\n    vowel, but only when it is at the end of the given word.\n\n    Example:\n    >>> vowels_count("abcde")\n    2\n    >>> vowels_count("ACEDY")\n    3\n    """\n\nEntry point: vowels_count\n\n\nPlease inference first, then provide the final answer (code) at the end, after \'Final Answer:\'\nBe careful about the spacing. The code after \'Final Answer: \' should start with 4 whitespace for indentation because it is under the entrypoint.\n'
        example_previous_prediction_2 = \
'Based on the problem statement, the function `vowels_count` is expected to take a string as input and return the number of vowels in that string. The vowels are defined as \'a\', \'e\', \'i\', \'o\', \'u\' and \'y\' only when it is at the end of the given word.\n\nHere\'s the final answer:\n\n```python\n# Importing no external functions or modules.\n\ndef vowels_count(s):\n    """Write a function vowels_count which takes a string representing\n    a word as input and returns the number of vowels in the string.\n    """\n    vowels = \'aeiouAEIOU\'\n    count = 0\n    for char in s:\n        if char in vowels:\n            count += 1\n    if s[-1] in vowels:\n        count += 1\n    return count\n```\n\nFinal Answer:\n```python\n# Indentation is important!\n\ndef vowels_count(s):\n    """Write a function vowels_count which takes a string representing\n    a word as input and returns the number of vowels in the string.\n    """\n    vowels = \'aeiouAEIOU\'\n    count = 0\n    for char in s:\n        if char in vowels:\n            count += 1\n    if s[-1] in vowels:\n        count += 1\n    return count\n```'
        example_mc_prediction_2 = \
"""Based on the problem statement, the function `vowels_count` is expected to take a string as input and return the number of vowels in that string. The vowels are defined as 'a', 'e', 'i', 'o', 'u' and 'y' only when it is at the end of the given word.

Here's the final answer:

Final Answer:
vowels = 'aeiouAEIOU'
count = 0
for char in s:
    if char in vowels:
        count += 1
if s[-1] == 'y' or s[-1] == 'Y':
    count += 1
return count
"""

        example_quesiton_3 = \
""""""
        example_previous_prediction_3 = \
""""""
        example_mc_prediction_3 = \
""""""


# EXAMPLE-1: The code is already correct, so we directly copy the previous prediction and put the code after 'Final Answer: ' we also keep as much original words as possible. we remove the part that the previous prediction run test. 
# EXAMPLE-1 QUESTION: {example_quesiton_1}
# EXAMPLE-1 PREVIOUS PREDICTION: {example_previous_prediction_1}
# EXAMPLE-1 MINIMUM CHANGED PREDICTION: {example_mc_prediction_1}

# EXAMPLE-2: Modify the wrong part while keep as much original words as possible. 
# EXAMPLE-2 QUESTION: {example_quesiton_2}
# EXAMPLE-2 PREVIOUS PREDICTION: {example_previous_prediction_2}
# EXAMPLE-2 MINIMUM CHANGED PREDICTION: {example_mc_prediction_2}

# EXAMPLE-3: Modify the wrong part while keep as much original words as possible. it remove the part that run the test. It place the code after 'Final Answer:'
# EXAMPLE-3 QUESTION: {example_quesiton_3}
# EXAMPLE-3 PREVIOUS PREDICTION: {example_previous_prediction_3}
# EXAMPLE-3 MINIMUM CHANGED PREDICTION: {example_mc_prediction_3}

        minimum_change_prompt = \
f"""You have need to do two things. You need to perform minimum change to the previous prediction to make it correct. 
Explaination of minimum change:
Given the question and the previous prediciton, please 
1. if the previous prediction contains the entrypoint {entry_point}, please remove it because it is already provided in question. 
2. please perform minimum change to the previous prediction to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the original words from the previous prediction while only change where is wrong to make it correct. 
3. directly give me the modified prediction(the code part does not contains entrypoint) Do not say anything like here is the modifeid prediction or this is wrong because. 
4. You do not need to add `````` or add python in the front of the code. Add the code after 'Final Answer:' at the end with proper spacing.
5. the provided minimum changed code has to be runnable on the code test while entrypoint {entry_point} is given. 
6. space is important. since the entry point is given, the minimum changed code has start with 4 white space


Remember Final Answer: code is part of the previous prediction and the modified prediction.
For example, i give you 

previous prediction: aaa
final answer: bbb


it means the previous prediction is 
"aaa
final answer: bbb"

aaa and final answer bbb is not seperate.


EXAMPLE-1:
EXAMPLE-1 QUESTION: {example_quesiton_1}
EXAMPLE-1 PREVIOUS PREDICTION: "{example_previous_prediction_1}"
EXAMPLE-1 MINIMUM CHANGED PREDICTION: "{example_mc_prediction_1}"

EXAMPLE-2: Modify the wrong part while keep as much original words as possible. 
EXAMPLE-2 QUESTION: {example_quesiton_2}
EXAMPLE-2 PREVIOUS PREDICTION: "{example_previous_prediction_2}"
EXAMPLE-2 MINIMUM CHANGED PREDICTION: "{example_mc_prediction_2}"

please carefully read the 2 examples and learn how to correctly perform minimum change. You have to change in the same way
please perform minimum change on everything instead of just the code after Final Answer:

Please perform minimum change to the following previous aprediction. 
You have to change as few previous prediction as possible. Remmeber it. please!!! remmeber this please!
QUESTION: {question}
PREVIOUS PREDICTION(including Final Answer which is part of code): "{previous_prediction}"
MINIMUM CHANGED PREDICTION:


please do not only change the code after Final answer. please read my instruction carefully. you need to perform minimum change on everything in previous prediction.
"""


        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)

        item['prompt'] = question
        item['input'] = ''
        item['answer'] = answer
        item['canonical_solution'] = ground_truth
 
        input_json_list.append(item)
    return input_json_list


def minimum_change_data_creation_CODE_code_only(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        question = item['question']
        previous_prediction = predict_list[i]
        entry_point = item['entry_point']
        prompt = item['prompt']

        extract_code_prompt = \
f"""You need to extract the code from the previous prediction. The prediction answer the QUESTION.
QUESTION: {question}
Previous prediction: {previous_prediction}

Explaination of extract code: 
1. You have to extract code from the previous prediction. Your answer will be evaluated to see if it will pass the test, thus of course you need to extract the code and comments without including any other words.
2. Please directly show me the extracted code with no other words. We will run the code to test if it is correct, thus do not extract other words.
3. you suppose to extract the code from the previous prediction. You should never correct the code on your own even if it is wrong.
4. If you see 'Final Answer:'  you should not include it in your extracted code.
"""
        code = create_gpt_completion(extract_code_prompt, MODEL_ENGINE, client)
        code = code.replace('```python', '')
        code = code.replace('```', '')
        data_list[i]['previous_code'] = code



        original_question = f"""
1. Please solve the following problem using only the information and resources provided in the question. Do not use external functions not mentioned or implied in the problem statement.
2. You need to try your best to solve this problem. 
3. Place your code after 'Final Answer:'

Problem: {prompt}
Entry point: {entry_point}"""

        minimum_change_prompt = \
f"""You need to perform minimum change on the previous code to make sure it is correct. when performining minimum change, make sure to change as little as possible to make it correct. 
QUESTION: {original_question}
Previous code: {code}

Explaination of minimum change: 
1. Given the question and the previous code, please perform minimum change. When I say minimum change, i mean you should try to modify as little previous prediction as possible to make it correct. Your are trying your best to preserve as much original words as possible. do not remove anything that is not wrong, such as comments.
2. Directly give me the minimum changed code.  The minimum changed code has to be code only. Do not say anything like 'here is the modifeid code', or 'this is wrong because'.
4. You do not need to modify the previous code if it is already correct. Please be careful about the spacing in code.

minimum changed code: 
"""

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        answer = answer.replace('```python', '')
        answer = answer.replace('```', '')

        answer_final = f"""Final Answer:{answer}"""
        data_list[i]['answer'] = answer_final
        data_list[i]['intput'] = ''
        data_list[i]['previous_prediction'] = previous_prediction
    return data_list


def extract_code(train_data_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    data_list = []
    for i, item in enumerate(train_data_list):
        item_temp = {}
        question = item['question']
        previous_prediction = item['answer']
        # entry_point = item['entry_point']
        # prompt = item['prompt']

        extract_code_prompt = \
f"""You need to extract the code from the previous prediction. The prediction answer the QUESTION.
QUESTION: {question}
Previous prediction: {previous_prediction}

Explaination of extract code: 
1. You have to extract code from the previous prediction. Your answer will be evaluated to see if it will pass the test, thus of course you need to extract the code and comments without including any other words.
2. Please directly show me the extracted code with no other words. We will run the code to test if it is correct, thus do not extract other words.
3. you suppose to extract the code from the previous prediction. You should never correct the code on your own even if it is wrong.
4. If you see 'Final Answer:'  you should not include it in your extracted code.
"""
        code = create_gpt_completion(extract_code_prompt, MODEL_ENGINE, client)
        code = code.replace('```python', '')
        code = code.replace('```', '')
        item_temp['previous_prediction'] = previous_prediction
        answer_final = f"""Final Answer:{code}"""
        item_temp['answer'] = answer_final
        item_temp['intput'] = ''
        data_list.append(item_temp)
    return data_list

def minimum_change_data_creation_CODE_step_by_step(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        previous_prediction = predict_list[i]
        ground_truth = item['ground_truth']
        task_id = item['task_id']
        entry_point = item['entry_point']
        test = item['test']
        

        minimum_change_prompt = f"""Given the question and the PREVIOUS PREDICTION, 

QUESTION: {question}
PREVIOUS PREDICTION: {previous_prediction}

1. Please perform minimum change to the PREVIOUS PEDICTION to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the origininal PREVIOUS PREDICTION while only change where is wrong to make it correct. 
2. Directly give me the minimum changed PREVIOUS PREDICTION. Do not say anything like here is the modifeid prediction or this is wrong because.
3. The prediction needs to output the code in step by step, but it does not need to run test. if the prediction contains the process of running test, you should delete it.
4. put the final answer of the code at the end after the minimum changed prediction. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
5. The modified answer format would be 
MINIMUM_CHANGED_PREDICTION_HERE
FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE

6. Please noticed that the entry point is already given, thus the FINAL_ANSWER does not have to contain the entry point {entry_point}. It just need to be the pieces of code in {entry_point}. It should start with 4 white space since it is in the function
7. The PREVIOUS PREDICTION has to be minimum changed in a way such it logically support the final answer. For example, the PREVIOUS PREDICTION might not solve the problem. it might just give you some hint. you have to minimum change it in such a way that it becomes logical steps guiding you to generate the FINAL ANSWER CODE """

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)

        temp = {}
        temp['prompt'] = question
        temp['input'] = ''
        temp['answer'] = answer
        temp['canonical_solution'] = ground_truth
        temp['previous_prediction'] = previous_prediction
        temp['task_id'] = task_id
        temp['entry_point'] = entry_point
        temp['test'] = test
        
        input_json_list.append(temp)
    return input_json_list


def minimum_change_data_creation_CODE_step_by_step_optimized(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        previous_prediction = predict_list[i]
        ground_truth = item['ground_truth']
        task_id = item['task_id']
        entry_point = item['entry_point']
        test = item['test']
        

        minimum_change_prompt = f"""Given the question and the PREVIOUS PREDICTION, 

QUESTION: {question}
PREVIOUS PREDICTION: {previous_prediction}

1. Please perform minimum change to the PREVIOUS PEDICTION to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the origininal PREVIOUS PREDICTION while only change where is wrong to make it correct. 
2. Directly give me the minimum changed PREVIOUS PREDICTION. Do not say anything like here is the modifeid prediction or this is wrong because.
3. The prediction needs to output the code in step by step, but it does not need to run test. if the prediction contains the process of running test, you should delete it.
4. put the final answer of the code at the end after the minimum changed prediction. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
5. The modified answer format would be 
MINIMUM_CHANGED_PREDICTION_HERE
FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE

6. Please noticed that the entry point is already given, thus the FINAL_ANSWER does not have to contain the entry point {entry_point}. It just need to be the pieces of code in {entry_point}. It should start with 4 white space since it is in the function
7. The PREVIOUS PREDICTION has to be minimum changed in a way such it logically support the final answer. For example, the PREVIOUS PREDICTION might not solve the problem. it might just give you some hint. you have to minimum change it in such a way that it becomes logical steps guiding you to generate the FINAL ANSWER CODE 

Example 1:
Example1-Question: 
Example1-Previous Prediction: 
Example1-Modified_Answer: 

"""

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)

        temp = {}
        temp['prompt'] = question
        temp['input'] = ''
        temp['answer'] = answer
        temp['canonical_solution'] = ground_truth
        temp['previous_prediction'] = previous_prediction
        temp['task_id'] = task_id
        temp['entry_point'] = entry_point
        temp['test'] = test
        
        input_json_list.append(temp)
    return input_json_list


def minimum_change_data_creation_CODE_step_by_step_optimized_new(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        previous_prediction = predict_list[i]
        ground_truth = item['ground_truth']
        task_id = item['task_id']
        entry_point = item['entry_point']
        test = item['test']
        

        example_q1 = \
"""from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"""
        example_a1 = \
"""Sure, I'd be happy to help you solve this problem! Here's my step-by-step approach:\n\nStep 1: Understand the problem\n-----------------------------\n\nThe problem statement is asking us to write a function `has_close_elements` that takes two arguments: `numbers` and `threshold`. The function should return `True` if there are any two numbers in the list `numbers` that are closer to each other than the threshold, and `False` otherwise.\n\nStep 2: Analyze the given information\n---------------------------------\n\nThe function `has_close_elements` is given a list of numbers `numbers` and a threshold `threshold`. The function should check if there are any two numbers in the list that are closer to each other than the threshold.\n\nStep 3: Plan the solution\n-------------------------\n\nTo solve this problem, we can iterate through the list `numbers` and compare each number with every other number. If we find two numbers that are closer to each other than the threshold, we return `True`. If we don't find any such pairs, we return `False`.\n\nStep 4: Implement the solution\n----------------------------\n\nHere's the implementation of the function `has_close_elements`:\n```\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\n\nFINAL_ANSWER:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False"""
        example_p1 = \
""" Sure, I'd be happy to help you solve this problem! Here's my step-by-step approach:\n\nStep 1: Understand the problem\n-----------------------------\n\nThe problem statement is asking us to write a function `has_close_elements` that takes two arguments: `numbers` and `threshold`. The function should return `True` if there are any two numbers in the list `numbers` that are closer to each other than the threshold, and `False` otherwise.\n\nStep 2: Analyze the given information\n---------------------------------\n\nThe function `has_close_elements` is given a list of numbers `numbers` and a threshold `threshold`. The function should check if there are any two numbers in the list that are closer to each other than the threshold.\n\nStep 3: Plan the solution\n-------------------------\n\nTo solve this problem, we can iterate through the list `numbers` and compare each number with its neighbors. If we find two numbers that are closer to each other than the threshold, we return `True`. If we don't find any such pairs, we return `False`.\n\nStep 4: Implement the solution\n----------------------------\n\nHere's the implementation of the function `has_close_elements`:\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers) - 1):\n        for j in range(i, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\nStep 5: Test the solution\n-------------------------\n\nLet's test the function with some examples:\n```python\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True\n```\nAs we can see, the function correctly returns `True` when there are two numbers closer to each other than the threshold, and `False` otherwise.\n\nThat's it! We have successfully solved the problem using only the information and resources provided in the question."""
        example_analyzation_1 = \
""""""
        
        example_q2 = \
""""""
        example_a2 = \
""""""
        example_p2 = \
""""""
        example_analyzation_2 = \
""""""
        
        example_q3 = \
""""""
        example_a3 = \
""""""
        example_p3 = \
""""""
        example_analyzation_3 = \
""""""
        
        example_q4 = \
""""""
        example_a4 = \
""""""
        example_p4 = \
""""""
        example_analyzation_4 = \
""""""
        minimum_change_prompt = \
f"""We have a question and the prediction. 
1. Please perform minimum change on the prediction to make it correct. 
2. Please also remove the hallucination, create your own answer when the inference steps from the prediction is empty. \
3. Please put the final answer at the end after FINAL ANSWER:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. FINAL ANSWER: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

Example1:{example_analyzation_1} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example1-Question: {example_q1}
Example1-Prediction: {example_p1}
Example1-Minimum Changed Prediction: {example_a1}



Example2: {example_analyzation_2} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example2-Question: {example_q2}
Example2-Prediction: {example_p2}
Example2-Minimum Changed Prediction: {example_a2}



Example3: {example_analyzation_3} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example3-Question: {example_q3}
Example3-Prediction: {example_p3}
Example3-Minimum Changed Prediction: {example_a3}



Example4: {example_analyzation_4} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example4-Question: {example_q4}
Example4-Prediction: {example_p4}
Example4-Minimum Changed Prediction: {example_a4}"""





        minimum_change_prompt = f"""Given the question and the Prediction, 

1. Please perform minimum change to the Prediction to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the origininal Prediction while only change where is wrong to make it correct. 
2. Directly give me the minimum changed Prediction. Do not say anything like here is the modifeid prediction or this is wrong because.
3. The prediction needs to output the code in step by step, but it does not need to run test. if the prediction contains the process of running test, you should delete it.
4. put the final answer of the code at the end after the minimum changed prediction. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
5. The modified answer format would be 
MINIMUM_CHANGED_PREDICTION_HERE
FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE

6. Please noticed that the entry point is already given, thus the FINAL_ANSWER does not have to contain the entry point {entry_point}. It just need to be the pieces of code in {entry_point}. It should start with 4 white space since it is in the function
7. The PREVIOUS PREDICTION has to be minimum changed in a way such it logically support the final answer. For example, the PREVIOUS PREDICTION might not solve the problem. it might just give you some hint. you have to minimum change it in such a way that it becomes logical steps guiding you to generate the FINAL ANSWER CODE 


Example1:{example_analyzation_1} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example1-Question: {example_q1}
Example1-Prediction: {example_p1}
Example1-Minimum Changed Prediction: {example_a1}



Example2: {example_analyzation_2} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example2-Question: {example_q2}
Example2-Prediction: {example_p2}
Example2-Minimum Changed Prediction: {example_a2}



Example3: {example_analyzation_3} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example3-Question: {example_q3}
Example3-Prediction: {example_p3}
Example3-Minimum Changed Prediction: {example_a3}



Example4: {example_analyzation_4} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example4-Question: {example_q4}
Example4-Prediction: {example_p4}
Example4-Minimum Changed Prediction: {example_a4}



Question: {question}
Prediction: {previous_prediction}
Minimum Changed Prediction: 
"""

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)

        temp = {}
        temp['prompt'] = question
        temp['input'] = ''
        temp['answer'] = answer
        temp['canonical_solution'] = ground_truth
        temp['previous_prediction'] = previous_prediction
        temp['task_id'] = task_id
        temp['entry_point'] = entry_point
        temp['test'] = test
        
        input_json_list.append(temp)
    return input_json_list




def minimum_change_data_creation_CODE_vanilla_total_pure(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        previous_prediction = predict_list[i]
        ground_truth = item['ground_truth']
        task_id = item['task_id']
        entry_point = item['entry_point']
        test = item['test']
        

        example_q1 = \
"""\n1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n\n\nProblem: from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"""
        example_a1 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a helper function called `closer_than` that takes two numbers and returns `True` if they are closer to each other than the given threshold, and `False` otherwise:\n```\ndef closer_than(a, b, threshold):\n    return abs(a - b) < threshold\n```\nNow, I can write the `has_close_elements` function using this helper function:\n```\ndef has_close_elements(numbers, threshold):\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if closer_than(numbers[i], numbers[j], threshold):\n                return True\n    return False\n```"""
        example_p1 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a helper function called `closer_than` that takes two numbers and returns `True` if they are closer to each other than the given threshold, and `False` otherwise:\n```\ndef closer_than(a, b, threshold):\n    return abs(a - b) < threshold\n```\nNow, I can write the `has_close_elements` function using this helper function:\n```\ndef has_close_elements(numbers, threshold):\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if closer_than(numbers[i], numbers[j], threshold):\n                return True\n    return False\n```\nHere's how the function works:\n\n1. I iterate over the list of numbers twice, once from the start to the middle, and once from the middle to the end.\n2. For each pair of adjacent numbers, I check if they are closer to each other than the given threshold using the `closer_than` function. If they are, I return `True` immediately.\n3. If no pair of adjacent numbers is closer than the threshold, I return `False` at the end of the second iteration.\n\nHere's an example of how the function works:\n```\n>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\nTrue\n```\nIn this example, the function checks the pair `(1.0, 2.8)`, `(2.8, 3.0)`, `(3.0, 4.0)`, and `(4.0, 5.0)`, and finds that the pair `(2.8, 3.0)` is closer to each other than the threshold of 0.3. Therefore, it returns `True`.\n\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_analyzation_1 = \
"""The solution is already correct. We just filter out the words after it gets the solution."""
        
        example_q2 = \
"""\n1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n\n\nProblem: from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n"""
        example_a2 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:

First, I'll define a helper function called `is_open` that takes a single character as input and returns `True` if it is an open parenthesis, and `False` otherwise. Here's the implementation:
```
def is_open(char):
    return char == '('
```
Next, I'll define the `separate_paren_groups` function. Here's the implementation:
```
def separate_paren_groups(paren_string):
    # Initialize an empty list to store the separated groups
    groups = []
    
    # Variable to hold the current group of parentheses
    current_group = ""
    
    # Variable to keep track of open parentheses count
    open_count = 0
    
    # Iterate through the input string, checking for open and close parentheses
    for char in paren_string:
        if is_open(char):
            # Increment open parentheses count
            open_count += 1
            current_group += char
        elif char == ')':
            # Decrement open parentheses count
            open_count -= 1
            current_group += char
            if open_count == 0:
                # Add the current group to the list and reset the current group
                groups.append(current_group)
                current_group = ""
    
    # Return the list of separated groups
    return groups
```"""
        example_p2 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a helper function called `is_open` that takes a single character as input and returns `True` if it is an open parenthesis, and `False` otherwise. Here's the implementation:\n```\ndef is_open(char):\n    return char == '('\n```\nNext, I'll define the `separate_paren_groups` function. Here's the implementation:\n```\ndef separate_paren_groups(paren_string):\n    # Initialize an empty list to store the separated groups\n    groups = []\n    \n    # Iterate through the input string, keeping track of the current group\n    # and checking for open and close parentheses\n    for i, char in enumerate(paren_string):\n        if is_open(char):\n            # If we encounter an open parenthesis, start a new group\n            groups.append('')\n        elif char == ')':\n            # If we encounter a close parenthesis, add the current group to the list\n            # and reset the current group to an empty string\n            groups.append('')\n            groups.append(paren_string[i-1])\n        else:\n            # If we encounter a non-parentheses character, add it to the current group\n            groups[len(groups) - 1] += char\n    \n    # Return the list of separated groups\n    return groups\n```\nNow, let's test the function with the example input:\n```\nprint(separate_paren_groups('( ) (( )) (( )( ))'))\n```\nThe output should be:\n```\n['()', '(())', '(()())']\n```\nAs you can see, the function correctly separates the input string into three groups of nested parentheses."""
        example_analyzation_2 = \
"""The part wrong is the prediction is the separate_paren_groups function. We perform the minor changes to make it correct. As you can see, the comments, the way of coding and parameter naming are all similar after change. We try our best to correct it with minor changes."""
        
        example_q3 = \
"""\n1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n\n\nProblem: \n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n"""
        example_a3 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:\n\nTo truncate a positive floating point number `number` and return the decimal part, we can use the following approach:\n\n1. Convert `number` to an integer using the `int()` function. This will give us the largest integer smaller than `number`.\n2. Calculate the difference between `number` and the integer part we just calculated in step 1. This difference will be the decimal part of `number`.\n\nHere's the code for the `truncate_number()` function based on the above approach:\n```\ndef truncate_number(number: float) -> float:\n    integer = int(number)\n    decimal = number - integer\n    return decimal\n```"""
        example_p3 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nTo truncate a positive floating point number `number` and return the decimal part, we can use the following approach:\n\n1. Convert `number` to an integer using the `int()` function. This will give us the largest integer smaller than `number`.\n2. Calculate the difference between `number` and the integer part we just calculated in step 1. This difference will be the decimal part of `number`.\n\nHere's the code for the `truncate_number()` function based on the above approach:\n```\ndef truncate_number(number: float) -> float:\n    integer = int(number)\n    decimal = number - integer\n    return decimal\n```\nLet's test the function with some examples:\n```\nprint(truncate_number(3.5))  # output: 0.5\nprint(truncate_number(2.75))  # output: 0.75\nprint(truncate_number(1.25))  # output: 0.25\n```\nEach of these outputs confirms that the function is working correctly and returning the decimal part of the input number.\n\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_analyzation_3 = \
"""The prediction is already correct. We just filter out the words after it gets the solution"""
        
        example_q4 = \
"""\n1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n\n\nProblem: from typing import List\n\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n"""
        example_a4 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:

First, I'll define a variable `balance` to keep track of the current balance of the bank account. I'll set it to 0 to start.

```
balance = 0
```

Next, I'll loop through the `operations` list and for each operation, I'll update the `balance` variable accordingly by adding the operation amount to the balance.

```
for operation in operations:
    balance += operation
    if balance < 0:
        return True
```

After the loop, if the balance never goes below zero, I'll return `False`.

```
return False
```

Here's the complete code:

```
def below_zero(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False
```"""
        example_p4 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a variable `balance` to keep track of the current balance of the bank account. I'll set it to 0 to start.\n```\nbalance = 0\n```\nNext, I'll loop through the `operations` list and for each operation, I'll update the `balance` variable accordingly. If the operation is a deposit, I'll add the amount to the balance. If the operation is a withdrawal, I'll subtract the amount from the balance.\n```\nfor operation in operations:\n    if operation > 0:\n        balance += operation\n    else:\n        balance -= operation\n```\nAfter the loop, I'll check if the balance is below zero. If it is, I'll return `True`, otherwise I'll return `False`.\n```\nif balance < 0:\n    return True\nelse:\n    return False\n```\nHere's the complete code:\n```\ndef below_zero(operations: List[int]) -> bool:\n    balance = 0\n    for operation in operations:\n        if operation > 0:\n            balance += operation\n        else:\n            balance -= operation\n    return balance < 0\n```\nNow, let's test the function with some examples:\n```\nprint(below_zero([1, 2, 3]))  # Should return False\nprint(below_zero([1, 2, -4, 5]))  # Should return True\n```\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_analyzation_4 = \
"""below_zero function is wrong. there are also other few inference details that is wrong. thus we correct it with minimum change while keep as much original words as possible. we also remove the things after the prediction return the complete code. """

        minimum_change_prompt = f"""Given the question and the Prediction, 

1. Please perform minimum change to the Prediction to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the origininal Prediction while only change where is wrong to make it correct. 
2. Directly give me the minimum changed Prediction. Do not say anything like here is the modifeid prediction or this is wrong because.
3. The prediction needs to output the code in step by step, but it does not need to run test. if the prediction contains the process of running test, you should delete it.
4. put the final answer of the code at the end after the minimum changed prediction. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
5. The modified answer format would be 
MINIMUM_CHANGED_PREDICTION_HERE
FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE

6. Please noticed that the entry point is already given, thus the FINAL_ANSWER does not have to contain the entry point {entry_point}. It just need to be the pieces of code in {entry_point}. It should start with 4 white space since it is in the function
7. The PREVIOUS PREDICTION has to be minimum changed in a way such it logically support the final answer. For example, the PREVIOUS PREDICTION might not solve the problem. it might just give you some hint. you have to minimum change it in such a way that it becomes logical steps guiding you to generate the FINAL ANSWER CODE 


Example1:{example_analyzation_1} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script.
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example1-Question: {example_q1}
Example1-Prediction: {example_p1}
Example1-Minimum Changed Prediction: {example_a1}



Example2: {example_analyzation_2} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script.
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example2-Question: {example_q2}
Example2-Prediction: {example_p2}
Example2-Minimum Changed Prediction: {example_a2}



Example3: {example_analyzation_3} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script.
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example3-Question: {example_q3}
Example3-Prediction: {example_p3}
Example3-Minimum Changed Prediction: {example_a3}



Example4: {example_analyzation_4} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script.
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example4-Question: {example_q4}
Example4-Prediction: {example_p4}
Example4-Minimum Changed Prediction: {example_a4}



Question: {question}
Prediction: {previous_prediction}
Minimum Changed Prediction: 
"""

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)

        temp = {}
        temp['prompt'] = question
        temp['input'] = ''
        temp['answer'] = answer
        temp['canonical_solution'] = ground_truth
        temp['previous_prediction'] = previous_prediction
        temp['task_id'] = task_id
        temp['entry_point'] = entry_point
        temp['test'] = test
        
        input_json_list.append(temp)
    return input_json_list


def minimum_change_data_creation_CODE_vanilla_total_feb_1(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        previous_prediction = predict_list[i]
        ground_truth = item['ground_truth']
        task_id = item['task_id']
        entry_point = item['entry_point']
        test = item['test']
        

        example_q1 = \
f"""1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.
2. You need to try your best to solve this problem. 
3. You only have to complete and show the code inside the entrypoint has_close_elements. Please be careful about the spacing. The code inside the entrypoint should start with 4 whitespace for indentation.
\n\n\nProblem: from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"""
        example_a1 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a helper function called `closer_than` that takes two numbers and returns `True` if they are closer to each other than the given threshold, and `False` otherwise:\n```\ndef closer_than(a, b, threshold):\n    return abs(a - b) < threshold\n```\nNow, I can complete the code inside the `has_close_elements` function using this helper function. 
FINAL ANSWER:\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if closer_than(numbers[i], numbers[j], threshold):\n                return True\n    return False\n"""
        example_p1 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a helper function called `closer_than` that takes two numbers and returns `True` if they are closer to each other than the given threshold, and `False` otherwise:\n```\ndef closer_than(a, b, threshold):\n    return abs(a - b) < threshold\n```\nNow, I can write the `has_close_elements` function using this helper function:\n```\ndef has_close_elements(numbers, threshold):\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if closer_than(numbers[i], numbers[j], threshold):\n                return True\n    return False\n```\nHere's how the function works:\n\n1. I iterate over the list of numbers twice, once from the start to the middle, and once from the middle to the end.\n2. For each pair of adjacent numbers, I check if they are closer to each other than the given threshold using the `closer_than` function. If they are, I return `True` immediately.\n3. If no pair of adjacent numbers is closer than the threshold, I return `False` at the end of the second iteration.\n\nHere's an example of how the function works:\n```\n>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\nTrue\n```\nIn this example, the function checks the pair `(1.0, 2.8)`, `(2.8, 3.0)`, `(3.0, 4.0)`, and `(4.0, 5.0)`, and finds that the pair `(2.8, 3.0)` is closer to each other than the threshold of 0.3. Therefore, it returns `True`.\n\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_analyzation_1 = \
"""The solution is already correct. We just filter out the words after it gets the solution."""
        
        example_q2 = \
f"""1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.
2. You need to try your best to solve this problem. 
3. You only have to complete and show the code inside the entrypoint has_close_elements. Please be careful about the spacing. The code inside the entrypoint should start with 4 whitespace for indentation.
from typing import List\n\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\" Insert a number 'delimeter' between every two consecutive elements of input list `numbers'\n    >>> intersperse([], 4)\n    []\n    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n    \"\"\"\n"""
        example_a2 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\" Insert a number 'delimeter' between every two consecutive elements of input list `numbers' \"\"\"\n    result = []\n    for i in range(0, len(numbers), 2):\n        result.append(numbers[i])\n        result.append(delimeter)\n    for i in range(1, len(numbers) - 1, 2):\n        result.append(numbers[i])\n        result.append(delimeter)\n    return result\n\n
FINAL ANSWER:
\"\"\" Insert a number 'delimeter' between every two consecutive elements of input list `numbers' \"\"\"\n    result = []\n    for i in range(0, len(numbers), 2):\n        result.append(numbers[i])\n        result.append(delimeter)\n    for i in range(1, len(numbers) - 1, 2):\n        result.append(numbers[i])\n        result.append(delimeter)\n    return result"""
        example_p2 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\" Insert a number 'delimeter' between every two consecutive elements of input list `numbers' \"\"\"\n    result = []\n    for i in range(0, len(numbers), 2):\n        result.append(numbers[i])\n        result.append(delimeter)\n    for i in range(1, len(numbers) - 1, 2):\n        result.append(numbers[i])\n        result.append(delimeter)\n    return result\n\nHere's how the function works:\n\n1. For every even index (i.e., 0, 2, 4, etc.) in the input list, we append the current element to the result list, without any intervening delimeter.\n2. For every odd index (i.e., 1, 3, 5, etc.) in the input list, we append the current element to the result list, followed by the delimeter.\n\nWith these steps, we ensure that the delimeter is inserted between every two consecutive elements of the input list, as desired.\n\nI hope this helps! Let me know if you have any questions or need further clarification."""
        example_analyzation_2 = \
"""This prediction was too wrong. We therefore just extract and put the FINAL ANSWER at the end while removing the content after the complete code."""
        
        example_q3 = \
f"""1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.
2. You need to try your best to solve this problem. 
3. You only have to complete and show the code inside the function {entry_point}. Please be careful about the spacing. The code inside the function should start with 4 whitespace for indentation.
\n\n\nProblem: \n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n"""
        example_a3 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:\n\nTo truncate a positive floating point number `number` and return the decimal part, we can use the following approach:\n\n1. Convert `number` to an integer using the `int()` function. This will give us the largest integer smaller than `number`.\n2. Calculate the difference between `number` and the integer part we just calculated in step 1. This difference will be the decimal part of `number`. \n\nHere's the code for the `truncate_number()` function based on the above approach: 
FINAL ANSWER:
    integer = int(number)
    decimal = number - integer
    return decimal
"""
        example_p3 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nTo truncate a positive floating point number `number` and return the decimal part, we can use the following approach:\n\n1. Convert `number` to an integer using the `int()` function. This will give us the largest integer smaller than `number`.\n2. Calculate the difference between `number` and the integer part we just calculated in step 1. This difference will be the decimal part of `number`.\n\nHere's the code for the `truncate_number()` function based on the above approach:\n```\ndef truncate_number(number: float) -> float:\n    integer = int(number)\n    decimal = number - integer\n    return decimal\n```\nLet's test the function with some examples:\n```\nprint(truncate_number(3.5))  # output: 0.5\nprint(truncate_number(2.75))  # output: 0.75\nprint(truncate_number(1.25))  # output: 0.25\n```\nEach of these outputs confirms that the function is working correctly and returning the decimal part of the input number.\n\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_analyzation_3 = \
"""The prediction is already correct. We just filter out the words after it gets the solution"""
        
        example_q4 = \
f"""1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.
2. You need to try your best to solve this problem. 
3. You only have to complete and show the code inside the entrypoint has_close_elements. Please be careful about the spacing. The code inside the entrypoint should start with 4 whitespace for indentation.
\n\n\nProblem: from typing import List\n\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n"""
        example_a4 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:

First, I'll define a variable `balance` to keep track of the current balance of the bank account. I'll set it to 0 to start.

```
balance = 0
```

Next, I'll loop through the `operations` list and for each operation, I'll update the `balance` variable accordingly by adding the operation amount to the balance.

```
for operation in operations:
    balance += operation
    if balance < 0:
        return True
```

After the loop, if the balance never goes below zero, I'll return `False`.

```
return False
```

Here's the complete code inside the entrypoint below_zero:
FINAL ANSWER:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False
"""
        example_p4 = \
""" Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a variable `balance` to keep track of the current balance of the bank account. I'll set it to 0 to start.\n```\nbalance = 0\n```\nNext, I'll loop through the `operations` list and for each operation, I'll update the `balance` variable accordingly. If the operation is a deposit, I'll add the amount to the balance. If the operation is a withdrawal, I'll subtract the amount from the balance.\n```\nfor operation in operations:\n    if operation > 0:\n        balance += operation\n    else:\n        balance -= operation\n```\nAfter the loop, I'll check if the balance is below zero. If it is, I'll return `True`, otherwise I'll return `False`.\n```\nif balance < 0:\n    return True\nelse:\n    return False\n```\nHere's the complete code:\n```\ndef below_zero(operations: List[int]) -> bool:\n    balance = 0\n    for operation in operations:\n        if operation > 0:\n            balance += operation\n        else:\n            balance -= operation\n    return balance < 0\n```\nNow, let's test the function with some examples:\n```\nprint(below_zero([1, 2, 3]))  # Should return False\nprint(below_zero([1, 2, -4, 5]))  # Should return True\n```\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_analyzation_4 = \
"""below_zero function is wrong. there are also other few inference details that is wrong. thus we correct it with minimum change while keep as much original words as possible. we also remove the things after the prediction return the complete code."""
        
        example_q5 = \
f"""1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.
2. You need to try your best to solve this problem. 
3. You only have to complete and show the code inside the entrypoint has_close_elements. Please be careful about the spacing. The code inside the entrypoint should start with 4 whitespace for indentation.
from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n"""
        example_a5 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a helper function called `is_open` that takes a single character as input and returns `True` if it is an open parenthesis, and `False` otherwise. Here's the implementation:\n```\ndef is_open(char):\n    return char == '('\n```\nNext, I'll define the `separate_paren_groups` function. Here's the implementation:\n```\ndef separate_paren_groups(paren_string):\n    # Initialize an empty list to store the separated groups\n    groups = []\n    \n    # Iterate through the input string, keeping track of the current group\n    # and checking for open and close parentheses\n    for i, char in enumerate(paren_string):\n        if is_open(char):\n            # If we encounter an open parenthesis, start a new group\n            groups.append('')\n        elif char == ')':\n            # If we encounter a close parenthesis, add the current group to the list\n            # and reset the current group to an empty string\n            groups.append('')\n            groups.append(paren_string[i-1])\n        else:\n            # If we encounter a non-parentheses character, add it to the current group\n            groups[len(groups) - 1] += char\n    \n    # Return the list of separated groups\n    return groups\n```

FINAL ANSWER:\n    # Initialize an empty list to store the separated groups\n    groups = []\n    \n    # Iterate through the input string, keeping track of the current group\n    # and checking for open and close parentheses\n    for i, char in enumerate(paren_string):\n        if is_open(char):\n            # If we encounter an open parenthesis, start a new group\n            groups.append('')\n        elif char == ')':\n            # If we encounter a close parenthesis, add the current group to the list\n            # and reset the current group to an empty string\n            groups.append('')\n            groups.append(paren_string[i-1])\n        else:\n            # If we encounter a non-parentheses character, add it to the current group\n            groups[len(groups) - 1] += char\n    \n    # Return the list of separated groups\n    return groups\n
"""
        example_p5 = \
"""Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a helper function called `is_open` that takes a single character as input and returns `True` if it is an open parenthesis, and `False` otherwise. Here's the implementation:\n```\ndef is_open(char):\n    return char == '('\n```\nNext, I'll define the `separate_paren_groups` function. Here's the implementation:\n```\ndef separate_paren_groups(paren_string):\n    # Initialize an empty list to store the separated groups\n    groups = []\n    \n    # Iterate through the input string, keeping track of the current group\n    # and checking for open and close parentheses\n    for i, char in enumerate(paren_string):\n        if is_open(char):\n            # If we encounter an open parenthesis, start a new group\n            groups.append('')\n        elif char == ')':\n            # If we encounter a close parenthesis, add the current group to the list\n            # and reset the current group to an empty string\n            groups.append('')\n            groups.append(paren_string[i-1])\n        else:\n            # If we encounter a non-parentheses character, add it to the current group\n            groups[len(groups) - 1] += char\n    \n    # Return the list of separated groups\n    return groups\n```\nNow, let's test the function with the example input:\n```\nprint(separate_paren_groups('( ) (( )) (( )( ))'))\n```\nThe output should be:\n```\n['()', '(())', '(()())']\n```\nAs you can see, the function correctly separates the input string into three groups of nested parentheses."""
        example_analyzation_5 = \
"""This previous prediction was too wrong thus it is hard to perform minimum change without rewriting the entire prediction. We therefore give un minimum change and just copy the inference steps. We also remove the things after the prediction return the complete code and place the FINAL ANSWER after the inference steps."""

        minimum_change_prompt = f"""Given the question and the Prediction, 

1. Please perform minimum change to the Prediction to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the origininal Prediction while only change where is wrong to make it correct. 
2. If the inference steps in the prediction is completely wrong which make it too hard to perfrom minimum change, as shown in example 5, then please change some details in the problem and keep the most parts as the way it is. Our primary task is minimum change which means we will try our best to keep the inference part of the prediction as the way it was. We would rather have a minimum changed prediction that is not 100% correct than have a correct prediction that is completely different with our previous prediction.
2. Directly give me the minimum changed Prediction. Do not say anything like here is the modifeid prediction or this is wrong because.
3. The prediction does not need to run test. if the prediction contains the process of running test, you should remove it.
4. put the final answer of the code at the end after the minimum changed prediction. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
5. The modified answer format would be 
MINIMUM_CHANGED_PREDICTION_HERE
FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE

6. Please noticed that the entrypoint of the function is already given, thus the FINAL_ANSWER does not have to include the code of entrypoint of the function. It just need to be the pieces of code inside the function. It should start with 4 white space since it is in the function. you will need to write before FINAL ANSWER:


Example1:{example_analyzation_1} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script.
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example1-Question: {example_q1}
Example1-Prediction: {example_p1}
Example1-Minimum Changed Prediction: {example_a1}



Example2: {example_analyzation_2}
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example2-Question: {example_q2}
Example2-Prediction: {example_p2}
Example2-Minimum Changed Prediction: {example_a2}



Example3: {example_analyzation_3} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script. 
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example3-Question: {example_q3}
Example3-Prediction: {example_p3}
Example3-Minimum Changed Prediction: {example_a3}



Example4: {example_analyzation_4} We add the FINAL ANSWER: at the end with proper spacing. We keep the proper spacing because our script will read the things after ANSWER: , put it under the entry point and run it using the script.
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example4-Question: {example_q4}
Example4-Prediction: {example_p4}
Example4-Minimum Changed Prediction: {example_a4}

Example5: {example_analyzation_5} 
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example5-Question: {example_q5}
Example5-Prediction: {example_p5}
Example5-Minimum Changed Prediction: {example_a5}




Your task:
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Question: {question}
Prediction: {previous_prediction}
Minimum Changed Prediction: 
"""

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)

        temp = {}
        temp['prompt'] = question
        temp['input'] = ''
        temp['answer'] = answer
        temp['canonical_solution'] = ground_truth
        temp['previous_prediction'] = previous_prediction
        temp['task_id'] = task_id
        temp['entry_point'] = entry_point
        temp['test'] = test
        
        input_json_list.append(temp)
    return input_json_list



def minimum_change_data_creation_CODE_vanilla_total_feb_2_tiny_change_step1(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        previous_prediction = predict_list[i]
        ground_truth = item['ground_truth']
        task_id = item['task_id']
        entry_point = item['entry_point']
        test = item['test']

        q_1 = \
f"""1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n3. You only have to complete and show the code inside the entrypoint has_close_elements. Please be careful about the spacing. The code inside the entrypoint should start with 4 whitespace for indentation.\n\nProblem: from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"""
        p_1 = \
f"""Sure, I\'d be happy to help! Here\'s my solution to the problem:\n```\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"\n    Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\nHere\'s how the code works:\n\n1. We iterate over the list of numbers using two nested for loops.\n2. We check the distance between each pair of numbers using the `abs()` function to get the absolute difference.\n3. If the distance is less than the given threshold, we return `True` immediately.\n4. If no pair of numbers has a distance less than the threshold, we return `False`.\n\nHere\'s an example of how the code would work for the given test cases:\n\n* `has_close_elements([1.0, 2.0, 3.0], 0.5)`: No pairs of numbers have a distance less than 0.5, so we return `False`.\n* `has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)`: The pair `(2.8, 3.0)` has a distance less than 0.3, so we return `True`.\n\nI hope this helps! Let me know if you have any questions or need further clarification."""
        a_1 = \
f"""INFERENCE:
Sure, I\'d be happy to help! Here\'s my solution to the problem:\n```\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"\n    Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```

After removing the entry point, we got the FINAL ANSWER:\n    \"\"\"\n    Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n"""

        q_2 = \
f"""1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n3. You only have to complete and show the code inside the entrypoint separate_paren_groups. Please be careful about the spacing. The code inside the entrypoint should start with 4 whitespace for indentation.\n\nProblem: from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups(\'( ) (( )) (( )( ))\')\n    [\'()\', \'(())\', \'(()())\']\n    \"\"\"\n"""
        p_2 = \
f"""Sure, I\'d be happy to help! Here\'s the solution to the problem:\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\"\n    Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those groups into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    \"\"\"\n    groups = []\n    current_group = []\n    for char in paren_string:\n        if char == \'(\':\n            current_group.append(\'(\')\n        elif char == \')\':\n            current_group.append(\')\')\n            if not current_group:\n                groups.append(\'\')\n                current_group = []\n            else:\n                groups.append(\'\'.join(current_group))\n                current_group = []\n        else:\n            current_group.append(char)\n    if current_group:\n        groups.append(\'\'.join(current_group))\n    return groups\n\nHere\'s an explanation of how the code works:\n\n1. We define a function `separate_paren_groups` that takes a string `paren_string` as input.\n2. We initialize two lists: `groups` to store the separated groups, and `current_group` to store the current group of parentheses.\n3. We loop through each character in the input string.\n4. If the current character is an open parenthesis `(\'), we append it to the current group `current_group`.\n5. If the current character is a close parenthesis `)`, we check if the current group is empty. If it is, we append an empty string to the `groups` list and reset the `current_group` to an empty list. If it\'s not empty, we append the current group to the `groups` list and reset the `current_group` to an empty list.\n6. If the current character is not a parenthesis, we append it to the current group `current_group`.\n7. After looping through all the characters, we check if the current group is not empty, and if it is, we append it to the `groups` list.\n8. Finally, we return the `groups` list.\n\nHere\'s an example of how the function works:\n\n>>> separate_paren_groups(\'( ) (( )) (( )( ))\')\n[\'()\', \'(())\', \'(()())\']\n\nI hope this helps! Let me know if you have any questions or need further clarification."""
        a_2 = \
f"""INFERENCE:
Sure, I\'d be happy to help! Here\'s the solution to the problem:\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\"\n    Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those groups into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    \"\"\"\n    groups = []\n    current_group = []\n    for char in paren_string:\n        if char == \'(\':\n            current_group.append(\'(\')\n        elif char == \')\':\n            current_group.append(\')\')\n            if not current_group:\n                groups.append(\'\')\n                current_group = []\n            else:\n                groups.append(\'\'.join(current_group))\n                current_group = []\n        else:\n            current_group.append(char)\n    if current_group:\n        groups.append(\'\'.join(current_group))\n    return groups

After removing the entry point, we got the FINAL ANSWER:\n    \"\"\"\n    Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those groups into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    \"\"\"\n    groups = []\n    current_group = []\n    for char in paren_string:\n        if char == \'(\':\n            current_group.append(\'(\')\n        elif char == \')\':\n            current_group.append(\')\')\n            if not current_group:\n                groups.append(\'\')\n                current_group = []\n            else:\n                groups.append(\'\'.join(current_group))\n                current_group = []\n        else:\n            current_group.append(char)\n    if current_group:\n        groups.append(\'\'.join(current_group))\n    return groups"""
        

        bad_a_1 = \
f"""INFERENCE:
\n```\n    \"\"\"\n    Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```

After removing the entry point, we got the FINAL ANSWER:\n    \"\"\"\n    Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n"""
        

        bad_a_2 = \
f"""INFERENCE:
After removing the entry point, we got the FINAL ANSWER:\n    \"\"\"\n    Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those groups into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    \"\"\"\n    groups = []\n    current_group = []\n    for char in paren_string:\n        if char == \'(\':\n            current_group.append(\'(\')\n        elif char == \')\':\n            current_group.append(\')\')\n            if not current_group:\n                groups.append(\'\')\n                current_group = []\n            else:\n                groups.append(\'\'.join(current_group))\n                current_group = []\n        else:\n            current_group.append(char)\n    if current_group:\n        groups.append(\'\'.join(current_group))\n    return groups"""

        minimum_change_prompt = f"""Given the question and the Prediction, 

1. Please directly remove the content after the final code solution and attach this at the end "We remove the content after getting the code, then place the code after FINAL ANSWER:". Then attach the code at the next line. The code cannot contains the entry point.
3. Directly give me the minimum changed Prediction. Do not say anything like here is the modifeid prediction or this is wrong because.
4. The prediction does not need to run test. if the prediction contains the process of running test, you should remove it.
5. put the final answer of the code at the end after the minimum changed inference process. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
6. Please noticed that the entrypoint of the function is already given, thus the FINAL_ANSWER does not have to include the code of entrypoint of the function. It just need to be the pieces of code inside the function. It should start with 4 white space since it is in the function. you will need to write before FINAL ANSWER:

If you cannot make it correct without rewrite it, then just improve some details.
        

GOOD EXAMPLES:
Example-1: we just remove the content after the code and attach the FINAL ANSWER after the end. Look! We do not change words in the previous prediction. Look at the INFERENCE example. It includes everything before the final code!!!!!!!!!!!!!
Example-1-Question: {q_1}
Example-1-Previous-Prediction: {p_1}
GOOD-Example-1-Minimum-Changed-Prediction: {a_1}
BAD-Example-1-Minimum-Changed-Prediction: {bad_a_1}


Example-2: we just remove the content after the code and attach the FINAL ANSWER after the end. Look! We do not change words in the previous prediction. Look at the INFERENCE example. It includes everything before the final code!!!!!!!!!!!!!
Example-2-Question: {q_2}
Example-2-Previous-Prediction: {p_2}
GOOD-Example-2-Minimum-Changed-Prediction: {a_2}
BAD-Example-2-Minimum-Changed-Prediction: {bad_a_2}



Your task:
please modify the prediction while keeping most of the original words. you only need to improve it a little bit. you do not have to completely rewrite it.
Question: {question}
Prediction: {previous_prediction}
Minimum Changed Prediction: 
INFERENCE: PUT_THE_INFERENCE_PROCESS_HERE_AS_SHOWN_IN_EXAMPLES

After removing the entry point, we got the FINAL ANSWER: PUT_THE_CODE_HERE_AS_SHOWN_IN_EXAMPLES


Please read carefully about BAD-Example-Minimum-Changed-Prediction and GOOD-Example-2-Minimum-Changed-Prediction and think about why it is good/bad. please follow my instruction.
""" 
        
        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        if 'FINAL ANSWER' not in answer:
            answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        if 'FINAL ANSWER' not in answer:
            answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        
        temp = {}
        temp['prompt'] = question
        temp['input'] = ''
        temp['answer'] = answer
        temp['canonical_solution'] = ground_truth
        temp['previous_prediction'] = previous_prediction
        temp['task_id'] = task_id
        temp['entry_point'] = entry_point
        temp['test'] = test
        
        input_json_list.append(temp)
    return input_json_list




def minimum_change_data_creation_CODE_vanilla_total_feb_2_tiny_change_step2(data_list, predict_list, CODE_modified_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        previous_prediction = predict_list[i]
        ground_truth = item['ground_truth']
        task_id = item['task_id']
        entry_point = item['entry_point']
        test = item['test']

        q_1 = \
f"""1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n3. You only have to complete and show the code inside the entrypoint has_close_elements. Please be careful about the spacing. The code inside the entrypoint should start with 4 whitespace for indentation.\n\nProblem: from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"""
        p_1 = \
f"""Sure, I\'d be happy to help! Here\'s my solution to the problem:\n```\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"\n    Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```\nHere\'s how the code works:\n\n1. We iterate over the list of numbers using two nested for loops.\n2. We check the distance between each pair of numbers using the `abs()` function to get the absolute difference.\n3. If the distance is less than the given threshold, we return `True` immediately.\n4. If no pair of numbers has a distance less than the threshold, we return `False`.\n\nHere\'s an example of how the code would work for the given test cases:\n\n* `has_close_elements([1.0, 2.0, 3.0], 0.5)`: No pairs of numbers have a distance less than 0.5, so we return `False`.\n* `has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)`: The pair `(2.8, 3.0)` has a distance less than 0.3, so we return `True`.\n\nI hope this helps! Let me know if you have any questions or need further clarification."""
        a_1 = \
f"""INFERENCE:
Sure, I\'d be happy to help! Here\'s my solution to the problem:\n```\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"\n    Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```

After removing the entry point, we got the FINAL ANSWER:\n    \"\"\"\n    Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n"""

        q_2 = \
f"""1. Please solve the following problem using only the information and resources provided in the question. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. You need to try your best to solve this problem. \n3. You only have to complete and show the code inside the entrypoint separate_paren_groups. Please be careful about the spacing. The code inside the entrypoint should start with 4 whitespace for indentation.\n\nProblem: from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups(\'( ) (( )) (( )( ))\')\n    [\'()\', \'(())\', \'(()())\']\n    \"\"\"\n"""
        p_2 = \
f"""Sure, I\'d be happy to help! Here\'s the solution to the problem:\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\"\n    Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those groups into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    \"\"\"\n    groups = []\n    current_group = []\n    for char in paren_string:\n        if char == \'(\':\n            current_group.append(\'(\')\n        elif char == \')\':\n            current_group.append(\')\')\n            if not current_group:\n                groups.append(\'\')\n                current_group = []\n            else:\n                groups.append(\'\'.join(current_group))\n                current_group = []\n        else:\n            current_group.append(char)\n    if current_group:\n        groups.append(\'\'.join(current_group))\n    return groups\n\nHere\'s an explanation of how the code works:\n\n1. We define a function `separate_paren_groups` that takes a string `paren_string` as input.\n2. We initialize two lists: `groups` to store the separated groups, and `current_group` to store the current group of parentheses.\n3. We loop through each character in the input string.\n4. If the current character is an open parenthesis `(\'), we append it to the current group `current_group`.\n5. If the current character is a close parenthesis `)`, we check if the current group is empty. If it is, we append an empty string to the `groups` list and reset the `current_group` to an empty list. If it\'s not empty, we append the current group to the `groups` list and reset the `current_group` to an empty list.\n6. If the current character is not a parenthesis, we append it to the current group `current_group`.\n7. After looping through all the characters, we check if the current group is not empty, and if it is, we append it to the `groups` list.\n8. Finally, we return the `groups` list.\n\nHere\'s an example of how the function works:\n\n>>> separate_paren_groups(\'( ) (( )) (( )( ))\')\n[\'()\', \'(())\', \'(()())\']\n\nI hope this helps! Let me know if you have any questions or need further clarification."""
        a_2 = \
f"""INFERENCE:
Sure, I\'d be happy to help! Here\'s the solution to the problem:\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\"\n    Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those groups into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    \"\"\"\n    groups = []\n    current_group = []\n    for char in paren_string:\n        if char == \'(\':\n            current_group.append(\'(\')\n        elif char == \')\':\n            current_group.append(\')\')\n            if not current_group:\n                groups.append(\'\')\n                current_group = []\n            else:\n                groups.append(\'\'.join(current_group))\n                current_group = []\n        else:\n            current_group.append(char)\n    if current_group:\n        groups.append(\'\'.join(current_group))\n    return groups

After removing the entry point, we got the FINAL ANSWER:\n    \"\"\"\n    Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those groups into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    \"\"\"\n    groups = []\n    current_group = []\n    for char in paren_string:\n        if char == \'(\':\n            current_group.append(\'(\')\n        elif char == \')\':\n            current_group.append(\')\')\n            if not current_group:\n                groups.append(\'\')\n                current_group = []\n            else:\n                groups.append(\'\'.join(current_group))\n                current_group = []\n        else:\n            current_group.append(char)\n    if current_group:\n        groups.append(\'\'.join(current_group))\n    return groups"""
        

        bad_a_1 = \
f"""INFERENCE:
\n```\n    \"\"\"\n    Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```

After removing the entry point, we got the FINAL ANSWER:\n    \"\"\"\n    Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n"""
        

        bad_a_2 = \
f"""INFERENCE:
After removing the entry point, we got the FINAL ANSWER:\n    \"\"\"\n    Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those groups into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    \"\"\"\n    groups = []\n    current_group = []\n    for char in paren_string:\n        if char == \'(\':\n            current_group.append(\'(\')\n        elif char == \')\':\n            current_group.append(\')\')\n            if not current_group:\n                groups.append(\'\')\n                current_group = []\n            else:\n                groups.append(\'\'.join(current_group))\n                current_group = []\n        else:\n            current_group.append(char)\n    if current_group:\n        groups.append(\'\'.join(current_group))\n    return groups"""

        minimum_change_prompt = f"""Given the question and the Prediction, 

1. Please directly remove the content after the final code solution and attach this at the end "We remove the content after getting the code, then place the code after FINAL ANSWER:". Then attach the code at the next line. The code cannot contains the entry point.
3. Directly give me the minimum changed Prediction. Do not say anything like here is the modifeid prediction or this is wrong because.
4. The prediction does not need to run test. if the prediction contains the process of running test, you should remove it.
5. put the final answer of the code at the end after the minimum changed inference process. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
6. Please noticed that the entrypoint of the function is already given, thus the FINAL_ANSWER does not have to include the code of entrypoint of the function. It just need to be the pieces of code inside the function. It should start with 4 white space since it is in the function. you will need to write before FINAL ANSWER:

If you cannot make it correct without rewrite it, then just improve some details.
        

GOOD EXAMPLES:
Example-1: we just remove the content after the code and attach the FINAL ANSWER after the end. Look! We do not change words in the previous prediction. Look at the INFERENCE example. It includes everything before the final code!!!!!!!!!!!!!
Example-1-Question: {q_1}
Example-1-Previous-Prediction: {p_1}
GOOD-Example-1-Minimum-Changed-Prediction: {a_1}
BAD-Example-1-Minimum-Changed-Prediction: {bad_a_1}


Example-2: we just remove the content after the code and attach the FINAL ANSWER after the end. Look! We do not change words in the previous prediction. Look at the INFERENCE example. It includes everything before the final code!!!!!!!!!!!!!
Example-2-Question: {q_2}
Example-2-Previous-Prediction: {p_2}
GOOD-Example-2-Minimum-Changed-Prediction: {a_2}
BAD-Example-2-Minimum-Changed-Prediction: {bad_a_2}



Your task:
please modify the prediction while keeping most of the original words. you only need to improve it a little bit. you do not have to completely rewrite it.
Question: {question}
Prediction: {previous_prediction}
Minimum Changed Prediction: 
INFERENCE: PUT_THE_INFERENCE_PROCESS_HERE_AS_SHOWN_IN_EXAMPLES

After removing the entry point, we got the FINAL ANSWER: PUT_THE_CODE_HERE_AS_SHOWN_IN_EXAMPLES


Please read carefully about BAD-Example-Minimum-Changed-Prediction and GOOD-Example-2-Minimum-Changed-Prediction and think about why it is good/bad. please follow my instruction.
""" 
        
        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        if 'FINAL ANSWER' not in answer:
            answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        if 'FINAL ANSWER' not in answer:
            answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        
        temp = {}
        temp['prompt'] = question
        temp['input'] = ''
        temp['answer'] = answer
        temp['canonical_solution'] = ground_truth
        temp['previous_prediction'] = previous_prediction
        temp['task_id'] = task_id
        temp['entry_point'] = entry_point
        temp['test'] = test
        
        input_json_list.append(temp)
    return input_json_list



def minimum_change_data_creation_APPS(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['answer']
        question = item['question']
        ground_truth = item['solutions']

#         minimum_change_prompt = f"""

# QUESTION: {question}

# PREVIOUS PREDICTION: {previous_prediction}

# Given the QUESTION, the PREVIOUS PREDICTION, and the CODE
# 1. Please perform minimum change to the PREVIOUS PEDICTION to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the origininal PREVIOUS PREDICTION while only change where is wrong to make it correct. 
# 2. Directly give me the minimum changed PREVIOUS PREDICTION. Do not say anything like here is the modifeid prediction or this is wrong because. (very important)
# 3. The prediction needs to output the code in step by step, but it does not need to run test. if the prediction contains the process of running test, you should delete it.
# 4. put the final answer of the code at the end after the minimum changed prediction. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
# 5. The modified answer format would be 
# MINIMUM_CHANGED_PREDICTION_HERE
# FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE
# 6. The PREVIOUS PREDICTION has to be minimum changed in a way such it logically support the final answer. For example, the PREVIOUS PREDICTION might not solve the problem. it might just give you some hint. you have to minimum change it in such a way that it becomes logical steps guiding you to generate the FINAL ANSWER CODE """
        

#         minimum_change_prompt = \
# f"""Given the QUESTION and the GROUNDTRUTH, please modify the PREVIOUS PREDICTION. When modifying, please use minimum change, which means you need to modify as little as possible to make it correct. This is very important.

# QUESTION: {question}

# PREVIOUS PREDICTION: {previous_prediction}

# Given the QUESTION, the PREVIOUS PREDICTION, and the GROUNDTRUTH,
# 1. Please perform minimum change to the PREVIOUS PEDICTION to make it correct. When i say minimum change, i mean you should try to keep the origininal PREVIOUS PREDICTION while only change where is wrong to make it correct. 
# 2. Directly give me the minimum changed PREVIOUS PREDICTION. Do not say anything like here is the modifeid prediction or this is wrong because. (very important)"""
        



# GROUNDTRUTH: {ground_truth}

# 3. GROUNDTRUTH only tell you the expected input format. Please do not directly copy GROUNDTRUTH. You may only correcting the PREVIOUS PREDICTION through minimum change instead of copying GROUNDTRUTH.
# 4. put the final answer of the code at the end after the minimum changed prediction. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code. 
# 5. The PREVIOUS PREDICTION has to be minimum changed in a way such it logically support the final answer. For example, the PREVIOUS PREDICTION might not solve the problem. it might just give you some hint. you have to minimum change it in such a way that it becomes logical steps guiding you to generate the FINAL ANSWER CODE
# 6. The modified answer format would be 
# MINIMUM_CHANGED_PREDICTION_HERE
# FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE

# The format is:
# MINIMUM_CHANGED_PREDICTION_HERE
# FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE"""
        




        Example_question = \
"""'1. When solving the code challenge, you cannot import the library except the library in the AVAILABLE LIBRARY. Do not use any external libraries or additional functions not mentioned or implied in the problem statement.\n2. Please try your best to solve the python code challenge step by step. \n\n\nAVAILABLE LIBRARY:\nimport sysimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n\nPython Code Challenge: You are given an array $a$ of length $n$ consisting of zeros. You perform $n$ actions with this array: during the $i$-th action, the following sequence of operations appears:  Choose the maximum by length subarray (continuous subsegment) consisting only of zeros, among all such segments choose the leftmost one;  Let this segment be $[l; r]$. If $r-l+1$ is odd (not divisible by $2$) then assign (set) $a[\\frac{l+r}{2}] := i$ (where $i$ is the number of the current action), otherwise (if $r-l+1$ is even) assign (set) $a[\\frac{l+r-1}{2}] := i$. \n\nConsider the array $a$ of length $5$ (initially $a=[0, 0, 0, 0, 0]$). Then it changes as follows:  Firstly, we choose the segment $[1; 5]$ and assign $a[3] := 1$, so $a$ becomes $[0, 0, 1, 0, 0]$;  then we choose the segment $[1; 2]$ and assign $a[1] := 2$, so $a$ becomes $[2, 0, 1, 0, 0]$;  then we choose the segment $[4; 5]$ and assign $a[4] := 3$, so $a$ becomes $[2, 0, 1, 3, 0]$;  then we choose the segment $[2; 2]$ and assign $a[2] := 4$, so $a$ becomes $[2, 4, 1, 3, 0]$;  and at last we choose the segment $[5; 5]$ and assign $a[5] := 5$, so $a$ becomes $[2, 4, 1, 3, 5]$. \n\nYour task is to find the array $a$ of length $n$ after performing all $n$ actions. Note that the answer exists and unique.\n\nYou have to answer $t$ independent test cases.\n\n\n-----Input-----\n\nThe first line of the input contains one integer $t$ ($1 \\le t \\le 10^4$) — the number of test cases. Then $t$ test cases follow.\n\nThe only line of the test case contains one integer $n$ ($1 \\le n \\le 2 \\cdot 10^5$) — the length of $a$.\n\nIt is guaranteed that the sum of $n$ over all test cases does not exceed $2 \\cdot 10^5$ ($\\sum n \\le 2 \\cdot 10^5$).\n\n\n-----Output-----\n\nFor each test case, print the answer — the array $a$ of length $n$ after performing $n$ actions described in the problem statement. Note that the answer exists and unique.\n\n\n-----Example-----\nInput\n6\n1\n2\n3\n4\n5\n6\n\nOutput\n1 \n1 2 \n2 1 3 \n3 1 2 4 \n2 4 1 3 5 \n3 4 1 5 2 6'
"""
        Example_mc = """Sure, I'd be happy to help! Here's my solution to the Python code challenge:

1. First, we'll define a function `action` that takes an integer `i` and an array `a` of length `n` as input, and performs the `i`-th action on `a`. The function will return the updated array `a` after the action is performed.
```python
def action(i, a):
    # Find the longest subarray of zeros and choose the leftmost one
    start, end, max_len = 0, 0, 0
    temp_start, temp_len = 0, 0
    for j in range(len(a)):
        if a[j] == 0:
            temp_len += 1
            if temp_len > max_len:
                max_len = temp_len
                start, end = temp_start, j
        else:
            temp_len = 0
            temp_start = j + 1

    mid = (start + end) // 2
    if (end - start) % 2 == 0:
        mid -= 1
    a[mid] = i
    return a
```
2. Next, we'll define a function `perform_actions` that takes an integer `n` as input and performs the actions described in the problem statement on an array `a` of length `n`. The function will return the updated array `a` after all actions are performed.
```python
def perform_actions(n):
    a = [0] * n
    for i in range(1, n + 1):
        action(i, a)
    return a
```

FINAL ANSWER:
def action(i, a):
    # Find the longest subarray of zeros and choose the leftmost one
    start, end, max_len = 0, 0, 0
    temp_start, temp_len = 0, 0
    for j in range(len(a)):
        if a[j] == 0:
            temp_len += 1
            if temp_len > max_len:
                max_len = temp_len
                start, end = temp_start, j
        else:
            temp_len = 0
            temp_start = j + 1

    mid = (start + end) // 2
    if (end - start) % 2 == 0:
        mid -= 1
    a[mid] = i
    return a

def perform_actions(n):
    a = [0] * n
    for i in range(1, n + 1):
        action(i, a)
    return a
"""
        Example_previous_prediction = \
"""Sure, I'd be happy to help! Here's my solution to the Python code challenge:\n\n1. First, we'll define a function `action` that takes an integer `i` and an array `a` of length `n` as input, and performs the `i`-th action on `a`. The function will return the updated array `a` after the action is performed.\n```python\ndef action(i, a):\n    # Choose the maximum by length subarray (continuous subsegment) consisting only of zeros, among all such segments choose the leftmost one\n    max_subarray = max(accumulate(a[::-1]), key=len)\n    # If the length of the max subarray is odd, assign the value of i to the middle index of the subarray\n    if len(max_subarray) % 2 == 1:\n        a[int(len(max_subarray) / 2)] = i\n    # Otherwise (if the length of the max subarray is even), assign the value of i to the leftmost index of the subarray\n    else:\n        a[int(len(max_subarray) / 2) - 1] = i\n    return a\n```\n2. Next, we'll define a function `perform_actions` that takes an integer `n` as input and performs the actions described in the problem statement on an array `a` of length `n`. The function will return the updated array `a` after all actions are performed.\n```python\ndef perform_actions(n):\n    a = [0] * n\n    for i in range(n):\n        action(i, a)\n    return a\n```\n3. Now, we'll test the function `perform_actions` on some sample inputs. We'll use the sample inputs from the problem statement to verify that our solution works correctly.\n```python\n# Test case 1\nprint(perform_actions(1))\n# Output: [1]\n\n# Test case 2\nprint(perform_actions(2))\n# Output: [2, 1]\n\n# Test case 3\nprint(perform_actions(3))\n# Output: [2, 1, 3]\n\n# Test case 4\nprint(perform_actions(4))\n# Output: [2, 4, 1, 3]\n\n# Test case 5\nprint(perform_actions(5))\n# Output: [2, 4, 1, 3, 5]\n```\n4. Finally, we'll write a function `solve` that takes an integer `t` as input and returns the answer to the $t$ test cases.\n```python\ndef solve(t):\n    for _ in range(t):\n        a = [0] * 5\n        perform_actions(5)\n    return a\n```\n5. Now, we can run the function `solve` on the input `t` to get the answer to the $t$ test cases.\n```python\n# Test case 1\nprint(solve(1))\n# Output: [1]\n\n# Test case 2\nprint(solve(2))\n# Output: [2, 1]\n\n# Test case 3\nprint(solve(3))\n# Output: [2, 1, 3]\n\n# Test case 4\nprint(solve(4))\n# Output: [2, 4, 1, 3]\n\n# Test case 5\nprint(solve(5))\n# Output: [2, 4, 1, 3, 5]\n```\nThat's it! Our solution should pass all $t$ test cases."""

        minimum_change_prompt = \
f"""Given the question and the PREVIOUS PREDICTION, Please perform minimum change to the PREVIOUS PEDICTION to make it correct.
We have minimum change example below: 
Example_question: {Example_question}
Previous Prediction: {Example_previous_prediction}
Minimum Changed Prediction: {Example_mc}



Right now we have 
QUESTION: {question}
PREVIOUS PREDICTION: {previous_prediction}

1. Please perform minimum change to the PREVIOUS PEDICTION to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the origininal PREVIOUS PREDICTION while only change where is wrong to make it correct. 
2. Directly give me the minimum changed PREVIOUS PREDICTION. Do not say anything like "here is the modifeid prediction" or "this is wrong because".
3. The prediction needs to output the code in step by step, but it does not need to run test. if the prediction contains the process of running test, you should delete it.
4. put the final answer of the code at the end after the minimum changed prediction. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
5. The modified answer format would be 
MINIMUM_CHANGED_PREDICTION_HERE
FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE
6. The PREVIOUS PREDICTION has to be minimum changed in a way such it logically support the final answer. For example, the PREVIOUS PREDICTION might not solve the problem. it might just give you some hint. you have to minimum change it in such a way that it becomes logical steps guiding you to generate the FINAL ANSWER CODE """

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction
    return data_list


# prompt = f"""We have some examples, a question and a solution for the qurstion. Please perform minimum change to modify the prediction to make it correct. You need to modify as little details as possible to correct it. Remember the prediction has to pass a code test, so the minimum_changed prediction has contains no other words such as 'sure i can help you with this' or 'here is your answer', otherwise the code test will fail.

# examples and questions: {question}
# Groundtruth: {ground_truth}




# When asking GPT3 the same question, this is its answer
# Previous prediction: "{previous_prediction}"

# Please perform minimum change to the prediction. Your will directly output the minimum changed version of the prediction, which means your output will be runable on code test. You do not need to add ``` ```.

# The entrypoint is already provided. Please not write the definition of the entrypoint {entry_point}. Thus after performing minimum change, you need to show the minimum changed code inside the entrypoint {entry_point}

# DIRECTLY_OUTPUT_THE_MINIMUM_CHANGED_CODE_PREDICTION
# """





def add_indentation(code, spaces=4):
    indented_code = code
    if code[0] != ' ':
        # Split the code into lines
        lines = code.split('\n')
        # Add the specified number of spaces to the start of each line
        indented_lines = [(' ' * spaces) + line for line in lines]
        # Join the lines back together
        indented_code = '\n'.join(indented_lines)
    return indented_code

def Code_Extraction(question_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    extracted_code_list = []
    for i in range(len(question_list)):
        question = question_list[i]
        previous_prediction = predict_list[i]

        code_extraction_prompt = \
f"""Given the question and the prediction, please directly extract the code in prediction. please do not extract the entrypoint because it is already provided in the question.
        
Question: {question}
Prediction: {previous_prediction}

1. When you extract, please directly extract the code without saying anything like sure I can help you  with or here is the extracted code. 
2. You do not need to call python, or saying anthing like ```. just directly extract the code without the provided entrypoint in question. For example, if the extracted is ```python\nbalance = 0\nfor operation in operations:\n    balance += operation\n    if balance < 0:\n        return True\nreturn False\n```, you should delete ``` and python so the extracted code is \nbalance = 0\nfor operation in operations:\n    balance += operation\n    if balance < 0:\n        return True\nreturn False\n
3. Please please do not solve the problem by yourself. your job is just extract the original code from prediction. I will run script to check if the prediction is correct by myself so please only extract code without solve it yourself.
4. Please remove the entry point from the extracted code. We will place the extracted code under the entry point to check if it is runnable. 



EXAMPLE1: 
EXAMPLE1-PREDICTION: Sure, I'd be happy to help! Here's my solution to the problem:\n\nFirst, I'll define a helper function called `closer_than` that takes two numbers and returns `True` if they are closer to each other than the given threshold, and `False` otherwise:\n```\ndef closer_than(a, b, threshold):\n    return abs(a - b) < threshold\n```\nNow, I can write the `has_close_elements` function using this helper function:\n```python\ndef has_close_elements(numbers, threshold):\n    for i in range(len(numbers) - 1):\n        for j in range(i + 1, len(numbers)):\n            if closer_than(numbers[i], numbers[j], threshold):\n                return True\n    return False\n```\nHere's how the function works:\n\n1. I iterate over the list of numbers twice, once from the start to the middle, and once from the middle to the end.\n2. For each pair of adjacent numbers, I check if they are closer to each other than the given threshold using the `closer_than` function. If they are, I return `True` immediately.\n3. If no pair of adjacent numbers is closer than the threshold, I return `False` at the end of the second iteration.\n\nHere's an example of how the function works:\n```\n>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\nTrue\n```\nIn this example, the function checks the pair `(1.0, 2.8)`, `(2.8, 3.0)`, `(3.0, 4.0)`, and `(4.0, 5.0)`, and finds that the pair `(2.8, 3.0)` is closer to each other than the threshold of 0.3. Therefore, it returns `True`.\n\nI hope this helps! Let me know if you have any questions or need further assistance.

please notice that entry point is removed and the space '    ' is kept. ```python is removed.
EXAMPLE1-EXTRACTED CODE:
\n    for i in range(len(numbers) - 1):
        for j in range(i + 1, len(numbers)):
            if closer_than(numbers[i], numbers[j], threshold):
                return True
    return False

    
EXAMPLE2: 
EXAMPLE2-PREDICTION: Sure, I\'d be happy to help! Here\'s my solution to the problem:\n\n```def rounded_avg(n, m):\n    if n > m:\n        return -1\n    total = 0\n    for i in range(n, m+1):\n        total += i\n    avg = total / (m - n + 1)\n    avg = round(avg)\n    return format(avg, \'b\')```\n\nHere\'s how the code works:\n\n1. First, we check if n is greater than m. If it is, we return -1, since there is no valid range of integers to average.\n2. Next, we initialize a running total of the integers from n through m. We use a for loop to iterate over the range of integers, and we add each integer to the total.\n3. After we\'ve added all the integers, we compute the average by dividing the total by the number of integers in the range (m - n + 1).\n4. Finally, we round the average to the nearest integer and return it as a binary string.\n\nHere are some examples of how the function behaves:\n\n* rounded_avg(1, 5) => "0b11" (since the average of 1 through 5 is 3, rounded to the nearest integer)\n* rounded_avg(7, 5) => -1 (since 7 is greater than 5)\n* rounded_avg(10, 20) => "0b1111" (since the average of 10 through 20 is 15, rounded to the nearest integer)\n* rounded_avg(20, 33) => "0b11010" (since the average of 20 through 33 is 26.5, rounded to the nearest integer)\n\nI hope this helps! Let me know if you have any questions or need further clarification.

please notice that entry point is removed and the space '    ' is kept. ``` is removed.
EXAMPLE2-EXTRACTED CODE:
\n    if n > m:
        return -1
    total = 0
    for i in range(n, m+1):
        total += i
    avg = total / (m - n + 1)
    avg = round(avg)
    return format(avg, 'b')





EXAMPLE3: 
EXAMPLE3-PREDICTION: Sure, I\'d be happy to help! Here\'s the solution to the problem:\n\nFirst, let\'s sort the input array using the `sorted` function:\n\narr = [2, 1, 1, 4, 5, 8, 2, 3]\nsorted_arr = sorted(arr)\n\nThis will give us the sorted array:\n\nsorted_arr = [1, 1, 2, 2, 3, 4, 5, 8]\n\nNext, let\'s reverse the sorted array using the `reversed` function:\n\nreversed_arr = list(reversed(sorted_arr))\n\nThis will give us the reversed array:\n\nreversed_arr = [8, 5, 4, 3, 2, 2, 1, 1]\n\nNow, let\'s replace each digit in the reversed array with its corresponding name from the list "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine":\n\nnames = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]\nreversed_arr_with_names = [names[int(x)] for x in reversed_arr]\n\nThis will give us the final answer:\n\nreversed_arr_with_names = ["Eight", "Five", "Four", "Three", "Two", "Two", "One", "One"]\n\nHere\'s the complete code:\n\ndef by_length(arr):\n    sorted_arr = sorted(arr)\n    reversed_arr = list(reversed(sorted_arr))\n    names = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]\n    reversed_arr_with_names = [names[int(x)] for x in reversed_arr]\n    return reversed_arr_with_names\n\nI hope this helps! Let me know if you have any questions or need further assistance.

please notice that entry point is removed and the space '    ' is kept. 
EXAMPLE3-EXTRACTED CODE:
\n    sorted_arr = sorted(arr)
    reversed_arr = list(reversed(sorted_arr))
    names = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    reversed_arr_with_names = [names[int(x)-1] for x in reversed_arr]
    return reversed_arr_with_names


"""
        

        answer = create_gpt_completion(code_extraction_prompt, MODEL_ENGINE, client)

        # Add additional indentation
        indented_code = add_indentation(answer, 4)

        extracted_code_list.append(indented_code)
    return extracted_code_list



def CODE_eval_given_list(question_list, test_data_list, predict_list, output_folder_name, unique_file_name_label = ''):
    extracted_code_list = Code_Extraction(question_list, predict_list)
    extracted_code_list_path = f"{HOME_DIRECTORY}/code_eval/extracted_code_list.json"
    with open(extracted_code_list_path, 'w') as json_file:
        json.dump(extracted_code_list, json_file, indent=4)
    with open(extracted_code_list_path, 'r') as file:
        extracted_code_list = json.load(file)

    predict_list = extracted_code_list

    for i in range(len(test_data_list)):
        test_data_list[i]['ground_truth'] = test_data_list[i]['canonical_solution']

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)

    path = f"{HOME_DIRECTORY}/code_eval/code_extract_code.json"
    with open(path, 'w') as file:
        for item in test_data_list: #problem_list:
            json_line = json.dumps(item)
            file.write(json_line + '\n')

    from evaluation.eval import calc_accuracy_CODE
    file_name = 'CODE_Evaluator'
    accuracy, cover_ratio = calc_accuracy_CODE(test_data_list, predict_list, output_folder_name, file_name, train_method = unique_file_name_label)
    return accuracy, extracted_code_list

def feedback_net_training_data_creation_ANLI_gpt4_groundtruth(data_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i in range(len(data_list)):
        instruction = data_list[i]['original_instruction']
        output = answer_list[i]
        prompt = f"""
We have an instruction and a groundtruth api-request.

INSTRUCTION: "{instruction}"
GROUNDTRUTH {output}



Given the instruction and the groundtruth {output}, please provide the inference process to get the groundtruth {output}. We need to know the inference which can lead to the conclusion of the groundtruth api-request given the instruction. Please directly give me the inference process without say things like 'sure, i can help you with it'


You need to Answer in proper format
Format:
Inference process: INFERENCE_PROCESS_HERE
"""
        
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        answer = f"""Analyzation: {answer}
API-Request: {output}"""
        temp = {}
        temp['instruction'] = instruction
        temp['input'] = ''
        temp['groundtruth'] = output
        temp['output'] = answer
        input_json_list.append(temp)
    return input_json_list



def minimum_change_data_creation_MATH(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['pred']
        question = item['question']
        ANSWER = item['numerical_final_answer']
        # ground_truth = item['solutions']

#         minimum_change_prompt = f"""Given the question and the PREVIOUS PREDICTION, 

# QUESTION: {question}
# PREVIOUS PREDICTION: {previous_prediction}"""
        


        minimum_change_prompt = \
f"""We have a question, grouondtruth and the prediction. 
1. Please perform minimum change on the prediction to make it correct. 
2. Please also remove the hallucination, add groundtruth as answer when the inference steps from the prediction is empty. \
3. Please put the final answer at the end in $\\boxed{ANSWER}$. 
For example:
MINIMUM_CHANGED_PREDICTION_HERE. FINAL ANSWER: $\\boxed{181}$
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

example1:
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
question: Ming ordered a 9 inches pizza, he ate half of it. How many inches left?
Groundtruth: 9/2 = 4.5 inches pizza left on the table.
Prediction: \n\nA) 2 inches \nB) 3 inches \nC) 5 inches \nD) 7 inches \n\nAnswer: B) 3 inches \n\nExplanation:\n Ming order 9 inches pizza and ate half of it, so 9/3 = 3 inches left on the table.
Analyzation: As you can see, GOT3 contains hallucinations where it creates multiple choice that does not exist in the groundtruth. In addition, it made a mistakenly divide 9 inches by 3 when calculating the half of the pizza size. We only change the part that is wrong from the prediction. We add the FINAL ANSWER: at the end
Answer: Ming order 9 inches pizza and ate half of it, so 9/2 = 4.5 inches left on the table. \n\nFINAL ANSWER: $\\boxed{4.5}$ inches



example2:
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
question: Jack drives to the company for two hours, then he return home with double speed. How much time does he spend on the road today?
Groundtruth: Jack takes 2 hours to go to the company and one hour come back, thus in total he drives 3 hours.
Prediction: To go to the company, Jack drives 2 hours. When he comeback, it tooks him 1 hours. Thus in todal, it takes him 3 hours.
Analyzation: The prediction is correct. thus we do not change the prediction and just add \n\nFINAL ANSWER: 3 hours at the end
Answer: To go to the company, Jack drives 2 hours. When he comeback, it tooks him 1 hours. Thus in todal, it takes him 3 hours. \n\nFINAL ANSWER: $\\boxed{3} hours



example3:
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
question: if every day, kobe shoots 100 times. How many times Kobe choots per year?
Groundtruth: kobe shoot 100 * 365 = 36500 times/ year.
Prediction: 2 times/ year.
Analyzation: As you can see, GOT3 contains hallucinations. It give a random guess of the final answer without inference. Since there is no inference steps, so no minimum change can be performed. We copy over the groundtruth as the final answer and add the FINAL ANSWER: at the end
Answer: 100 * 365 = 36500 times/ year. \n\nFINAL ANSWER: $\\boxed{36500} times/ year





Question:

given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
question: {question}
Groundtruth: {ANSWER}
Prediction: {previous_prediction}
Analyzation: NO_NEED_TO_FILL_THIS
Answer: """
#         minimum_change_prompt = \
# f"""We have a question, grouondtruth and prediction by GPT3. Please perform minimum change on the prediction by GPT3 to make it correct. 

# For example:
# MINIMUM_CHANGED_PREDICTION_HERE. 
# FINAL ANSWER: $\\boxed{ANSWER}$

# Question: {question}
# Groundtruth: {ANSWER}
# Prediction by GPT3: {previous_prediction}

# 1. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help you with that !" or "Here is the modified answer."
# 2. Please also remove the hallucination, add groundtruth as answer when the inference steps from GPT3 is empty. Please put the final answer at the end in $\\boxed{ANSWER}$. 
# 3. When performing minimum change, make sure to only correct where is wrong. please keep as much original words as possible
# 4. Format:
# MINIMUM_CHANGED_PREDICTION_HERE. 
# FINAL ANSWER: $\\boxed{ANSWER}$
# """


        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction
    return data_list




def minimum_change_data_creation_MATH_number_theory(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['pred']
        question = item['question']
        ANSWER = item['numerical_final_answer']

        example_q1 = \
"""The decimal expansion of $8/11$ is a repeating decimal. What is the least number of digits in a repeating block of 8/11?"""
        example_a1 = \
"""Certainly! The decimal expansion of $8/11$ is approximately equal to 0.7272727272727272. We can see that the digits "72" repeat in the decimal expansion. Therefore, $8/11$ is a repeating decimal with a block of 2 digits repeating indefinitely. The least number of digits in this repeating block is 2. FINAL ANSWER: 2"""
        example_p1 = \
""" Certainly! The decimal expansion of $8/11$ is a repeating decimal with a block of 3 digits repeating indefinitely. The least number of digits in this repeating block is 3."""
        example_analyzation_1 = \
"""The prediction mispredict the answer and does not include how it gets the answer. Thus we use as much original words as possible and we only add the short inference process of how we get the answer."""
        
        example_q2 = \
"""There are 54 chips in a box. Each chip is either small or large. If the number of small chips is greater than the number of large chips by a prime number of chips, what is the greatest possible number of large chips?"""
        example_a2 = \
"""Hello! I'd be happy to help you with that. Let's see...\n\nIf the number of small chips is greater than the number of large chips by a prime number of chips, that means the difference between the number of small chips and the number of large chips is a prime number.\n\nIn this case, we know that the total number of chips is 54, so we can set up an equation based on that:\n\nLet S be the number of small chips, and L be the number of large chips.\n\nWe know that S > L by a prime number of chips, so we can write:\n\nS - L = p\n\nwhere p is a prime number.\n\nNow, we have two equations:\n\n1. S + L = 54 (since there are 54 chips in total)\n2. S - L = p (since the difference between the number of small and large chips is a prime number)\n\nUsing the first equation to subtract the second equation we get 2 * L = 54 - p. We should minimize the prime number difference. The smallest prime number is 2. Thus we have 2 * L = 54 - 2 = 52. L = 26. Therefore, the greatest possible number of large chips is 26. FINAL ANSWER: 26"""
        example_p2 = \
"""" Hello! I'd be happy to help you with that. Let's see...\n\nIf the number of small chips is greater than the number of large chips by a prime number of chips, that means the difference between the number of small chips and the number of large chips is a prime number.\n\nIn this case, we know that the total number of chips is 54, so we can set up an equation based on that:\n\nLet S be the number of small chips, and L be the number of large chips.\n\nWe know that S > L by a prime number of chips, so we can write:\n\nS - L = p\n\nwhere p is a prime number.\n\nNow, we have two equations:\n\n1. S + L = 54 (since there are 54 chips in total)\n2. S - L = p (since the difference between the number of small and large chips is a prime number)\n\nCan you solve these equations to find the greatest possible number of large chips?"""
        example_analyzation_2 = \
"""The prediction has not finishing answering the quesiton, but its inference process is not worng.  Since we try to perform minimum change. We keep as much original useful inference as possible and continue the inference process."""
        
        example_q3 = \
"""How many positive multiples of 5 that are less than 100 have a units digit of 5?"""
        example_a3 = \
"""Sure, I'd be happy to help!\n\nTo find the positive multiples of 5 that are less than 100 with a units digit of 5, we can start by listing out the multiples of 5:\n\n5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95\n\nNext, we can check each of these numbers to see if they have a units digit of 5. We can do this by looking at the last digit of each number.\n\nHere are the numbers that have a units digit of 5:\n\n5, 15, 25, 35, 45, 55, 65, 75, 85, 95\n\nSo, there are 10 positive multiples of 5 that are less than 100 with a units digit of 5. FINAL ANSWER: 5"""
        example_p3 = \
"""Sure, I'd be happy to help!\n\nTo find the positive multiples of 5 that are less than 100 with a units digit of 5, we can start by listing out the multiples of 5:\n\n5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95\n\nNext, we can check each of these numbers to see if they have a units digit of 5. We can do this by looking at the last digit of each number.\n\nHere are the numbers that have a units digit of 5:\n\n5, 15, 25, 35, 45, 55, 65, 75, 85, 95\n\nSo, there are 9 positive multiples of 5 that are less than 100 with a units digit of 5."""
        example_analyzation_3 = \
"""The only mistakes the prediction made was it miscount the numbers that have a units digit of 5. Since we try to perform minimum change. We keep as much original useful inference as possible and only change the count of numbers from 9 to 10."""
        
        example_q4 = \
"""What is the smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16?"""
        example_a4 = \
"""The smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16 is 120.

Here's the calculation:

* 30 x 4 = 120

Since 120 is not a multiple of 16, it meets the condition of being a multiple of both 30 and 40 but not a multiple of 16. FINAL ANSWER: 120"""
        example_p4 = \
"""The smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16 is 120.

Here's the calculation:

* 30 x 4 = 120

Since 120 is not a multiple of 16, it meets the condition of being a multiple of both 30 and 40 but not a multiple of 16."""
        example_analyzation_4 = \
"""The prediction is already correct. Thus we use the original prediction."""
        minimum_change_prompt = \
f"""We have a question and the prediction. 
1. Please perform minimum change on the prediction to make it correct. 
2. Please also remove the hallucination, create your own answer when the inference steps from the prediction is empty. \
3. Please put the final answer at the end after FINAL ANSWER:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. FINAL ANSWER: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

Example1:{example_analyzation_1} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example1-Question: {example_q1}
Example1-Prediction: {example_p1}
Example1-Minimum Changed Prediction: {example_a1}



Example2: {example_analyzation_2} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example2-Question: {example_q2}
Example2-Prediction: {example_p2}
Example2-Minimum Changed Prediction: {example_a2}



Example3: {example_analyzation_3} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example3-Question: {example_q3}
Example3-Prediction: {example_p3}
Example3-Minimum Changed Prediction: {example_a3}



Example4: {example_analyzation_4} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example4-Question: {example_q4}
Example4-Prediction: {example_p4}
Example4-Minimum Changed Prediction: {example_a4}




Question:

given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
question: {question}
Groundtruth: {ANSWER}
Prediction: {previous_prediction}
Minimum Changed Prediction: """

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction
    return data_list

# ii = 6
# print('answer: ', data_list[ii]['answer'])
# print()
# print()
# print('previous_prediction: ', data_list[ii]['previous_prediction'])
# print('-----------------')

def minimum_change_data_creation_MATH_algebra(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['pred']
        question = item['question']
        question = question.replace('\n### Instruction:\n', '')
        question = question.replace("\n### Response:\nLet's think step by step.\n    ", '')

        
        ANSWER = item['numerical_final_answer']

        example_q1 = \
"""The decimal expansion of $8/11$ is a repeating decimal. What is the least number of digits in a repeating block of 8/11?"""
        example_a1 = \
"""Certainly! The decimal expansion of $8/11$ is approximately equal to 0.7272727272727272. We can see that the digits "72" repeat in the decimal expansion. Therefore, $8/11$ is a repeating decimal with a block of 2 digits repeating indefinitely. The least number of digits in this repeating block is 2. FINAL ANSWER: 2"""
        example_p1 = \
""" Certainly! The decimal expansion of $8/11$ is a repeating decimal with a block of 3 digits repeating indefinitely. The least number of digits in this repeating block is 3."""
        example_analyzation_1 = \
"""The prediction mispredict the answer and does not include how it gets the answer. Thus we use as much original words as possible and we only add the short inference process of how we get the answer."""
        
        example_q2 = \
"""There are 54 chips in a box. Each chip is either small or large. If the number of small chips is greater than the number of large chips by a prime number of chips, what is the greatest possible number of large chips?"""
        example_a2 = \
"""Hello! I'd be happy to help you with that. Let's see...\n\nIf the number of small chips is greater than the number of large chips by a prime number of chips, that means the difference between the number of small chips and the number of large chips is a prime number.\n\nIn this case, we know that the total number of chips is 54, so we can set up an equation based on that:\n\nLet S be the number of small chips, and L be the number of large chips.\n\nWe know that S > L by a prime number of chips, so we can write:\n\nS - L = p\n\nwhere p is a prime number.\n\nNow, we have two equations:\n\n1. S + L = 54 (since there are 54 chips in total)\n2. S - L = p (since the difference between the number of small and large chips is a prime number)\n\nUsing the first equation to subtract the second equation we get 2 * L = 54 - p. We should minimize the prime number difference. The smallest prime number is 2. Thus we have 2 * L = 54 - 2 = 52. L = 26. Therefore, the greatest possible number of large chips is 26. FINAL ANSWER: 26"""
        example_p2 = \
"""" Hello! I'd be happy to help you with that. Let's see...\n\nIf the number of small chips is greater than the number of large chips by a prime number of chips, that means the difference between the number of small chips and the number of large chips is a prime number.\n\nIn this case, we know that the total number of chips is 54, so we can set up an equation based on that:\n\nLet S be the number of small chips, and L be the number of large chips.\n\nWe know that S > L by a prime number of chips, so we can write:\n\nS - L = p\n\nwhere p is a prime number.\n\nNow, we have two equations:\n\n1. S + L = 54 (since there are 54 chips in total)\n2. S - L = p (since the difference between the number of small and large chips is a prime number)\n\nCan you solve these equations to find the greatest possible number of large chips?"""
        example_analyzation_2 = \
"""The prediction has not finishing answering the quesiton, but its inference process is not worng.  Since we try to perform minimum change. We keep as much original useful inference as possible and continue the inference process."""
        
        example_q3 = \
"""How many positive multiples of 5 that are less than 100 have a units digit of 5?"""
        example_a3 = \
"""Sure, I'd be happy to help!\n\nTo find the positive multiples of 5 that are less than 100 with a units digit of 5, we can start by listing out the multiples of 5:\n\n5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95\n\nNext, we can check each of these numbers to see if they have a units digit of 5. We can do this by looking at the last digit of each number.\n\nHere are the numbers that have a units digit of 5:\n\n5, 15, 25, 35, 45, 55, 65, 75, 85, 95\n\nSo, there are 10 positive multiples of 5 that are less than 100 with a units digit of 5. FINAL ANSWER: 5"""
        example_p3 = \
"""Sure, I'd be happy to help!\n\nTo find the positive multiples of 5 that are less than 100 with a units digit of 5, we can start by listing out the multiples of 5:\n\n5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95\n\nNext, we can check each of these numbers to see if they have a units digit of 5. We can do this by looking at the last digit of each number.\n\nHere are the numbers that have a units digit of 5:\n\n5, 15, 25, 35, 45, 55, 65, 75, 85, 95\n\nSo, there are 9 positive multiples of 5 that are less than 100 with a units digit of 5."""
        example_analyzation_3 = \
"""The only mistakes the prediction made was it miscount the numbers that have a units digit of 5. Since we try to perform minimum change. We keep as much original useful inference as possible and only change the count of numbers from 9 to 10."""
        
        example_q4 = \
"""What is the smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16?"""
        example_a4 = \
"""The smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16 is 120.

Here's the calculation:

* 30 x 4 = 120

Since 120 is not a multiple of 16, it meets the condition of being a multiple of both 30 and 40 but not a multiple of 16. FINAL ANSWER: 120"""
        example_p4 = \
"""The smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16 is 120.

Here's the calculation:

* 30 x 4 = 120

Since 120 is not a multiple of 16, it meets the condition of being a multiple of both 30 and 40 but not a multiple of 16."""
        example_analyzation_4 = \
"""The prediction is already correct. Thus we use the original prediction."""
        minimum_change_prompt = \
f"""We have a question and the prediction. 
1. Please perform minimum change on the prediction to make it correct. 
2. Please also remove the hallucination, create your own answer when the inference steps from the prediction is empty. \
3. Please put the final answer at the end after FINAL ANSWER:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. FINAL ANSWER: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

Example1:{example_analyzation_1} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example1-Question: {example_q1}
Example1-Prediction: {example_p1}
Example1-Minimum Changed Prediction: {example_a1}



Example2: {example_analyzation_2} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example2-Question: {example_q2}
Example2-Prediction: {example_p2}
Example2-Minimum Changed Prediction: {example_a2}



Example3: {example_analyzation_3} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example3-Question: {example_q3}
Example3-Prediction: {example_p3}
Example3-Minimum Changed Prediction: {example_a3}



Example4: {example_analyzation_4} We add the FINAL ANSWER: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example4-Question: {example_q4}
Example4-Prediction: {example_p4}
Example4-Minimum Changed Prediction: {example_a4}

Question:

given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
question: {question}
Groundtruth: {ANSWER}
Prediction: {previous_prediction}
Minimum Changed Prediction: """

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        if 'correct' not in question.lower():
            if 'correct' in answer.lower() or 'previous pre' in answer.lower() or 'previous ans' in answer.lower():
                answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
                if 'correct' in answer.lower() or 'previous pre' in answer.lower() or 'previous ans' in  answer.lower():
                    answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction

    return data_list






def minimum_change_data_creation_MATH_algebra_total(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['pred']
        question = item['question']
        ANSWER = item['solution']

        example_q1 = \
"""The decimal expansion of $8/11$ is a repeating decimal. What is the least number of digits in a repeating block of 8/11?"""
        example_a1 = \
"""Certainly! The decimal expansion of $8/11$ is approximately equal to 0.7272727272727272. We can see that the digits "72" repeat in the decimal expansion. Therefore, $8/11$ is a repeating decimal with a block of 2 digits repeating indefinitely. The least number of digits in this repeating block is 2. Final Answer: 2"""
        example_p1 = \
""" Certainly! The decimal expansion of $8/11$ is a repeating decimal with a block of 3 digits repeating indefinitely. The least number of digits in this repeating block is 3."""
        example_analyzation_1 = \
"""The prediction mispredict the answer and does not include how it gets the answer. Thus we use as much original words as possible and we only add the short inference process of how we get the answer."""
        
        example_q2 = \
"""There are 54 chips in a box. Each chip is either small or large. If the number of small chips is greater than the number of large chips by a prime number of chips, what is the greatest possible number of large chips?"""
        example_a2 = \
"""Hello! I'd be happy to help you with that. Let's see...\n\nIf the number of small chips is greater than the number of large chips by a prime number of chips, that means the difference between the number of small chips and the number of large chips is a prime number.\n\nIn this case, we know that the total number of chips is 54, so we can set up an equation based on that:\n\nLet S be the number of small chips, and L be the number of large chips.\n\nWe know that S > L by a prime number of chips, so we can write:\n\nS - L = p\n\nwhere p is a prime number.\n\nNow, we have two equations:\n\n1. S + L = 54 (since there are 54 chips in total)\n2. S - L = p (since the difference between the number of small and large chips is a prime number)\n\nUsing the first equation to subtract the second equation we get 2 * L = 54 - p. We should minimize the prime number difference. The smallest prime number is 2. Thus we have 2 * L = 54 - 2 = 52. L = 26. Therefore, the greatest possible number of large chips is 26. Final Answer: 26"""
        example_p2 = \
"""" Hello! I'd be happy to help you with that. Let's see...\n\nIf the number of small chips is greater than the number of large chips by a prime number of chips, that means the difference between the number of small chips and the number of large chips is a prime number.\n\nIn this case, we know that the total number of chips is 54, so we can set up an equation based on that:\n\nLet S be the number of small chips, and L be the number of large chips.\n\nWe know that S > L by a prime number of chips, so we can write:\n\nS - L = p\n\nwhere p is a prime number.\n\nNow, we have two equations:\n\n1. S + L = 54 (since there are 54 chips in total)\n2. S - L = p (since the difference between the number of small and large chips is a prime number)\n\nCan you solve these equations to find the greatest possible number of large chips?"""
        example_analyzation_2 = \
"""The prediction has not finishing answering the quesiton, but its inference process is not worng.  Since we try to perform minimum change. We keep as much original useful inference as possible and continue the inference process."""
        
        example_q3 = \
"""How many positive multiples of 5 that are less than 100 have a units digit of 5?"""
        example_a3 = \
"""Sure, I'd be happy to help!\n\nTo find the positive multiples of 5 that are less than 100 with a units digit of 5, we can start by listing out the multiples of 5:\n\n5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95\n\nNext, we can check each of these numbers to see if they have a units digit of 5. We can do this by looking at the last digit of each number.\n\nHere are the numbers that have a units digit of 5:\n\n5, 15, 25, 35, 45, 55, 65, 75, 85, 95\n\nSo, there are 10 positive multiples of 5 that are less than 100 with a units digit of 5. Final Answer: 5"""
        example_p3 = \
"""Sure, I'd be happy to help!\n\nTo find the positive multiples of 5 that are less than 100 with a units digit of 5, we can start by listing out the multiples of 5:\n\n5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95\n\nNext, we can check each of these numbers to see if they have a units digit of 5. We can do this by looking at the last digit of each number.\n\nHere are the numbers that have a units digit of 5:\n\n5, 15, 25, 35, 45, 55, 65, 75, 85, 95\n\nSo, there are 9 positive multiples of 5 that are less than 100 with a units digit of 5."""
        example_analyzation_3 = \
"""The only mistakes the prediction made was it miscount the numbers that have a units digit of 5. Since we try to perform minimum change. We keep as much original useful inference as possible and only change the count of numbers from 9 to 10."""
        
        example_q4 = \
"""What is the smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16?"""
        example_a4 = \
"""The smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16 is 120.

Here's the calculation:

* 30 x 4 = 120

Since 120 is not a multiple of 16, it meets the condition of being a multiple of both 30 and 40 but not a multiple of 16. Final Answer: 120"""
        example_p4 = \
"""The smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16 is 120.

Here's the calculation:

* 30 x 4 = 120

Since 120 is not a multiple of 16, it meets the condition of being a multiple of both 30 and 40 but not a multiple of 16."""
        example_analyzation_4 = \
"""The prediction is already correct. Thus we use the original prediction."""
        minimum_change_prompt = \
f"""We have a question and the prediction. 
1. Please perform minimum change on the prediction to make it correct. 
2. Please also remove the hallucination, create your own answer when the inference steps from the prediction is empty. \
3. Please put the final answer at the end after Final Answer:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. Final Answer: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

Example1:{example_analyzation_1} We add the Final Answer: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example1-Question: {example_q1}
Example1-Prediction: {example_p1}
Example1-Minimum Changed Prediction: {example_a1}



Example2: {example_analyzation_2} We add the Final Answer: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example2-Question: {example_q2}
Example2-Prediction: {example_p2}
Example2-Minimum Changed Prediction: {example_a2}



Example3: {example_analyzation_3} We add the Final Answer: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example3-Question: {example_q3}
Example3-Prediction: {example_p3}
Example3-Minimum Changed Prediction: {example_a3}



Example4: {example_analyzation_4} We add the Final Answer: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example4-Question: {example_q4}
Example4-Prediction: {example_p4}
Example4-Minimum Changed Prediction: {example_a4}

Question:

given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
question: {question}
Groundtruth: {ANSWER}
Prediction: {previous_prediction}
Minimum Changed Prediction: """

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction
    return data_list



def minimum_change_data_creation_MATH_algebra_total_given_answer(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['pred']
        question = item['question']
        ANSWER = item['solution']

        example_q1 = \
"""The decimal expansion of $8/11$ is a repeating decimal. What is the least number of digits in a repeating block of 8/11?"""
        example_a1 = \
"""Certainly! The decimal expansion of $8/11$ is approximately equal to 0.7272727272727272. We can see that the digits "72" repeat in the decimal expansion. Therefore, $8/11$ is a repeating decimal with a block of 2 digits repeating indefinitely. The least number of digits in this repeating block is 2. Final Answer: 2"""
        example_p1 = \
""" Certainly! The decimal expansion of $8/11$ is a repeating decimal with a block of 3 digits repeating indefinitely. The least number of digits in this repeating block is 3."""
        example_analyzation_1 = \
"""The prediction mispredict the answer and does not include how it gets the answer. Thus we use as much original words as possible and we only add the short inference process of how we get the answer."""
        
        example_q2 = \
"""There are 54 chips in a box. Each chip is either small or large. If the number of small chips is greater than the number of large chips by a prime number of chips, what is the greatest possible number of large chips?"""
        example_a2 = \
"""Hello! I'd be happy to help you with that. Let's see...\n\nIf the number of small chips is greater than the number of large chips by a prime number of chips, that means the difference between the number of small chips and the number of large chips is a prime number.\n\nIn this case, we know that the total number of chips is 54, so we can set up an equation based on that:\n\nLet S be the number of small chips, and L be the number of large chips.\n\nWe know that S > L by a prime number of chips, so we can write:\n\nS - L = p\n\nwhere p is a prime number.\n\nNow, we have two equations:\n\n1. S + L = 54 (since there are 54 chips in total)\n2. S - L = p (since the difference between the number of small and large chips is a prime number)\n\nUsing the first equation to subtract the second equation we get 2 * L = 54 - p. We should minimize the prime number difference. The smallest prime number is 2. Thus we have 2 * L = 54 - 2 = 52. L = 26. Therefore, the greatest possible number of large chips is 26. Final Answer: 26"""
        example_p2 = \
"""" Hello! I'd be happy to help you with that. Let's see...\n\nIf the number of small chips is greater than the number of large chips by a prime number of chips, that means the difference between the number of small chips and the number of large chips is a prime number.\n\nIn this case, we know that the total number of chips is 54, so we can set up an equation based on that:\n\nLet S be the number of small chips, and L be the number of large chips.\n\nWe know that S > L by a prime number of chips, so we can write:\n\nS - L = p\n\nwhere p is a prime number.\n\nNow, we have two equations:\n\n1. S + L = 54 (since there are 54 chips in total)\n2. S - L = p (since the difference between the number of small and large chips is a prime number)\n\nCan you solve these equations to find the greatest possible number of large chips?"""
        example_analyzation_2 = \
"""The prediction has not finishing answering the quesiton, but its inference process is not worng.  Since we try to perform minimum change. We keep as much original useful inference as possible and continue the inference process."""
        
        example_q3 = \
"""How many positive multiples of 5 that are less than 100 have a units digit of 5?"""
        example_a3 = \
"""Sure, I'd be happy to help!\n\nTo find the positive multiples of 5 that are less than 100 with a units digit of 5, we can start by listing out the multiples of 5:\n\n5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95\n\nNext, we can check each of these numbers to see if they have a units digit of 5. We can do this by looking at the last digit of each number.\n\nHere are the numbers that have a units digit of 5:\n\n5, 15, 25, 35, 45, 55, 65, 75, 85, 95\n\nSo, there are 10 positive multiples of 5 that are less than 100 with a units digit of 5. Final Answer: 5"""
        example_p3 = \
"""Sure, I'd be happy to help!\n\nTo find the positive multiples of 5 that are less than 100 with a units digit of 5, we can start by listing out the multiples of 5:\n\n5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95\n\nNext, we can check each of these numbers to see if they have a units digit of 5. We can do this by looking at the last digit of each number.\n\nHere are the numbers that have a units digit of 5:\n\n5, 15, 25, 35, 45, 55, 65, 75, 85, 95\n\nSo, there are 9 positive multiples of 5 that are less than 100 with a units digit of 5."""
        example_analyzation_3 = \
"""The only mistakes the prediction made was it miscount the numbers that have a units digit of 5. Since we try to perform minimum change. We keep as much original useful inference as possible and only change the count of numbers from 9 to 10."""
        
        example_q4 = \
"""What is the smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16?"""
        example_a4 = \
"""The smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16 is 120.

Here's the calculation:

* 30 x 4 = 120

Since 120 is not a multiple of 16, it meets the condition of being a multiple of both 30 and 40 but not a multiple of 16. Final Answer: 120"""
        example_p4 = \
"""The smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16 is 120.

Here's the calculation:

* 30 x 4 = 120

Since 120 is not a multiple of 16, it meets the condition of being a multiple of both 30 and 40 but not a multiple of 16."""
        example_analyzation_4 = \
"""The prediction is already correct. Thus we use the original prediction."""
        minimum_change_prompt = \
f"""We have a question and the prediction. 
1. Please perform minimum change on the prediction to make it correct. 
2. Please also remove the hallucination, create your own answer when the inference steps from the prediction is empty. \
3. Please put the final answer at the end after Final Answer:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. Final Answer: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

Example1:{example_analyzation_1} We add the Final Answer: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example1-Question: {example_q1}
Example1-Prediction: {example_p1}
Example1-Minimum Changed Prediction: {example_a1}



Example2: {example_analyzation_2} We add the Final Answer: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example2-Question: {example_q2}
Example2-Prediction: {example_p2}
Example2-Minimum Changed Prediction: {example_a2}



Example3: {example_analyzation_3} We add the Final Answer: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example3-Question: {example_q3}
Example3-Prediction: {example_p3}
Example3-Minimum Changed Prediction: {example_a3}



Example4: {example_analyzation_4} We add the Final Answer: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example4-Question: {example_q4}
Example4-Prediction: {example_p4}
Example4-Minimum Changed Prediction: {example_a4}

Question:

given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
question: {question}
Groundtruth: {ANSWER}
Prediction: {previous_prediction}
Minimum Changed Prediction: """

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction
    return data_list

def minimum_change_data_creation_MATH_algebra_total_mistral(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = predict_list[i]
        question = item['question']
        # ANSWER = item['solution']

        question = question.removeprefix("Input:")

        # Remove the ending "\n\nResponse:"
        question = question.removesuffix("\n\nResponse:")
        question = question.strip()

        example_q1 = \
"""Find the distance between the points (0,15) and (8,0).\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"""
        example_a1 = \
"""To find the distance between two points, we use the distance formula:\n\ndistance = √(x2 - x1)^2 + (y2 - y1)^2\n\nIn this case, the points are (0,15) and (8,0)\n\nx2 = 8, x1 = 0\ny2 = 0, y1 = 15\n\nPlugging these values into the formula, we get:\n\ndistance = √(8 - 0)^2 + (0 - 15)^2\ndistance = √(8)^2 + (15)^2\ndistance = √64 + 225\ndistance = √289\ndistance = 17\n\nFinal Answer: 17\n\nSo the distance between the points (0,15) and (8,0) is 17."""
        example_p1 = \
"""To find the distance between two points, we use the distance formula:\n\ndistance = √(x2 - x1)^2 + (y2 - y1)^2\n\nIn this case, the points are (0,15) and (8,0)\n\nx2 = 8, x1 = 0\ny2 = 0, y1 = 15\n\nPlugging these values into the formula, we get:\n\ndistance = √(8 - 0)^2 + (0 - 15)^2\ndistance = √(8)^2 + (15)^2\ndistance = √64 + 225\ndistance = √249\ndistance = 15\n\nFinal Answer: 15\n\nSo the distance between the points (0,15) and (8,0) is 15."""
        example_analyzation_1 = \
"""The inference was mostly correct. there is just a calculation error. We keep as much original useful inference as possible and only change the final answer. We make sure the Final Answer: a number is placed at the end"""
        
        example_q2 = \
"""Evaluate $(x-a+3)$ if the value of $x$ is $(a+7)$.\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'"""
        example_a2 = \
"""First, we need to evaluate the expression $(x-a+3)$ when $x = (a+7)$.\n\nSubstituting $x = (a+7)$ into the expression, we get:\n\n$$(a+7) - a + 3$$\n\n= $$7 + 3$$\n\n= $$10$$\n\nFinal Answer: 10"""
        example_p2 = \
"""First, we need to evaluate the expression $(x-a+3)$ when $x = (a+7)$.\n\nSubstituting $x = (a+7)$ into the expression, we get:\n\n$$(a+7) - a + 3$$\n\n= $$7 + 3$$\n\n= $$10$$\n\nFinal Answer: $10$\n\nTherefore, the value of $(x-a+3)$ when $x = (a+7)$ is $10$."""
        example_analyzation_2 = \
"""The previous prediction is already correct, so we copy the previous prediction(including the inference process). Just modify it to make sure Final Answer: 10 is placed at the end."""
        example_q3 = \
"""How many positive multiples of 5 that are less than 100 have a units digit of 5?"""
        example_a3 = \
"""To find the positive multiples of 5 that are less than 100 with a units digit of 5, we can start by listing out the multiples of 5:\n\n5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95\n\nNext, we can check each of these numbers to see if they have a units digit of 5. We can do this by looking at the last digit of each number.\n\nHere are the numbers that have a units digit of 5:\n\n5, 15, 25, 35, 45, 55, 65, 75, 85, 95\n\nSo, there are 10 positive multiples of 5 that are less than 100 with a units digit of 5. Final Answer: 5"""
        example_p3 = \
"""To find the positive multiples of 5 that are less than 100 with a units digit of 5, we can start by listing out the multiples of 5:\n\n5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95\n\nNext, we can check each of these numbers to see if they have a units digit of 5. We can do this by looking at the last digit of each number.\n\nHere are the numbers that have a units digit of 5:\n\n5, 15, 25, 35, 45, 55, 65, 75, 85, 95\n\nSo, there are 9 positive multiples of 5 that are less than 100 with a units digit of 5."""
        example_analyzation_3 = \
"""The only mistakes the prediction made was it miscount the numbers that have a units digit of 5. Since we try to perform minimum change. We keep as much original useful inference as possible and only change the count of numbers from 9 to 10."""
       
        minimum_change_prompt = \
f"""We have a question and the previous prediction. 
1. Please perform minimum change on the previous prediction(including the inference process) to make it correct. when i say minimum change, i mean you should keep as much original words from the previous prediction as possible and only correct where the previous prediction(including the inference process) is wrong.
3. Please put the final answer at the end after Final Answer:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. Final Answer: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

Example1:{example_analyzation_1}
given the question and the previous prediction, please perform minimum change to make the prediction correct.
Example1-Question: {example_q1}
Example1-Previous Prediction: {example_p1}
Example1-Minimum Changed Prediction: {example_a1}



Example2: {example_analyzation_2}
given the question and the previous prediction, please perform minimum change to make the prediction correct.
Example2-Question: {example_q2}
Example2-Previous Prediction: {example_p2}
Example2-Minimum Changed Prediction: {example_a2}

Example3: {example_analyzation_3} We add the Final Answer: at the end
given the question and prediction, please perform minimum change to make the prediction correct.
Example3-Question: {example_q3}
Example3-Prediction: {example_p3}
Example3-Minimum Changed Prediction: {example_a3}


Let me repeat my most important requirement. You are trying to keep as much original words as possible and only correct where the previous prediction(including the inference process) is wrong. Keep the original words is very important!


given the question and the previous prediction, please perform minimum change to make the prediction correct.
Question: {question}
Preivous Prediction: {previous_prediction}
Minimum Changed Prediction: """
# Groundtruth: {ANSWER}

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        if 'correct' not in question.lower():
            if 'correct' in answer.lower() or 'previous pre' in answer.lower() or 'previous ans' in answer.lower():
                answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
                if 'correct' in answer.lower() or 'previous pre' in answer.lower() or 'previous ans' in  answer.lower():
                    answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction
    return data_list




def minimum_change_data_creation_MATH_algebra_total_llama3(data_list, predict_list, use_groundtruth = False):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = predict_list[i]
        question = item['question']
        numerical_final_answer = item['numerical_final_answer']
        ANSWER = item['solution']
        ANSWER += f'Final Answer: {numerical_final_answer}'
        question = question.removeprefix("Input:")

        # Remove the ending "\n\nResponse:"
        question = question.removesuffix("\n\nResponse:")
        question = question.strip()

        example_q1 = \
"""Find the coefficient of $x^2$ when $3(x^2 - x^3) +2(x - 2x^2 + 3x^5) -(4x^3 - x^2)$ is simplified.\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"""
        example_a1 = \
"""We have that \begin{align*}
3(x^2 - x^3) +2(x - 2x^2 + 3x^5) -(4x^3 - x^2) &= 3x^2 - 3x^3 + 2x - 4x^2 + 6x^5 - 4x^3 + x^2 \\\\
&= 6x^5 + (-3x^3 - 4x^3) + (3x^2 - 4x^2 + x^2) + 2x \\\\
&= 6x^5 - 7x^3 + 0x^2 + 2x.
\end{align*}
The coefficient of $x^2$ is $\boxed{0}$.
Final Answer: 0"""
        example_p1 = \
"""We have that \\begin{align*}\n3(x^2 - x^3) +2(x - 2x^2 + 3x^5) -(4x^3 - x^2) &= 3x^2 - 3x^3 + 2x - 4x^2 + 6x^5 - 4x^3 + x^2 \\\\\n&= 6x^5 + (-3x^3 - 4x^3 + 3x^2 - 4x^2 + 2x) \\\\\n&= 6x^5 + (-7x^3 + (-2)x^2 + 2x) \\\\\n&= 6x^5 - 7x^3 - 2x^2 + 2x.\n\\end{align*}The coefficient of $x^2$ is $\\boxed{-2}$.\nFinal Answer: -2."""
        example_gt1 = \
"""When we expand we get  \\begin{align*}\n&3(x^2 - x^3) +2(x - 2x^2 + 3x^5) -(4x^3 - x^2) \\\\\n&\\qquad =3x^2 - 3x^3 +2x - 4x^2 + 6x^5 -4x^3 + x^2\\\\\n&\\qquad =6x^5-7x^3+2x.\n\\end{align*}The coefficient of $x^2$ is $3-4+1=\\boxed{0}$. Final Answer: 0"""
        example_analyzation_1 = \
"""The previous prediction made some minor calculation mistakes, so we correct it. We also make sure Final Answer: NUMBER is placed at the end."""
        
        example_q2 = \
"""What value of $x$ will give the minimum value for $x^2- 10x + 24$?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"""
        example_a2 = \
"""To find the minimum value of the function $x^2-10x+24$, we can complete the square. \n\n\\begin{align*}\nx^2-10x+24 &= x^2-10x+25-1\\\\\n&= (x-5)^2-1.\n\\end{align*}The minimum value of $(x-5)^2$ is 0, and when $x=5$, $(x-5)^2$ equals 0. Therefore, the minimum value of $x^2-10x+24$ is $-1$, and the value of $x$ that gives this minimum value is $\\boxed{5}$. \n\nFinal Answer: 5"""
        example_gt2 = \
"""We start by completing the square.  \\[x^2-10x+24=(x-5)^2-1.\\] Since the square of a real number is at least 0, $(x-5)^2\\ge 0$ and $(x-5)^2-1 \\ge -1.$ Thus, the minimum value of the quadratic is $-1,$ which occurs when $x=\\boxed{5}. Final Answer: 5"""
        example_p2 = \
"""To find the minimum value of the function $x^2-10x+24$, we can complete the square. \n\n\\begin{align*}\nx^2-10x+24 &= x^2-10x+25-1\\\\\n&= (x-5)^2-1.\n\\end{align*}The minimum value of $(x-5)^2$ is 0, and when $x=5$, $(x-5)^2$ equals 0. Therefore, the minimum value of $x^2-10x+24$ is $-1$, and the value of $x$ that gives this minimum value is $\\boxed{5}$. \n\nFinal Answer: The final answer is 5. I hope it is correct."""

        example_analyzation_2 = \
"""The previous prediction is correct. There is only a format issue. We make sure Final Answer: NUMBER is placed at the end."""

        if not use_groundtruth:
            minimum_change_prompt = \
f"""We have a question and the previous prediction. 
1. Please perform minimum change on the previous prediction(including the inference process) to make it correct. when i say minimum change, i mean you should keep as much original words from the previous prediction as possible and only correct where the previous prediction(including the inference process) is wrong.
3. Please put the final answer at the end after Final Answer:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. Final Answer: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

Example1:{example_analyzation_1}
given the question and the previous prediction, please perform minimum change to make the prediction correct.
Example1-Question: {example_q1}
Example1-Previous Prediction: {example_p1}
Example1-Minimum Changed Prediction: {example_a1}



Example2: {example_analyzation_2}
given the question and the previous prediction, please perform minimum change to make the prediction correct.
Example2-Question: {example_q2}
Example2-Previous Prediction: {example_p2}
Example2-Minimum Changed Prediction: {example_a2}


Let me repeat my most important requirement. You are trying to keep as much original words as possible and only correct where the previous prediction(including the inference process) is wrong. Keep the original words is very important!


given the question and the previous prediction, please perform minimum change to make the prediction correct.
Question: {question}
Preivous Prediction: {previous_prediction}
Minimum Changed Prediction: """
        else:
            minimum_change_prompt = \
f"""We have a question, goroundtruth and the previous prediction. 
1. Please perform minimum change on the previous prediction(including the inference process) to make it correct. when i say minimum change, i mean you should keep as much original words from the previous prediction as possible and only correct where the previous prediction(including the inference process) is wrong.
2. You should pretend as you do not know the groundtruth. so don't saying anything like according to groundtruth. Your task is performing minimum change so you are trying to sue the words from previous prediction instead of direct copying groundtruth.
3. Please put the final answer at the end after Final Answer:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. Final Answer: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

Example1:{example_analyzation_1}
given the question and the previous prediction, please perform minimum change to make the prediction correct.
Example1-Question: {example_q1}
Example1-Groundtruth: {example_gt1}
Example1-Previous Prediction: {example_p1}
Example1-Minimum Changed Prediction: {example_a1}



Example2: {example_analyzation_2}
given the question and the previous prediction, please perform minimum change to make the prediction correct.
Example2-Question: {example_q2}
Example2-Groundtruth: {example_gt2}
Example2-Previous Prediction: {example_p2}
Example2-Minimum Changed Prediction: {example_a2}


Let me repeat my most important requirement. You are trying to keep as much original words as possible and only correct where the previous prediction(including the inference process) is wrong. Keep the original words is very important!


given the question and the previous prediction, please perform minimum change to make the prediction correct.
Question: {question}
Groundtruth: {ANSWER}
Preivous Prediction: {previous_prediction}
Minimum Changed Prediction: """


        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        if 'correct' not in question.lower():
            if 'groundtruth' in answer.lower() or 'correct' in answer.lower() or 'previous pre' in answer.lower() or 'previous ans' in answer.lower():
                answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
                if 'groundtruth' in answer.lower() or 'correct' in answer.lower() or 'previous pre' in answer.lower() or 'previous ans' in  answer.lower():
                    answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction
    return data_list


def minimum_change_data_creation_GSM8K(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = predict_list[i]
        question = item['question']
        ANSWER = item['answer']

        example_q1 = \
"""The decimal expansion of $8/11$ is a repeating decimal. What is the least number of digits in a repeating block of 8/11?"""
        example_a1 = \
"""Certainly! The decimal expansion of $8/11$ is approximately equal to 0.7272727272727272. We can see that the digits "72" repeat in the decimal expansion. Therefore, $8/11$ is a repeating decimal with a block of 2 digits repeating indefinitely. The least number of digits in this repeating block is 2. Final Answer: 2"""
        example_p1 = \
""" Certainly! The decimal expansion of $8/11$ is a repeating decimal with a block of 3 digits repeating indefinitely. The least number of digits in this repeating block is 3."""
        example_analyzation_1 = \
"""The prediction mispredict the answer and does not include how it gets the answer. Thus we use as much original words as possible and we only add the short inference process of how we get the answer."""
        
        example_q2 = \
"""There are 54 chips in a box. Each chip is either small or large. If the number of small chips is greater than the number of large chips by a prime number of chips, what is the greatest possible number of large chips?"""
        example_a2 = \
"""Hello! I'd be happy to help you with that. Let's see...\n\nIf the number of small chips is greater than the number of large chips by a prime number of chips, that means the difference between the number of small chips and the number of large chips is a prime number.\n\nIn this case, we know that the total number of chips is 54, so we can set up an equation based on that:\n\nLet S be the number of small chips, and L be the number of large chips.\n\nWe know that S > L by a prime number of chips, so we can write:\n\nS - L = p\n\nwhere p is a prime number.\n\nNow, we have two equations:\n\n1. S + L = 54 (since there are 54 chips in total)\n2. S - L = p (since the difference between the number of small and large chips is a prime number)\n\nUsing the first equation to subtract the second equation we get 2 * L = 54 - p. We should minimize the prime number difference. The smallest prime number is 2. Thus we have 2 * L = 54 - 2 = 52. L = 26. Therefore, the greatest possible number of large chips is 26. Final Answer: 26"""
        example_p2 = \
"""" Hello! I'd be happy to help you with that. Let's see...\n\nIf the number of small chips is greater than the number of large chips by a prime number of chips, that means the difference between the number of small chips and the number of large chips is a prime number.\n\nIn this case, we know that the total number of chips is 54, so we can set up an equation based on that:\n\nLet S be the number of small chips, and L be the number of large chips.\n\nWe know that S > L by a prime number of chips, so we can write:\n\nS - L = p\n\nwhere p is a prime number.\n\nNow, we have two equations:\n\n1. S + L = 54 (since there are 54 chips in total)\n2. S - L = p (since the difference between the number of small and large chips is a prime number)\n\nCan you solve these equations to find the greatest possible number of large chips?"""
        example_analyzation_2 = \
"""The prediction has not finishing answering the quesiton, but its inference process is not worng.  Since we try to perform minimum change. We keep as much original useful inference as possible and continue the inference process."""
        
        example_q3 = \
"""How many positive multiples of 5 that are less than 100 have a units digit of 5?"""
        example_a3 = \
"""Sure, I'd be happy to help!\n\nTo find the positive multiples of 5 that are less than 100 with a units digit of 5, we can start by listing out the multiples of 5:\n\n5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95\n\nNext, we can check each of these numbers to see if they have a units digit of 5. We can do this by looking at the last digit of each number.\n\nHere are the numbers that have a units digit of 5:\n\n5, 15, 25, 35, 45, 55, 65, 75, 85, 95\n\nSo, there are 10 positive multiples of 5 that are less than 100 with a units digit of 5. Final Answer: 5"""
        example_p3 = \
"""Sure, I'd be happy to help!\n\nTo find the positive multiples of 5 that are less than 100 with a units digit of 5, we can start by listing out the multiples of 5:\n\n5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95\n\nNext, we can check each of these numbers to see if they have a units digit of 5. We can do this by looking at the last digit of each number.\n\nHere are the numbers that have a units digit of 5:\n\n5, 15, 25, 35, 45, 55, 65, 75, 85, 95\n\nSo, there are 9 positive multiples of 5 that are less than 100 with a units digit of 5."""
        example_analyzation_3 = \
"""The only mistakes the prediction made was it miscount the numbers that have a units digit of 5. Since we try to perform minimum change. We keep as much original useful inference as possible and only change the count of numbers from 9 to 10."""
        
        example_q4 = \
"""What is the smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16?"""
        example_a4 = \
"""The smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16 is 120.

Here's the calculation:

* 30 x 4 = 120

Since 120 is not a multiple of 16, it meets the condition of being a multiple of both 30 and 40 but not a multiple of 16. Final Answer: 120"""
        example_p4 = \
"""The smallest positive integer that is a multiple of both 30 and 40 but not a multiple of 16 is 120.

Here's the calculation:

* 30 x 4 = 120

Since 120 is not a multiple of 16, it meets the condition of being a multiple of both 30 and 40 but not a multiple of 16."""
        example_analyzation_4 = \
"""The prediction is already correct. Thus we use the original prediction."""
        minimum_change_prompt = \
f"""We have a question and the prediction. 
1. Please perform minimum change on the prediction to make it correct. 
2. Please also remove the hallucination, create your own answer when the inference steps from the prediction is empty. \
3. Please put the final answer at the end after Final Answer:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. Final Answer: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

Example1:{example_analyzation_1} We add the Final Answer: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example1-Question: {example_q1}
Example1-Prediction: {example_p1}
Example1-Minimum Changed Prediction: {example_a1}



Example2: {example_analyzation_2} We add the Final Answer: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example2-Question: {example_q2}
Example2-Prediction: {example_p2}
Example2-Minimum Changed Prediction: {example_a2}



Example3: {example_analyzation_3} We add the Final Answer: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example3-Question: {example_q3}
Example3-Prediction: {example_p3}
Example3-Minimum Changed Prediction: {example_a3}



Example4: {example_analyzation_4} We add the Final Answer: at the end
given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
Example4-Question: {example_q4}
Example4-Prediction: {example_p4}
Example4-Minimum Changed Prediction: {example_a4}

Question:

given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
question: {question}
Groundtruth: {ANSWER}
Prediction: {previous_prediction}
Minimum Changed Prediction: """

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction
    return data_list


def minimum_change_data_creation_GSM8K_mistral(data_list, predict_list, use_groundtruth = True, groundtruth_list = []):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = predict_list[i]
        question = item['question']
        question = question.replace('\n### Instruction:\n', '')
        question = question.replace("\n### Response:\nLet's think step by step.\n    ", '')

        # Remove the beginning "Input:"
        question = question.removeprefix("Input:")

        # Remove the ending "\n\nResponse:"
        question = question.removesuffix("\n\nResponse:")
        question = question.strip()
        if use_groundtruth:
            ANSWER = item['answer']
        if use_groundtruth:
            minimum_change_prompt = \
f"""We have a question and the previous prediction. 
1. Please perform minimum change on the prediction to make it correct. 
2. Please also remove the hallucination, create your own answer when the inference steps from the prediction is empty. \
3. Please put the final answer at the end after Final Answer:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. Final Answer: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

Example1: The inference steps of the previous prediction is mostly correct, but it made some mistakes and it does not use the correct format. The minimum changed prediction correct the mistakes and keep as much original words as possible. We also remove the $ sign after 'Final Answer:' to make sure what is placed at the end is a number.
Example1-question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'
Example1-groundtruth: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n#### 5
Example1-previous prediction: Betty's parents gave her $15, and her grandparents gave her twice that amount. So, her grandparents gave her $15 * 2 = $30.\n\nBetty has received a total of $15 (from her parents) + $30 (from her grandparents) = $45.\n\nThe wallet costs $100, so Betty still needs $100 - $45 = $55 to buy the wallet.\n\nFinal Answer: $55
Example1-minimum changed prediction: Betty's parents gave her $15, and her grandparents gave her twice that amount. So, her grandparents gave her $15 * 2 = $30.\n\nBetty originally had $50, which is half of what she needs for the wallet.\n\nBetty has received a total of $15 (from her parents) + $30 (from her grandparents) + $50 (originally had) = $95.\n\nThe wallet costs $100, so Betty still needs $100 - $95 = $5 to buy the wallet.\n\nFinal Answer: 5

Example2: The previous prediction is already correct, but it does not use the correct format. It suppose to add the Final Answer: at the end with a number. So we copy the previous prediction and replace '72 clips' with '72' at the end.
Example2-question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'
Example2-groundtruth: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72
Example2-previous prediction: Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips.\n\nSo, Natalia sold a total of 48 (April) + 24 (May) = 72 clips in April and May.\n\nFinal Answer: 72 clips
Example2-minimum changed prediction: Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips.\n\nSo, Natalia sold a total of 48 (April) + 24 (May) = 72 clips in April and May.\n\nFinal Answer: 72


given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
question: {question}
Groundtruth: {ANSWER}
Previous Prediction: {previous_prediction}
Minimum Changed Prediction: """
        else:
            minimum_change_prompt = \
f"""We have a question and the previous prediction. 
1. Please perform minimum change on the prediction to make it correct. 
2. Please also remove the hallucination, create your own answer when the inference steps from the prediction is empty. \
3. Please put the final answer at the end after Final Answer:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. Final Answer: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

Example1: The inference steps of the previous prediction is mostly correct, but it made some mistakes and it does not use the correct format. The minimum changed prediction correct the mistakes and keep as much original words as possible. We also remove the $ sign after 'Final Answer:' to make sure what is placed at the end is a number.
Example1-question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'
Example1-previous prediction: Betty's parents gave her $15, and her grandparents gave her twice that amount. So, her grandparents gave her $15 * 2 = $30.\n\nBetty has received a total of $15 (from her parents) + $30 (from her grandparents) = $45.\n\nThe wallet costs $100, so Betty still needs $100 - $45 = $55 to buy the wallet.\n\nFinal Answer: $55
Example1-minimum changed prediction: Betty's parents gave her $15, and her grandparents gave her twice that amount. So, her grandparents gave her $15 * 2 = $30.\n\nBetty originally had $50, which is half of what she needs for the wallet.\n\nBetty has received a total of $15 (from her parents) + $30 (from her grandparents) + $50 (originally had) = $95.\n\nThe wallet costs $100, so Betty still needs $100 - $95 = $5 to buy the wallet.\n\nFinal Answer: 5

Example2: The previous prediction is already correct, but it does not use the correct format. It suppose to add the Final Answer: at the end with a number. So we copy the previous prediction and replace '72 clips' with '72' at the end.
Example2-question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'
Example2-previous prediction: Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips.\n\nSo, Natalia sold a total of 48 (April) + 24 (May) = 72 clips in April and May.\n\nFinal Answer: 72 clips
Example2-minimum changed prediction: Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips.\n\nSo, Natalia sold a total of 48 (April) + 24 (May) = 72 clips in April and May.\n\nFinal Answer: 72


given the question and prediction, please perform minimum change to make the prediction correct.
question: {question}
Previous Prediction: {previous_prediction}
Minimum Changed Prediction: """


        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        if 'correct' not in question.lower():
            if 'correct' in answer.lower() or 'previous pre' in answer.lower() or 'previous ans' in answer.lower():
                answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
                if 'correct' in answer.lower() or 'previous pre' in answer.lower() or 'previous ans' in  answer.lower():
                    answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction
    return data_list





def minimum_change_data_creation_AQuaRAT(data_list, predict_list, use_groundtruth = True, groundtruth_list = []):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = predict_list[i]
        question = item['question']
        question = question.replace('\n### Instruction:\n', '')
        question = question.replace("\n### Response:\nLet's think step by step.\n    ", '')

        # Remove the beginning "Input:"
        question = question.removeprefix("Input:")

        # Remove the ending "\n\nResponse:"
        question = question.removesuffix("\n\nResponse:")
        question = question.strip()
           
        minimum_change_prompt = \
f"""We have a question and the previous prediction. 
1. Please perform minimum change on the prediction to make it correct. 
2. If the previous prediction and its inference steps are already correct, you can just copy it.
3. Make sure the format is correct. You have to put the Final Answer: (A)/(B)/(C)/(D)/(E) at the end after Final Answer:
For example:
MINIMUM_CHANGED_PREDICTION_HERE.\n\nFinal Answer: (E)
4. Please also remove the hallucination, create your own answer when the inference steps from the prediction is empty. \
5. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !", "Here is the modified answer." or "the previous prediction is correct because"

Example1: Some of the inference steps of the previous prediction is correct. We only change where is wrong and try to keep original words. The minimum changed prediction correct the mistakes and keep as much original words as possible. We also make sure the Final Answer: (A)/(B)/(C)/(D)/(E) is placed at the end according to the question format.
Example1-question: "Two friends plan to walk along a 43-km trail, starting at opposite ends of the trail at the same time. If Friend P's rate is 15% faster than Friend Q's, how many kilometers will Friend P have walked when they pass each other?\n\nOptions:\n(A)21\n(B)21.5\n(C)22\n(D)22.5 \n(E)23\nPlease choose the correct answer (A)/(B)/(C)/(D)/(E) and place it at the end, after '\n\nFinal Answer: '\n\nLet's think step by step."
Example1-previous prediction: We are given that Friend P's rate is 15% faster than Friend Q's.\nThis means that Friend P walks 15% faster than Friend Q, or that Friend P walks 1.15 times the distance Friend Q walks.\n\nLet's represent the distance Friend Q walks as $x$ kilometers.\nThen, the distance Friend P walks is $1.15x$ kilometers.\n\nSince they start at opposite ends of the trail, they will meet at a point that is halfway along the trail.\nThis means that Friend P will have walked $\\frac{{1.15x}}{2}$ kilometers when they pass each other.\n\nWe are also given that the total distance of the trail is 43 kilometers.\nSince Friend Q walks $x$ kilometers and Friend P walks $1.15x$ kilometers, we have the equation $x + 1.15x = 43$.\n\nCombining like terms, we get $2.15x = 43$.\nDividing both sides by 2.15, we find $x = \\frac{43}{2.15}$.\n\nSubstituting this value back into the equation for Friend P's distance, we have $\\frac{1.15}{2} \\cdot \\frac{43}{2.15}$.\n\nEvaluating this expression, we find that Friend P will have walked approximately 21.5 kilometers when they pass each other.\n\nTherefore, the correct answer is (B) 21.5 kilometers.The answer is: 21.5
Example1-minimum changed prediction: We are given that Friend P's rate is 15% faster than Friend Q's.\nThis means that Friend P walks 15% faster than Friend Q, or that Friend P walks 1.15 times the distance Friend Q walks.\n\nLet's represent the distance Friend Q walks as $x$ kilometers.
Then, the distance Friend P walks is $1.15x$ kilometers.\n\nSince they start at opposite ends of the trail, they will meet at a point that is halfway along the trail.\nThis means that Friend P will have walked $\\frac{{1.15x}}{2}$ kilometers when they pass each other.\n\nWe are also given that the total distance of the trail is 43 kilometers.\nSince Friend Q walks $x$ kilometers and Friend P walks $1.15x$ kilometers, we have the equation $x + 1.15x = 43$.\n\nCombining like terms, we get $2.15x = 43$.\nDividing both sides by 2.15, we find $x = 20$.\n\nSubstituting $x = 20$ into the equation $x + 1.15x = 43$ for Friend P's distance, Friend P would have walked $1.15 \times 20 = 23$ kilometers when they meet.\n\nTherefore, the correct answer is (E) 21.5 kilometers.\n\nFinal Answer: (E)

Example2: Some of the inference steps of the previous prediction is correct. We only change where is wrong and try to keep original words. The minimum changed prediction correct the mistakes and keep as much original words as possible. We also make sure the Final Answer: (A)/(B)/(C)/(D)/(E) is placed at the end according to the question format.
Example1-question: "In the coordinate plane, points (x, 1) and (5, y) are on line k. If line k passes through the origin and has slope 1/5, then what are the values of x and y respectively?\n\nOptions:\n(A)4 and 1\n(B)1 and 5\n(C)5 and 1\n(D)3 and 5 \n(E)5 and 3\nPlease choose the correct answer (A)/(B)/(C)/(D)/(E) and place it at the end, after '\n\nFinal Answer: '\n\n\n### Response:\nLet's think step by step."
Example1-previous prediction: We are given that line k passes through the origin and has slope 1/5.\nThis means that for every increase of 1 unit in the x-coordinate, there is an increase of 1/5 unit in the y-coordinate.\nSince point (x, 1) lies on line k, we can conclude that the x-coordinate of point (x, 1) is 1 unit away from the origin, and the y-coordinate of point (x, 1) is 1/5 unit away from the origin.\nTherefore, the x-coordinate of point (x, 1) is 1 * 1 = 1 unit.\nSimilarly, the y-coordinate of point (5, y) is 1/5 * 5 = 1 unit.\nSo the values of x and y are 1 and 1 respectively.\nThe answer is: 1
Example1-minimum changed prediction: We are given that line k passes through the origin and has slope 1/5.\nThis means that for every increase of 5 units in the x-direction, there is an increase of 1 unit in the y-direction.\nSince point (x, 1) lies on line k, its y-coordinate is 1, implying that its x-coordinate must be 5 to maintain the slope of 1/5.\nSimilarly, point (5, y) lies on line k, its x-coordinate is 5, which implies that its y-coordinate must be 1, to maintain the slope of 1/5 when passing through the origin.\nSo, the values of x and y are 5 and 1 respectively.\n\nFinal Answer: (C)


given the question and prediction, please perform minimum change to make the prediction correct.
question: {question}
Previous Prediction: {previous_prediction}
Minimum Changed Prediction: """


        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        if 'correct' not in question.lower():
            if 'correct' in answer.lower() or 'previous pre' in answer.lower() or 'previous ans' in answer.lower():
                answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
                if 'correct' in answer.lower() or 'previous pre' in answer.lower() or 'previous ans' in  answer.lower():
                    answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction
    return data_list


def reformat_groundtruth_GSM8K_mistral_training_data_creation(minimum_change_data_list, groundtruth_data_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(minimum_change_data_list):
        previous_prediction = item['previous_prediction'] 
        question = item['question']
        answer = item['answer']
        groundtruth = groundtruth_data_list[i]['answer']

        minimum_change_prompt = \
f"""You have a question, a previous answer and a new answer. You have to modify the previous answer to follow the logic of the new answer. In addition, you have to talk in a similar way as the previous answer even though the logic is modified.

Example 1: The previous answer follow the exact logic logic of the new answer, thus we output modifed the answer exactly the same as the previous answer.
question: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"
new answer: 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.'\nFinal Answer: 72
previous answer: 'Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips.\n\nSo, Natalia sold a total of 48 (April) + 24 (May) = 72 clips in April and May.\n\nFinal Answer: 72 clips'
Modified answer: Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips.\n\nSo, Natalia sold a total of 48 (April) + 24 (May) = 72 clips in April and May.\n\nFinal Answer: 72 clips

Example 2: The previous answer does not follow the calculation logic of the new answer, thus we output the modified answer with the new calculation logic. We try to keep the language style from the previous answer as much as possible.
question: "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"
new answer: "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\nFinal Answer: 5"
previous answer: "Betty's parents gave her $15, and her grandparents gave her twice that amount. So, her grandparents gave her $15 * 2 = $30.\n\nBetty has received a total of $15 (from her parents) + $30 (from her grandparents) = $45.\n\nThe wallet costs $100, so Betty still needs $100 - $45 = $55 to buy the wallet.\n\nFinal Answer: $55"
Modified answer: Betty originally had half the money needed for the wallet, which is $100 / 2 = $50. Betty's parents gave her $15, and her grandparents gave her twice that amount. So, her grandparents gave her $15 * 2 = $30.\n\nAdding up, Betty has $50 (initial) + $15 (from her parents) + $30 (from her grandparents) = $95.\n\nThe wallet costs $100, so Betty still needs $100 - $95 = $5 to buy the wallet.\n\nFinal Answer: $5"

In summary, you have to keep the language style of theprevious answer while modifying it to follow the logic of the new answer. 
Please directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

question: {question}
new answer: {groundtruth}
previous answer: {previous_prediction}
Modified answer: """

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        minimum_change_data_list[i]['modified_groundtruth'] = answer
        minimum_change_data_list[i]['previous_prediction'] = previous_prediction
    return minimum_change_data_list


def minimum_change_data_creation_GSM8K_mistral_solution_optimization(data_list, use_groundtruth = True):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = predict_list[i]
        question = item['question']
        if use_groundtruth:
            ANSWER = item['answer']
        else:
            ANSWER = groundtruth_list[i]
        if use_groundtruth:
            minimum_change_prompt = \
f"""We have a question and the previous prediction. 
1. Please perform minimum change on the prediction to make it correct. 
2. Please also remove the hallucination, create your own answer when the inference steps from the prediction is empty. \
3. Please put the final answer at the end after Final Answer:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. Final Answer: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

Example1: The inference steps of the previous prediction is mostly correct, but it made some mistakes and it does not use the correct format. The minimum changed prediction correct the mistakes and keep as much original words as possible. We also remove the $ sign after 'Final Answer:' to make sure what is placed at the end is a number.
Example1-question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'
Example1-groundtruth: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n#### 5
Example1-previous prediction: Betty's parents gave her $15, and her grandparents gave her twice that amount. So, her grandparents gave her $15 * 2 = $30.\n\nBetty has received a total of $15 (from her parents) + $30 (from her grandparents) = $45.\n\nThe wallet costs $100, so Betty still needs $100 - $45 = $55 to buy the wallet.\n\nFinal Answer: $55
Example1-minimum changed prediction: Betty's parents gave her $15, and her grandparents gave her twice that amount. So, her grandparents gave her $15 * 2 = $30.\n\nBetty originally had $50, which is half of what she needs for the wallet.\n\nBetty has received a total of $15 (from her parents) + $30 (from her grandparents) + $50 (originally had) = $95.\n\nThe wallet costs $100, so Betty still needs $100 - $95 = $5 to buy the wallet.\n\nFinal Answer: 5

Example2: The previous prediction is already correct, but it does not use the correct format. It suppose to add the Final Answer: at the end with a number. So we copy the previous prediction and replace '72 clips' with '72' at the end.
Example2-question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'
Example2-groundtruth: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72
Example2-previous prediction: Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips.\n\nSo, Natalia sold a total of 48 (April) + 24 (May) = 72 clips in April and May.\n\nFinal Answer: 72 clips
Example2-minimum changed prediction: Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips.\n\nSo, Natalia sold a total of 48 (April) + 24 (May) = 72 clips in April and May.\n\nFinal Answer: 72


given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
question: {question}
Groundtruth: {ANSWER}
Previous Prediction: {previous_prediction}
Minimum Changed Prediction: """
        else:
            minimum_change_prompt = \
f"""We have a question and the previous prediction. 
1. Please perform minimum change on the prediction to make it correct. 
2. Please also remove the hallucination, create your own answer when the inference steps from the prediction is empty. \
3. Please put the final answer at the end after Final Answer:
For example:
MINIMUM_CHANGED_PREDICTION_HERE. Final Answer: 181
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."

Example1: The inference steps of the previous prediction is mostly correct, but it made some mistakes and it does not use the correct format. The minimum changed prediction correct the mistakes and keep as much original words as possible. We also remove the $ sign after 'Final Answer:' to make sure what is placed at the end is a number.
Example1-question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'
Example1-previous prediction: Betty's parents gave her $15, and her grandparents gave her twice that amount. So, her grandparents gave her $15 * 2 = $30.\n\nBetty has received a total of $15 (from her parents) + $30 (from her grandparents) = $45.\n\nThe wallet costs $100, so Betty still needs $100 - $45 = $55 to buy the wallet.\n\nFinal Answer: $55
Example1-minimum changed prediction: Betty's parents gave her $15, and her grandparents gave her twice that amount. So, her grandparents gave her $15 * 2 = $30.\n\nBetty originally had $50, which is half of what she needs for the wallet.\n\nBetty has received a total of $15 (from her parents) + $30 (from her grandparents) + $50 (originally had) = $95.\n\nThe wallet costs $100, so Betty still needs $100 - $95 = $5 to buy the wallet.\n\nFinal Answer: 5

Example2: The previous prediction is already correct, but it does not use the correct format. It suppose to add the Final Answer: at the end with a number. So we copy the previous prediction and replace '72 clips' with '72' at the end.
Example2-question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'
Example2-previous prediction: Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips.\n\nSo, Natalia sold a total of 48 (April) + 24 (May) = 72 clips in April and May.\n\nFinal Answer: 72 clips
Example2-minimum changed prediction: Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips.\n\nSo, Natalia sold a total of 48 (April) + 24 (May) = 72 clips in April and May.\n\nFinal Answer: 72


given the question and prediction, please perform minimum change to make the prediction correct.
question: {question}
Previous Prediction: {previous_prediction}
Minimum Changed Prediction: """


        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction
    return data_list

def minimum_change_data_creation_MBPP(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['prediction']
        question = item['question']

        example_quesiton_1 = \
"""You are an expert Python programmer, and you have a computer task. Your code should pass the tests:\n\nHere is your task. You only need to complete this task. Please be careful about whitespace between each line of the code.\nTask: Write a function to find the shared elements from the given two lists.\nTest Example: assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\nAnswer: PUT_YOUR_ANSWER_HERE"""
        example_previous_prediction_1 = \
"""Sure, I'd be happy to help! Here's my solution:\n```\ndef similar_elements(list1, list2):\n    return set(list(filter(lambda x: x in list2, list1)))\n```\nHere's an explanation of how the function works:\n\n1. `list1` and `list2` are the two lists we want to find the shared elements in.\n2. `list(filter(lambda x: x in list2, list1))` creates a new list containing only the elements in `list1` that are also in `list2`. The `filter()` function takes a lambda function as an argument, which in this case checks if an element is in `list2`. If it is, the element is included in the new list.\n3. `set()` converts the list of shared elements into a set, which is an unordered collection of unique elements.\n\nHere's an example of how to use the function:\n```\nprint(similar_elements((3, 4, 5, 6), (5, 7, 4, 10)))  # Output: {4, 5}\n```\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_mc_prediction_1 = \
"""Here's my solution:\n\n```\ndef similar_elements(list1, list2):\n    return list(set(list1).intersection(list2))\n```\n\nHere's an explanation of how the function works:\n\n1. `list1` and `list2` are the two lists we want to find the shared elements in.\n2. `set(list1).intersection(list2)` creates a new set containing only the elements that are common between `list1` and `list2`.\n3. `list()` converts the set of shared elements back into a list.\n\nFINAL ANSWER:\ndef similar_elements(list1, list2):\n    return list(set(list1).intersection(list2))"""

        example_quesiton_2 = \
'You are an expert Python programmer, and you have a computer task. Your code should pass the tests:\n\nHere is your task. You only need to complete this task. Please be careful about whitespace between each line of the code.\nTask: Write a python function to identify non-prime numbers.\nTest Example: assert is_not_prime(2) == False\nAnswer: PUT_YOUR_ANSWER_HERE'
        example_previous_prediction_2 = \
" Sure, I'd be happy to help! Here's my solution:\n```\ndef heap_queue_largest(numbers, n):\n    # Create a max heap using a list of lists\n    heap = [numbers[0]]\n    for i in range(1, len(numbers)):\n        if numbers[i] > heap[0]:\n            heap.append(numbers[i])\n            heap.pop(0)\n\n    # Return the n largest integers in descending order\n    return heap[:n]\n\n# Test the function\nprint(heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3))\n```\nHere's how the function works:\n\n1. We create a list of lists, `heap`, to store the integers in a max heap. We start by adding the first integer to the heap.\n2. We loop through the rest of the integers in the list, and for each one, we check if it's greater than the top element of the heap. If it is, we add it to the heap and remove the top element.\n3. Finally, we return the n largest integers in descending order by slicing the heap from the top down to the nth element.\n\nHere's the output of the test example:\n```\n[85, 75, 65]\n```\nI hope this helps! Let me know if you have any questions or need further assistance."
        example_mc_prediction_2 = \
"Sure, I'd be happy to help! Here is a Python function that identifies non-prime numbers:\n\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False\n\nHere's a breakdown of how the function works:\n\n1. We first check if `n` is less than or equal to 1, in which case it is definitely not prime.\n2. We then iterate from 2 to `n ** 0.5` (inclusive) and check if `n` is divisible by any of the numbers in this range. If it is, we return `True`, indicating that `n` is not prime.\n3. If no divisors are found, we return `False`, indicating that `n` is prime.\n\nFINAL ANSWER:\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False\n"

        example_quesiton_3 = \
"""You are an expert Python programmer, and you have a computer task. Your code should pass the tests:\n\nHere is your task. You only need to complete this task. Please be careful about whitespace between each line of the code.\nTask: Write a python function to identify non-prime numbers.\nTest Example: assert is_not_prime(2) == False\nAnswer: PUT_YOUR_ANSWER_HERE"""
        example_previous_prediction_3 = \
"""1	
Sure, I'd be happy to help! Here is a Python function that identifies non-prime numbers:\n```\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```\nHere's a breakdown of how the function works:\n\n1. We first check if `n` is less than or equal to 1, in which case it is definitely not prime.\n2. We then iterate from 2 to `n ** 0.5` (inclusive) and check if `n` is divisible by any of the numbers in this range. If it is, we return `False`, indicating that `n` is not prime.\n3. If no divisors are found, we return `True`, indicating that `n` is prime.\n\nHere are some test examples to ensure the function is working correctly:\n```\nprint(is_not_prime(2))  # Should print False\nprint(is_not_prime(3))  # Should print True\nprint(is_not_prime(4))  # Should print False\nprint(is_not_prime(5))  # Should print True\nprint(is_not_prime(6))  # Should print False\n```\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_mc_prediction_3 = \
"""Sure, I'd be happy to help! Here is a Python function that identifies non-prime numbers:\n\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False\n\nHere's a breakdown of how the function works:\n\n1. We first check if `n` is less than or equal to 1, in which case it is definitely not prime.\n2. We then iterate from 2 to `n ** 0.5` (inclusive) and check if `n` is divisible by any of the numbers in this range. If it is, we return `True`, indicating that `n` is not prime.\n3. If no divisors are found, we return `False`, indicating that `n` is prime.\n\nFINAL ANSWER:\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False"""

        minimum_change_prompt = \
f"""Given the question and the PREVIOUS PREDICTION, please perform minimum change. When I say minimum change, i mean you should try to modify as little previous prediction as possible to make it correct. Your are trying your best to preserve the original words while making the previous prediction correct. The following are the examples of minimum change. 

EXAMPLE-1: Modify the wrong part while keep as much original words as possible
EXAMPLE-1 QUESTION: {example_quesiton_1}
EXAMPLE-1 PREVIOUS PREDICTION: {example_previous_prediction_1}
EXAMPLE-1 MINIMUM CHANGED PREDICTION: {example_mc_prediction_1}

EXAMPLE-2: Modify the wrong part while keep as much original words as possible
EXAMPLE-2 QUESTION: {example_quesiton_2}
EXAMPLE-2 PREVIOUS PREDICTION: {example_previous_prediction_2}
EXAMPLE-2 MINIMUM CHANGED PREDICTION: {example_mc_prediction_2}

EXAMPLE-3: Modify the wrong part while keep as much original words as possible
EXAMPLE-3 QUESTION: {example_quesiton_3}
EXAMPLE-3 PREVIOUS PREDICTION: {example_previous_prediction_3}
EXAMPLE-3 MINIMUM CHANGED PREDICTION: {example_mc_prediction_3}



Please perform minimum change to the following previous prediction. 
QUESTION: {question}
PREVIOUS PREDICTION: {previous_prediction}

1. Please perform minimum change to the PREVIOUS PEDICTION to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the origininal PREVIOUS PREDICTION while only change where is wrong to make it correct. 
2. Directly give me the minimum changed PREVIOUS PREDICTION. Do not say anything like 'here is the modifeid prediction', or 'this is wrong because'.
3. The prediction needs to output the code in step by step, but it does not need to run test. if the prediction contains the process of running test, you should delete it.
4. put the final answer of the code at the end after the minimum changed prediction. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
5. The modified answer format would be 
MINIMUM_CHANGED_PREDICTION_HERE
FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE

6. The PREVIOUS PREDICTION has to be minimum changed in a way such it logically support the final answer. For example, the PREVIOUS PREDICTION might not solve the problem. it might just give you some hint. you have to minimum change it in such a way that it becomes logical steps guiding you to generate the FINAL ANSWER CODE """

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        # answer = answer.replace('```python', '')
        # answer = answer.replace('```', '')
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction

        print('previous_prediction', previous_prediction)
        print('answer', answer)
        print()
        print()
        print('----------------------------')
    return data_list




def minimum_change_data_creation_MBPP_march_18(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['prediction']
        question = item['question']

        
        minimum_change_prompt = \
f"""Given the question and the PREVIOUS PREDICTION, please perform minimum change. When I say minimum change, i mean you should try to modify as little previous prediction as possible to make it correct. Your are trying your best to preserve the original words while making the previous prediction correct. 

Please perform minimum change to the following previous prediction. 
QUESTION: {question}
PREVIOUS PREDICTION: {previous_prediction}

1. Please perform minimum change to the PREVIOUS PEDICTION to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the origininal PREVIOUS PREDICTION while only change where is wrong to make it correct. 
2. Directly give me the minimum changed PREVIOUS PREDICTION. Do not say anything like 'here is the modifeid prediction', or 'this is wrong because'.
3. The prediction needs to output the code in step by step, but it does not need to run test. if the prediction contains the process of running test, you should delete it.
4. put the final answer of the code at the end after the minimum changed prediction. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
5. The modified answer format would be 
MINIMUM_CHANGED_PREDICTION_HERE
FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE

6. The PREVIOUS PREDICTION has to be minimum changed in a way such it logically support the final answer. For example, the PREVIOUS PREDICTION might not solve the problem. it might just give you some hint. you have to minimum change it in such a way that it becomes logical steps guiding you to generate the FINAL ANSWER CODE """

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        # answer = answer.replace('```python', '')
        # answer = answer.replace('```', '')
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction

        print('previous_prediction', previous_prediction)
        print('answer', answer)
        print()
        print()
        print('----------------------------')
    return data_list


def minimum_change_data_creation_MBPP_march_8(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['prediction']
        question = item['question']

        example_quesiton_1 = \
"""You are an expert Python programmer, and you have a computer task. Your code should pass the tests:\n\nHere is your task. You only need to complete this task. Please be careful about whitespace between each line of the code.\nTask: Write a function to find the shared elements from the given two lists.\nTest Example: assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\nAnswer: PUT_YOUR_ANSWER_HERE"""
        example_previous_prediction_1 = \
"""Sure, I'd be happy to help! Here's my solution:\n```\ndef similar_elements(list1, list2):\n    return set(list(filter(lambda x: x in list2, list1)))\n```\nHere's an explanation of how the function works:\n\n1. `list1` and `list2` are the two lists we want to find the shared elements in.\n2. `list(filter(lambda x: x in list2, list1))` creates a new list containing only the elements in `list1` that are also in `list2`. The `filter()` function takes a lambda function as an argument, which in this case checks if an element is in `list2`. If it is, the element is included in the new list.\n3. `set()` converts the list of shared elements into a set, which is an unordered collection of unique elements.\n\nHere's an example of how to use the function:\n```\nprint(similar_elements((3, 4, 5, 6), (5, 7, 4, 10)))  # Output: {4, 5}\n```\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_mc_prediction_1 = \
"""Here's my solution:\n\n```\ndef similar_elements(list1, list2):\n    return list(set(list1).intersection(list2))\n```\n\nHere's an explanation of how the function works:\n\n1. `list1` and `list2` are the two lists we want to find the shared elements in.\n2. `set(list1).intersection(list2)` creates a new set containing only the elements that are common between `list1` and `list2`.\n3. `list()` converts the set of shared elements back into a list.\n\nFINAL ANSWER:\ndef similar_elements(list1, list2):\n    return list(set(list1).intersection(list2))"""

        example_quesiton_2 = \
'You are an expert Python programmer, and you have a computer task. Your code should pass the tests:\n\nHere is your task. You only need to complete this task. Please be careful about whitespace between each line of the code.\nTask: Write a python function to identify non-prime numbers.\nTest Example: assert is_not_prime(2) == False\nAnswer: PUT_YOUR_ANSWER_HERE'
        example_previous_prediction_2 = \
" Sure, I'd be happy to help! Here's my solution:\n```\ndef heap_queue_largest(numbers, n):\n    # Create a max heap using a list of lists\n    heap = [numbers[0]]\n    for i in range(1, len(numbers)):\n        if numbers[i] > heap[0]:\n            heap.append(numbers[i])\n            heap.pop(0)\n\n    # Return the n largest integers in descending order\n    return heap[:n]\n\n# Test the function\nprint(heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3))\n```\nHere's how the function works:\n\n1. We create a list of lists, `heap`, to store the integers in a max heap. We start by adding the first integer to the heap.\n2. We loop through the rest of the integers in the list, and for each one, we check if it's greater than the top element of the heap. If it is, we add it to the heap and remove the top element.\n3. Finally, we return the n largest integers in descending order by slicing the heap from the top down to the nth element.\n\nHere's the output of the test example:\n```\n[85, 75, 65]\n```\nI hope this helps! Let me know if you have any questions or need further assistance."
        example_mc_prediction_2 = \
"Sure, I'd be happy to help! Here is a Python function that identifies non-prime numbers:\n\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False\n\nHere's a breakdown of how the function works:\n\n1. We first check if `n` is less than or equal to 1, in which case it is definitely not prime.\n2. We then iterate from 2 to `n ** 0.5` (inclusive) and check if `n` is divisible by any of the numbers in this range. If it is, we return `True`, indicating that `n` is not prime.\n3. If no divisors are found, we return `False`, indicating that `n` is prime.\n\nFINAL ANSWER:\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False\n"

        example_quesiton_3 = \
"""You are an expert Python programmer, and you have a computer task. Your code should pass the tests:\n\nHere is your task. You only need to complete this task. Please be careful about whitespace between each line of the code.\nTask: Write a python function to identify non-prime numbers.\nTest Example: assert is_not_prime(2) == False\nAnswer: PUT_YOUR_ANSWER_HERE"""
        example_previous_prediction_3 = \
"""1	
Sure, I'd be happy to help! Here is a Python function that identifies non-prime numbers:\n```\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```\nHere's a breakdown of how the function works:\n\n1. We first check if `n` is less than or equal to 1, in which case it is definitely not prime.\n2. We then iterate from 2 to `n ** 0.5` (inclusive) and check if `n` is divisible by any of the numbers in this range. If it is, we return `False`, indicating that `n` is not prime.\n3. If no divisors are found, we return `True`, indicating that `n` is prime.\n\nHere are some test examples to ensure the function is working correctly:\n```\nprint(is_not_prime(2))  # Should print False\nprint(is_not_prime(3))  # Should print True\nprint(is_not_prime(4))  # Should print False\nprint(is_not_prime(5))  # Should print True\nprint(is_not_prime(6))  # Should print False\n```\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_mc_prediction_3 = \
"""Sure, I'd be happy to help! Here is a Python function that identifies non-prime numbers:\n\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False\n\nHere's a breakdown of how the function works:\n\n1. We first check if `n` is less than or equal to 1, in which case it is definitely not prime.\n2. We then iterate from 2 to `n ** 0.5` (inclusive) and check if `n` is divisible by any of the numbers in this range. If it is, we return `True`, indicating that `n` is not prime.\n3. If no divisors are found, we return `False`, indicating that `n` is prime.\n\nFINAL ANSWER:\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False"""

        minimum_change_prompt = \
f"""Given the question and the PREVIOUS PREDICTION, please perform minimum change. When I say minimum change, i mean you should try to modify as little previous prediction as possible to make it correct. Your are trying your best to preserve the original words while making the previous prediction correct. The following are the examples of minimum change. 

EXAMPLE-1: Modify the wrong part while keep as much original words as possible
EXAMPLE-1 QUESTION: {example_quesiton_1}
EXAMPLE-1 PREVIOUS PREDICTION: {example_previous_prediction_1}
EXAMPLE-1 MINIMUM CHANGED PREDICTION: {example_mc_prediction_1}

EXAMPLE-2: Modify the wrong part while keep as much original words as possible
EXAMPLE-2 QUESTION: {example_quesiton_2}
EXAMPLE-2 PREVIOUS PREDICTION: {example_previous_prediction_2}
EXAMPLE-2 MINIMUM CHANGED PREDICTION: {example_mc_prediction_2}

EXAMPLE-3: Modify the wrong part while keep as much original words as possible
EXAMPLE-3 QUESTION: {example_quesiton_3}
EXAMPLE-3 PREVIOUS PREDICTION: {example_previous_prediction_3}
EXAMPLE-3 MINIMUM CHANGED PREDICTION: {example_mc_prediction_3}



Please perform minimum change to the following previous aprediction. 
QUESTION: {question}
PREVIOUS PREDICTION: {previous_prediction}

1. Please perform minimum change to the PREVIOUS PEDICTION to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the origininal PREVIOUS PREDICTION while only change where is wrong to make it correct. 
2. Directly give me the minimum changed PREVIOUS PREDICTION. Do not say anything like 'here is the modifeid prediction', or 'this is wrong because'.
3. The prediction needs to output the code in step by step, but it does not need to run test. if the prediction contains the process of running test, you should delete it.
4. put the final answer of the code at the end after the minimum changed prediction. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
5. The modified answer format would be 
MINIMUM_CHANGED_PREDICTION_HERE
FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE

6. The PREVIOUS PREDICTION has to be minimum changed in a way such it logically support the final answer. For example, the PREVIOUS PREDICTION might not solve the problem. it might just give you some hint. you have to minimum change it in such a way that it becomes logical steps guiding you to generate the FINAL ANSWER CODE """

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        # answer = answer.replace('```python', '')
        # answer = answer.replace('```', '')
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction

        print('previous_prediction', previous_prediction)
        print('answer', answer)
        print()
        print()
        print('----------------------------')
    return data_list





def minimum_change_data_creation_MBPP_march_9(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['prediction']
        question = item['question']

        example_quesiton_1 = \
"""You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code.\n\nTask: Write a function to find the shared elements from the given two lists.\nTest Example: assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n\n\nYou should analyze first\n\nThen provide the answer at the end."""
        example_previous_prediction_1 = \
"""' Sure! I\'d be happy to help you with your Python programming task.\n\nSo, you want me to write a function that finds the shared elements from two given lists, right? And you\'ve provided a test example to show me the input format and the function structure.\n\nLet me take a closer look at the test example. You\'ve given two lists, (3, 4, 5, 6) and (5, 7, 4, 10), and you want to find the shared elements between them.\n\nHmmm... it looks like the shared elements between these two lists are 4 and 5. Is that correct?\n\nNow, let me think about how I can write a function to find these shared elements.\n\nOkay, I\'ve got it! Here\'s my solution:\n\ndef similar_elements(list1, list2):\n    shared_elements = set()\n    for element1 in list1:\n        for element2 in list2:\n            if element1 == element2:\n                shared_elements.add(element1)\n    return shared_elements\n\nHere\'s how the function works:\n\n1. First, we create an empty set called "shared_elements".\n2. Then, we loop through each element in list1 and list2 separately.\n3. For each element in list1, we loop through each element in list2 to find any matching elements.\n4. If we find a matching element in list2, we add that element to the "shared_elements" set.\n5. Finally, we return the "shared_elements" set.\n\nNow, let me run the test example to make sure my function works correctly.\n\nOkay, I\'ve run the test example, and it looks like my function passes! The shared elements between the two lists are indeed 4 and 5.\n\nSo, there you have it! That\'s how you can find the shared elements between two lists in Python using a set comprehension.'"""
        example_mc_prediction_1 = \
"""Sure! I'd be happy to help you with your Python programming task.\n\nSo, you want me to write a function that finds the shared elements from two given lists, right? And you've provided a test example to show me the input format and the function structure.\n\nHmmm... it looks like the shared elements between these two lists are 4 and 5. Is that correct?\n\nNow, let me think about how I can write a function to find these shared elements.\n\nHere's my solution:\n\ndef similar_elements(list1, list2):\n    return list(set(list1) & set(list2))\n\nHere's how the function works:\n\n1. First, we convert both lists into sets using the set() constructor.\n2. Then, we use the & operator to find the intersection of the two sets, which gives us the shared elements.\n3. Finally, we convert the resulting set back into a list and return it.\n\nFINAL ANSWER: \ndef similar_elements(list1, list2):\n    return list(set(list1) & set(list2))"""

        example_quesiton_2 = \
'You are an expert Python programmer, and you have a computer task. Your code should pass the tests:\n\nHere is your task. You only need to complete this task. Please be careful about whitespace between each line of the code.\nTask: Write a python function to identify non-prime numbers.\nTest Example: assert is_not_prime(2) == False\nAnswer: PUT_YOUR_ANSWER_HERE'
        example_previous_prediction_2 = \
" Sure, I'd be happy to help! Here's my solution:\n```\ndef heap_queue_largest(numbers, n):\n    # Create a max heap using a list of lists\n    heap = [numbers[0]]\n    for i in range(1, len(numbers)):\n        if numbers[i] > heap[0]:\n            heap.append(numbers[i])\n            heap.pop(0)\n\n    # Return the n largest integers in descending order\n    return heap[:n]\n\n# Test the function\nprint(heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3))\n```\nHere's how the function works:\n\n1. We create a list of lists, `heap`, to store the integers in a max heap. We start by adding the first integer to the heap.\n2. We loop through the rest of the integers in the list, and for each one, we check if it's greater than the top element of the heap. If it is, we add it to the heap and remove the top element.\n3. Finally, we return the n largest integers in descending order by slicing the heap from the top down to the nth element.\n\nHere's the output of the test example:\n```\n[85, 75, 65]\n```\nI hope this helps! Let me know if you have any questions or need further assistance."
        example_mc_prediction_2 = \
"Sure, I'd be happy to help! Here is a Python function that identifies non-prime numbers:\n\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False\n\nHere's a breakdown of how the function works:\n\n1. We first check if `n` is less than or equal to 1, in which case it is definitely not prime.\n2. We then iterate from 2 to `n ** 0.5` (inclusive) and check if `n` is divisible by any of the numbers in this range. If it is, we return `True`, indicating that `n` is not prime.\n3. If no divisors are found, we return `False`, indicating that `n` is prime.\n\nFINAL ANSWER:\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False\n"

        example_quesiton_3 = \
"""You are an expert Python programmer, and you have a computer task. Your code should pass the tests:\n\nHere is your task. You only need to complete this task. Please be careful about whitespace between each line of the code.\nTask: Write a python function to identify non-prime numbers.\nTest Example: assert is_not_prime(2) == False\nAnswer: PUT_YOUR_ANSWER_HERE"""
        example_previous_prediction_3 = \
"""1	
Sure, I'd be happy to help! Here is a Python function that identifies non-prime numbers:\n```\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```\nHere's a breakdown of how the function works:\n\n1. We first check if `n` is less than or equal to 1, in which case it is definitely not prime.\n2. We then iterate from 2 to `n ** 0.5` (inclusive) and check if `n` is divisible by any of the numbers in this range. If it is, we return `False`, indicating that `n` is not prime.\n3. If no divisors are found, we return `True`, indicating that `n` is prime.\n\nHere are some test examples to ensure the function is working correctly:\n```\nprint(is_not_prime(2))  # Should print False\nprint(is_not_prime(3))  # Should print True\nprint(is_not_prime(4))  # Should print False\nprint(is_not_prime(5))  # Should print True\nprint(is_not_prime(6))  # Should print False\n```\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_mc_prediction_3 = \
"""Sure, I'd be happy to help! Here is a Python function that identifies non-prime numbers:\n\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False\n\nHere's a breakdown of how the function works:\n\n1. We first check if `n` is less than or equal to 1, in which case it is definitely not prime.\n2. We then iterate from 2 to `n ** 0.5` (inclusive) and check if `n` is divisible by any of the numbers in this range. If it is, we return `True`, indicating that `n` is not prime.\n3. If no divisors are found, we return `False`, indicating that `n` is prime.\n\nFINAL ANSWER:\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False"""


# The following are the examples of minimum change. 

# EXAMPLE-1: Modify the wrong part while keep as much original words as possible
# EXAMPLE-1 QUESTION: {example_quesiton_1}
# EXAMPLE-1 PREVIOUS PREDICTION: {example_previous_prediction_1}
# EXAMPLE-1 MINIMUM CHANGED PREDICTION: {example_mc_prediction_1}

# EXAMPLE-2: Modify the wrong part while keep as much original words as possible
# EXAMPLE-2 QUESTION: {example_quesiton_2}
# EXAMPLE-2 PREVIOUS PREDICTION: {example_previous_prediction_2}
# EXAMPLE-2 MINIMUM CHANGED PREDICTION: {example_mc_prediction_2}

# EXAMPLE-3: Modify the wrong part while keep as much original words as possible
# EXAMPLE-3 QUESTION: {example_quesiton_3}
# EXAMPLE-3 PREVIOUS PREDICTION: {example_previous_prediction_3}
# EXAMPLE-3 MINIMUM CHANGED PREDICTION: {example_mc_prediction_3}



        minimum_change_prompt = \
f"""Given the question and the PREVIOUS PREDICTION, please perform minimum change. When I say minimum change, i mean you should try to modify as little previous prediction as possible to make it correct. Your are trying your best to preserve the original words while making the previous prediction correct. 


Please perform minimum change to the following previous aprediction. 
QUESTION: {question}
PREVIOUS PREDICTION: {previous_prediction}

1. Please perform minimum change to the PREVIOUS PEDICTION to make it correct when there is anything wrong. When i say minimum change, i mean you should try to keep the origininal PREVIOUS PREDICTION while only change where is wrong to make it correct. 
2. Directly give me the minimum changed PREVIOUS PREDICTION. Do not say anything like 'here is the modifeid prediction', or 'this is wrong because'.
3. The prediction needs to output the code in step by step, but it does not need to run test. if the prediction contains the process of running test, you should delete it.
4. put the final answer of the code at the end after the minimum changed prediction. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
5. The modified answer format would be 
MINIMUM_CHANGED_PREDICTION_HERE
FINAL ANSWER: YOUR_FINAL_CODE_ANSWER_HERE

6. The PREVIOUS PREDICTION has to be minimum changed in a way such it logically support the final answer. For example, the PREVIOUS PREDICTION might not solve the problem. it might just give you some hint. you have to minimum change it in such a way that it becomes logical steps guiding you to generate the FINAL ANSWER CODE """

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        # answer = answer.replace('```python', '')
        # answer = answer.replace('```', '')
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction

        # print('previous_prediction', previous_prediction)
        # print('answer', answer)
        # print()
        # print()
        # print('----------------------------')
    return data_list




def minimum_change_data_creation_MBPP_march_27(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['prediction']
        question = item['question']

        example_quesiton_1 = \
"""You are an expert Python programmer, and you have a computer task. Your code should pass the tests:\n\nTask: Write a function to find the shared elements from the given two lists.\nTest Example: assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n\n\nYou do not need to run the test example. It is only used to instruct the input format and the function name. Directly generate the code with no explaination."""
        example_previous_prediction_1 = \
""" Sure! Here's the code for the function `similar_elements` that finds the shared elements from two lists:\n```python\ndef similar_elements(list1, list2):\n    return set(list(filter(lambda x: x in list2, list1)))\n```\nThis function takes two lists as input, and returns a set of the elements that are present in both lists.\n\nHere's an example usage of the function:\n```python\nprint(similar_elements([3, 4, 5, 6], [5, 7, 4, 10]))  # Output: {4, 5}\n```\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_mc_prediction_1 = \
"""Final Answer: \ndef similar_elements(list1, list2):\n    return set(filter(lambda x: x in list2, list1))"""

        example_quesiton_2 = \
'You are an expert Python programmer, and you have a computer task. Your code should pass the tests:\n\nTask: Write a python function to identify non-prime numbers.\nTest Example: assert is_not_prime(2) == False\n\n\nYou do not need to run the test example. It is only used to instruct the input format and the function name. Directly generate the code with no explaination.'
        example_previous_prediction_2 = \
" Sure, here's the code for the Python function to identify non-prime numbers:\n```python\ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False\n```\nYou can use this function by passing in any integer as an argument, and it will return `True` if the input is a non-prime number and `False` otherwise.\n\nFor example, if you call the function with the argument `2`, it will return `False` because 2 is a prime number. If you call it with the argument `4`, it will return `True` because 4 is not a prime number (it can be divided by 2)."
        example_mc_prediction_2 = \
"Final Answer: \ndef is_not_prime(n):\n    if n <= 1:\n        return True\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return True\n    return False"

        example_quesiton_3 = \
"""You are an expert Python programmer, and you have a computer task. Your code should pass the tests:\n\nTask: Write a function to find the n largest integers from a given list of numbers, returned in descending order.\nTest Example: assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]\n\n\nYou do not need to run the test example. It is only used to instruct the input format and the function name. Directly generate the code with no explaination."""
        example_previous_prediction_3 = \
""" Sure! Here is the code for the function `heap_queue_largest` that finds the n largest integers from a given list of numbers, returned in descending order:\n```\ndef heap_queue_largest(numbers, n):\n    # Create a heap queue\n    heap = [(num, len(numbers) - i - 1) for i, num in enumerate(numbers)]\n    heapq.heapify(heap)\n\n    # Extract the n largest numbers from the heap\n    result = []\n    for i in range(n):\n        num, index = heapq.heappop(heap)\n        result.append(num)\n\n    return result\n```\nThis function first creates a heap queue from the input list of numbers using the `heapq.heapify` function. Then, it extracts the n largest numbers from the heap using the `heapq.heappop` function, and returns them in a list.\n\nHere is an example usage of the function:\n```\nnumbers = [25, 35, 22, 85, 14, 65, 75, 22, 58]\nresult = heap_queue_largest(numbers, 3)\nprint(result)  # Output: [85, 75, 65]\n```"""
        example_mc_prediction_3 = \
"""Final Answer: \ndef heap_queue_largest(numbers, n):\n    import heapq\n    \n    # Convert numbers to negative for max heap simulation\n    for i in range(len(numbers)):\n        numbers[i] = -numbers[i]\n    \n    # Create a max heap\n    heapq.heapify(numbers)\n    \n    # Extract the n largest numbers from the max heap, convert them back to positive\n    result = [-heapq.heappop(numbers) for _ in range(n)]\n    \n    return result"""



# The following are the examples of minimum change. 

# EXAMPLE-1: The code is already correct, so we directly copy the code and put after 'Final Answer: '
# EXAMPLE-1 QUESTION: {example_quesiton_1}
# EXAMPLE-1 PREVIOUS PREDICTION: {example_previous_prediction_1}
# EXAMPLE-1 MINIMUM CHANGED PREDICTION: {example_mc_prediction_1}

# EXAMPLE-2: Modify the wrong part while keep as much original words as possible
# EXAMPLE-2 QUESTION: {example_quesiton_2}
# EXAMPLE-2 PREVIOUS PREDICTION: {example_previous_prediction_2}
# EXAMPLE-2 MINIMUM CHANGED PREDICTION: {example_mc_prediction_2}

# EXAMPLE-3: Modify the wrong part while keep as much original words as possible
# EXAMPLE-3 QUESTION: {example_quesiton_3}
# EXAMPLE-3 PREVIOUS PREDICTION: {example_previous_prediction_3}
# EXAMPLE-3 MINIMUM CHANGED PREDICTION: {example_mc_prediction_3}


        minimum_change_prompt = \
f"""You need to extract the code from the previous prediction. 

Previous prediction: {previous_prediction}

Explaination of extract code: 
1. You have to extract code from the previous prediction. Your answer will be used as the groundtruth to pass the test, thus of course you need to first extract the code and comments without including any other words.
2. You only need to extract the final code for the fuction for the test case{item['test'][0]}. You do not care about the test cases.
3. Please directly show me the extracted code with no other words. We will run the code to test if it is correct, thus do not extract other words.
"""

        previous_code = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)


        minimum_change_prompt = \
f"""You have need to do two things. You need to perform minimum change to the code extracted from the previous prediction to make it correct.

Explaination of minimum change:
1. Given the question and the PREVIOUS CODE, please perform minimum change. When I say minimum change, i mean you should try to modify as little previous code as possible to make it correct. Your are trying your best to preserve the code and comments while making the previous code correct. 
2. Directly give me the minimum changed Code. Do not say anything like 'here is the modifeid prediction', or 'this is wrong because'.
3. put the modified code at the end after the word 'Final Answer: ' with proper spacing. The code being putted after Final Answer: should be runnable code which means You do not need to add ``` ``` or add 'python' or the words like 'here is the modified code' in front of the code .
5. The modified answer format would be 
4. You do not need to modify the code if it is already correct.
Final Answer: YOUR_FINAL_CODE_HERE



Please perform minimum change to the following previous aprediction. 
QUESTION: {question}
PREVIOUS PREDICTION: {previous_code}
"""

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        # answer = answer.replace('```python', '')
        # answer = answer.replace('```', '')
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction

        # print('previous_prediction', previous_prediction)
        # print('answer', answer)
        # print()
        # print()
        # print('----------------------------')
    return data_list






def minimum_change_data_creation_MBPP_april_5(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['prediction']
        question = item['question']

        example_quesiton_1 = \
"""You are an expert Python programmer, and you have a computer task. Your code should pass the tests:\n\nTask: Write a function to find the shared elements from the given two lists.\nTest Example: assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n\n\nYou do not need to run the test example. It is only used to instruct the input format and the function name. Directly generate the code with no explaination."""
        example_previous_prediction_1 = \
""" Sure! Here's the code for the function `similar_elements` that finds the shared elements from two lists:\n```python\ndef similar_elements(list1, list2):\n    return set(list(filter(lambda x: x in list2, list1)))\n```\nThis function takes two lists as input, and returns a set of the elements that are present in both lists.\n\nHere's an example usage of the function:\n```python\nprint(similar_elements([3, 4, 5, 6], [5, 7, 4, 10]))  # Output: {4, 5}\n```\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_mc_prediction_1 = \
"""Sure! Here's the code for the function `similar_elements` that finds the shared elements from two lists:\npython\ndef similar_elements(list1, list2):\n    return set(list(filter(lambda x: x in list2, list1)))\n\nThis function takes two lists as input, and returns a set of the elements that are present in both lists.

Final Answer: 
def similar_elements(list1, list2):
    return set(filter(lambda x: x in list2, list1))"""

        example_quesiton_2 = \
"You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. The test is only used to show you the input structure. You do not need to run the test.\n\n\nTask: Write a python function to identify non-prime numbers.\nTest Example: assert is_not_prime(2) == False\n\nPlease inference first, then provide the final answer (code) at the end, after 'Final Answer:'"
        example_previous_prediction_2 = \
"Sure, I'd be happy to help! To identify non-prime numbers, we can use a simple algorithm that checks if a number is divisible by any prime number less than or equal to its square root.\n\nHere's the algorithm in detail:\n\n1. Start with the given number n.\n2. Check if n is divisible by 2. If it is, return False (since 2 is a prime number and n is divisible by it, so n is not prime).\n3. Check if n is divisible by 3. If it is, return False (since 3 is a prime number and n is divisible by it, so n is not prime).\n4. Check if n is divisible by 5. If it is, return False (since 5 is a prime number and n is divisible by it, so n is not prime).\n5. Check if n is divisible by any prime number less than or equal to the square root of n. If it is, return False (since n is divisible by a prime number, so it is not prime).\n6. If n is not divisible by any prime number less than or equal to its square root, return True (since n is not divisible by any prime number, so it is prime).\n\nNow, let's write the Python function to implement this algorithm:\n\ndef is_not_prime(n):\n    # Step 1: Check if n is divisible by 2\n    if n % 2 == 0:\n        return False\n    # Step 2: Check if n is divisible by 3\n    if n % 3 == 0:\n        return False\n    # Step 3: Check if n is divisible by 5\n    if n % 5 == 0:\n        return False\n    # Step 4: Check if n is divisible by any prime number less than or equal to its square root\n    for p in range(2, int(n ** 0.5) + 1):\n        if n % p == 0:\n            return False\n    return True\n\nFinal Answer:\n\ndef is_not_prime(n):\n    # Step 1: Check if n is divisible by 2\n    if n % 2 == 0:\n        return False\n    # Step 2: Check if n is divisible by 3\n    if n % 3 == 0:\n        return False\n    # Step 3: Check if n is divisible by 5\n    if n % 5 == 0:\n        return False\n    # Step 4: Check if n is divisible by any prime number less than or equal to its square root\n    for p in range(2, int(n ** 0.5) + 1):\n        if n % p == 0:\n            return False\n    return True\n\nI hope this helps! Let me know if you have any questions or need further clarification."
        example_mc_prediction_2 = \
"""Sure, I'd be happy to help! To identify non-prime numbers, we can use a simple algorithm that checks if a number is divisible by any number other than 1 and itself.

Here's the algorithm in detail:

1. Start with the given number n.
2. Check if n is less than 2. If it is, it's not a prime, so return True.
3. Check if n is divisible by 2. If it is, and n is not 2, return True (since 2 is the only even prime number, any other even number is not prime).
4. Check if n is divisible by any number from 3 up to the square root of n, stepping by 2 (since even numbers were already checked). If it is, return True (since it's divisible by a number other than 1 and itself, it is not prime).
5. If n is not divisible by any such number, return False (since it's not divisible by any number other than 1 and itself, it is prime).

Now, let's write the Python function to implement this algorithm:

```python
def is_not_prime(n):
    if n < 2:
        return True  # Non-primes are numbers less than 2
    if n % 2 == 0:
        return n != 2  # If n is even and not 2, then it's not a prime
    for p in range(3, int(n ** 0.5) + 1, 2):
        if n % p == 0:
            return True  # If divisible by any number other than 1 and itself, it's not a prime
    return False  # If not divisible by any such number, it's a prime
```

I hope this helps! Let me know if you have any questions or need further clarification.
Final Answer: 
def is_not_prime(n):
    if n < 2:
        return True  # Non-primes are numbers less than 2
    if n % 2 == 0:
        return n != 2  # If n is even and not 2, then it's not a prime
    for p in range(3, int(n ** 0.5) + 1, 2):
        if n % p == 0:
            return True  # If divisible by any number other than 1 and itself, it's not a prime
    return False  # If not divisible by any such number, it's a prime"""

        example_quesiton_3 = \
"""You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. The test is only used to show you the input structure. You do not need to run the test.\n\n\nTask: Write a function to find the n largest integers from a given list of numbers, returned in descending order.\nTest Example: assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]\n\nPlease inference first, then provide the final answer (code) at the end, after 'Final Answer:'"""
        example_previous_prediction_3 = \
"""Sure! I'd be happy to help you with this task.\n\nBased on the test example provided, it seems that the function `heap_queue_largest` should take a list of integers as input and return the n largest integers from the list, in descending order. The test example shows that the function should return the integers 85, 75, and 65 when given the list `[25, 35, 22, 85, 14, 65, 75, 22, 58]` and `n=3`.\n\nHere's my inference of the function's structure:\n\n1. The function should take a list of integers as input.\n2. It should return a list of integers, where the integers are the n largest integers from the input list, in descending order.\n3. The function should use a heap data structure to keep track of the largest integers.\n4. The function should use a queue data structure to keep track of the integers that need to be processed.\n5. The function should pop the largest integer from the heap and add it to the queue when the queue is not full.\n6. The function should pop the smallest integer from the queue and add it to the output list when the output list is not full.\n7. The function should repeat steps 5 and 6 until the queue is empty and the output list is full.\n\nNow, here's the final answer:\n\ndef heap_queue_largest(arr, n):\n```\n# Create an empty heap and queue\nheap = [(0, arr[0])]\nqueue = [arr[0]]\n\n# Add the rest of the elements to the heap and queue\nfor i in range(1, len(arr)):\n    # Add the current element to the heap\n    heap.append((arr[i], arr[i]))\n    \n    # Remove the smallest element from the heap and add it to the queue\n    while heap[0][1] < arr[i]:\n        heap.pop(0)\n        queue.pop(0)\n    \n    # Add the current element to the end of the queue\n    queue.append(arr[i])\n\n# Return the n largest elements from the queue, in descending order\nreturn queue[:n]\n\nFinal Answer:\n\nassert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]\n```\nThis function uses a heap data structure to keep track of the largest integers and a queue data structure to keep track of the integers that need to be processed. It adds the elements to the heap and queue in the correct order, and then returns the n largest integers from the queue, in descending order."""
        example_mc_prediction_3 = \
"""Sure! I'd be happy to help you with this task.\n\nBased on the test example provided, it seems that the function `heap_queue_largest` should take a list of integers as input and return the n largest integers from the list, in descending order.\n\nHere's my inference of the function's structure:\n\n1. The function should take a list of integers and an integer n as input.\n2. It should use the heapq module to find the n largest elements in the list.\n3. The elements should then be returned in descending order.\n\nNow, here's the function:\n\n```python\nimport heapq\n\ndef heap_queue_largest(arr, n):\n    # Find the n largest elements and return them in descending order\n    return sorted(heapq.nlargest(n, arr), reverse=True)\n```\n\nThe function `heap_queue_largest` uses the `heapq` module to find the n largest elements in the input array and then returns them sorted in descending order.\n\nFinal Answer:\n\nimport heapq\n\ndef heap_queue_largest(arr, n):\n    return sorted(heapq.nlargest(n, arr), reverse=True)"""



# The following are the examples of minimum change. 

# EXAMPLE-1: The code is already correct, so we directly copy the code and put after 'Final Answer: '
# EXAMPLE-1 QUESTION: {example_quesiton_1}
# EXAMPLE-1 PREVIOUS PREDICTION: {example_previous_prediction_1}
# EXAMPLE-1 MINIMUM CHANGED PREDICTION: {example_mc_prediction_1}

# EXAMPLE-2: Modify the wrong part while keep as much original words as possible
# EXAMPLE-2 QUESTION: {example_quesiton_2}
# EXAMPLE-2 PREVIOUS PREDICTION: {example_previous_prediction_2}
# EXAMPLE-2 MINIMUM CHANGED PREDICTION: {example_mc_prediction_2}

# EXAMPLE-3: Modify the wrong part while keep as much original words as possible
# EXAMPLE-3 QUESTION: {example_quesiton_3}
# EXAMPLE-3 PREVIOUS PREDICTION: {example_previous_prediction_3}
# EXAMPLE-3 MINIMUM CHANGED PREDICTION: {example_mc_prediction_3}





        minimum_change_prompt = \
f"""You have need to do two things. You need to perform minimum change to the previous prediction to make it correct. Not only make sure the code is correct, but also make sure the inference, explaination, etc from the previous prediction are correct.

EXAMPLE-1: The code is already correct, so we directly copy the previous prediction and put the code after 'Final Answer: ' we also keep as much original words as possible. we remove the part that the previous prediction run test. 
EXAMPLE-1 QUESTION: {example_quesiton_1}
EXAMPLE-1 PREVIOUS PREDICTION: {example_previous_prediction_1}
EXAMPLE-1 MINIMUM CHANGED PREDICTION: {example_mc_prediction_1}

EXAMPLE-2: Modify the wrong part while keep as much original words as possible. 
EXAMPLE-2 QUESTION: {example_quesiton_2}
EXAMPLE-2 PREVIOUS PREDICTION: {example_previous_prediction_2}
EXAMPLE-2 MINIMUM CHANGED PREDICTION: {example_mc_prediction_2}

EXAMPLE-3: Modify the wrong part while keep as much original words as possible. it remove the part that run the test. It place the code after 'Final Answer:'
EXAMPLE-3 QUESTION: {example_quesiton_3}
EXAMPLE-3 PREVIOUS PREDICTION: {example_previous_prediction_3}
EXAMPLE-3 MINIMUM CHANGED PREDICTION: {example_mc_prediction_3}

Explaination of minimum change:
1. Given the question and the previous prediction, please perform minimum change. When I say minimum change, i mean you should try to modify as little previous prediction as possible to make it correct. Your are trying your best to preserve as much original words as possible.
2. the prediction does not required to run the test. If you see it run the test, plese remove it.
3. Directly give me the minimum changed prediction. Do not say anything like 'here is the modifeid prediction', or 'this is wrong because'.
4. put the modified code at the end after the word 'Final Answer: ' with proper spacing. The code being putted after Final Answer: should be runnable code which means You do not need to add ``` ``` or add 'python' or the words like 'here is the modified code' in front of the code .
5. You do not need to modify the previous prediction if it is already correct. Please be careful about the spacing in code.

Please perform minimum change to the following previous aprediction. 
QUESTION: {question}
PREVIOUS PREDICTION: {previous_prediction}
MINIMUM CHANGED PREDICTION:
"""

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        # answer = answer.replace('```python', '')
        # answer = answer.replace('```', '')
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction

        # print('previous_prediction', previous_prediction)
        # print('answer', answer)
        # print()
        # print()
        # print('----------------------------')
    return data_list




def minimum_change_data_creation_MBPP_april_7(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['prediction']
        question = item['question']

        example_quesiton_1 = \
"""You are an expert Python programmer, and you have a computer task. Your code should pass the tests:\n\nTask: Write a function to find the shared elements from the given two lists.\nTest Example: assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n\n\nYou do not need to run the test example. It is only used to instruct the input format and the function name. Directly generate the code with no explaination."""
        example_previous_prediction_1 = \
""" Sure! Here's the code for the function `similar_elements` that finds the shared elements from two lists:\n```python\ndef similar_elements(list1, list2):\n    return set(list(filter(lambda x: x in list2, list1)))\n```\nThis function takes two lists as input, and returns a set of the elements that are present in both lists.\n\nHere's an example usage of the function:\n```python\nprint(similar_elements([3, 4, 5, 6], [5, 7, 4, 10]))  # Output: {4, 5}\n```\nI hope this helps! Let me know if you have any questions or need further assistance."""
        example_mc_prediction_1 = \
"""Sure! Here's the code for the function `similar_elements` that finds the shared elements from two lists:
Final Answer: 
def similar_elements(list1, list2):
    return set(filter(lambda x: x in list2, list1))"""

        example_quesiton_2 = \
"You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. The test is only used to show you the input structure. You do not need to run the test.\n\n\nTask: Write a python function to identify non-prime numbers.\nTest Example: assert is_not_prime(2) == False\n\nPlease inference first, then provide the final answer (code) at the end, after 'Final Answer:'"
        example_previous_prediction_2 = \
"Sure, I'd be happy to help! To identify non-prime numbers, we can use a simple algorithm that checks if a number is divisible by any prime number less than or equal to its square root.\n\nHere's the algorithm in detail:\n\n1. Start with the given number n.\n2. Check if n is divisible by 2. If it is, return False (since 2 is a prime number and n is divisible by it, so n is not prime).\n3. Check if n is divisible by 3. If it is, return False (since 3 is a prime number and n is divisible by it, so n is not prime).\n4. Check if n is divisible by 5. If it is, return False (since 5 is a prime number and n is divisible by it, so n is not prime).\n5. Check if n is divisible by any prime number less than or equal to the square root of n. If it is, return False (since n is divisible by a prime number, so it is not prime).\n6. If n is not divisible by any prime number less than or equal to its square root, return True (since n is not divisible by any prime number, so it is prime).\n\nNow, let's write the Python function to implement this algorithm:\n\ndef is_not_prime(n):\n    # Step 1: Check if n is divisible by 2\n    if n % 2 == 0:\n        return False\n    # Step 2: Check if n is divisible by 3\n    if n % 3 == 0:\n        return False\n    # Step 3: Check if n is divisible by 5\n    if n % 5 == 0:\n        return False\n    # Step 4: Check if n is divisible by any prime number less than or equal to its square root\n    for p in range(2, int(n ** 0.5) + 1):\n        if n % p == 0:\n            return False\n    return True\n\nFinal Answer:\n\ndef is_not_prime(n):\n    # Step 1: Check if n is divisible by 2\n    if n % 2 == 0:\n        return False\n    # Step 2: Check if n is divisible by 3\n    if n % 3 == 0:\n        return False\n    # Step 3: Check if n is divisible by 5\n    if n % 5 == 0:\n        return False\n    # Step 4: Check if n is divisible by any prime number less than or equal to its square root\n    for p in range(2, int(n ** 0.5) + 1):\n        if n % p == 0:\n            return False\n    return True\n\nI hope this helps! Let me know if you have any questions or need further clarification."
        example_mc_prediction_2 = \
"""Sure, I'd be happy to help! To identify non-prime numbers, we can use a simple algorithm that checks if a number is divisible by any number other than 1 and itself.

Here's the algorithm in detail:

1. Start with the given number n.
2. Check if n is less than 2. If it is, it's not a prime, so return True.
3. Check if n is divisible by 2. If it is, and n is not 2, return True (since 2 is the only even prime number, any other even number is not prime).
4. Check if n is divisible by any number from 3 up to the square root of n, stepping by 2 (since even numbers were already checked). If it is, return True (since it's divisible by a number other than 1 and itself, it is not prime).
5. If n is not divisible by any such number, return False (since it's not divisible by any number other than 1 and itself, it is prime).

Now, let's write the Python function to implement this algorithm:
Final Answer: 
def is_not_prime(n):
    if n < 2:
        return True  # Non-primes are numbers less than 2
    if n % 2 == 0:
        return n != 2  # If n is even and not 2, then it's not a prime
    for p in range(3, int(n ** 0.5) + 1, 2):
        if n % p == 0:
            return True  # If divisible by any number other than 1 and itself, it's not a prime
    return False  # If not divisible by any such number, it's a prime"""

        example_quesiton_3 = \
"""You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. The test is only used to show you the input structure. You do not need to run the test.\n\n\nTask: Write a function to find the n largest integers from a given list of numbers, returned in descending order.\nTest Example: assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]\n\nPlease inference first, then provide the final answer (code) at the end, after 'Final Answer:'"""
        example_previous_prediction_3 = \
"""Sure! I'd be happy to help you with this task.\n\nBased on the test example provided, it seems that the function `heap_queue_largest` should take a list of integers as input and return the n largest integers from the list, in descending order. The test example shows that the function should return the integers 85, 75, and 65 when given the list `[25, 35, 22, 85, 14, 65, 75, 22, 58]` and `n=3`.\n\nHere's my inference of the function's structure:\n\n1. The function should take a list of integers as input.\n2. It should return a list of integers, where the integers are the n largest integers from the input list, in descending order.\n3. The function should use a heap data structure to keep track of the largest integers.\n4. The function should use a queue data structure to keep track of the integers that need to be processed.\n5. The function should pop the largest integer from the heap and add it to the queue when the queue is not full.\n6. The function should pop the smallest integer from the queue and add it to the output list when the output list is not full.\n7. The function should repeat steps 5 and 6 until the queue is empty and the output list is full.\n\nNow, here's the final answer:\n\ndef heap_queue_largest(arr, n):\n```\n# Create an empty heap and queue\nheap = [(0, arr[0])]\nqueue = [arr[0]]\n\n# Add the rest of the elements to the heap and queue\nfor i in range(1, len(arr)):\n    # Add the current element to the heap\n    heap.append((arr[i], arr[i]))\n    \n    # Remove the smallest element from the heap and add it to the queue\n    while heap[0][1] < arr[i]:\n        heap.pop(0)\n        queue.pop(0)\n    \n    # Add the current element to the end of the queue\n    queue.append(arr[i])\n\n# Return the n largest elements from the queue, in descending order\nreturn queue[:n]\n\nFinal Answer:\n\nassert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]\n```\nThis function uses a heap data structure to keep track of the largest integers and a queue data structure to keep track of the integers that need to be processed. It adds the elements to the heap and queue in the correct order, and then returns the n largest integers from the queue, in descending order."""
        example_mc_prediction_3 = \
"""Sure! I'd be happy to help you with this task.\n\nBased on the test example provided, it seems that the function `heap_queue_largest` should take a list of integers as input and return the n largest integers from the list, in descending order.\n\nHere's my inference of the function's structure:\n\n1. The function should take a list of integers and an integer n as input.\n2. It should use the heapq module to find the n largest elements in the list.\n3. The elements should then be returned in descending order.\n\nNow, here's the function:\n\nFinal Answer:\n\nimport heapq\n\ndef heap_queue_largest(arr, n):\n    return sorted(heapq.nlargest(n, arr), reverse=True)"""



        minimum_change_prompt = \
f"""You have need to do two things. You need to perform minimum change to the previous prediction to make it correct. Not only make sure the code is correct, but also make sure the inference, explaination, etc from the previous prediction are correct.

EXAMPLE-1: The code is already correct, so we directly copy the previous prediction and put the code after 'Final Answer: ' we also keep as much original words as possible. we remove the part that the previous prediction run test.  Please notice that we make sure the code after Final Answer is directly runnable, thus we did not add ``` or 'python' in front of def similar_elements(list1, list2):
EXAMPLE-1 QUESTION: {example_quesiton_1}
EXAMPLE-1 PREVIOUS PREDICTION: {example_previous_prediction_1}
EXAMPLE-1 MINIMUM CHANGED PREDICTION: {example_mc_prediction_1}

EXAMPLE-2: Modify the wrong part while keep as much original words as possible. Please notice that we make sure the code after Final Answer is directly runnable, thus we did not add ``` or 'python' in front of def is_not_prime(n):
EXAMPLE-2 QUESTION: {example_quesiton_2}
EXAMPLE-2 PREVIOUS PREDICTION: {example_previous_prediction_2}
EXAMPLE-2 MINIMUM CHANGED PREDICTION: {example_mc_prediction_2}

EXAMPLE-3: Modify the wrong part while keep as much original words as possible. it remove the part that run the test. It place the code after 'Final Answer:' Please notice that we make sure the code after Final Answer is directly runnable, thus we did not add ``` or 'python' in front of import heapq
EXAMPLE-3 QUESTION: {example_quesiton_3}
EXAMPLE-3 PREVIOUS PREDICTION: {example_previous_prediction_3}
EXAMPLE-3 MINIMUM CHANGED PREDICTION: {example_mc_prediction_3}

Explaination of minimum change:
1. Given the question and the previous prediction, please perform minimum change. When I say minimum change, i mean you should try to modify as little previous prediction as possible to make it correct. Your are trying your best to preserve as much original words as possible.
2. the prediction does not required to run the test. If you see it run the test, plese remove it.
3. Directly give me the minimum changed prediction. Do not say anything like 'here is the modifeid prediction', or 'this is wrong because'.
4. put the modified code at the end after the word 'Final Answer: ' with proper spacing. The code being putted after Final Answer: should be runnable code which means You do not need to add ``` ``` or add 'python' or the words like 'here is the modified code' in front of the code , because we will directly extract the content after Final Answer: which means the content after Final Answer has to be runnable.
5. You do not need to modify the previous prediction if it is already correct. Please be careful about the spacing in code.

Please perform minimum change to the following previous aprediction. 
QUESTION: {question}
PREVIOUS PREDICTION: {previous_prediction}
MINIMUM CHANGED PREDICTION:
"""

        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        # answer = answer.replace('```python', '')
        # answer = answer.replace('```', '')
        data_list[i]['answer'] = answer
        data_list[i]['previous_prediction'] = previous_prediction

        # print('previous_prediction', previous_prediction)
        # print('answer', answer)
        # print()
        # print()
        # print('----------------------------')
    return data_list



def minimum_change_data_creation_MBPP_not_changing_anything(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['prediction']
        original_question = item['original_question']
        question = item['question']

        example_quesiton_1 = \
"""You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. The test is only used to show you the input structure. You do not need to run the test.\n\n\nTask: Write a function to find the shared elements from the given two lists.\nTest Example: assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))"""
        example_previous_prediction_1 = \
" Sure! I'd be happy to help you with that.\n\nSo, based on the test example provided, it seems that we need to find the shared elements between two lists. The function we need to write is called `similar_elements`, and it should return a set of the shared elements.\n\nHere's my inference:\n\nThe function `similar_elements` should take two lists as input, and it should return a set of the elements that are present in both lists. To do this, we can use the `set` data structure, which can uniquely identify the elements in the lists.\n\nHere's the final answer:\n\n```\ndef similar_elements(list1, list2):\n    return set(list1).intersection(list2)\n```\n\nExplanation:\n\nThe `set` data structure can uniquely identify the elements in the lists, so we can use it to find the shared elements. The `intersection` method of the set will return a new set containing all the elements that are present in both lists.\n\nFor example, if we call the function with the input lists `(3, 4, 5, 6)` and `(5, 7, 4, 10)`, the function will return the set `{(4, 5)}`, which contains the shared elements between the two lists.\n\nI hope this helps! Let me know if you have any questions or need further clarification."
        example_mc_prediction_1 = \
"""Sure! I'd be happy to help you with that.

So, based on the test example provided, it seems that we need to find the shared elements between two lists. The function we need to write is called `similar_elements`, and it should return a set of the shared elements.

Here's my inference:

The function `similar_elements` should take two lists as input, and it should return a set of the elements that are present in both lists. To do this, we can use the `set` data structure, which can uniquely identify the elements in the lists.

Here's the final answer:

```python
def similar_elements(list1, list2):
    return set(list1).intersection(list2)
```

Explanation:

The `set` data structure can uniquely identify the elements in the lists, so we can use it to find the shared elements. The `intersection` method of the set will return a new set containing all the elements that are present in both lists.

For example, if we call the function with the input lists `(3, 4, 5, 6)` and `(5, 7, 4, 10)`, the function will return the set `{4, 5}`, which contains the shared elements between the two lists.

I hope this helps! Let me know if you have any questions or need further clarification."""

        example_quesiton_2 = \
"You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. The test is only used to show you the input structure. You do not need to run the test.\n\n\nTask: Write a python function to identify non-prime numbers.\nTest Example: assert is_not_prime(2) == False\n\nPlease inference first, then provide the final answer (code) at the end.'"
        example_previous_prediction_2 = \
"Sure, I'd be happy to help! To identify non-prime numbers, we can use a simple algorithm that checks if a number is divisible by any prime number less than or equal to its square root.\n\nHere's the algorithm in detail:\n\n1. Start with the given number n.\n2. Check if n is divisible by 2. If it is, return False (since 2 is a prime number and n is divisible by it, so n is not prime).\n3. Check if n is divisible by 3. If it is, return False (since 3 is a prime number and n is divisible by it, so n is not prime).\n4. Check if n is divisible by 5. If it is, return False (since 5 is a prime number and n is divisible by it, so n is not prime).\n5. Check if n is divisible by any prime number less than or equal to the square root of n. If it is, return False (since n is divisible by a prime number, so it is not prime).\n6. If n is not divisible by any prime number less than or equal to its square root, return True (since n is not divisible by any prime number, so it is prime).\n\nNow, let's write the Python function to implement this algorithm:\n\ndef is_not_prime(n):\n    # Step 1: Check if n is divisible by 2\n    if n % 2 == 0:\n        return False\n    # Step 2: Check if n is divisible by 3\n    if n % 3 == 0:\n        return False\n    # Step 3: Check if n is divisible by 5\n    if n % 5 == 0:\n        return False\n    # Step 4: Check if n is divisible by any prime number less than or equal to its square root\n    for p in range(2, int(n ** 0.5) + 1):\n        if n % p == 0:\n            return False\n    return True\n\nI hope this helps! Let me know if you have any questions or need further clarification."
        example_mc_prediction_2 = \
"""Sure, I'd be happy to help! To identify non-prime numbers, we can use a simple algorithm that checks if a number is divisible by any prime number less than or equal to its square root.

Here's the algorithm in detail:

1. Start with the given number n.
2. Check if n is divisible by 2. If it is, return True (since n is not prime).
3. Check if n is divisible by 3. If it is, return True (since n is not prime).
4. Check if n is divisible by any number from 2 up to the square root of n. If it is, return True (since n is not prime).
5. If n is not divisible by any of the numbers checked, return False (indicating that n is prime).

Now, let's slightly modify the Python function to implement this algorithm correctly:

```python
def is_not_prime(n):
    # Step 1: Check if n is less than 2
    if n < 2:
        return True
    # Step 2: Check if n is divisible by 2
    if n % 2 == 0 and n > 2:
        return True
    # Step 3: Check if n is divisible by any number up to its square root
    for p in range(3, int(n ** 0.5) + 1, 2):
        if n % p == 0:
            return True
    return False
```

I hope this helps! Let me know if you have any questions or need further clarification."""

        example_quesiton_3 = \
"""'You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. The test is only used to show you the input structure. You do not need to run the test.\n\n\nTask: Write a function to find the shared elements from the given two lists.\nTest Example: assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))'"""
        example_previous_prediction_3 = \
"""Sure! I'd be happy to help you with this task.\n\nBased on the test example provided, it seems that the function `heap_queue_largest` should take a list of integers as input and return the n largest integers from the list, in descending order. The test example shows that the function should return the integers 85, 75, and 65 when given the list `[25, 35, 22, 85, 14, 65, 75, 22, 58]` and `n=3`.\n\nHere's my inference of the function's structure:\n\n1. The function should take a list of integers as input.\n2. It should return a list of integers, where the integers are the n largest integers from the input list, in descending order.\n3. The function should use a heap data structure to keep track of the largest integers.\n4. The function should use a queue data structure to keep track of the integers that need to be processed.\n5. The function should pop the largest integer from the heap and add it to the queue when the queue is not full.\n6. The function should pop the smallest integer from the queue and add it to the output list when the output list is not full.\n7. The function should repeat steps 5 and 6 until the queue is empty and the output list is full.\n\nNow, here's the final answer:\n\ndef heap_queue_largest(arr, n):\n```\n# Create an empty heap and queue\nheap = [(0, arr[0])]\nqueue = [arr[0]]\n\n# Add the rest of the elements to the heap and queue\nfor i in range(1, len(arr)):\n    # Add the current element to the heap\n    heap.append((arr[i], arr[i]))\n    \n    # Remove the smallest element from the heap and add it to the queue\n    while heap[0][1] < arr[i]:\n        heap.pop(0)\n        queue.pop(0)\n    \n    # Add the current element to the end of the queue\n    queue.append(arr[i])\n\n# Return the n largest elements from the queue, in descending order\nreturn queue[:n]\n\nFinal Answer:\n\nassert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]\n```\nThis function uses a heap data structure to keep track of the largest integers and a queue data structure to keep track of the integers that need to be processed. It adds the elements to the heap and queue in the correct order, and then returns the n largest integers from the queue, in descending order."""
        example_mc_prediction_3 = \
"""Sure! I'd be happy to help you with this task.\n\nBased on the test example provided, it seems that the function `heap_queue_largest` should take a list of integers as input and return the n largest integers from the list, in descending order.\n\nHere's my inference of the function's structure:\n\n1. The function should take a list of integers and an integer n as input.\n2. It should use the heapq module to find the n largest elements in the list.\n3. The elements should then be returned in descending order.\n\nNow, here's the function:\n\nFinal Answer:\n\nimport heapq\n\ndef heap_queue_largest(arr, n):\n    return sorted(heapq.nlargest(n, arr), reverse=True)"""



# EXAMPLE-3: Modify the wrong part while keep as much original words as possible. it remove the part that run the test. It place the code after 'Final Answer:' Please notice that we make sure the code after Final Answer is directly runnable, thus we did not add ``` or 'python' in front of import heapq
# EXAMPLE-3 QUESTION: {example_quesiton_3}
# EXAMPLE-3 PREVIOUS PREDICTION: {example_previous_prediction_3}
# EXAMPLE-3 MINIMUM CHANGED PREDICTION: {example_mc_prediction_3}

        modified_question = question.replace(", after 'Final Answer:'\n", '.')

        minimum_change_prompt = \
f"""You need to perform minimum change to the previous prediction to make it correct. Not only make sure the code is correct, but also make sure the inference, explaination, etc from the previous prediction are correct.

Explaination of minimum change:
1. Given the question and the previous prediction, please perform minimum change. When I say minimum change, i mean you should try to modify as little previous prediction as possible to make it correct. Your are trying your best to preserve as much original words as possible.
2. the prediction does not required to run the test. If you see it run the test, plese remove it.
3. Directly give me the minimum changed prediction. Do not say anything like 'here is the modifeid prediction', or 'this is wrong because'.
4. You do not need to modify the previous prediction if it is already correct. Please be careful about the spacing in code.

EXAMPLE-1: The code is already correct, so we directly copy the previous prediction so as to keep as much original words as possible. 
EXAMPLE-1 QUESTION: {example_quesiton_1}
EXAMPLE-1 PREVIOUS PREDICTION: {example_previous_prediction_1}
EXAMPLE-1 MINIMUM CHANGED PREDICTION: {example_mc_prediction_1}

EXAMPLE-2: Modify the wrong part while keep as much original words as possible. 
EXAMPLE-2 QUESTION: {example_quesiton_2}
EXAMPLE-2 PREVIOUS PREDICTION: {example_previous_prediction_2}
EXAMPLE-2 MINIMUM CHANGED PREDICTION: {example_mc_prediction_2}


Please perform minimum change to the following previous aprediction. 
QUESTION: {modified_question}
PREVIOUS PREDICTION: {previous_prediction}
MINIMUM CHANGED PREDICTION:
"""

        modified_prediction_step1 = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        a = 1
        
        minimum_change_prompt = \
f"""You need to extract the code from the previous prediction. The prediction answer the QUESTION.
QUESTION: {original_question}
Previous prediction: {modified_prediction_step1}

Explaination of extract code: 
1. You have to extract code from the previous prediction. Your answer will be used as the groundtruth to pass the test, thus of course you need to first extract the code and comments without including any other words.
2. You only need to extract the final code for the fuction for the test case{item['test'][0]}. You do not care about the test cases.
3. Please directly show me the extracted code with no other words. We will run the code to test if it is correct, thus do not extract other words.
"""
        code = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        code = code.replace('```python', '')
        code = code.replace('```', '')

        # index = modified_prediction_step1.lower().rfind('final answer:')
        # if index != -1:
        #     modified_prediction_step1 = modified_prediction_step1[:index]
        #     answer_final = f"""{modified_prediction_step1}Final Answer: {code}"""
        # else:
        answer_final = f"""{modified_prediction_step1}

Final Answer: {code}"""
        data_list[i]['answer'] = answer_final
        data_list[i]['previous_prediction'] = previous_prediction

        # print('previous_prediction', previous_prediction)
        # print('answer', answer)
        # print()
        # print()
        # print('----------------------------')
    return data_list


def minimum_change_data_creation_MBPP_not_changing_anything_apr_7(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['prediction']
        original_question = item['original_question']
        question = item['question']

        example_quesiton_1 = \
"""You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. The test is only used to show you the input structure. You do not need to run the test.\n\n\nTask: Write a function to find the shared elements from the given two lists.\nTest Example: assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))"""
        example_previous_prediction_1 = \
" Sure! I'd be happy to help you with that.\n\nSo, based on the test example provided, it seems that we need to find the shared elements between two lists. The function we need to write is called `similar_elements`, and it should return a set of the shared elements.\n\nHere's my inference:\n\nThe function `similar_elements` should take two lists as input, and it should return a set of the elements that are present in both lists. To do this, we can use the `set` data structure, which can uniquely identify the elements in the lists.\n\nHere's the final answer:\n\n```\ndef similar_elements(list1, list2):\n    return set(list1).intersection(list2)\n```\n\nExplanation:\n\nThe `set` data structure can uniquely identify the elements in the lists, so we can use it to find the shared elements. The `intersection` method of the set will return a new set containing all the elements that are present in both lists.\n\nFor example, if we call the function with the input lists `(3, 4, 5, 6)` and `(5, 7, 4, 10)`, the function will return the set `{(4, 5)}`, which contains the shared elements between the two lists.\n\nI hope this helps! Let me know if you have any questions or need further clarification."
        example_mc_prediction_1 = \
"""Sure! I'd be happy to help you with that.

So, based on the test example provided, it seems that we need to find the shared elements between two lists. The function we need to write is called `similar_elements`, and it should return a set of the shared elements.

Here's my inference:

The function `similar_elements` should take two lists as input, and it should return a set of the elements that are present in both lists. To do this, we can use the `set` data structure, which can uniquely identify the elements in the lists.

Here's the final answer:

```python
def similar_elements(list1, list2):
    return set(list1).intersection(list2)
```

Explanation:

The `set` data structure can uniquely identify the elements in the lists, so we can use it to find the shared elements. The `intersection` method of the set will return a new set containing all the elements that are present in both lists.

For example, if we call the function with the input lists `(3, 4, 5, 6)` and `(5, 7, 4, 10)`, the function will return the set `{4, 5}`, which contains the shared elements between the two lists.

I hope this helps! Let me know if you have any questions or need further clarification."""

        example_quesiton_2 = \
"You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. The test is only used to show you the input structure. You do not need to run the test.\n\n\nTask: Write a python function to identify non-prime numbers.\nTest Example: assert is_not_prime(2) == False\n\nPlease inference first, then provide the final answer (code) at the end.'"
        example_previous_prediction_2 = \
"Sure, I'd be happy to help! To identify non-prime numbers, we can use a simple algorithm that checks if a number is divisible by any prime number less than or equal to its square root.\n\nHere's the algorithm in detail:\n\n1. Start with the given number n.\n2. Check if n is divisible by 2. If it is, return False (since 2 is a prime number and n is divisible by it, so n is not prime).\n3. Check if n is divisible by 3. If it is, return False (since 3 is a prime number and n is divisible by it, so n is not prime).\n4. Check if n is divisible by 5. If it is, return False (since 5 is a prime number and n is divisible by it, so n is not prime).\n5. Check if n is divisible by any prime number less than or equal to the square root of n. If it is, return False (since n is divisible by a prime number, so it is not prime).\n6. If n is not divisible by any prime number less than or equal to its square root, return True (since n is not divisible by any prime number, so it is prime).\n\nNow, let's write the Python function to implement this algorithm:\n\ndef is_not_prime(n):\n    # Step 1: Check if n is divisible by 2\n    if n % 2 == 0:\n        return False\n    # Step 2: Check if n is divisible by 3\n    if n % 3 == 0:\n        return False\n    # Step 3: Check if n is divisible by 5\n    if n % 5 == 0:\n        return False\n    # Step 4: Check if n is divisible by any prime number less than or equal to its square root\n    for p in range(2, int(n ** 0.5) + 1):\n        if n % p == 0:\n            return False\n    return True\n\nI hope this helps! Let me know if you have any questions or need further clarification."
        example_mc_prediction_2 = \
"""Sure, I'd be happy to help! To identify non-prime numbers, we can use a simple algorithm that checks if a number is divisible by any prime number less than or equal to its square root.

Here's the algorithm in detail:

1. Start with the given number n.
2. Check if n is divisible by 2. If it is, return True (since n is not prime).
3. Check if n is divisible by 3. If it is, return True (since n is not prime).
4. Check if n is divisible by any number from 2 up to the square root of n. If it is, return True (since n is not prime).
5. If n is not divisible by any of the numbers checked, return False (indicating that n is prime).

Now, let's slightly modify the Python function to implement this algorithm correctly:

```python
def is_not_prime(n):
    # Step 1: Check if n is less than 2
    if n < 2:
        return True
    # Step 2: Check if n is divisible by 2
    if n % 2 == 0 and n > 2:
        return True
    # Step 3: Check if n is divisible by any number up to its square root
    for p in range(3, int(n ** 0.5) + 1, 2):
        if n % p == 0:
            return True
    return False
```

I hope this helps! Let me know if you have any questions or need further clarification."""

        modified_question = question.replace("Please inference first, then provide the final answer (code) at the end, after 'Final Answer:'\n", '.')

        minimum_change_prompt = \
f"""You need to perform minimum change to the previous prediction to make it correct. Not only make sure the code is correct, but also make sure the inference, explaination, etc from the previous prediction are correct.

Explaination of minimum change:
1. Given the question and the previous prediction, please perform minimum change. When I say minimum change, i mean you should try to modify as little previous prediction as possible to make it correct. Your are trying your best to preserve as much original words as possible.
2. the prediction does not required to run the test. If you see it run the test, plese remove it.
3. Directly give me the minimum changed prediction. Do not say anything like 'here is the modifeid prediction', or 'this is wrong because'.
4. You do not need to modify the previous prediction if it is already correct. Please be careful about the spacing in code.

EXAMPLE-1: The code is already correct, so we directly copy the previous prediction so as to keep as much original words as possible. 
EXAMPLE-1 QUESTION: {example_quesiton_1}
EXAMPLE-1 PREVIOUS PREDICTION: {example_previous_prediction_1}
EXAMPLE-1 MINIMUM CHANGED PREDICTION: {example_mc_prediction_1}

EXAMPLE-2: Modify the wrong part while keep as much original words as possible. 
EXAMPLE-2 QUESTION: {example_quesiton_2}
EXAMPLE-2 PREVIOUS PREDICTION: {example_previous_prediction_2}
EXAMPLE-2 MINIMUM CHANGED PREDICTION: {example_mc_prediction_2}


Please perform minimum change to the following previous aprediction. 
QUESTION: {modified_question}
PREVIOUS PREDICTION: {previous_prediction}
MINIMUM CHANGED PREDICTION:
"""

        modified_prediction_step1 = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        a = 1
        
        minimum_change_prompt = \
f"""You need to extract the code from the previous prediction. The prediction answer the QUESTION.
QUESTION: {original_question}
Previous prediction: {modified_prediction_step1}

Explaination of extract code: 
1. You have to extract code from the previous prediction. Your answer will be used as the groundtruth to pass the test, thus of course you need to first extract the code and comments without including any other words.
2. You only need to extract the final code for the fuction for the test case{item['test'][0]}. You do not care about the test cases.
3. Please directly show me the extracted code with no other words. We will run the code to test if it is correct, thus do not extract other words.
"""
        code = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        code = code.replace('```python', '')
        code = code.replace('```', '')

        # index = modified_prediction_step1.lower().rfind('final answer:')
        # if index != -1:
        #     modified_prediction_step1 = modified_prediction_step1[:index]
        #     answer_final = f"""{modified_prediction_step1}Final Answer: {code}"""
        # else:
        answer_final = f"""{modified_prediction_step1}

Final Answer: {code}"""
        data_list[i]['answer'] = answer_final
        data_list[i]['previous_prediction'] = previous_prediction

        # print('previous_prediction', previous_prediction)
        # print('answer', answer)
        # print()
        # print()
        # print('----------------------------')
    return data_list


def minimum_change_data_creation_MBPP_code_only(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    for i, item in enumerate(data_list):
        previous_prediction = item['prediction']
        original_question = item['original_question']

        modified_question = \
f"""You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. The test is only used to show you the input structure. You do not need to run the test.


Task: {original_question}
Test Example: {item['test_list'][0]}"""


        previous_prediction = predict_list[i]
        
        prompt = \
f"""You need to extract the code from the previous prediction. The prediction answer the QUESTION.
QUESTION: {modified_question}
Previous prediction: {previous_prediction}

Explaination of extract code: 
1. You have to extract code from the previous prediction. Your answer will be evaluated to see if it will pass the test, thus of course you need to extract the code and comments without including any other words.
2. You only need to extract the final code for the fuction for the test case{item['test_list'][0]}. You do not care about the test cases.
3. Please directly show me the extracted code with no other words. We will run the code to test if it is correct, thus do not extract other words.
4. you suppose to extract the code from the previous prediction. You should never correct the code on your own even if it is wrong.
"""
        code = create_gpt_completion(prompt, MODEL_ENGINE, client)
        code = code.replace('```python', '')
        code = code.replace('```', '')
        data_list[i]['previous_code'] = code

        prompt = \
f"""You need to perform minimum change on the previous code to make sure it is correct. when performining minimum change, make sure to change as little as possible to make it correct. 
QUESTION: {modified_question}
Previous code: {code}

Explaination of minimum change: 
1. Given the question and the previous code, please perform minimum change. When I say minimum change, i mean you should try to modify as little previous prediction as possible to make it correct. Your are trying your best to preserve as much original words as possible. do not remove anything that is not wrong, such as comments.
2. Directly give me the minimum changed code.  The minimum changed code has to be code only. Do not say anything like 'here is the modifeid code', or 'this is wrong because'.
4. You do not need to modify the previous code if it is already correct. Please be careful about the spacing in code.
5. The test example is only used to show you the input structure. You do not need to run the test.

minimum changed code: 
"""
        minimum_changed_code = create_gpt_completion(prompt, MODEL_ENGINE, client)
        minimum_changed_code = minimum_changed_code.replace('```python', '')
        minimum_changed_code = minimum_changed_code.replace('```', '')

        # print('previous_prediction: ', previous_prediction)
        # print('code', code)
        # print('minimum_changed_code', minimum_changed_code)
        # print()
        # print()
        answer_final = f"""Final Answer:{minimum_changed_code}"""
        data_list[i]['answer'] = answer_final
        data_list[i]['intput'] = ''
        data_list[i]['previous_prediction'] = previous_prediction
    return data_list

def minimum_change_data_creation_BOOLQ(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        gold_label = item['answer']
        previous_prediction = predict_list[i]

        prompt = f"""We have the question, the previous prediction and the groundtruth.
Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the "Final Answer: {gold_label}" at the end.


Let me repeat the most important cretiria: When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 

Example1: the previous prediction is wrong. thus we perform minimum change to correctit. as you can see, the minimum changed prediction is similar to the previous prediction. In addition, make sure Place the Final Answer: {gold_label} at the end.
Example1-question: Given the context: Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.\n\ndo iran and afghanistan speak the same language?\n\nPlease inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'
Example1-groundtruth: True
Example1-previous prediction: Persian, also known as Farsi, is the predominant language in Iran. Dari is the name used for the Persian language in Afghanistan since 1958. While there are some differences between the two, they are mutually intelligible to a large extent. This means that speakers of both languages can understand each other with some effort. However, there are distinct variations in vocabulary, pronunciation, and grammar between the two.\n\nFinal Answer:\nWhile Persian and Dari (Afghan Persian) are closely related and mutually intelligible to a significant degree, they are not exactly the same language. Therefore, the answer is False.
Example1-minimum changed prediction: Persian, also known as Farsi, is the predominant language in Iran. Dari is the name used for the Persian language in Afghanistan since 1958. The two are dialects of the same language, and while there are some differences between them, they are mutually intelligible. This means that speakers of both can generally understand each other without too much effort. There may be variations in vocabulary, pronunciation, and grammar, but Iranians and Afghans essentially speak the same language, Persian.\n\nFinal Answer: True


Example2: The previous predection is 100% correct. We just copy it with the priod at the end removed. In this way, we keep as much original words as possible.
Example2-question: Given the context: Good Samaritan laws offer legal protection to people who give reasonable assistance to those who are, or who they believe to be, injured, ill, in peril, or otherwise incapacitated. The protection is intended to reduce bystanders' hesitation to assist, for fear of being sued or prosecuted for unintentional injury or wrongful death. An example of such a law in common-law areas of Canada: a good Samaritan doctrine is a legal principle that prevents a rescuer who has voluntarily helped a victim in distress from being successfully sued for wrongdoing. Its purpose is to keep people from being reluctant to help a stranger in need for fear of legal repercussions should they make some mistake in treatment. By contrast, a duty to rescue law requires people to offer assistance and holds those who fail to do so liable.\n\ndo good samaritan laws protect those who help at an accident?\n\nPlease inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'
Example2-groundtruth: True
Example2-previous prediction: Inference: Based on the context provided, Good Samaritan laws do protect individuals who provide assistance at the scene of an accident, as long as their actions are considered reasonable.\n\nFinal Answer: True.
Example2-minimum changed prediction: Inference: Based on the context provided, Good Samaritan laws do protect individuals who provide assistance at the scene of an accident, as long as their actions are considered reasonable.\n\nFinal Answer: True


We have the Question: {question} 
Groundtruth: {gold_label}
Previous prediction: {previous_prediction}
Minimum Changed Prediction: 
"""




        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        item['answer'] = answer
        item['groundtruth'] = gold_label
        item['previous_prediction'] = previous_prediction
        input_json_list.append(item)
    return input_json_list






def minimum_change_data_creation_PIQA(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

        prompt = f"""We have the question, the previous prediction and the groundtruth.
Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the "Final Answer: {gold_label}" at the end.


Let me repeat the most important cretiria: When modify the previous prediction, please perform minimum change to make it correct and make the inference part support the answer label. 


Example1: The previous prediction is incorrect. We try our best to keep as much original words as possible to make it correct. we place the "Final Answer: {gold_label}" at the end.
Example1-question: Given the question: how do you indent something?\n\nWhat option is correct?\nOption 1: leave a space before starting the writing\nOption 2: press the spacebar\n\nPlease inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'\n
Example1-groundtruth: 1
Example1-previous prediction: Based on the context of the question, it appears that the term "indent" is being used in the context of text formatting, specifically to refer to the act of setting text or a paragraph in from the left margin. Therefore, the correct option would be to use the Tab key or press the spacebar multiple times to create a consistent indentation.\n\nHowever, the options provided in the question do not accurately reflect the correct method for indenting text. Option 1 suggests leaving a space before starting to write, which is not the same as indenting text. Option 2 suggests pressing the spacebar, which is a valid way to create a small indentation, but it may not be sufficient for larger indentations or consistent formatting.\n\nFinal Answer: Neither Option 1 nor Option 2 are the correct answers, but the correct method for indenting text involves using the Tab key or pressing the spacebar multiple times to create a consistent indentation.
Example1-minimum changed prediction: Based on the context of the question, it appears that the term "indent" is being used in the context of text formatting, specifically to refer to the act of setting text or a paragraph in from the left margin. Therefore, the correct option would be to leave a space before starting the writing, which can be a way to create a simple form of indentation, especially in contexts where specific tools or keys like the Tab key are not mentioned or available.\nFinal Answer: 1


Example2: The previous prediction is already correct, so we just copy it., which means we try to keep the original words. We made a formatting correction--we place the "Final Answer: {gold_label}" at the end.
Example2-question: Given the question: To permanently attach metal legs to a chair, you can\n\nWhat option is correct?\nOption 1: Weld the metal together to get it to stay firmly in place\nOption 2: Nail the metal together to get it to stay firmly in place\n\nPlease inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'
Example2-groundtruth: 1
Example2-previous prediction: Inference: The given question is about attaching metal legs permanently to a chair. The options provided suggest two possible methods: welding and nailing. Welding involves melting metal to join it together, creating a strong bond. Nailing, on the other hand, involves hammering nails through the metal and into the chair to secure it in place.\n\nGiven the nature of the task and the materials involved, it's more likely that welding would be the better option for permanently attaching metal legs to a chair. Welding creates a strong, permanent bond between the metal parts.\n\nFinal Answer: Option 1: Weld the metal together to get it to stay firmly in place.
Example2-minimum changed prediction: Inference: The given question is about attaching metal legs permanently to a chair. The options provided suggest two possible methods: welding and nailing. Welding involves melting metal to join it together, creating a strong bond. Nailing, on the other hand, involves hammering nails through the metal and into the chair to secure it in place.\n\nGiven the nature of the task and the materials involved, it's more likely that welding would be the better option for permanently attaching metal legs to a chair. Welding creates a strong, permanent bond between the metal parts.\n\nFinal Answer: 1


We have the Question: {question} 
Groundtruth: {gold_label}
Previous prediction: {previous_prediction}
Minimum Changed Prediction: 
"""
        




        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        item['answer'] = answer
        item['groundtruth'] = gold_label
        item['previous_prediction'] = previous_prediction
        input_json_list.append(item)
    return input_json_list




def minimum_change_data_creation_ECQA(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

        prompt = f"""We have the question, the previous prediction and the groundtruth.
Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the "Answer: {gold_label}" at the end.


Let me repeat the most important cretiria: When modify the previous prediction, please perform minimum change to make it correct and make the inference part support the answer label. 

Example1: as you can see, we perform minimum change to make the previous question correct. We keep as much original words from previous prediction as possible to make it correct. there is a high lexical overlap between the previous prediction and the minimum changed prediction.
Example1-question: We have the question: What might the result of unwanted flirting be?\nand the options:\n(1): attraction\n(2): problems\n(3): the gallows\n(4): being slapped\n(5): curiosity\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'
Example1-gold-label: 4
Example1-previous-prediction: Sure, I\'d be happy to help!\n\nBased on the question "What might the result of unwanted flirting be?", I can infer that the possible outcomes could be negative or undesirable. Unwanted flirting can make the person on the receiving end feel uncomfortable, annoyed, or even harassed.\n\nHere are the possible outcomes:\n\n1. Attraction: Unwanted flirting may not necessarily lead to attraction, as the person being flirted with may not be interested.\n2. Problems: Unwanted flirting can cause problems in the workplace or other social situations, leading to tension or conflict.\n3. The gallows: This is an extreme and unlikely outcome, as flirting is not a criminal offense.\n4. Being slapped: This is also an extreme and unlikely outcome, as physical violence is never an appropriate response to flirting.\n5. Curiosity: Unwanted flirting may spark curiosity about the person\'s intentions or motivations.\n\nFinal Answer:\n\nBased on the possible outcomes, the correct answer is (2) Problems. Unwanted flirting can cause problems in the workplace or other social situations, leading to tension or conflict.
Exameple1-minimum-changed-prediction: Based on the question "What might the result of unwanted flirting be?", I can infer that the possible outcomes could be negative or undesirable. Unwanted flirting can make the person on the receiving end feel uncomfortable, annoyed, or even harassed. Unwanted flirting may lead to a reaction such as being slapped.\n\nHere are the possible outcomes:\n\n1. Attraction: Unwanted flirting may not necessarily lead to attraction, as the person being flirted with may not be interested.\n2. Problems: Unwanted flirting can cause problems, but this is not the specific outcome we\'re considering here.\n3. The gallows: This is an extreme and unrelated outcome, as flirting is not a capital offense.\n4. Being slapped: Being slapped could be a reaction to unwanted flirting due to discomfort or anger.\n5. Curiosity: Unwanted flirting is unlikely to spark curiosity in a positive manner, as the attention is not desired.\n\nFinal Answer: 4


Example2: The previous prediction is already correct, so we do not need to change anything. We just copy it. there is a tiny formatting issue so we modify the ending from 'Final Answer: (3) the bedroom' to Final Answer: 3
Example2-question: We have the question: If you want to set a romantic atmosphere you might light a candle where?\nand the options:\n(1): dimly lit room\n(2): synagogue\n(3): bedroom\n(4): birthday cake\n(5): roses\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'
Example2-gold-label: 3
Example2-previous-prediction: Sure, I'd be happy to help!\n\nBased on the information provided, the correct option for setting a romantic atmosphere by lighting a candle would be (3) the bedroom.\n\nThe other options do not seem to be relevant to setting a romantic atmosphere. Dimly lit room (1) may be too general and could apply to any room, not necessarily a romantic one. Synagogue (2) is a place of worship and would not be associated with romance. Birthday cake (4) is a dessert typically served on a birthday, not a romantic setting. Roses (5) are a symbol of love and romance, but they do not provide lighting.\n\nFinal Answer: (3) the bedroom.
Exameple2-minimum-changed-prediction: Sure, I'd be happy to help!\n\nBased on the information provided, the correct option for setting a romantic atmosphere by lighting a candle would be (3) the bedroom.\n\nThe other options do not seem to be relevant to setting a romantic atmosphere. Dimly lit room (1) may be too general and could apply to any room, not necessarily a romantic one. Synagogue (2) is a place of worship and would not be associated with romance. Birthday cake (4) is a dessert typically served on a birthday, not a romantic setting. Roses (5) are a symbol of love and romance, but they do not provide lighting.\n\nFinal Answer: 3



We have the Question: {question} 
Groundtruth: {gold_label}
Previous prediction: {previous_prediction}
Minimum Changed Prediction: 
"""

# Example3: 
# Example3-question: 
# Example3-gold-label: 
# Example3-previous-prediction: 
# Exameple3-minimum-changed-prediction: 
        # if not item['use_gpt4_answer']:
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        # if 'correct' not in question.lower():
        if 'previous pre' in answer.lower() or 'previous ans' in answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
            if 'previous pre' in answer.lower() or 'previous ans' in  answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        item['answer'] = answer
        # else:
        #     a = 1
        # answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        
        item['groundtruth'] = gold_label
        item['previous_prediction'] = previous_prediction
        input_json_list.append(item)
    return input_json_list





# def minimum_change_data_creation_WINOGRANDE(data_list, predict_list):
#     from openai import OpenAI
#     import openai
#     from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
#     client = OpenAI(api_key=GPT_API)
#     @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
#     def create_gpt_completion(qa_, model_engine, client):
#         response = client.chat.completions.create(
#             model=model_engine,
#             messages=[
#                 {"role": "system", "content": "provide answer"},
#                 {"role": "user", "content": f"{qa_}"}
#             ]
#         )
#         answer = response.choices[0].message.content
#         return answer

#     input_json_list = []
#     for i, item in enumerate(data_list):
#         question = item['question']
#         option1 = item['option1']
#         option2 = item['option2']
#         gold_label = item['gold_label']
#         previous_prediction = item['prediction']

#         prompt = f"""We have the question, the previous prediction and the groundtruth.
# Your task:
# 1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 
# 2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
# 3. If the previous prediction is already correct, then do not change anything. just copy it.
# 4. Place the "Answer: {gold_label}" at the end.


# Let me repeat the most important cretiria: When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 

# We have the Question: {question} 
# Groundtruth: {gold_label}
# Previous prediction: {previous_prediction}
# Minimum Changed Prediction: 
# """


#         answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

#         temp = {}
#         temp['question'] = question
#         temp['input'] = ''
#         temp['answer'] = answer
#         temp['groundtruth'] = gold_label
#         temp['previous_prediction'] = previous_prediction
#         temp['option1'] = item['option1']
#         temp['option2'] = item['option2']
#         input_json_list.append(temp)
#     return input_json_list




def minimum_change_data_creation_WINOGRANDE(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    
    WINOGRANDE_GPT_Initial_prediction_minimum_change_list = []
    for i, item in enumerate(data_list):
        previous_prediction = predict_list[i]
        question = item['question']
        gold_label = item['gold_label']

        prompt = f"""We have the question, the previous prediction and the groundtruth.
Your task:
1. According to groundtruth answer, please use Minimum Change to modify the previous prediction to make it correct. When modify the previous prediction, please try to keep the original words and logic styles and correct the wrong part. make sure the corrected prediction support the answer label. In other words, the corrected prediction should sounds similar and inference in similar way as the previous prediction.
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the "Final Answer: {gold_label}" at the end.


Let me repeat the most important cretiria-Minimum Change: When modify the previous prediction, please try to keep the original words and logic styles and correct the wrong part. make sure the corrected prediction support the answer label. In other words, the corrected prediction should sounds similar and inference in similar way as the previous prediction.

Example1: the previous prediction is wrong. thus we perform minimum change to correctit. as you can see, the minimum changed prediction is similar to the previous prediction. In addition, make sure Place the Final Answer: 2 at the end.
Example1-question: Given the question: Ian volunteered to eat Dennis's menudo after already having a bowl because _ despised eating intestine.\n\nWhat option is correct?\nOption 1: Ian\nOption 2: Dennis\n\nPlease inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'\n
Example1-groundtruth: 2
Example1-previous prediction: The sentence suggests that Ian volunteered to eat the menudo, which is a traditional Mexican soup that typically includes tripe (cow's stomach). The blank, \"_ despised eating intestine,\" is intended to refer to the person who disliked eating intestine, leading to Ian's decision to eat Dennis's menudo. Since Ian is the one who volunteered, it implies that Ian is the person who despised eating intestine, as it would be unlikely for Dennis to volunteer to eat something he himself despised. \n\nFINAL ANSWER: 1
Example1-minimum changed prediction: The sentence suggests that Ian volunteered to eat the menudo, which is a traditional Mexican soup that typically includes tripe (cow's stomach). The blank, \"_ despised eating intestine,\" is intended to refer to the person who disliked eating intestine, leading to Ian's decision to eat Dennis's menudo. Since Ian is the one who volunteered, it implies that Dennis is the person who despised eating intestine, as it would be unlikely for Ian to volunteer to eat something he himself despised. \n\nFinal Answer: 2



Example2: The previous predection is 100% correct. We just copy it with the priod at the end removed. In this way, we keep as much original words as possible. In addition, make sure Place the Final Answer: 1 at the end.
Example2-question: Given the question: Ian volunteered to eat Dennis's menudo after already having a bowl because _ enjoyed eating intestine.\n\nWhat option is correct?\nOption 1: Ian\nOption 2: Dennis\n\nPlease inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'\n
Example2-groundtruth: 1
Example2-previous prediction: The sentence specifies that Ian volunteered to eat Dennis's menudo, indicating that Ian was willing to eat the dish. The implication here is that Ian did this after already having a bowl himself, which suggests that Ian is the one who enjoyed eating intestine, as he was willing to have more of the menudo.\n\nFINAL ANSWER: Ian
Example2-minimum changed prediction: The sentence specifies that Ian volunteered to eat Dennis's menudo, indicating that Ian was willing to eat the dish. The implication here is that Ian did this after already having a bowl himself, which suggests that Ian is the one who enjoyed eating intestine, as he was willing to have more of the menudo. \n\nFinal Answer: 1


We have the Question: {question} 
Groundtruth: {gold_label}
Previous prediction: {previous_prediction}
Minimum Changed Prediction: 
"""

        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        if 'previous pre' in answer.lower() or 'previous ans' in answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
            if 'previous pre' in answer.lower() or 'previous ans' in  answer.lower() or 'minimum modi' in answer.lower() or 'minimum cha' in answer.lower():
                answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['answer'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        WINOGRANDE_GPT_Initial_prediction_minimum_change_list.append(temp)

    return WINOGRANDE_GPT_Initial_prediction_minimum_change_list





def minimum_change_data_creation_TRIVIA_QA(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        gold_label = item['answer']
        sol1 = item['sol1']
        sol2 = item['sol2']
        previous_prediction = predict_list[i]

        prompt = f"""We have the question, the previous prediction and the groundtruth.
Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the "Answer: {gold_label}" at the end.


Let me repeat the most important cretiria: When modify the previous prediction, please perform minimum change to make it correct and make the inference part support the answer label. 

We have the Question: {question} 
Groundtruth: {gold_label}
Previous prediction: {previous_prediction}
Minimum Changed Prediction: 
"""



        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['output'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['gold_label'] = gold_label
        temp['sol1'] = item['sol1']
        temp['sol2'] = item['sol2']
        input_json_list.append(temp)
    return input_json_list




def minimum_change_data_creation_MMLU(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    batch_api_list = []
    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        gold_label = item['answer']
        previous_prediction = predict_list[i]

        prompt = f"""We have the question, the previous prediction and the groundtruth.
Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the "Final Answer: {gold_label}" at the end.


Let me repeat the most important cretiria: When modify the previous prediction, please perform minimum change to make it correct and make the inference part support the answer label. 

Example1: the previous prediction made a mistake. In addition, it does not follow the format of Place the "Final Answer: A/B/C/D" at the end. We performed minimum change to keep as much original words as possible well correcting the prediction.
Example1-question: Given the question: A spherical balloon filled with helium is floating in air. If the balloon is inflated until its radius is doubled, how will the buoyant force on the balloon be affected?\n\nand the options:\nA: It will decrease by a factor of 4.\nB: It will increase by a factor of 4.\nC: It will increase by a factor of 8.\nD: It will not be affected.\n\nWhat is the answer?\n\nPlease inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'
Example1-groudntruth: C
Example1-previous prediction: The buoyant force on a spherical object in a fluid is given by the formula:\n\nBuoyant force = density of fluid * (4/3) * π * r³\n\nwhere r is the radius of the object.\n\nWhen the radius of the balloon is doubled, its volume will increase by a factor of 8 (since volume of a sphere is given by (4/3) * π * r³). However, the buoyant force is dependent on the cube of the radius. Therefore, the buoyant force will increase by a factor of 8³ = 512.\n\nSo, the answer is:\n\nFinal Answer: B. It will increase by a factor of 4 (actually, it increases by a factor of 8 cubed, but the question asks for a factor of 4 increase, which is equivalent to the cube root of 512).
Example1-minimum changed prediction: The buoyant force on a spherical object in a fluid is given by the formula:\n\nBuoyant force = density of fluid * (4/3) * π * r³\n\nwhere r is the radius of the object.\n\nWhen the radius of the balloon is doubled, its volume will increase by a factor of 8 (since volume of a sphere is given by (4/3) * π * r³). Therefore, the buoyant force will also increase by a factor of 8 since the buoyant force is directly proportional to the displaced volume of fluid.\n\nSo the answer is:\n\nFinal Answer: C


Example2: the previous prediction made a mistake in the logic step even though the inference in other place is correct. In addition, it does not follow the format of Place the "Final Answer: A/B/C/D" at the end. We performed minimum change to keep as much original words as possible well correcting the prediction.
Example1-question: Given the question: You have a wire of length L, radius Q, and resistance Q. You need to obtain half that resistance using the same material and changing only one factor. You could\n\nand the options:\nA: use half the length\nB: use twice the length\nC: use half the radius\nD: use twice the radius\n\nWhat is the answer?\n\nPlease inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'\n
Example1-groudntruth: A
Example1-previous prediction: The resistance of a wire is given by the formula: R = ρ * (L / A), where R is resistance, ρ is the resistivity of the material, L is length, and A is the cross-sectional area. The cross-sectional area is given by the formula: A = π * r², where r is the radius.\n\nTo obtain half the resistance with the same material, we need to find which factor to change to reduce the resistance by a factor of two.\n\nSince resistance is given by the formula R = ρ * (L / A), we can see that resistance is directly proportional to the length (L) and inversely proportional to the cross-sectional area (A).\n\nTo reduce the resistance by a factor of two, we need to either halve the length or double the cross-sectional area. Since we are only allowed to change one factor, and we cannot change the length to zero to double the cross-sectional area, the only option is to halve the length.\n\nFinal Answer: A. use half the length.
Example1-minimum changed prediction: The resistance of a wire is given by the formula: R = ρ * (L / A), where R is resistance, ρ is the resistivity of the material, L is length, and A is the cross-sectional area. The cross-sectional area is given by the formula: A = π * r², where r is the radius.\n\nTo obtain half the resistance with the same material, we need to find which factor to change to reduce the resistance by a factor of two.\n\nSince resistance is given by the formula R = ρ * (L / A), we can see that resistance is directly proportional to the length (L) and inversely proportional to the cross-sectional area (A).\n\nTo reduce the resistance by a factor of two, we need to halve the length of the wire as changing the radius would not give us an exact half adjustement in resistance.\n\nFinal Answer: A



We have the Question: {question} 
Groundtruth: {gold_label}
Previous prediction: {previous_prediction}
Minimum Changed Prediction: 
"""



        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        item['groundtruth'] = gold_label
        item['previous_prediction'] = previous_prediction
        item['answer'] = answer
        input_json_list.append(item)


    #     batch_api_list.append({"custom_id": f"{i}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": MODEL_ENGINE, "messages": [{"role": "system", "content": "provide answer"}, {"role": "user", "content": f"{prompt}"}], "max_tokens": 1000}})


    # model_name = MODEL_ENGINE
    # file_path = f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/gpt4_generated_data/batch_api/mmlu_{model_name}_1000.jsonl"
    # file_txt_path = f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/gpt4_generated_data/batch_api/mmlu_{model_name}_1000.txt"
    # with open(file_path, 'w') as file:
    #     for obj in batch_api_list:
    #         file.write(json.dumps(obj) + '\n')

    # batch_input_file = client.files.create(
    #     file=open(file_path, "rb"),
    #     purpose="batch"
    # )

    # batch_input_file_id = batch_input_file.id
    # batch_job = client.batches.create(
    #     input_file_id=batch_input_file_id,
    #     endpoint="/v1/chat/completions",
    #     completion_window="24h",
    #     metadata={
    #     "description": f"mmlu_{model_name}_1000"
    #     }
    # )

    # batch_job_id = batch_job.id

    # with open(file_txt_path, 'w') as file:
    #     file.write(f"batch_job_id: {batch_job_id}")
    return input_json_list

def minimum_change_data_creation_AGIEVAL(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

# Example 1: 
# We have the Question: {question} 
# Groundtruth: A
# Previous prediction: Sure, I'd be happy to help! Here's my inference and final answer:\n\nInference:\n\nFrom the statement, we know that there are three young staff members on a business trip, and they are sitting in a row. We also know that at least one of the two people sitting on the right side of the 24-year-old is 20 years old. Additionally, we know that at least one of the two people sitting on the left of the accountant is a salesperson.\n\nBased on this information, we can infer that the three young staff members are on a business trip related to sales, and the accountant is likely not one of the salespeople. This is because the statement says that at least one of the two people sitting on the right side of the 24-year-old is 20 years old, which suggests that the 24-year-old is not an accountant.\n\nFinal Answer:\n\nTherefore, the three young staff members on a business trip are:\n\n(A) 20-year-old accountant, 20-year-old salesperson, 24-year-old salesperson\n\nThe 24-year-old accountant is not one of the salespeople, and the 20-year-old salesperson is sitting on the left of the accountant. The 20-year-old salesperson on the right side of the 24-year-old is not the accountant.
# Minimum Changed Prediction: "Based on the statement, we know that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area. This implies that the Civic Park, which is central to the cultural area, could be considered north of the administrative service area when looking at the overall layout relative to the mentioned directions."

# Final Answer: A

        prompt = f"""We have the question, the previous prediction and the groundtruth.
Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the "Final Answer: {gold_label}" at the end.

Let me repeat the most important cretiria: When modify the previous prediction, please perform minimum change to make it correct and make the inference part support the answer label. 

We have the Question: {question} 
Groundtruth: {gold_label}
Previous prediction: {previous_prediction}
Minimum Changed Prediction: 
"""

        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        item['answer'] = answer
        input_json_list.append(item)
    return input_json_list






def minimum_change_data_creation_AGIEVAL_with_examples(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

        prompt = f"""We have the question, the previous prediction and the groundtruth.

Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Place the "Final Answer: {gold_label}" at the end.

Let me repeat the most important cretiria: When modify the previous prediction, please perform minimum change to make it correct and make the inference part support the answer label. 

Example1: We perform minimum change to the previous prediction to make it correct, which means we try to keep the original words. We make sure Place the "Final Answer: {gold_label}" at the end.
Example1-question: Given the statement: In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park.These four communities are designated as cultural area, leisure area, commercial area and administrative service area.It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area.\n\nand the question: Based on the above statement, which of the following can be derived?\n\nand the options:\n(A)Civic Park is north of the administrative service area\n(B)The leisure area is southwest of the cultural area\n(C)The cultural district is in the northeast of the business district\n(D)The business district is southeast of the leisure area\n\n\nWhat is the answer?\n\n\nPlease inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'
Example1-groundtruth: A
Example1-previous prediction: Based on the given statement, we know that the citizen park is the central point of the new district, and there are four communities designated around it: cultural area, leisure area, commercial area, and administrative service area. We also know that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area.\n\nFrom this information, we can infer that:\n\n1. The cultural area is to the southeast of the leisure area.\n2. The administrative service area is to the southwest of the cultural area.\n3. Citizen park is at the center of the new district, and it is not explicitly stated where it is in relation to the other communities, but since the cultural area is southeast of the leisure area, and the administrative service area is southwest of the cultural area, we can infer that the citizen park is likely north of the administrative service area, but we cannot be completely sure based on the given information.\n\nSo, the answer is:\n\nFinal Answer: (A) Civic Park is north of the administrative service area (but this is an inference and not a direct statement from the given information)\n\nThe other options are not correct based on the given statement.\n\n(C) The cultural district is in the northeast of the business district - This is not correct based on the given information as there is no mention of the business district or its location in the statement.\n\n(B) The leisure area is southwest of the cultural area - This is correct based on the given information.\n\n(D) The business district is southeast of the leisure area - This is not correct based on the given information as there is no mention of the business district or its location in the statement.
Example1-minimum changed prediction: Based on the given statement, we know that the citizen park is the central point of the new district, and there are four communities designated around it: cultural area, leisure area, commercial area, and administrative service area. We also know that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area.\n\nFrom this information, we can infer that:\n\n1. The cultural area is to the southeast of the leisure area.\n2. The administrative service area must be to the southwest of the cultural area.\n3. Because the Citizen Park is at the center of these districts, and based on the relative positions of the cultural and administrative service areas, we can deduce the Citizen Park is north, or more accurately north-northeast, of the administrative service area.\n\nSo, the answer is:\n\nFinal Answer: A


Example2: The previous prediction is correct. there is a tiny formating issue where it mention Final Answer: D. (Ding) we copy the original previous prediciton so as to keep as much original words as possible and change Final Answer: D. (Ding) to Final Answer: D
Example2-question: Given the statement: A unit conducted the year-end assessment and after democratic voting, five people were identified as candidates for the first prize.In the selection of five in four, the following factors need to be considered? 1) At least one person of Bingding is selected.2) If E is selected, then A and B are also selected.3) A maximum of 2 people are selected.\n\nand the question: According to the above statement, it can be concluded that who is not in the fourth?\n\nand the options:\n(A)A.\n(B)B.\n(C)C.\n(D)Ding.\n\n\nWhat is the answer?\n\n\nPlease inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'
Example2-groundtruth: D
Example2-previous prediction: Based on the given statement, only one person from Bingding is required to be selected, and a maximum of 2 people can be selected. Additionally, if E is selected, then A and B must also be selected.\n\nGiven these conditions, let's analyze the options:\n\n(A) A: A can be selected since if E is selected, then A and B must also be selected.\n\n(B) B: B can be selected since at least one person from Bingding must be selected.\n\n(C) C: Since only a maximum of 2 people can be selected, and A and B are already selected due to E being selected, C cannot be in the fourth position.\n\n(D) Ding: Ding is not mentioned in the statement as being one of the five candidates, so they cannot be in the fourth position.\n\nFinal Answer: D. (Ding)
Example2-minimum changed prediction: Based on the given statement, only one person from Bingding is required to be selected, and a maximum of 2 people can be selected. Additionally, if E is selected, then A and B must also be selected.\n\nGiven these conditions, let's analyze the options:\n\n(A) A: A can be selected since if E is selected, then A and B must also be selected.\n\n(B) B: B can be selected since at least one person from Bingding must be selected.\n\n(C) C: Since only a maximum of 2 people can be selected, and A and B are already selected due to E being selected, C cannot be in the fourth position.\n\n(D) Ding: Ding is not mentioned in the statement as being one of the five candidates, so they cannot be in the fourth position.\n\nFinal Answer: D


Now please perform minimum change here
We have the Question: {question} 
Groundtruth: {gold_label}
Previous prediction: {previous_prediction}
Minimum Changed Prediction: 
"""

        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        item['answer'] = answer
        input_json_list.append(item)
    return input_json_list


# Example 1: Please notice that we only correct the minimum part that is wrong and we keep the original words as much as possible.
# Example1-Question: Given the statement: In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park.These four communities are designated as cultural area, leisure area, commercial area and administrative service area.It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area.\n\nand the question: Based on the above statement, which of the following can be derived?\n\nand the options:\n(A)Civic Park is north of the administrative service area\n(B)The leisure area is southwest of the cultural area\n(C)The cultural district is in the northeast of the business district\n(D)The business district is southeast of the leisure area\n\n\nWhat is the answer?\n\n\nPlease inference first, then provide the final answer (A/B/C/D) at the end, after 'Final Answer:'
# Example1-Groundtruth: A
# Example1-Previous prediction: Sure, I'd be happy to help! Here's my inference and final answer:\n\nInference:\n\nFrom the statement, we know that there are three young staff members on a business trip, and they are sitting in a row. We also know that at least one of the two people sitting on the right side of the 24-year-old is 20 years old. Additionally, we know that at least one of the two people sitting on the left of the accountant is a salesperson.\n\nBased on this information, we can infer that the three young staff members are on a business trip related to sales, and the accountant is likely not one of the salespeople. This is because the statement says that at least one of the two people sitting on the right side of the 24-year-old is 20 years old, which suggests that the 24-year-old is not an accountant.\n\nFinal Answer:\n\nTherefore, the three young staff members on a business trip are:\n\n(A) 20-year-old accountant, 20-year-old salesperson, 24-year-old salesperson\n\nThe 24-year-old accountant is not one of the salespeople, and the 20-year-old salesperson is sitting on the left of the accountant. The 20-year-old salesperson on the right side of the 24-year-old is not the accountant.
# Example1-Minimum Changed Prediction: Based on the statement, we know that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area. We can infer that the civic park is north of the administrative service area since the cultural area is centered on the citizen park, which means the civic park must be located in the center of the cultural area. Since the cultural area is southeast of the leisure area and the administrative service area is southwest of the cultural area, we can conclude that the citizen park is north of the administrative service area.\n\nNow, let's examine the options:\n\n(A) Civic Park is north of the administrative service area. This statement is correct because the citizen park is in the center of the cultural area, which is southeast of the leisure area and hence north of the administrative service area.\n\n(B) The leisure area is southwest of the cultural area. This statement is incorrect because we know that the cultural area is southeast of the leisure area.\n\n(C) The cultural district is in the northeast of the business district. We cannot infer this information from the given statement.\n\n(D) The business district is southeast of the leisure area. We cannot infer this information from the given statement.\n\nFinal Answer: A

def minimum_change_data_creation_SQUAD(data_list, predict_list):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    input_json_list = []
    for i, item in enumerate(data_list):
        question = item['question']
        gold_label = item['gold_label']
        previous_prediction = predict_list[i]

        prompt = f"""We have the question, the previous prediction and the groundtruth.
Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !" or "Here is the modified answer."
5. Place the "Final Answer: {gold_label}" at the end. Make sure there is no extra information after "Final Answer: {gold_label}"


Let me repeat the most important cretiria: When modify the previous prediction, please perform minimum change to make it correct and make the inference part support the answer label. When I say minimum change, I mean you have to change as little as possible to make it correct.

please do not remove analyze process if the analyzation is correct.



example1: example 1 perform minimum change by delete the wrong content and copy the correct content with few modification.
example1-Question: 'Given the question: When did Beyonce start becoming popular? and the context: Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".\n\nWhat is the answer?\n\nPlease inference first, then provide the final answer (text span) at the end, after \'Final Answer:\'\n'
example1-previous prediction: Beyoncé started becoming popular in the late 1990s as a member of the R&B girl-group Destiny's Child. However, her solo career took off with the release of her debut album, Dangerously in Love, in 2003. This album established her as a solo artist worldwide, earned five Grammy Awards, and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy.
example1-minimum changed prediction: Beyoncé started becoming popular in the late 1990s as a member of the R&B girl-group Destiny's Child.

Final Answer: in the late 1990s


example 2 perform minimum change by only change where is wrong. it keeps as much original words as possible when it is correct.
example2-Question: Given the question: When did Beyonce leave Destiny\'s Child and become a solo singer? and the context: Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".\n\nWhat is the answer?\n\nPlease inference first, then provide the final answer (text span) at the end, after \'Final Answer:\'\n
example2-previous prediction: Beyoncé left Destiny\'s Child and began her solo career after the release of their last album, "Destiny Fulfilled" in 2004. However, her debut solo album, "Dangerously in Love," was released in 2003. So technically, she started her solo career before the official group hiatus, but the wider recognition and success came after the release of the album.\n\nFinal Answer: Beyoncé left Destiny\'s Child and became a solo singer around 2003.
example2-minimum changed prediction: Beyoncé left Destiny\'s Child and began her solo career after the release of their last album, "Destiny Fulfilled" in 2004. However, her debut solo album, "Dangerously in Love," was released in 2003. So technically, she started her solo career before the official group hiatus, but the wider recognition and success came after the release of the album.\n\nFinal Answer: 2003




Your task:
1. According to groundtruth answer, please modify the previous prediction. When modify the previous prediction, please perform minimum change to make it correct and make it support the answer label. You will try to use its original words with a minimum modification to make it correct. 
2. If the previous prediction contains some hallucination that is not relevant to answer the question, such as "Please help! I'm having trouble with these types of questions.", then remove it. 
3. If the previous prediction is already correct, then do not change anything. just copy it.
4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
you with that !", "Here is the modified answer." or "the previous prediction is wrong because"
5. Place the "Final Answer: {gold_label}" at the end. Make sure there is no extra sentence after "Final Answer: {gold_label}"

We have the Question: {question} 
Groundtruth: {gold_label}
Previous prediction: "{previous_prediction}"
Minimum Changed Prediction: 
"""

        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp = {}
        temp['question'] = item['question']
        temp['input'] = ''
        temp['output'] = answer
        temp['groundtruth'] = gold_label
        temp['previous_prediction'] = previous_prediction
        temp['answer'] = answer
        input_json_list.append(temp)
    return input_json_list



def gpt4_create_groundtruth(question_list, groundtruth_list = [], step_by_step = False, task_name = '', train_data_list = [], provide_groundtruth_with_inference_steps = False, model_company = 'openai', answer_without_groundtruth = False, enable_mini_gpt4 = False, enable_batch_api = False):
    batch_api_list = []
    if model_company == 'openai':
        from openai import OpenAI
        import openai
        from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
        client = OpenAI(api_key=GPT_API)
        @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
        def create_gpt_completion(qa_, model_engine, client):
            response = client.chat.completions.create(
                model=model_engine,
                messages=[
                    {"role": "system", "content": "provide answer"},
                    {"role": "user", "content": f"{qa_}"}
                ]
            )
            answer = response.choices[0].message.content
            return answer
    elif model_company == 'anthropic':
        import anthropic
        my_api_key = os.environ.get("ANTHROPIC_API_KEY")
        # print('111111111', my_api_key)
        client = anthropic.Anthropic(
            api_key=f"{my_api_key}",
        )

        # print('22222222', my_api_key)
        def create_gpt_completion(qa_, model_engine, client):
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                temperature=0,
                messages=[
                        {"role": "user", "content": f"{qa_}"}
                ]
            )
            answer = message.content[0].text
            return answer

    
    gpt4_answer_list = []
    if step_by_step:
        step_by_step_insertion_1 = ' step by step'
        step_by_step_insertion_2 = 'Step by Step '
    else:
        step_by_step_insertion_1 = ''
        step_by_step_insertion_2 = ''
    
    for i, question in enumerate(question_list):
        gold_label = groundtruth_list[i]
        temp = {}
        temp['question'] = question
        temp['input'] = ''
        # AQuaRAT
        if task_name == 'MATH_ALGEBRA' or task_name == 'MATH_NUMBER_THEORY' or task_name == 'MATH_COUNTING_AND_PROBABILITY' or 'GSM8K' in task_name or 'AQUARAT' in task_name or 'MATH_GEOMETRY' in task_name:
            if 'GSM8K' in task_name and 'GSM8K' != task_name:
                prompt = question
            elif 'AQUARAT' in task_name:
                prompt = f"""We have the question {question} 


1. We wish you to answer the question{step_by_step_insertion_1}.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (A)/(B)/(C)/(D)/(E) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: ({gold_label})""" 
            else:
                if not provide_groundtruth_with_inference_steps:
                    prompt = f"""We have the {question} 


1. We wish you to answer the question{step_by_step_insertion_1}.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (NUMBER_HERE) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: NUMBER_HERE""" 
                else:
                    prompt = f"""We have the 
question: {question} 
and the groundtruth: {groundtruth_list[i]}


1. Based on the groundtruth, we wish you to answer the question{step_by_step_insertion_1} in more detailed way.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (NUMBER_HERE) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: NUMBER_HERE""" 
        elif task_name == 'ANLI':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer: (Entailment/Contradiction/Neutral) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
        elif task_name == 'ESNLI':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (Entailment/Contradiction/Neutral) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
FINAL ANSWER: {gold_label}""" 
        elif task_name == 'SCITAIL':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (Entailment/Neutral) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
FINAL ANSWER: {gold_label}""" 
        elif task_name == 'BOOLQ':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (True/False) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
        elif task_name == 'PIQA':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (a number) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
FINAL ANSWER: {gold_label}""" 
        elif task_name == 'WINOGRANDE':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (a number) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
FINAL ANSWER: {gold_label}""" 
        elif task_name == 'MMLU':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (A/B/C/D) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
FINAL ANSWER: {gold_label}""" 
        elif task_name == 'AGIEVAL':
            prompt = f"""We have the {question} and the gold label {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer (A/B/C/D) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
FINAL ANSWER: {gold_label}""" 
        elif task_name == 'CODE':
            temp = train_data_list[i]
            entry_point = temp['entry_point']
            question = question.replace('Since entry point is already given, you only need to fill out the context under the entry point. Remember that you need to start with 4 white space(indentation) since you your answer will be placed under the entry point.', 'The entry point is given. please complete the code in the entrypoint').replace(' The code after \'Final Answer: \' should start with 4 whitespace for indentation because it is under the entrypoint.', '')
            prompt = f"""We have the {question}


1. We wish you to answer the question{step_by_step_insertion_1}.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer: at the end of the prediction like this {step_by_step_insertion_2}

YOUR_INFERENCE_HERE
FINAL ANSWER:
YOUR_CODE_ANSWER_HERE (don't put ``` or ```python here.)
4. The code being putted after Final Answer should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
5. if you need to import something, please make sure to place the module under the entrypoint({entry_point}).
"""

        elif task_name == 'MBPP':
            temp = train_data_list[i]
#             prompt = f"""We have the {question}


# 1. We wish you to answer the question{step_by_step_insertion_1}.
# 2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
# 3. You will inference first then put the FINAL ANSWER: at the end of the prediction like this {step_by_step_insertion_2}

# YOUR_INFERENCE_HERE
# FINAL ANSWER:
# YOUR_CODE_ANSWER_HERE (don't put ``` or ```python here.)
# 4. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
# """
            prompt = f"""We have the {question}


1. We wish you to answer the question.
2. You must directly provide the code answer without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'. 
3. The code being putted after FINAL_ANSWER should be runnable code which means You do not need to add ``` ``` or add python in front of the code.
"""
            
        elif task_name == 'API_BANK':
            # 3. You must explain why you choose the particular API-Request. For example, if the previous API-Request is ToolSearch, then at this step you may need to use the tool found by the previous step. If the previous step alreaady used an API as a tool, then you may need to use ToolSearch to find the API for the next Action.
            temp = train_data_list[i]
            prompt = f"""We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            if step_by_step_insertion_1 != '':
                prompt = f"""We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You will solve the problem in step by step manner.
3. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
4. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            
            temp['question'] = question
            temp['input'] = ''
        elif task_name == 'API_BANK_VANILLA':
            temp = train_data_list[i]
            question = question + f"""

The task requires multi API-Request generation. We only generate the next API-Request at this time. What should be generate for the next API-Request? The question might already provide the previous API-CALL history."""
            prompt = f"""We have the {question} and the groundtruth {gold_label}

1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the FINAL ANSWER: {gold_label} at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
FINAL ANSWER: {gold_label}""" 
        elif task_name == 'PLAN_BENCH':
            temp = train_data_list[i]
            prompt = f"""We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            if step_by_step_insertion_1 != '':
                prompt = f"""We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You will solve the problem in step by step manner.
3. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
4. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            
            temp['question'] = question
            temp['input'] = ''
        elif task_name == 'ECQA':
            temp = train_data_list[i]
            prompt = f"""We have the {question} and the groundtruth {gold_label}

1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the FINAL ANSWER: {gold_label} at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            
            temp['question'] = question
            temp['input'] = ''
        
        elif task_name == 'SQUAD':
            temp = train_data_list[i]
            prompt = f"""We have the question "{question}" and the groundtruth {gold_label}

1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer: {gold_label} at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            
            temp['question'] = question
            temp['input'] = ''
        

        if answer_without_groundtruth:
            # original_question_item = train_data_list[i]['original_question']
            # prompt = original_question_item
            # temp['question'] = original_question_item

            prompt = question
            temp['question'] = question
            temp['input'] = ''
        
        # print(prompt)
        if not enable_batch_api:
            if not enable_mini_gpt4:
                answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
            else:
                answer = create_gpt_completion(prompt, MINI_MODEL_ENGINE, client)
            temp['answer'] = answer
            temp['gold_label'] = gold_label
            gpt4_answer_list.append(temp)
        else:
            if not enable_mini_gpt4:
                batch_api_list.append({"custom_id": f"{i}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": MODEL_ENGINE, "messages": [{"role": "system", "content": "provide answer"}, {"role": "user", "content": f"{prompt}"}], "max_tokens": 1000}})
            else:
                batch_api_list.append({"custom_id": f"{i}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": MINI_MODEL_ENGINE, "messages": [{"role": "system", "content": "provide answer"}, {"role": "user", "content": f"{prompt}"}], "max_tokens": 1000}})

            a = 1
        
    if enable_batch_api:
        if not enable_mini_gpt4:
            model_name = MODEL_ENGINE
        else:
            model_name = MINI_MODEL_ENGINE
        file_path = f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/gpt4_generated_data/batch_api/{task_name}_{step_by_step}_{model_name}_{len(question_list)}.jsonl"
        file_txt_path = f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/gpt4_generated_data/batch_api/{task_name}_{step_by_step}_{model_name}_{len(question_list)}.txt"
        with open(file_path, 'w') as file:
            for obj in batch_api_list:
                file.write(json.dumps(obj) + '\n')

        batch_input_file = client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        
        batch_job = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": f"{task_name}_{model_name}_{len(question_list)}"
            }
        )

        batch_job_id = batch_job.id

        with open(file_txt_path, 'w') as file:
            file.write(f"batch_job_id: {batch_job_id}")
    return gpt4_answer_list



def api_bank_create_groundtruth(question_list, groundtruth_list = [], step_by_step = False, task_name = '', train_data_list = [], model_company = 'openai', variation = '', previous_prediction_list = [], enable_batch_api = True, enable_mini_gpt4 = False):
    if model_company == 'openai':
        from openai import OpenAI
        import openai
        from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
        client = OpenAI(api_key=GPT_API)
        @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
        def create_gpt_completion(qa_, model_engine, client):
            response = client.chat.completions.create(
                model=model_engine,
                messages=[
                    {"role": "system", "content": "provide answer"},
                    {"role": "user", "content": f"{qa_}"}
                ]
            )
            answer = response.choices[0].message.content
            return answer
    elif model_company == 'anthropic':
        import anthropic
        my_api_key = os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(
            api_key=f"{my_api_key}",
        )
        def create_gpt_completion(qa_, model_engine, client):
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                temperature=0,
                messages=[
                        {"role": "user", "content": f"{qa_}"}
                ]
            )
            answer = message.content[0].text
            return answer

    batch_api_list = []
    gpt4_answer_list = []
    if step_by_step:
        step_by_step_insertion_1 = ' step by step'
        step_by_step_insertion_2 = 'Step by Step '
    else:
        step_by_step_insertion_1 = ''
        step_by_step_insertion_2 = ''
    for i, question in enumerate(question_list):
        gold_label = groundtruth_list[i]
        temp = {}
        temp['question'] = question
        temp['input'] = ''
        # AQuaRAT
        
        if task_name == 'API_BANK':
            temp = train_data_list[i]
            prompt = f"""We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}""" 
            if variation == 'step_by_step':
                prompt = f"""We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You will solve the problem in step by step manner.
3. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
4. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            if variation == 'provide_gpt4_example':

                q1 = \
'\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User\'s utterence\n\nExpected output:\nAPI-Request: [ApiName(key1=\'value1\', key2=\'value2\', ...)]\n\nAPI descriptions:\n{"apiCode": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nGenerate next API Request: '
                
                q2 = \
'\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User\'s utterence\n\nExpected output:\nAPI-Request: [ApiName(key1=\'value1\', key2=\'value2\', ...)]\n\nAPI descriptions:\n{"apiCode": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nAPI-Request: [ToolSearcher(keywords=\'healthcare provider appointment availability checker\')]->{\'name\': \'HealthcareProviderAppointmentChecker\', \'description\': \'API for checking the availability of appointments with healthcare providers.\', \'input_parameters\': {\'specialty\': {\'type\': \'str\', \'description\': \'The specialty of the healthcare provider.\'}, \'location\': {\'type\': \'str\', \'description\': \'The city location.\'}}, \'output_parameters\': {\'appointments\': {\'type\': \'list\', \'description\': \'A list of available appointment slots.\'}}}\nGenerate next API Request: '

                q3 = \
'\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User\'s utterence\n\nExpected output:\nAPI-Request: [ApiName(key1=\'value1\', key2=\'value2\', ...)]\n\nAPI descriptions:\n{"apiCode": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nAPI-Request: [ToolSearcher(keywords=\'healthcare provider appointment availability checker\')]->{\'name\': \'HealthcareProviderAppointmentChecker\', \'description\': \'API for checking the availability of appointments with healthcare providers.\', \'input_parameters\': {\'specialty\': {\'type\': \'str\', \'description\': \'The specialty of the healthcare provider.\'}, \'location\': {\'type\': \'str\', \'description\': \'The city location.\'}}, \'output_parameters\': {\'appointments\': {\'type\': \'list\', \'description\': \'A list of available appointment slots.\'}}}\nAPI-Request: [HealthcareProviderAppointmentChecker(specialty=\'cardiologist\', location=\'Los Angeles\')]->{\'appointments\': [\'2034-04-18 14:30:00\', \'2034-04-19 11:00:00\', \'2034-04-20 09:45:00\']}\nGenerate next API Request: '
                prompt = f"""We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}

Example 1:

Question : {q1}

groundtruth: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]

Inference: The user is requesting to find a healthcare provider (specifically a cardiologist) in Los Angeles for a check-up appointment. The available API description indicates that the ToolSearcher API can be used to search for relevant tools based on the provided keywords. Therefore, the first step is to search for a tool that can help find a healthcare provider appointment availability checker.

Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]


Example 2:

question: {q2}

groundtruth: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]

Inference: The first API request was successfully made to find a tool for checking healthcare provider appointment availability. The HealthcareProviderAppointmentChecker API was identified, which requires specialty and location as input parameters to search for available appointment slots. Based on the user's request to find a cardiologist in Los Angeles for a check-up appointment, the next API call should use this information.

Final Answer: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]


Example 3:

question: {q3}

groundtruth: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]

Inference: The user initially searched for an availability checker and found available appointment slots for a cardiologist in Los Angeles. Now, the user needs to schedule an appointment, so the next step is to find a tool for scheduling healthcare provider appointments using the ToolSearcher API with relevant keywords.

Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]

We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}

""" 

            if variation == 'human_written_examples':

                q1 = \
'\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User\'s utterence\n\nExpected output:\nAPI-Request: [ApiName(key1=\'value1\', key2=\'value2\', ...)]\n\nAPI descriptions:\n{"apiCode": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nGenerate next API Request: '
                
                q2 = \
'\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User\'s utterence\n\nExpected output:\nAPI-Request: [ApiName(key1=\'value1\', key2=\'value2\', ...)]\n\nAPI descriptions:\n{"apiCode": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nAPI-Request: [ToolSearcher(keywords=\'healthcare provider appointment availability checker\')]->{\'name\': \'HealthcareProviderAppointmentChecker\', \'description\': \'API for checking the availability of appointments with healthcare providers.\', \'input_parameters\': {\'specialty\': {\'type\': \'str\', \'description\': \'The specialty of the healthcare provider.\'}, \'location\': {\'type\': \'str\', \'description\': \'The city location.\'}}, \'output_parameters\': {\'appointments\': {\'type\': \'list\', \'description\': \'A list of available appointment slots.\'}}}\nGenerate next API Request: '

                q3 = \
'\nGenerate an API request in the format of [ApiName(key1=\'value1\', key2=\'value2\', ...)] based on the user\'s utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User\'s utterence\n\nExpected output:\nAPI-Request: [ApiName(key1=\'value1\', key2=\'value2\', ...)]\n\nAPI descriptions:\n{"apiCode": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nAPI-Request: [ToolSearcher(keywords=\'healthcare provider appointment availability checker\')]->{\'name\': \'HealthcareProviderAppointmentChecker\', \'description\': \'API for checking the availability of appointments with healthcare providers.\', \'input_parameters\': {\'specialty\': {\'type\': \'str\', \'description\': \'The specialty of the healthcare provider.\'}, \'location\': {\'type\': \'str\', \'description\': \'The city location.\'}}, \'output_parameters\': {\'appointments\': {\'type\': \'list\', \'description\': \'A list of available appointment slots.\'}}}\nAPI-Request: [HealthcareProviderAppointmentChecker(specialty=\'cardiologist\', location=\'Los Angeles\')]->{\'appointments\': [\'2034-04-18 14:30:00\', \'2034-04-19 11:00:00\', \'2034-04-20 09:45:00\']}\nGenerate next API Request: '
                
                prompt = f"""We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}

Example 1:

Question : {q1}

groundtruth: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]

Inference: The user is requesting to find a healthcare provider (specifically a cardiologist) in Los Angeles for a check-up appointment. The first step should be to search for a tool that can help find a healthcare provider appointment availability checker. To accomplish this, we choose the ToolSearcher API from the available APIs. The ToolSearcher API is used to search for relevant tools based on the provided keywords according to the description. We need to fill out the keywords according to the description. The keywords could be 'healthcare provider appointment availability checker.' Therefore, the next step (which is also the first step) is:

Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]



Example 2:

question: {q2}

groundtruth: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]

Inference: According to the API call history, the user has called the ToolSearcher API and found the HealthcareProviderAppointmentChecker API. The next step is to fill out the input parameters for HealthcareProviderAppointmentChecker and use it to find healthcare provider appointment availability. The input parameters are specialty and location. The user wants to find a cardiologist in Los Angeles for a check-up appointment. Therefore, the next API request should be:

Final Answer: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]


Example 3:

question: {q3}

groundtruth: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]

Inference: The user previously called the HealthcareProviderAppointmentChecker API and found three appointment times, which are '2034-04-18 14:30:00', '2034-04-19 11:00:00', and '2034-04-20 09:45:00'. The next step is to find the scheduler for the appointment. Since there is no available tool, the user needs to search for a tool that can schedule healthcare provider appointments. The ToolSearcher API can be used to search for relevant tools based on the keywords according to the description. The keywords should be 'healthcare provider appointment scheduler'. Therefore, the answer is:

Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]




We have a question and a groundtruth 

question: {question}

groundtruth: {gold_label}


1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}

""" 
            if 'paraphrase_anthropic_examples' in variation:
                previous_prediction_item = previous_prediction_list[i]
                prompt = f"""Given the question: {question}
            
and the prediction: {previous_prediction_item} 

1. Please paraphrase the prediction. The prediction might be wrong sometimes, but you do not need to correct it. just paraphrase it.
2. (important format) You must create the paraphrased prediction directly without say anything else. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
4. (important format) Please make sure the Final Answer: {gold_label} is placed at the end, which means you will not paraphrase the phrase 'Final Answer'.""" 

#             if variation == 'plan_then_generate':
#                 prompt = f"""We have the {question} and the groundtruth {gold_label}


# 1. We wish you to plan then generate the final answer. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
# 2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
# 3. In your plan, you need to explain why you choose the particular API-Request. For example, if the previous API-Request is ToolSearch, then at this step you may need to use the tool found by the previous step. If the previous step alreaady used an API as a tool, then you may need to use ToolSearch to find the API for the next Action.
# 4. You will make a plan first then inference based on the plan then generate the Final Answer: {gold_label} at the end of the prediction like this

# PLAN_HERE
# INFERENCE_HERE
# Final Answer: {gold_label}
# """ 
            
        
        


        if not enable_batch_api:
            if not enable_mini_gpt4:
                answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
            else:
                answer = create_gpt_completion(prompt, MINI_MODEL_ENGINE, client)
            temp['answer'] = answer
            temp['gold_label'] = gold_label
            gpt4_answer_list.append(temp)
        else:
            if not enable_mini_gpt4:
                batch_api_list.append({"custom_id": f"{i}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": MODEL_ENGINE, "messages": [{"role": "system", "content": "provide answer"}, {"role": "user", "content": f"{prompt}"}], "max_tokens": 1000}})
            else:
                batch_api_list.append({"custom_id": f"{i}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": MINI_MODEL_ENGINE, "messages": [{"role": "system", "content": "provide answer"}, {"role": "user", "content": f"{prompt}"}], "max_tokens": 1000}})

            a = 1
        


    if enable_batch_api:
        if not enable_mini_gpt4:
            model_name = MODEL_ENGINE
        else:
            model_name = MINI_MODEL_ENGINE
        file_path = f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/gpt4_generated_data/batch_api/{task_name}_{step_by_step}_{model_name}_{len(question_list)}.jsonl"
        file_txt_path = f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/gpt4_generated_data/batch_api/{task_name}_{step_by_step}_{model_name}_{len(question_list)}.txt"
        with open(file_path, 'w') as file:
            for obj in batch_api_list:
                file.write(json.dumps(obj) + '\n')

        batch_input_file = client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        
        batch_job = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": f"{task_name}_{model_name}_{len(question_list)}"
            }
        )

        batch_job_id = batch_job.id

        with open(file_txt_path, 'w') as file:
            file.write(f"batch_job_id: {batch_job_id}")
    return gpt4_answer_list






def plan_bench_create_groundtruth(question_list, groundtruth_list = [], step_by_step = False, task_name = '', train_data_list = [], model_company = 'openai', variation = '', enable_batch_api = True, enable_mini_gpt4 = False):
    if model_company == 'openai':
        from openai import OpenAI
        import openai
        from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
        client = OpenAI(api_key=GPT_API)
        @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
        def create_gpt_completion(qa_, model_engine, client):
            response = client.chat.completions.create(
                model=model_engine,
                messages=[
                    {"role": "system", "content": "provide answer"},
                    {"role": "user", "content": f"{qa_}"}
                ]
            )
            answer = response.choices[0].message.content
            return answer
    elif model_company == 'anthropic':
        import anthropic
        my_api_key = os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(
            api_key=f"{my_api_key}",
        )
        def create_gpt_completion(qa_, model_engine, client):
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                temperature=0,
                messages=[
                        {"role": "user", "content": f"{qa_}"}
                ]
            )
            answer = message.content[0].text
            return answer

    batch_api_list = []
    gpt4_answer_list = []
    if step_by_step:
        step_by_step_insertion_1 = ' step by step'
        step_by_step_insertion_2 = 'Step by Step '
    else:
        step_by_step_insertion_1 = ''
        step_by_step_insertion_2 = ''
    for i, question in enumerate(question_list):
        gold_label = groundtruth_list[i]
        temp = {}
        temp['question'] = question
        temp['input'] = ''
        # AQuaRAT
        
        if task_name == 'PLAN_BENCH':
            temp = train_data_list[i]
            prompt = f"""We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}""" 
            if variation == 'step_by_step':
                prompt = f"""We have the {question} and the groundtruth {gold_label}


1. We wish you to answer the question{step_by_step_insertion_1}. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You will solve the problem in step by step manner.
3. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
4. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

{step_by_step_insertion_2}INFERENCE HERE
Final Answer: {gold_label}""" 
            if variation == 'provide_gpt4_example':

                q1 = \
'I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the orange block is on top of the blue block.\n\nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\nput down the blue block\npick up the orange block\nstack the orange block on top of the blue block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the orange block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]'
                
                q2 = \
'I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the orange block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]\nunstack the yellow block from on top of the orange block\nput down the yellow block\npick up the orange block\nstack the orange block on top of the red block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the blue block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the yellow block, the yellow block is on top of the red block and the red block is on the table.\nMy goal is to have that the red block is on top of the orange block and the yellow block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]'

                # q3 = \
# 'I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the blue block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the yellow block, the yellow block is on top of the red block and the red block is on the table.\nMy goal is to have that the red block is on top of the orange block and the yellow block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\nput down the blue block\nunstack the orange block from on top of the yellow block\nput down the orange block\nunstack the yellow block from on top of the red block\nstack the yellow block on top of the blue block\npick up the red block\nstack the red block on top of the orange block\nunstack the yellow block from on top of the blue block\nstack the yellow block on top of the red block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the yellow block is clear, the hand is empty, the red block is on top of the orange block, the orange block is on top of the blue block, the yellow block is on top of the red block and the blue block is on the table.\nMy goal is to have that the red block is on top of the yellow block and the yellow block is on top of the blue block.\n\nMy plan is as follows:\n\n[PLAN]'


# Example 3:

# question: "{q3}"

# groundtruth: 

# Inference: 
# To solve this problem, we need to sequence actions that respect the constraints and achieve the desired tower configuration. The initial configuration has the yellow block as the topmost and clear, and beneath it, the stack is: red on orange on blue on the table. The goal configuration has the red block on top of the yellow block, and the yellow block on top of the blue block, with the blue block on the table.\n\n1. We start by unstacking the yellow block from the red block because it is clear and our hand is empty initially. This clears the red block for the next action.\n2. We then put down the yellow block on the table so that our hand is empty.\n3. Next, we unstack the red block from the orange block because it is now clear.\n4. We put down the red block to clear our hand. Now, it's sitting on the table.\n5. Unstack the orange block from the blue block because now the orange block is clear.\n6. We put down the orange block to clear our hand.\n7. Now, the plan needs us to pick up the yellow block again.\n8. Then stack the yellow block onto the blue one since the blue block is clear on the table.\n9. Pick up the red block, which is clear on the table.\n10. Lastly, stack the red block on top of the yellow block. \n\nThe sequence of moves respects the constraints and results in achieving the configuration goal: red on yellow on blue.

# Final Answer: \n(unstack yellow red)\n(put-down yellow)\n(unstack red orange)\n(put-down red)\n(unstack orange blue)\n(put-down orange)\n(pick-up yellow)\n(stack yellow blue)\n(pick-up red)\n(stack red yellow)

                prompt = f"""We have the "{question}" and the groundtruth {gold_label}


1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer: {gold_label} at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}

Example 1:

Question : "{q1}"

groundtruth: (unstack yellow orange)\n(put-down yellow)\n(pick-up orange)\n(stack orange red)\n

Inference: 
Initial conditions:\n- Red block is on top of the blue block.\n- Yellow block is on top of the orange block.\n- Both yellow and red blocks are clear.\n- Hand is empty.\n\nGoal:\n- Orange block should be on top of the red block.\n\nPlan steps:\n1. **Unstack the yellow block from the orange block.** \n   - The yellow block is clear and can be unstacked. This action makes the orange block clear. The action is valid.\n   \n2. **Put down the yellow block.** \n   - After unstacking, you're holding the yellow block. You can put it down since the hand is not empty.\n   \n3. **Pick up the orange block.** \n   - The orange block is clear and now on the table. With an empty hand, you can pick it up.\n   \n4. **Stack the orange block on top of the red block.**\n   - The red block is clear after putting down the yellow block. While holding the orange block, you can stack it on the red block.\n\nEach step adheres to the initial conditions and satisfies the goals through logical, permissible operations.

Final Answer:\n(unstack yellow orange)\n(put-down yellow)\n(pick-up orange)\n(stack orange red)



Example 2:

question: "{q2}"

groundtruth: (unstack blue orange)\n(put-down blue)\n(unstack orange yellow)\n(put-down orange)\n(unstack yellow red)\n(stack yellow blue)\n(pick-up red)\n(stack red orange)\n(unstack yellow blue)\n(stack yellow red)

Inference: 
1. Start by unstacking the blue block from the orange block. This action is possible because the blue block is clear. Now the hand is holding the blue block, and the orange block is clear.\n\n2. Put down the blue block on the table. This clears the orange block completely and empties the hand.\n\n3. Unstack the orange block from the yellow block. This is possible because the orange block is now clear. Now the hand is holding the orange block, and the yellow block is clear.\n\n4. Put down the orange block on the table. This action ensures that the orange block is not disturbing the intended goal and empties the hand. \n\n5. Unstack the yellow block from the red block. This action is possible because the yellow block is clear. The hand now holds the yellow block, and the red block is clear.\n\n6. Stack the yellow block onto the blue block. This ensures that the yellow block is appropriately repositioned, necessary for the subsequent correct arrangement. Now the hand is empty, and the blue block is not clear as the yellow block is on top of it.\n\n7. Pick up the red block. This action is possible because the red block is clear. The hand now holds the red block.\n\n8. Stack the red block on top of the orange block. This places the red block in one part of the final goal, and the hand is now empty.\n\n9. Unstack the yellow block from the blue block. This action is necessary to correctly place the yellow block for the final configuration. The yellow block is now in hand, and the blue block is clear.\n\n10. Stack the yellow block on top of the red block. This places the yellow block in the correct final position, sitting on top of the red block to achieve the goal.

Final Answer: (unstack blue orange)\n(put-down blue)\n(unstack orange yellow)\n(put-down orange)\n(unstack yellow red)\n(stack yellow blue)\n(pick-up red)\n(stack red orange)\n(unstack yellow blue)\n(stack yellow red)



We have the "{question}" and the groundtruth {gold_label}


1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}

""" 

            if variation == 'human_written_examples':
                q1 = \
'I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the orange block is on top of the blue block.\n\nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\nput down the blue block\npick up the orange block\nstack the orange block on top of the blue block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the orange block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]'
                
                q2 = \
'I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the orange block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]\nunstack the yellow block from on top of the orange block\nput down the yellow block\npick up the orange block\nstack the orange block on top of the red block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the blue block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the yellow block, the yellow block is on top of the red block and the red block is on the table.\nMy goal is to have that the red block is on top of the orange block and the yellow block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]'

#                 q3 = \
# 'I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the blue block is clear, the hand is empty, the blue block is on top of the orange block, the orange block is on top of the yellow block, the yellow block is on top of the red block and the red block is on the table.\nMy goal is to have that the red block is on top of the orange block and the yellow block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\nput down the blue block\nunstack the orange block from on top of the yellow block\nput down the orange block\nunstack the yellow block from on top of the red block\nstack the yellow block on top of the blue block\npick up the red block\nstack the red block on top of the orange block\nunstack the yellow block from on top of the blue block\nstack the yellow block on top of the red block\n[PLAN END]\n\n[STATEMENT]\nAs initial conditions I have that, the yellow block is clear, the hand is empty, the red block is on top of the orange block, the orange block is on top of the blue block, the yellow block is on top of the red block and the blue block is on the table.\nMy goal is to have that the red block is on top of the yellow block and the yellow block is on top of the blue block.\n\nMy plan is as follows:\n\n[PLAN]'
                
                prompt = f"""We have the "{question}" and the groundtruth {gold_label}


1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}

Example 1:

Question : "{q1}"

groundtruth: (unstack yellow orange)\n(put-down yellow)\n(pick-up orange)\n(stack orange red)\n

Inference: The goal is to place the orange block on top of the red block. According to the initial conditions, the red block is clear, meaning there is nothing on top of it. Therefore, after we remove the orange block, we can directly place it onto the red block. Based on the initial conditions, the orange block is underneath the yellow block, and the yellow block is clear, meaning there is nothing on top of it. Since our hand is empty, we can directly pick up the yellow block and then put it down on the table. After that, we pick up the orange block and place it on the red block.

Final Answer:
(unstack yellow orange)  
(put-down yellow)  
(pick-up orange)  
(stack orange red)  



Example 2:

question: "{q2}"

groundtruth: (unstack blue orange)\n(put-down blue)\n(unstack orange yellow)\n(put-down orange)\n(unstack yellow red)\n(stack yellow blue)\n(pick-up red)\n(stack red orange)\n(unstack yellow blue)\n(stack yellow red)\n

Inference: According to the problem, we need to place the red block on top of the orange block. We find that only the red block is on the table. The yellow block is on top of the red block, the orange block is on top of the yellow block, and the blue block is on top of the orange block. The blue block is clear. Therefore, the arrangement order is blue, orange, yellow, and red, with nothing on top of the blue block.

Our goal is to place the red block on top of the orange block and the yellow block on top of the red block. To achieve this, we need to ensure that the bottom block in the final arrangement is the orange block. Based on our initial conditions, the red block is at the bottom, but in our goal, the red block is above the orange block. Therefore, we need to move all the blocks from the table and place them on the table in the correct order, then pick them up one by one and stack them according to the goal's order: yellow > red > orange.

Final Answer:
(unstack blue orange)  
(put down blue)  
(unstack orange yellow)  
(put down orange)  
(unstack yellow red)  
(stack yellow blue)  
(pick up red)  
(stack red orange)  
(unstack yellow blue)  
(stack yellow red)  



Now, please answer.

We have a question and a groundtruth 

question: "{question}"

groundtruth: {gold_label}


1. We wish you to answer the question. We will use your answer to train our model, thus you will answer and pretend as not knowing the gold_label.
2. You must answer the question (with inference process) directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
3. You will inference first then put the Final Answer ({gold_label}) at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}

""" 
            if variation == 'rewirte_groundtruth_in_own_words':
                prompt = f"""Given the question: {question} 
and the groundtruth: {gold_label}

Please states the prediction in your own words. The groundtruth is 100% correct. You should not change the problem solving logic of the groundtruth. just restates it in your own words.

1. You will pretend as you do not know the groundtruth, because we will use your prediction as target labels to train our model.
2. (important format) You must generate the groundtruth directly. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
3. (important format) Please make sure the Final Answer: {gold_label} is placed at the end of the modified prediction."""



        if not enable_batch_api:
            if not enable_mini_gpt4:
                answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
            else:
                answer = create_gpt_completion(prompt, MINI_MODEL_ENGINE, client)
            temp['answer'] = answer
            temp['gold_label'] = gold_label
            gpt4_answer_list.append(temp)
        else:
            if not enable_mini_gpt4:
                batch_api_list.append({"custom_id": f"{i}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": MODEL_ENGINE, "messages": [{"role": "system", "content": "provide answer"}, {"role": "user", "content": f"{prompt}"}], "max_tokens": 1000}})
            else:
                batch_api_list.append({"custom_id": f"{i}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": MINI_MODEL_ENGINE, "messages": [{"role": "system", "content": "provide answer"}, {"role": "user", "content": f"{prompt}"}], "max_tokens": 1000}})

            a = 1
        


    if enable_batch_api:
        if not enable_mini_gpt4:
            model_name = MODEL_ENGINE
        else:
            model_name = MINI_MODEL_ENGINE
        file_path = f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/gpt4_generated_data/batch_api/{task_name}_{step_by_step}_{model_name}_{len(question_list)}.jsonl"
        file_txt_path = f"/gpfs/users/a1796450/ACL_2024/Minimum_Change/gpt4_generated_data/batch_api/{task_name}_{step_by_step}_{model_name}_{len(question_list)}.txt"
        with open(file_path, 'w') as file:
            for obj in batch_api_list:
                file.write(json.dumps(obj) + '\n')

        batch_input_file = client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        
        batch_job = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": f"{task_name}_{model_name}_{len(question_list)}"
            }
        )

        batch_job_id = batch_job.id

        with open(file_txt_path, 'w') as file:
            file.write(f"batch_job_id: {batch_job_id}")
    return gpt4_answer_list



def create_groundtruth_varient(question_list, groundtruth_list = [], task_name = '', train_data_list = [], varient_name = '', gpt4_prediction_list = [], model_company = 'openai', API_from_tangquan = False):
    if model_company == 'openai':
        from openai import OpenAI
        import openai
        from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
        client = OpenAI(api_key=GPT_API)
        @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
        def create_gpt_completion(qa_, model_engine, client):
            response = client.chat.completions.create(
                model=model_engine,
                messages=[
                    {"role": "system", "content": "provide answer"},
                    {"role": "user", "content": f"{qa_}"}
                ]
            )
            answer = response.choices[0].message.content
            return answer
    elif model_company == 'anthropic':
        import anthropic
        import os
        import time
        import requests

        if not API_from_tangquan:
            my_api_key = os.environ.get("ANTHROPIC_API_KEY")
            client = anthropic.Anthropic(api_key=f"{my_api_key}")

            def create_gpt_completion(qa_, model_engine, client, retries=4, delay=5):
                for attempt in range(retries):
                    try:
                        message = client.messages.create(
                            model="claude-3-5-sonnet-20240620",
                            max_tokens=1000,
                            temperature=0,
                            messages=[
                                {"role": "user", "content": f"{qa_}"}
                            ]
                        )
                        answer = message.content[0].text
                        return answer
                    except:
                        a = 1
                        # break
                    time.sleep(delay)
                
                return None
        else:
            import requests
            import json
            import os
            import time

            OPENROUTER_API_KEY = os.environ.get("TANGQUAN_API")
            client = 1

            def create_gpt_completion(qa_, model_engine, client, retries=4, delay=5):
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}"
                }
                data = json.dumps({
                    "model": "anthropic/claude-3.5-sonnet",
                    "messages": [
                        {"role": "user", "content": f"{qa_}"}
                    ]
                })
                
                for attempt in range(retries):
                    try:
                        response = requests.post(url, headers=headers, data=data, timeout=10)
                        
                        if response.status_code == 200:
                            response_data = response.json()
                            message_content = response_data['choices'][0]['message']['content']
                            return message_content
                        # if response.status_code == 500:
                        #     print(f"Attempt {attempt+1} failed with 500 Internal Server Error. Retrying in {delay} seconds...")
                        # else:
                        #     print(f"Request failed with status code {response.status_code}: {response.text}")
                        #     break
                    except: 
                        a = 1                   
                        # break

                    time.sleep(delay)
                
                return None



    data_list = []
    for i, question in enumerate(question_list):
        if task_name.upper() == 'ECQA':
            gold_label = train_data_list[i]['gold_label']
            a = 1
        elif task_name.upper() == 'GSM8K' or task_name.upper() == 'MATH_ALGEBRA':
            gold_label = train_data_list[i]['numerical_final_answer']
            a = 1
        elif task_name.upper() == 'ANLI' or task_name.upper() == 'ESNLI' or task_name.upper() == 'BOOLQ' or task_name.upper() == 'PLAN_BENCH':
            # premise = train_data_list[i]['premise']
            # hypothesis = train_data_list[i]['hypothesis']
            gold_label = train_data_list[i]['gold_label']
            a = 1
        elif task_name.upper() == 'API_BANK':
            gold_label = train_data_list[i]['answer']
        temp = {}

        groundtruth = train_data_list[i]['answer']
        gpt4_prediction = gpt4_prediction_list[i]
        original_question = question.replace("\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n", '')
        original_question = original_question.replace("\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'.\n", '')
        original_question = original_question.replace("\n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'\n", '')
        original_question = original_question.replace("\n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'.\n", '')
        original_question = original_question.replace("\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'\n", '')
        original_question = original_question.replace("\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'.\n", '')
        if 'GSM8K' in task_name.upper() or 'MATH' in task_name.upper():
            gold_label_type = 'A_NUMBER'
        elif 'CODE' in task_name.upper() or 'MBPP' in task_name.upper():
            a = 1
        elif 'ANLI' in task_name.upper() or 'ESNLI' in task_name.upper():
            gold_label_type = 'Entailment/Contradiction/Neutral'
            parts = question.split("\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context.")
            original_question = parts[0]
            original_question_after = "\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context." + parts[1] if len(parts) > 1 else ''        
        elif 'ECQA' in task_name.upper():
            gold_label_type = '1/2/3/4/5'
            parts = question.split('\nand the options:\n')
            original_question = parts[0]
            original_question_after = '\nand the options:\n' + parts[1] if len(parts) > 1 else ''        
        if varient_name == 'step_by_step':
            prompt = f"""We have the question and the groundtruth. Please reformat the groundtruth in step by step manner with details.

Question: {question}
Groundtruth: {groundtruth}



1. We wish you to regenerate a new groundtruth. The new groundtruth solve the problem step by step. If you believe the groundtruth is not detail enough, you could add details.
2. You will pretend as you do not know the groundtruth, because we will use your prediction as target labels to train our model.
3. (important format) You must generate the groundtruth with the step by step inference process directly. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
4. (important format) You will inference first then put the Final Answer: {gold_label}

at the end like this

INFERENCE HERE
Final Answer: {gold_label}""" 
        
        # suppose to have good perofrmance
        elif varient_name == 'rewirte_groundtruth_in_own_words':
            prompt = f"""Given the question: {question} 
and the groundtruth: {groundtruth}

Please states the prediction in your own words. The groundtruth is 100% correct. You should not change the problem solving logic of the groundtruth. just restates it in your own words.

1. You will pretend as you do not know the groundtruth, because we will use your prediction as target labels to train our model.
2. (important format) You must generate the groundtruth directly. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
3. (important format) Please make sure the Final Answer: {gold_label} is placed at the end of the modified prediction."""

        elif varient_name == 'rewirte_gpt4_answer_with_different_styles':
            prompt = f"""Given the question: {question} 
and the prediction: {gpt4_prediction}

Please states the prediction with different language styles and logic styles(you only generate one prediction, not multiple prediction). Also make sure the details in the prediction is enough.

1. You will pretend as you do not know the prediction, because we will use your prediction with different styles as target labels to train our model.
2. (important format) You must generate the predictions with different styles directly. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label' or 'the previous prediction is wrong, i will transfer it to different styles'
3. (important format) Please make sure the Final Answer: {gold_label_type} is placed at the end of the modified prediction. Please make sure nothing is placed after Final Answer: {gold_label_type}"""
        elif varient_name == 'rewirte_gpt4_answer_with_different_logic_styles':
            prompt = f"""Given the question: {question} 
and the prediction: {gpt4_prediction}

Please states the prediction with different logic styles. Make sure to keep the original language styles.(you only generate one prediction, not multiple prediction). Also make sure the details in the prediction is enough.

1. You will pretend as you do not know the prediction, because we will use your prediction with different logic styles as target labels to train our model.
2. (important format) You must generate the predictions with different logic styles directly. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label' or 'the previous prediction is wrong, i will transfer it to different styles'
3. (important format) Please make sure the Final Answer: {gold_label_type} is placed at the end of the modified prediction. Please make sure nothing is placed after Final Answer: {gold_label_type}"""
        elif varient_name == 'paraphrase_the_language_of_gpt4_answer':
            gpt4_prediction = gpt4_prediction_list[i]
            prompt = f"""Given the question: {question}
            
and the prediction: {gpt4_prediction} 

please create a new varient of the prediction by paraphrasing it. 

1. the keywords in your prediction has to follow the question. For example, if the question is if a = 1, b = 2, what is a + b? and the given solution is a + b = 1 + 2 = 3, then the paraphrased solution can be given a = 1, b = 2, the solution is 1 + 2 = 3.  It cannot be like if c = 1, d = 2, then c + d = 3.
2. do not change the meaning of the sentence in the prediction even if the text is logically wrong. Do not add extra detail. just paraphrase it. 
3. (important format) You must create the varient of the prediction directly without say anything else. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
4. (important format) Please make sure the Final Answer: {gold_label_type} is placed at the end of the paraphrased prediction.""" 
        elif varient_name == 'paraphrase_the_inference_process_of_gpt4_answer':
            prompt = f"""Given the question: {question}
            
and the prediction: {gpt4_prediction} 

please create a new vairent of the prediction by rewriting the inference process of it(if possible).  You need to rewrite the inference process and also try to keep the original language styles (if possible).

1. the keywords in your prediction has to follow the question, but you are trying to modify the problem solving logic. 
2. Please rewrite the inference process of the prediction to make a new varient of the prediction. Try to keep the original language styles if you could.
3. (important format) You must create the varient of the prediction directly without say anything else. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
4. (important format) Please make sure the Final Answer: {gold_label_type} is placed at the end of the paraphrased prediction.""" 
        elif varient_name == 'paraphrase_gpt4_answer':
            prompt = f"""Given the question: {question}
            
and the original target labels: {gpt4_prediction} 

please create a new vairent of the prediction by rewriting the original target labels. You need to make sure that the new varient of the target labels can serve as a new target labels that is different from the original target labels.

1. the keywords in your the new varient of target labels has to follow the question.
2. (important format) You must create the varient of the prediction directly without say anything else. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
4. (important format) Please make sure the Final Answer: {gold_label_type} is placed at the end of the new varient of the target label.""" 
        elif varient_name == 'direct_paraphrase_gpt4_answer':
            prompt = f"""Given the question: {question}
            
and the prediction: {gpt4_prediction} 

1. Please paraphrase the prediction. The prediction might be wrong sometimes, but you do not need to correct it. just paraphrase it.
2. (important format) You must create the paraphrased prediction directly without say anything else. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
4. (important format) Please make sure the Final Answer: {gold_label_type} is placed at the end, which means you will not paraphrase the phrase 'Final Answer'.""" 
        elif varient_name == 'add_details':
            prompt = f"""We have the question and the groundtruth, but sometimes the groundtruth is not detailed enough. Please add more details to the groundtruth to make it better if you believe there needs to be more details.
Question: {question}
Groundtruth: {groundtruth}


1. (important format) We wish you to generate the detailed groundtruth directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
2. You will pretend as you do not know the groundtruth. We will to use your prediction as target labels to train our model.
3. (important format) You will inference first then put the Final Answer: {gold_label} at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}"""   
        elif varient_name == 'rewrite_with_more_details':
            prompt = f"""We have the question and the groundtruth, but sometimes the groundtruth is not detailed enough. Please add more details to the groundtruth to make better groundtruth if you believe there needs to be more details.
Question: {question}
Groundtruth: {groundtruth}


1. (important format) We wish you to generate the detailed groundtruth directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
2. You have to directly generate the rewritten groundtruth with more details(if the original groundtruth is detailed enough, then no need to add details).
3. (important format) Your new groundtruth will perform detailed inference first then put the Final Answer: {gold_label} at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}"""  
        elif varient_name == 'create_gpt4_style_question':
            prompt = f"""We have a math question. We wish to rewrite the question in the language style of GPT-4. The question is as follows: "{original_question}"


important format: We wish you to generate the gpt-4 style directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, here is the question that is in gpt-4 style'

PUT_THE_GPT4_STYLE_QUESTION_HERE"""
        elif varient_name == 'add_details_while_keep_original_distribution':
            prompt = \
f"""We have the question and the groundtruth, but sometimes the groundtruth is not detailed enough. Please add more details to the groundtruth to make better groundtruth if you believe there needs to be more details. 
When adding detail, you must try your best to keep the original words unchange. You can add more details, but you are trying to keep as much original words from groundtruth unchanged as possible.


Question: {question}
Groundtruth: {groundtruth}


1. (important format) We wish you to generate the detailed groundtruth directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
2. You have to directly generate the rewritten groundtruth with more details(if the original groundtruth is detailed enough, then no need to add details). When adding detail, you must try your best to keep the original words unchange. You can add more details, but you are trying to keep as much original words from groundtruth unchanged as possible.
3. When adding details, you should imitate the language styles of the original groundtruth.
4. (important format) Your new groundtruth will perform detailed inference first then put the Final Answer: {gold_label} at the end of the prediction like this

INFERENCE HERE
Final Answer: {gold_label}"""  
        elif varient_name == 'detailed_step_by_step_while_keep_original_distribution':
            prompt = \
f"""We have the question and the groundtruth. Please reformat the groundtruth in step by step manner with details for each step. When adding detail, you must try your best to keep the original words and logic unchange. You are reformating the groundtruth with more details on each step instead of rewrite the groundtruth, but you are trying to keep original words from groundtruth unchanged.


Question: {question}
Groundtruth: {groundtruth}


1. We wish you to reformat a new groundtruth. The new groundtruth are reformated a new groundtruth which solve the problem step by step with detail on each step.
2. You will pretend as you do not know the groundtruth, because we will use your prediction as target labels to train our model.
3. When adding step by step inference process, you should imitate the languaguage styles of the original groundtruth.
4. (important format) You must generate the groundtruth with the detailed step by step inference process directly. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
5. (important format) You will inference first then put the Final Answer: {gold_label}

at the end like this

INFERENCE HERE
Final Answer: {gold_label}""" 
        elif varient_name == 'redundant':
            prompt = \
f"""We have the question and the groundtruth. Given on the groundtruth, please reformat the groundtruth so that it answer the question in a step by step redundant manner. Be as repetitive and step by step and redundant as possible.


Question: {question}
Groundtruth: {groundtruth}


1. We wish you to reformat a new groundtruth. The new groundtruth are reformated a new groundtruth which solve the problem as steo by step and redundant as possible.
2. You will pretend as you do not know the groundtruth, because we will use your step by step redundant answer as target responses to train our model.
3. (important format) You must generate the groundtruth with the step by step redundant inference process directly. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
4. (important format) You will inference first then put the Final Answer: {gold_label}

at the end like this

INFERENCE HERE
Final Answer: {gold_label}"""
        elif varient_name == 'step_by_step_while_keep_original_distribution':
            prompt = \
f"""We have the question and the groundtruth. Please reformat the groundtruth in step by step manner. You must try your best to keep the original words and logic unchange. You are reformating the groundtruth into step by step solution. you could add new words, but you are trying to keep original words from groundtruth unchanged.


Question: {question}
Groundtruth: {groundtruth}



1. We wish you to reformat a new groundtruth. The new groundtruth are reformated a new groundtruth which solve the problem step by step. you could add new words, but you are trying to not change the original words.
2. You will pretend as you do not know the groundtruth, because we will use your prediction as target labels to train our model.
3. When adding step by step inference process, you should imitate the language styles of the original groundtruth.
4. (important format) You must generate the groundtruth with the step by step inference process directly. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
5. (important format) You will inference first then put the Final Answer: {gold_label}

at the end like this

INFERENCE HERE
Final Answer: {gold_label}""" 
        
        elif varient_name == 'paraphrase_question':
#             if 'GSM8K' in task_name.upper() or 'MATH' in task_name.upper():
#                 prompt = \
# f"""Given the solution '{groundtruth}', Please paraphrase this question {original_question}. Your goal is to create a new question that ask the same question as original question does but with different wording. The new question has mathch the solution by maintaining the keywords.
# 1. (most important format) Please do not change the keywords, you may paraphrase the sentence, but the keywords should remain the same(at least make sure the keywords in the groundtruth match the keywords in the original question).
# 2. make sure the new question ask the same question but the language style is very different.
# 3. (important format) please do not change the meaning of the question. makesure all of the details in the original question is maintained.
# 5. (important format) You must paraphrase the question directly without say anything else. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
# 6. (important format) Your must only paraphrase the question without solving it.
# """ 
#             elif 'ANLI' in task_name.upper() or 'ESNLI' in task_name.upper():
#                 prompt = \
# f"""Given the answer '{gold_label}', Please paraphrase this question {original_question}. 
# 4. (important format) Please do not change the keywords. you may paraphrase the sentence, but the keywords should remain the same. makesure the new question align with the options.
# 2. make sure the new question ask the same question but the language style is very different.
# 3. (important format) please do not change the meaning of the question. makesure all of the details in the original question is maintained.
# 5. (important format) You must paraphrase the question directly without say anything else. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
# 6. (important format) Your must only paraphrase the question without solving it.
# """ 
            # elif 'ECQA' in task_name.upper():
            
            prompt = \
f"""Given the solution:'{groundtruth}'

Please paraphrase this question: {original_question}

Your goal is to create a new question that ask the same question as original question does but with different wording. The new question has mathch the solution by maintaining the keywords. For example, if the given solution is a + b = 1 + 2 = 3, then the paraphrased question can be if a = 1, b = 2, what is a + b? It cannot be like if c = 1, d = 2, what is c + d?

1. (most important format) Please do not change the keywords, you may paraphrase the sentence, but the keywords should remain the same(at least make sure the keywords in the groundtruth match the keywords in the original question).
2. make sure the new question ask the same question but the language style is very different.
3. (important format) please do not change the meaning of the question. makesure all of the details in the original question is maintained.
5. (important format) You must paraphrase the question directly without say anything else. Please not saying anything like 'sure I can help you with' or 'sure, i will not mention the gold label'
6. (important format) Your must only paraphrase the question without solving it.
""" 
            if 'ANLI' in task_name.upper() or 'ESNLI' in task_name.upper():
                prompt += "7. You cannot paraphrase the keywords 'entailment', 'contradiction', 'neutral', 'context', 'statement'"
        elif 'imitate_human_detailed_example' in varient_name:
            if 'GSM8K' in task_name.upper():
                prompt = f"""Given the question: {question} 
and the groundtruth {groundtruth}


Please transform groundtruth in a more natural language and detailed way. We provide 2 transformation examples. Please transform the groundtruth into the similar styles. 
Please make sure the Final Answer (a number) is placed at the end of the modified prediction. 

Example 1: We modify the solution so that it sounds more fluent and it seems more like a human written solution. We also add more details to the solution.
Original question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Original groundtruth: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72
Transformed groundtruth: In April, 

In April, Natalia sold clips to 48 of her friends. In May, she sold half as many clips as in April. To find how many clips she sold in May, we divide the number of clips sold in April by 2.\n\nMay's sales: 48 / 2 = 24 clips.\n\nNow we add the clips sold in April to the clips sold in May to get the total.\n\nTotal clips sold in April and May: 48 (April) + 24 (May) = 72 clips.\n\Final Answer: 72

Example 2: We modify the solution so that it sounds more fluent and it seems more like a human written solution. We also add more details to the solution.
Original question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Original groundtruth: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10
Transformed groundtruth: Weng earns $12 per hour and babysat for 50 minutes. An hour has 60 minutes. To find out Weng's earnings for 50 minutes, we perform the following calculation:\n\n(50 minutes / 60 minutes) * $12/hour = X\n\n(5/6) * $12 = X\n\n$10 = X\n\nFinal Answer: 10


Given the question: {question}
and the groundtruth: {groundtruth}
Transformed groundtruth: 

Please give me the transformed groundtruth directly without saying anything like sure i can help you with.
Remember to make sure the Final Answer (a number) is placed at the end of the transformedd groundtruth. 
""" 


        elif 'generate_detailed_example' in varient_name:
            if 'GSM8K' in task_name.upper():
                prompt = f"""Given the question: {question} 
and the groundtruth {groundtruth}


Please transform groundtruth in a more natural language and detailed way. We provide 2 transformation examples. Please transform the groundtruth into the similar styles. 
Please make sure the Final Answer (a number) is placed at the end of the modified prediction. 

Example 1: We modify the solution so that it sounds more fluent and it seems more like a human written solution. We also add more details to the solution.
Original question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Original groundtruth: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72
Transformed groundtruth: In April, Natalia sold clips to 48 of her friends. In May, she sold half as many clips as in April. To find how many clips she sold in May, we divide the number of clips sold in April by 2.\n\nMay's sales: 48 / 2 = 24 clips.\n\nNow we add the clips sold in April to the clips sold in May to get the total.\n\nTotal clips sold in April and May: 48 (April) + 24 (May) = 72 clips.\n\Final Answer: 72

Example 2: We modify the solution so that it sounds more fluent and it seems more like a human written solution. We also add more details to the solution.
Original question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Original groundtruth: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10
Transformed groundtruth: Weng earns $12 per hour and babysat for 50 minutes. An hour has 60 minutes. To find out Weng's earnings for 50 minutes, we perform the following calculation:\n\n(50 minutes / 60 minutes) * $12/hour = X\n\n(5/6) * $12 = X\n\n$10 = X\n\nFinal Answer: 10


Given the question: {question}
and the groundtruth: {groundtruth}
Transformed groundtruth: 

Please give me the transformed groundtruth directly without saying anything like sure i can help you with.
Remember to make sure the Final Answer (a number) is placed at the end of the transformedd groundtruth. 
""" 
        
        
        
        if 'rewrite_in_gpt4_1106_style' in varient_name or 'rewrite_in_mc_style' in varient_name:
            if 'MATH' in task_name.upper():
                math_example_1_question = \
"Find the distance between the points (0,15) and (8,0).\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"
                math_example_1_gt = \
"We use the distance formula: $$\\sqrt{(8 - 0)^2 + (0 - 15)^2} = \\sqrt{64 + 225} = \\boxed {17}.$$- OR -\n\nWe note that the points $(0, 15)$, $(8, 0)$, and $(0, 0)$ form a right triangle with legs of length 8 and 15. This is a Pythagorean triple, so the hypotenuse must have length $\\boxed{17}$. Final Answer: 17"
                if 'rewrite_in_gpt4_1106_style ' in varient_name:
                    math_example_1_transformed_gt = \
"To find the distance between the two points, (0,15) and (8,0), you can use the distance formula derived from the Pythagorean Theorem:\n\nDistance = sqrt((x2 - x1)^2 + (y2 - y1)^2)\n\nWhere (x1, y1) = (0, 15) and (x2, y2) = (8, 0).\n\nPlugging these values into the formula gives us:\nDistance = sqrt((8 - 0)^2 + (0 - 15)^2)\nDistance = sqrt(8^2 + (-15)^2)\nDistance = sqrt(64 + 225)\nDistance = sqrt(289)\nDistance = 17\n\nFinal Answer: 17"
                elif 'rewrite_in_mc_style ' in varient_name:
                    math_example_1_transformed_gt = \
""

                math_example_2_question = \
"he graph of the line $x+y=b$ intersects the line segment from $(2,5)$ to $(4,9)$ at its midpoint. What is the value of $b$?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"
                math_example_2_gt = \
"If the line $x+y=b$ intersects the midpoint, which is: $$\\left(\\frac{2+4}{2},\\frac{5+9}{2}\\right)=(3,7)$$This point lies on the line $x+y=b$, so we must have $3+7=b$. Thus, $b=\\boxed{10}$. Final Answer: 10"
                if 'rewrite_in_gpt4_1106_style ' in varient_name:
                    math_example_2_transformed_gt = \
"To find the value of $b$, we'll first determine the midpoint of the line segment that connects the points $(2,5)$ and $(4,9)$. The midpoint $M$ of a line segment connecting two points $(x_1, y_1)$ and $(x_2, y_2)$ is given by the average of their coordinates:\n\n$$M = \\left(\\frac{x_1 + x_2}{2}, \\frac{y_1 + y_2}{2}\\right)$$\n\nPlugging in the given points $(2,5)$ and $(4,9)$:\n\n$$M = \\left(\\frac{2 + 4}{2}, \\frac{5 + 9}{2}\\right)$$\n\n$$M = \\left(\\frac{6}{2}, \\frac{14}{2}\\right)$$\n\n$$M = (3,7)$$\n\nNow, we need to find the value of $b$ in the equation $x + y = b$ that corresponds to this midpoint. Substituting the midpoint's coordinates, we get:\n\n$$3 + 7 = b$$\n\n$$b = 10$$\n\nFinal Answer:\n10"
                elif 'rewrite_in_mc_style ' in varient_name:
                    math_example_2_transformed_gt = \
""

                prompt = \
f"""Please transform the following groundtruth to better styles. The example below show you how to make groundtruth transformation to better style properly. Please perform the transformation directly without any explaination.

Example1 Question: "{math_example_1_question}",

Example1 Groundtruth: "{math_example_1_gt}"

Example1 Transformed Groundtruth: "{math_example_1_transformed_gt}"



Example2 Question: "{math_example_2_question}",

Example2 Groundtruth: "{math_example_2_gt}"

Example2 Transformed Groundtruth: "{math_example_2_transformed_gt}”



Now, please trasnform this groundtruth to better styles directly without any explaination.

Question: "{question}”

Groundtruth: “{groundtruth}”

Transformed Groundtruth: """

            if 'GSM8K' in task_name.upper():
                gsm8k_example_1_question = \
"Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"
                gsm8k_example_1_gt = \
"Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\nFinal Answer: 72"
                if 'rewrite_in_gpt4_1106_style ' in varient_name:
                    gsm8k_example_1_transformed_gt = \
"Natalia sold 48 clips to her friends in April. In May, she sold half as many, so in May, she sold 48 / 2 = 24 clips. Adding April and May sales together: 48 clips (April) + 24 clips (May) = 72 clips.\n\nFinal Answer: 72"
                elif 'rewrite_in_mc_style ' in varient_name:
                    gsm8k_example_1_transformed_gt = \
""

                gsm8k_example_2_question = \
"James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"
                gsm8k_example_2_gt = \
"He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\n#### 624"
                if 'rewrite_in_gpt4_1106_style ' in varient_name:
                    gsm8k_example_2_transformed_gt = \
"James writes a 3-page letter to each of 2 friends, which totals to 3 pages x 2 friends = 6 pages per writing session. He writes twice a week, so in a week, he writes 6 pages x 2 = 12 pages. There are 52 weeks in a year, so in a year, he writes 12 pages x 52 weeks = 624 pages.\n\nFinal Answer: 624"
                elif 'rewrite_in_mc_style ' in varient_name:
                    gsm8k_example_2_transformed_gt = \
""

                prompt = \
f"""Please transform the following groundtruth to better styles. The example below show you how to make groundtruth transformation to better style properly. Please perform the transformation directly without any explaination.

Example1 Question: "{gsm8k_example_1_question}",

Example1 Groundtruth: "{gsm8k_example_1_gt}"

Example1 Transformed Groundtruth: "{gsm8k_example_1_transformed_gt}"



Example2 Question: "{gsm8k_example_2_question}",

Example2 Groundtruth: "{gsm8k_example_2_gt}"

Example2 Transformed Groundtruth: "{gsm8k_example_2_transformed_gt}”



Now, please trasnform this groundtruth to better styles directly without any explaination.

Question: "{question}”

Groundtruth: “{groundtruth}”

Transformed Groundtruth: """


            if 'ECQA' in task_name.upper():
                ecqa_example_1_question = \
"We have the question: What might a person see at the scene of a brutal killing?\nand the options:\n(1): bloody mess\n(2): pleasure\n(3): being imprisoned\n(4): feeling of guilt\n(5): cake\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'\n"
                ecqa_example_1_gt = \
"Bloody mess"
                if 'rewrite_in_mc_style' in varient_name:
                    ecqa_example_1_transformed_gt = \
"""Based on the context of the question and the given options, it is likely that the scene of a brutal killing would be disturbing and, therefore, neither \"pleasure\" nor \"cake\" would be appropriate to the scene. Thus, we can eliminate options (2) and (5).

Given the nature of a brutal killing, it is reasonable to expect that the aftermath would be quite chaotic and messy, characterized by a significant amount of blood. Hence, the correct option is most likely to be (1) \"bloody mess,\" which corresponds with what one might typically see at such a tragic scene.

Final Answer: 1"""
                elif 'rewrite_in_gpt4_1106_style' in varient_name:
                    ecqa_example_1_transformed_gt = \
"""In the context of the scene of a brutal killing, the expected observation would be directly related to the immediate aftermath of a violent act. So, among the given options, the one that most appropriately reflects this would be a graphic description of the scene.

Option (1) "bloody mess" fits this because it describes the likely visual consequences of a brutal killing. Options (2) "pleasure," (3) "being imprisoned," and (4) "feeling of guilt" are psychological responses or potential subsequent developments rather than direct observations at the scene. Option (5) "cake" is completely incongruent and does not fit the context of a brutal killing scene.

Final Answer: 1"""


                ecqa_example_2_question = \
"We have the question: If you want to set a romantic atmosphere you might light a candle where?\nand the options:\n(1): dimly lit room\n(2): synagogue\n(3): bedroom\n(4): birthday cake\n(5): roses\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'\n"
                ecqa_example_2_gt = \
"Bedroom"
                if 'rewrite_in_mc_style' in varient_name:
                    ecqa_example_2_transformed_gt = \
"""Based on the context of the question, the correct answer would be (3) bedroom. The question mentions setting a romantic atmosphere, which is often associated with a quiet and intimate space like a bedroom. The other options, such as a dimly lit room (1), synagogue (2), birthday cake (4), and roses (5), while they can contribute to a romantic setting, are not the specific location where one would typically light a candle to set the mood.

Final Answer: 3"""
                elif 'rewrite_in_gpt4_1106_style' in varient_name:
                    ecqa_example_2_transformed_gt = \
"""To set a romantic atmosphere, you would typically look for a setting that's private, intimate, and conducive to romance. Here is the reasoning for each option:

(1): A dimly lit room can create a romantic atmosphere due to the subdued lighting, which often sets a mood of relaxation and intimacy.

(2): A synagogue is a place of worship and is not typically associated with romance. It's not a common choice for creating a romantic atmosphere.

(3): A bedroom is often considered a very private and intimate space, making it a prime location for romance.

(4): Birthday cake is associated with celebrations such as birthdays, not necessarily with romance.

(5): Roses are often associated with romance, but they are not a place where you light a candle. They instead might be an element of romantic decoration.

Considering the above, the setting that best fits the description of lighting a candle to create a romantic atmosphere would be a dimly lit room or a bedroom. Since a bedroom inherently implies privacy and intimacy, it is the most obvious choice for a romantic setting.

Final Answer: 3 """

                prompt = \
f"""Given the question and the groundtruth, please write a solution in GPT-4 to styles. We did provide 2 examples of how to write good solution given q and a. Please use the same styles to create solution.

Example1 Question: "{ecqa_example_1_question}",

Example1 Groundtruth: "{ecqa_example_1_gt}"

Example1 Solution: "{ecqa_example_1_transformed_gt}"



Example2 Question: "{ecqa_example_2_question}",

Example2 Groundtruth: "{ecqa_example_2_gt}"

Example2 Solution: "{ecqa_example_2_transformed_gt}”



Please use the same styles to create solution.

Question: "{question}”

Groundtruth: “{gold_label}”

Solution: """
            if 'ESNLI' in task_name.upper():
                esnli_example_1_question = \
"We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nWe have \nContext: A person on a horse jumps over a broken down airplane.\nStatement: A person is training his horse for a competition.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context."
                esnli_example_1_gt = \
"Neutral"
                if 'rewrite_in_mc_style' in varient_name:
                    esnli_example_1_transformed_gt = \
"Inference: The statement 'A person is training his horse for a competition' is neutral given the context. The context states that a person is on a horse jumping over a broken down airplane, but it does not provide any information about th e purpose of the jump or the training status of the horse. Therefore, the truth of the statement is undetermined or irrelevant given the context.\n\nFinal Answer: Neutral."

# """Analysis:
# The context describes a situation where a person is riding a horse and jumps over a broken-down airplane. The statement, on the other hand, asserts that a person is training his horse for a competition.

# There is no direct connection between the context and the statement. The context describes an action taken by the person with his horse, but it does not provide any information about the purpose or intent behind the action. Training a horse for a competition is a possible explanation for the person's actions, but it is not the only one. Other explanations could include practicing for fun, preparing for a long journey, or simply enjoying the experience of riding his horse.

# Therefore, based on the given context, the statement "A person is training his horse for a competition" is neutral. The truth of the statement is neither directly entailed nor contradicted by the context.\n\nFinal Answer: Neutral."""
                elif 'rewrite_in_gpt4_1106_style' in varient_name:
                    esnli_example_1_transformed_gt = \
"""The given statement, "A person is training his horse for a competition," given the context, "A person on a horse jumps over a broken down airplane," could be seen as potentially related, but it is not definitively confirmed by the context provided.

- It is not an entailment, because jumping over a broken down airplane does not necessarily mean that the person is training for a competition. They could be doing it for a variety of reasons, such as for a stunt, for a dare, for a film shoot, or just for personal enjoyment.

- It is not a contradiction, because the context does not provide information that directly contradicts the statement. There is nothing in the context that suggests the person is not training for a competition.

- The statement falls under the category of neutral. The context leaves the truth of the statement undetermined; there is no explicit information indicating that the person is specifically training for a competition, nor is there information explicitly stating otherwise.

Therefore, the relationship between the statement and the context is neutral.

Final Answer: Neutral"""
# """Analysis:
# - The given context describes an event where a person on a horse is jumping over a broken down airplane.
# - The given statement suggests that this person is training his horse for a competition.

# To determine the relationship between the context and the statement, we must evaluate whether the action described in the context can be reasonably associated with the activity mentioned in the statement.

# Training for a horse competition often involves a variety of exercises to improve the horse's agility, strength, discipline, and ability to navigate obstacles. Jumping over obstacles is a common training activity for horses, especially when preparing for competitive events that may include show jumping or cross-country events. While jumping over a broken down airplane is an unconventional obstacle, it could theoretically be part of an unusual or rigorous training routine.

# Given that the context does not explicitly negate the idea of training for a competition but instead depicts an action that could concievably be part of a training session, we can argue that the context and the statement are not in direct contradiction. However, the context also doesn't explicitly confirm that the person is training for a competition; it simply states what they are doing at the moment without explaining their intent.

# Therefore, the statement "A person is training his horse for a competition" is not definitely true or false based on the provided context, since the context doesn't sufficiently elaborate on the purpose or motivation behind the person's action.

# Final Answer: Neutral"""

# """Inference:
# The context suggests an unusual scenario where a person on a horse is jumping over a broken-down airplane. This indicates an action that might plausibly happen in a training situation for a particularly challenging or unusual equestrian competition where horses need to be prepared for unconventional obstacles. However, the context does not explicitly confirm this to be training for a competition; there could be other reasons for the person to jump over a broken-down airplane (e.g., for entertainment, in an emergency, as part of a stunt). Therefore, we cannot definitively conclude from the given context that the training is occurring for a competition, but the possibility is not ruled out either.

# Final Answer: Neutral"""

                esnli_example_2_question = \
"We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nWe have \nContext: Children smiling and waving at camera\nStatement: The kids are frowning\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context."
                esnli_example_2_gt = \
"Contradiction"
                if 'rewrite_in_mc_style' in varient_name:
                    esnli_example_2_transformed_gt = \
"""Based on the context provided, the statement 'The kids are frowning' is a contradiction. The context describes children smiling and waving at the camera, which is the opposite of frowning. Therefore, the statement is definitely false given the context.\n\nFinal Answer: Contradiction."""

# """Based on the context and statement provided, the statement "The kids are frowning" is a contradiction given the context of "Children smiling and waving at camera." The context describes children who are smiling and waving at the camera, which is an expression of happiness and positivity. The statement, however, asserts that the children are frowning, which is the opposite of smiling and waving. Therefore, the statement is a contradiction given the context.\n\nFinal Answer: Contradiction."""
                elif 'rewrite_in_gpt4_1106_style' in varient_name:
                    esnli_example_2_transformed_gt = \
"""Given the context of "Children smiling and waving at camera," and the statement "The kids are frowning," the relationship between the context and the statement is one of contradiction. The statement that the kids are frowning is definitely false because the context clearly indicates that they are smiling, which is the opposite of frowning. Therefore, the statement contradicts the given context.

Final Answer: Contradiction"""
# """Analysis:
# The context describes "Children smiling and waving at camera," which suggests a positive and happy action. Smiling is the opposite of frowning and indicates a joyful or friendly expression. When individuals smile, they are not frowning since these two expressions are mutually exclusive—one cannot do both simultaneously.

# Statement:
# "The kids are frowning" directly opposes the context, which states that the children are smiling. Therefore, because the statement contradicts the context provided, it cannot be true as long as the context holds.

# Conclusion:
# The statement "The kids are frowning" is a contradiction given the context "Children smiling and waving at camera.

# Final Answer: Contradiction"""
# """Inference: If the context is children smiling and waving at the camera, then a statement that says "The kids are frowning" would directly oppose the given context. Smiling is the opposite of frowning; therefore, the context and the statement cannot both be true.

# Final Answer: Contradiction"""


                    
#                 esnli_example_3_question = \
# "We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nWe have \nContext: Two women who just had lunch hugging and saying goodbye.\nStatement: There are two woman in this picture.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context. \n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'\n"
#                 esnli_example_3_gt = \
# "Entailment"
#                 if 'rewrite_in_mc_style' in varient_name:
#                     esnli_example_3_transformed_gt = \
# """Inference: The statement \"There are two women in this picture\" is an entailment given the context. The context describes two women hugging and saying goodbye, and the statement asserts the presence of two women in the picture, which is a direct observation based on the context.\n\nFinal Answer: Entailment."""
#                 elif 'rewrite_in_gpt4_1106_style' in varient_name:
#                     esnli_example_3_transformed_gt = \
# """Inference:
# The context provided describes two women who just had lunch and are hugging and saying goodbye. This implies that there are indeed two women present in the scenario mentioned.

# Final Answer: Entailment"""

# Example3 Question: "{esnli_example_3_question}",

# Example3 Groundtruth: "{esnli_example_3_gt}"

# Example3 Solution: "{esnli_example_3_transformed_gt}”

                prompt = \
f"""Given the question and the groundtruth, please write a solution in GPT-4 to styles. We did provide 2 examples of how to write good solution given q and a. Please use the same styles to create solution.

Example1 Question: "{esnli_example_1_question}",

Example1 Groundtruth: "{esnli_example_1_gt}"

Example1 Solution: "{esnli_example_1_transformed_gt}"



Example2 Question: "{esnli_example_2_question}",

Example2 Groundtruth: "{esnli_example_2_gt}"

Example2 Solution: "{esnli_example_2_transformed_gt}”




Please use the same styles to create solution.

Question: "{question}”

Groundtruth: “{gold_label}”

Solution: """
            
            if 'BOOLQ' in task_name.upper():
                boolq_example_1_question = \
"Given the context: Persian (/\u02c8p\u025c\u02d0r\u0292\u0259n, -\u0283\u0259n/), also known by its endonym Farsi (\u0641\u0627\u0631\u0633\u06cc f\u0101rsi (f\u0252\u02d0\u027e\u02c8si\u02d0) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.\n\ndo iran and afghanistan speak the same language?\n\nPlease inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'\n"
                boolq_example_1_gt = \
"True"
                if 'rewrite_in_mc_style' in varient_name:
                    boolq_example_1_transformed_gt = \
"""Persian, also known as Farsi, is the predominant language in Iran. Dari is the name used for the Persian language in Afghanistan since 1958. Both are variants of the same language, belonging to the Western Iranian languages within the Indo-Iranian branch of the Indo-European family. While there may be some differences in vocabulary, pronunciation, and grammar between the two, they are mutually intelligible to a significant degree.

Final Answer: True"""
                elif 'rewrite_in_gpt4_1106_style' in varient_name:
                    boolq_example_1_transformed_gt = \
"""Inference: Persian, or Farsi, is mentioned as the language spoken in Iran. In Afghanistan, the same language is known as Dari. While the languages are considered different dialects or varieties, they stem from the same Persian language roots and have a high degree of mutual intelligibility. Thus, while they are considered different entities for cultural and political reasons, linguistically, they are very similar and essentially variants of the same language.

Final Answer: True"""

                boolq_example_2_question = \
"Given the context: As with other games in The Elder Scrolls series, the game is set on the continent of Tamriel. The events of the game occur a millennium before those of The Elder Scrolls V: Skyrim and around 800 years before The Elder Scrolls III: Morrowind and The Elder Scrolls IV: Oblivion. It has a broadly similar structure to Skyrim, with two separate conflicts progressing at the same time, one with the fate of the world in the balance, and one where the prize is supreme power on Tamriel. In The Elder Scrolls Online, the first struggle is against the Daedric Prince Molag Bal, who is attempting to meld the plane of Mundus with his realm of Coldharbour, and the second is to capture the vacant imperial throne, contested by three alliances of the mortal races. The player character has been sacrificed to Molag Bal, and Molag Bal has stolen their soul, the recovery of which is the primary game objective.\n\nis elder scrolls online the same as skyrim?\n\nPlease inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'\n"
                boolq_example_2_gt = \
"False"
                if 'rewrite_in_mc_style' in varient_name:
                    boolq_example_2_transformed_gt = \
"""Inference: Although The Elder Scrolls Online and Skyrim share some similarities in terms of setting, game structure, and themes, they are not the same game. The Elder Scrolls Online is an MMORPG (Massively Multiplayer Online Role-Playing Game) and takes place a millennium before Skyrim. The main objective in The Elder Scrolls Online is to recover the player character's stolen soul from Molag Bal and contend for the imperial throne, whereas Skyrim focuses on the player's journey to become the Dragonborn and save the world from destruction. Additionally, The Elder Scrolls Online is an MMORPG, while Skyrim is a single-player game.

Final Answer: False."""
                elif 'rewrite_in_gpt4_1106_style' in varient_name:
                    boolq_example_2_transformed_gt = \
"""The Elder Scrolls Online (ESO) and The Elder Scrolls V: Skyrim are both part of The Elder Scrolls series of video games, which means they share the same fictional universe, the continent of Tamriel. However, they are distinct games with different gameplay mechanics, storylines, timelines, and objectives.

The Elder Scrolls Online is an MMORPG (massively multiplayer online role-playing game) that allows players to explore the continent of Tamriel together with others online. It features various factions and a storyline involving the conflict against the Daedric Prince Molag Bal as well as a power struggle for the imperial throne.

On the other hand, The Elder Scrolls V: Skyrim is a single-player action RPG that focuses on the player's journey as the Dragonborn to defeat a dragon prophesied to destroy the world. It is set in the province of Skyrim, and its gameplay revolves around exploring a vast open world, completing quests, and developing the character's skills.

Although both games are open-world RPGs set in the same universe, ESO has a more extensive multiplayer focus and is set roughly 800-1000 years before the events of Skyrim. Skyrim is more focused on the single-player experience with a narrative centered around the Dragonborn's role in the prophecy.

Final Answer: False"""

                prompt = \
f"""Given the question and the groundtruth, please write a solution in GPT-4 to styles. We did provide 2 examples of how to write good solution given question and groundtruth. Please use the same styles to create solution.

Example1 Question: "{boolq_example_1_question}",

Example1 Groundtruth: "{boolq_example_1_gt}"

Example1 Solution: "{boolq_example_1_transformed_gt}"



Example2 Question: "{boolq_example_2_question}",

Example2 Groundtruth: "{boolq_example_2_gt}"

Example2 Solution: "{boolq_example_2_transformed_gt}”



Please use the same styles to create solution.

Question: "{question}”

Groundtruth: “{gold_label}”

Solution: """
            


            if 'WINOGRANDE' in task_name.upper():
                winogrande_example_1_question = \
"Given the question: Ian volunteered to eat Dennis's menudo after already having a bowl because _ despised eating intestine.\n\nWhat option is correct?\nOption 1: Ian\nOption 2: Dennis\n\nPlease inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'\n"
                winogrande_example_1_gt = \
"2"
                if 'rewrite_in_mc_style' in varient_name:
                    winogrande_example_1_transformed_gt = \
"""Inference: The person who despised eating intestine is the one who did not want to eat Dennis's menudo initially. Since Ian volunteered to eat it after already having a bowl, it means he was not the one who disliked it. Therefore, Dennis must be the one who despised eating intestine.

Final Answer: 2"""
                elif 'rewrite_in_gpt4_1106_style' in varient_name:
                    winogrande_example_1_transformed_gt = \
"""Inference:
The sentence implies that one of the individuals disliked eating intestine. Since Ian volunteered to eat Dennis's menudo after already having a bowl of his own, it suggests that Ian does not have an issue with eating intestine. It is more likely that Dennis is the one who despises eating intestine, which is why Ian stepped in to eat Dennis's menudo.

Final Answer: 2"""

                winogrande_example_2_question = \
"Given the question: He never comes to my home, but I always go to his house because the _ is smaller.\n\nWhat option is correct?\nOption 1: home\nOption 2: house\n\nPlease inference first, then provide the final answer (1/2) at the end, after 'Final Answer:'\n"
                winogrande_example_2_gt = \
""
                if 'rewrite_in_mc_style' in varient_name:
                    winogrande_example_2_transformed_gt = \
"""Inference: The sentence is comparing the sizes of the speakers' homes and the other person's house. Since the reason given for the speaker not going to the other person's house is that it is smaller, it is clear that "home" in this context refers to the speaker's home, and "house" refers to the other person's house.

Final Answer: 1"""
                elif 'rewrite_in_gpt4_1106_style' in varient_name:
                    winogrande_example_2_transformed_gt = \
"""In the context of the sentence, it implies that one of the two places (the speaker's home or the other person's house) is smaller, which makes it more convenient or preferable for one person to visit the other. The sentence is looking to be completed with the subject that is smaller.

The two options given are 'home' and 'house'. These words are often used interchangeably in everyday language, and both refer to a place where someone lives. However, in the context of this sentence, we must pick one to indicate which location is smaller.

Given that the sentence has already used both 'my home' and 'his house' to distinguish between the two dwellings, it would be logical to assume that the blank should be filled with the word that refers back to the subject's own dwelling. In this case, since the subject refers to their own dwelling as 'home', the more coherent choice would be 'home' to maintain consistency within the sentence.

Final Answer: 1"""

                prompt = \
f"""Given the question and the groundtruth, please write a solution in GPT-4 to styles. We did provide 2 examples of how to write good solution given question and groundtruth. Please use the same styles to create solution.

Example1 Question: "{winogrande_example_1_question}",

Example1 Groundtruth: "{winogrande_example_1_gt}"

Example1 Solution: "{winogrande_example_1_transformed_gt}"



Example2 Question: "{winogrande_example_2_question}",

Example2 Groundtruth: "{winogrande_example_2_gt}"

Example2 Solution: "{winogrande_example_2_transformed_gt}”



Please use the same styles to create solution.

Question: "{question}”

Groundtruth: “{gold_label}”

Solution: """


#         if 'new_rewrite_strategy' in varient_name:
#             if 'MATH' in task_name.upper():
#                 math_example_1_question = \
# "Find the distance between the points (0,15) and (8,0).\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"
#                 math_example_1_gt = \
# "We use the distance formula: $$\\sqrt{(8 - 0)^2 + (0 - 15)^2} = \\sqrt{64 + 225} = \\boxed {17}.$$- OR -\n\nWe note that the points $(0, 15)$, $(8, 0)$, and $(0, 0)$ form a right triangle with legs of length 8 and 15. This is a Pythagorean triple, so the hypotenuse must have length $\\boxed{17}$. Final Answer: 17"
#                 math_example_1_transformed_gt = \
# "To find the distance between the two points, (0,15) and (8,0), you can use the distance formula derived from the Pythagorean Theorem:\n\nDistance = sqrt((x2 - x1)^2 + (y2 - y1)^2)\n\nWhere (x1, y1) = (0, 15) and (x2, y2) = (8, 0).\n\nPlugging these values into the formula gives us:\nDistance = sqrt((8 - 0)^2 + (0 - 15)^2)\nDistance = sqrt(8^2 + (-15)^2)\nDistance = sqrt(64 + 225)\nDistance = sqrt(289)\nDistance = 17\n\nFinal Answer: 17"

#                 math_example_2_question = \
# "he graph of the line $x+y=b$ intersects the line segment from $(2,5)$ to $(4,9)$ at its midpoint. What is the value of $b$?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n"
#                 math_example_2_gt = \
# "If the line $x+y=b$ intersects the midpoint, which is: $$\\left(\\frac{2+4}{2},\\frac{5+9}{2}\\right)=(3,7)$$This point lies on the line $x+y=b$, so we must have $3+7=b$. Thus, $b=\\boxed{10}$. Final Answer: 10"
#                 math_example_2_transformed_gt = \
# "To find the value of $b$, we'll first determine the midpoint of the line segment that connects the points $(2,5)$ and $(4,9)$. The midpoint $M$ of a line segment connecting two points $(x_1, y_1)$ and $(x_2, y_2)$ is given by the average of their coordinates:\n\n$$M = \\left(\\frac{x_1 + x_2}{2}, \\frac{y_1 + y_2}{2}\\right)$$\n\nPlugging in the given points $(2,5)$ and $(4,9)$:\n\n$$M = \\left(\\frac{2 + 4}{2}, \\frac{5 + 9}{2}\\right)$$\n\n$$M = \\left(\\frac{6}{2}, \\frac{14}{2}\\right)$$\n\n$$M = (3,7)$$\n\nNow, we need to find the value of $b$ in the equation $x + y = b$ that corresponds to this midpoint. Substituting the midpoint's coordinates, we get:\n\n$$3 + 7 = b$$\n\n$$b = 10$$\n\nFinal Answer:\n10"

#                 prompt = \
# f"""Please transform the following groundtruth to better styles. Please perform the transformation directly without any explaination.

# Example1 Question: "{math_example_1_question}",

# Example1 Groundtruth: "{math_example_1_gt}"

# Example1 Transformed Groundtruth: "{math_example_1_transformed_gt}"



# Example2 Question: "{math_example_2_question}",

# Example2 Groundtruth: "{math_example_2_gt}"

# Example2 Transformed Groundtruth: "{math_example_2_transformed_gt}”



# Now, please trasnform this groundtruth to better styles directly without any explaination.

# Question: "{question}”

# Groundtruth: “{groundtruth}”

# Transformed Groundtruth: """
#         if 'rewrite_in_gpt4_style' in varient_name:
#             if 'ECQA' in task_name.upper():
#                 ecqa_example_1_question = \
# "We have the question: What might a person see at the scene of a brutal killing?\nand the options:\n(1): bloody mess\n(2): pleasure\n(3): being imprisoned\n(4): feeling of guilt\n(5): cake\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'\n"
#                 ecqa_example_1_gt = \
# "Bloody mess"
#                 ecqa_example_1_transformed_gt = \
# """To infer the correct option, let's break down the context of the question and each option:

# 1. Bloody mess: In a scene described as a "brutal killing," it is likely that there would be visible blood and signs of violence. This option aligns with the context of a violent crime scene.

# 2. Pleasure: This option is not consistent with the context of a brutal killing, which is generally associated with distressing and horrific conditions rather than pleasure.

# 3. Being imprisoned: While imprisonment might be a consequence of a crime, it is not something typically observed directly at the scene of the killing itself.

# 4. Feeling of guilt: This option refers to an emotional state rather than something seen at the crime scene.

# 5. Cake: This option is unrelated to the context of a brutal killing and is not something one would expect to see at such a scene.

# Given the context of a brutal killing, the most appropriate option that describes what a person might see is a "bloody mess."

# Final Answer: 1"""

#                 ecqa_example_2_question = \
# "We have the question: If you want to set a romantic atmosphere you might light a candle where?\nand the options:\n(1): dimly lit room\n(2): synagogue\n(3): bedroom\n(4): birthday cake\n(5): roses\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'\n"
#                 ecqa_example_2_gt = \
# "Bedroom"
#                 ecqa_example_2_transformed_gt = \
# """To infer the correct option, let's analyze the context of setting a romantic atmosphere and each option:

# 1. Dimly lit room: A dimly lit room can be enhanced with candles to create a romantic atmosphere. This option is relevant to creating a romantic setting.

# 2. Synagogue: While candles might be used in religious settings, this is not typically associated with setting a romantic atmosphere.

# 3. Bedroom: A bedroom is a common place where people might light candles to set a romantic mood. This option aligns well with the idea of creating a romantic atmosphere.

# 4. Birthday cake: Candles on a birthday cake are typically used for celebrations rather than setting a romantic atmosphere.

# 5. Roses: While roses are romantic, they do not involve lighting candles.

# Given the context of setting a romantic atmosphere, the most appropriate place to light a candle is a "bedroom."

# Final Answer: 3"""

#                 prompt = \
# f"""Given the question and the groundtruth, please write a solution in GPT-4 to styles. We did provide 2 examples of how to write good solution given q and a. Please use the same styles to create solution.

# Example1 Question: "{ecqa_example_1_question}",

# Example1 Groundtruth: "{ecqa_example_1_gt}"

# Example1 Solution: "{ecqa_example_1_transformed_gt}"



# Example2 Question: "{ecqa_example_2_question}",

# Example2 Groundtruth: "{ecqa_example_2_gt}"

# Example2 Solution: "{ecqa_example_2_transformed_gt}”



# Please use the same styles to create solution.

# Question: "{question}”

# Groundtruth: “{gold_label}”

# Solution: """
#             if 'ESNLI' in task_name.upper():
#                 esnli_example_1_question = \
# "We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nWe have \nContext: A person on a horse jumps over a broken down airplane.\nStatement: A person is training his horse for a competition.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context. \nMake sure to analyze first, then provide the answer.\n"
#                 esnli_example_1_gt = \
# "Neutral"
#                 esnli_example_1_transformed_gt = \
# "Context: A person on a horse jumps over a broken down airplane.\nStatement: A person is training his horse for a competition.\n\nInference: The context does not provide any information about why the person is jumping over the broken down airplane. It could be part of a training for a competition, but it could also be for entertainment, a showcase, a survival scenario, or any other number of reasons. The provided statement that a person is training his horse for a competition is a conceivable explanation for the context; however, it is not the only possible explanation. Therefore, the context does not confirm or deny the statement.\n\nFINAL ANSWER: Neutral"

#                 esnli_example_2_question = \
# "We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nWe have \nContext: A person on a horse jumps over a broken down airplane.\nStatement: A person is outdoors, on a horse.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context. \nMake sure to analyze first, then provide the answer.\n"
#                 esnli_example_2_gt = \
# "Entailment"
#                 esnli_example_2_transformed_gt = \
# """Let's analyze the given context and statement:

# Context: A person on a horse jumps over a broken down airplane.

# Statement: A person is outdoors, on a horse.

# Analysis:

# 1. Entailment: For the statement to be an entailment, it must be definitely true based on the context. In this case, the context describes a person on a horse interacting with a broken-down airplane. Given that airplanes are typically found outdoors and not indoors, and since there's no indication of an indoor setting, it can be reasonably inferred that the person is outdoors. Therefore, the statement aligns with the context.

# 2. Contradiction: For the statement to be a contradiction, it must be definitely false based on the context. Since the context does not provide any evidence that the person is indoors, and the presence of an airplane suggests an outdoor setting, this statement is not contradicted by the context.

# 3. Neutral: For the statement to be neutral, the truth of the statement would be undetermined or irrelevant based on the context. In this case, the context does provide enough information to determine that the person is likely outdoors, making the statement relevant rather than neutral.

# Given the context of a person on a horse jumping over a broken-down airplane, which implies an outdoor setting, the statement that "a person is outdoors, on a horse" is definitely true.

# Final Answer: Entailment"""

#                 prompt = \
# f"""Given the question and the groundtruth, please write a solution in GPT-4 to styles. We did provide 2 examples of how to write good solution given q and a. Please use the same styles to create solution.

# Example1 Question: "{esnli_example_1_question}",

# Example1 Groundtruth: "{esnli_example_1_gt}"

# Example1 Solution: "{esnli_example_1_transformed_gt}"



# Example2 Question: "{esnli_example_2_question}",

# Example2 Groundtruth: "{esnli_example_2_gt}"

# Example2 Solution: "{esnli_example_2_transformed_gt}”



# Please use the same styles to create solution.

# Question: "{question}”

# Groundtruth: “{gold_label}”

# Solution: """
            
#             if 'BOOLQ' in task_name.upper():
#                 boolq_example_1_question = \
# "Given the context: Persian (/\u02c8p\u025c\u02d0r\u0292\u0259n, -\u0283\u0259n/), also known by its endonym Farsi (\u0641\u0627\u0631\u0633\u06cc f\u0101rsi (f\u0252\u02d0\u027e\u02c8si\u02d0) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.\n\ndo iran and afghanistan speak the same language?\n\nPlease inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'\n"
#                 boolq_example_1_gt = \
# "True"
#                 boolq_example_1_transformed_gt = \
# """Inference:

# Persian, also known as Farsi, is a language that is spoken in Iran, Afghanistan, and Tajikistan. In Iran, it is referred to as Farsi, while in Afghanistan it is called Dari. Dari is a variety of Persian that is spoken in Afghanistan. Although Dari and Farsi are essentially the same language, they have some differences in pronunciation, vocabulary, and usage. Tajikistan also uses a variant of Persian known as Tajiki, which is written in the Cyrillic script.

# Final Answer: True"""

#                 boolq_example_2_question = \
# "Given the context: As with other games in The Elder Scrolls series, the game is set on the continent of Tamriel. The events of the game occur a millennium before those of The Elder Scrolls V: Skyrim and around 800 years before The Elder Scrolls III: Morrowind and The Elder Scrolls IV: Oblivion. It has a broadly similar structure to Skyrim, with two separate conflicts progressing at the same time, one with the fate of the world in the balance, and one where the prize is supreme power on Tamriel. In The Elder Scrolls Online, the first struggle is against the Daedric Prince Molag Bal, who is attempting to meld the plane of Mundus with his realm of Coldharbour, and the second is to capture the vacant imperial throne, contested by three alliances of the mortal races. The player character has been sacrificed to Molag Bal, and Molag Bal has stolen their soul, the recovery of which is the primary game objective.\n\nis elder scrolls online the same as skyrim?\n\nPlease inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'\n"
#                 boolq_example_2_gt = \
# "False"
#                 boolq_example_2_transformed_gt = \
# """Inference:

# The Elder Scrolls Online (ESO) and The Elder Scrolls V: Skyrim are both set in the same universe of Tamriel, but they are distinct from each other. ESO is set a millennium before Skyrim, and it focuses on a different set of events, including conflicts involving Daedric Princes and the struggle for the imperial throne. Skyrim, on the other hand, is a single-player game that takes place in a different time period and focuses on the player's individual quest to deal with the Dragonborn prophecy and other localized conflicts.

# Final Answer: False"""

#                 prompt = \
# f"""Given the question and the groundtruth, please write a solution in GPT-4 to styles. We did provide 2 examples of how to write good solution given question and groundtruth. Please use the same styles to create solution.

# Example1 Question: "{boolq_example_1_question}",

# Example1 Groundtruth: "{boolq_example_1_gt}"

# Example1 Solution: "{boolq_example_1_transformed_gt}"



# Example2 Question: "{boolq_example_2_question}",

# Example2 Groundtruth: "{boolq_example_2_gt}"

# Example2 Solution: "{boolq_example_2_transformed_gt}”



# Please use the same styles to create solution.

# Question: "{question}”

# Groundtruth: “{gold_label}”

# Solution: """

        if 'rewrite_in_gpt4_1106_style' in varient_name or 'rewrite_in_mc_style' in varient_name:
            ecqa_example_1_question = \
"We have the question: What might a person see at the scene of a brutal killing?\nand the options:\n(1): bloody mess\n(2): pleasure\n(3): being imprisoned\n(4): feeling of guilt\n(5): cake\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'\n"
            ecqa_example_1_gt = \
"Bloody mess is covered or stained with blood. A person might see a bloody mess at the scene of a brutal killing.  Pleasure is about what a person sees at the scene of a brutal killing and one cannot be happy to see such brutality. You can’t see someone in jail at the brutal killing scene. Feeling of guilt doesn’t come as the killing is brutal or merciless. Cake is baseless and weird to think as it is a brutal killing scene and not a bakery.\n\nFinal Answer: 1"
            ecqa_example_1_transformed_gt = \
"Based on the context of the question and the given options, it is likely that the scene of a brutal killing would be disturbing and, therefore, neither \"pleasure\" nor \"cake\" would be appropriate to the scene. Thus, we can eliminate options (2) and (5).\n\nGiven the nature of a brutal killing, it is reasonable to expect that the aftermath would be quite chaotic and messy, characterized by a significant amount of blood. Hence, the correct option is most likely to be (1) \"bloody mess,\" which corresponds with what one might typically see at such a tragic scene.\n\nFinal Answer: 1"

            ecqa_example_2_question = \
"We have the question: Where would you borrow coffee if you do not have any?\nand the options:\n(1): meeting\n(2): convenience store\n(3): supermarket\n(4): fast food restaurant\n(5): friend's house\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'\n"
            ecqa_example_2_gt = \
"If you are finished with stock of coffee beans / powder and don't want to buy it, you can borrow it from friend's home because you can't borrow from meeting and other options are selling the coffee\n\nFinal Answer: 5"

            ecqa_example_2_transformed_gt = \
"Based on the context of the question, I would infer that the person is asking about a place where one can obtain coffee when they do not have any at home. Among the given options, a meeting, a convenience store, a supermarket, and a fast food restaurant are places associated with getting coffee, but not typically venues where one would borrow it.\n\nA friend's house, on the other hand, fits the social norm where borrowing is a common and acceptable practice. If you do not have any coffee, it would be most appropriate to ask a friend if you could borrow some, with the expectation that you would return the favor or replace it at a later time.\n\nFinal Answer: 5"




            esnli_example_1_question = \
"We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nWe have \nContext: A person on a horse jumps over a broken down airplane.\nStatement: A person is training his horse for a competition.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context. \n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'\n"
            esnli_example_1_gt = \
"Because the person is not necessarily training his horse the answer is neutral.\nFinal Answer: neutral"
            esnli_example_1_transformed_gt = \
"Inference: The context states that a person is on a horse jumping over a broken down airplane, but it does not provide any information about the purpose of the jump or the training status of the horse. Therefore, the truth of the statement is undetermined or irrelevant given the context.\n\nFinal Answer: Neutral."

            esnli_example_2_question = \
"We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nWe have \nContext: Children smiling and waving at camera\nStatement: The kids are frowning\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context. \n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'\n"
            ensli_example_2_gt = \
'Because one cannot be smiling and frowning at the same time, the answer is contradiction.\nFinal Answer: contradiction'
            ensli_example_2_transformed_gt = \
"The context describes children smiling and waving at the camera, which is the opposite of frowning. Therefore, the statement is definitely false given the context.\n\nFinal Answer: Contradiction."
            if 'ECQA' in task_name.upper():
                prompt = \
f"""There are question and groundtruth, but the groundtruth is not written in a good style. 

Please generate a new groundtruth in your own styles. Please genearte it directly without any explaination.

Example1 Question: "{ecqa_example_1_question}",

Example1 Groundtruth: "{ecqa_example_1_gt}"

Example1 Groundtruth in your own style: "{ecqa_example_1_transformed_gt}"



Example2 Question: "{ecqa_example_2_question}",

Example2 Groundtruth: "{ecqa_example_2_gt}"

Example2 Groundtruth in your own style: "{ecqa_example_2_transformed_gt}”




Now, please rewrite this groundtruth your own styles without any explaination.

Question: "{question}”

Groundtruth: “{groundtruth}”

Groundtruth in your own style: """
            if 'ESNLI' in task_name.upper():
                prompt = \
f"""There are question and groundtruth, but the groundtruth is not written in a good style. 

Please generate a new groundtruth in your own styles. Please genearte it directly without any explaination.

Example1 Question: "{esnli_example_1_question}",

Example1 Groundtruth: "{esnli_example_1_gt}"

Example1 Groundtruth in your own style: "{esnli_example_1_transformed_gt}"



Example2 Question: "{esnli_example_2_question}",

Example2 Groundtruth: "{ensli_example_2_gt}"

Example2 Groundtruth in your own style: "{ensli_example_2_transformed_gt}”




Now, please transform this groundtruth in your own styles without any explaination.

Question: "{question}”

Groundtruth: “{groundtruth}”

Groundtruth in your own style: """
        
        
        if varient_name == 'paraphrase_question' or varient_name == 'create_gpt4_style_question':
            question_paraphrased = create_gpt_completion(prompt, MODEL_ENGINE, client)
        else:
            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        if varient_name == 'paraphrase_question' or varient_name == 'create_gpt4_style_question':
            if 'GSM8K' in task_name.upper() or 'MATH' in task_name.upper():
                question_paraphrased += "\n\nPlease provide the final answer (a number) at the end, after 'Final Answer: '"
            elif 'ECQA' in task_name.upper():
                question_paraphrased += original_question_after
            elif 'ANLI' in task_name.upper() or 'ESNLI' in task_name.upper():
                question_paraphrased += "\n\nPlease inference first, then provide the final answer (entailment/contradiction/neutral) at the end, after 'Final Answer:'"
            question = question_paraphrased

        temp['question'] = question
        temp['input'] = ''
        if varient_name == 'paraphrase_question' or varient_name == 'create_gpt4_style_question':
            gpt4_prediction = gpt4_prediction_list[i]
            temp['answer'] = gpt4_prediction
        else:
            temp['answer'] = answer
        temp['gold_label'] = gold_label

        temp['groundtruth'] = groundtruth
        temp['original_gpt4_prediction'] = gpt4_prediction
        data_list.append(temp)
    return data_list


def back_translation_creator(minimum_change_list, task_name = ''):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    gpt4_answer_list = []
    for i, item in enumerate(minimum_change_list):
        prediction = item['answer']
        question = item['question']
        temp = {}
        temp['input'] = ''
        prompt = f"""We have the prediction： {prediction} 


1. 请把这段话翻译成中文
2. 你必须直接回答问题. 不要说任何其他的话。请不要说像“当然我可以帮你”，“这是您要的翻译过的对话”或”translated text : xxxxx”的话。
3. 请不要翻译'FINAL ANSWER:' 我们需要让这两个字以英文形式呈现.请不要移动'FINAL ANSWER:'的位置。""" 
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        prompt = f""" we have the prediction in chinese {answer}
1. please directly translate this to english. 
2. do not change anymeaning even if the text is logically wrong. just translate it.
3. You must answer the question directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label' 
4. """ 
    
        
        temp['question'] = question
        temp['input'] = ''
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        temp['answer'] = answer
        gpt4_answer_list.append(temp)
    return gpt4_answer_list



def paraphrase_question(minimum_change_list, task_name = ''):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    gpt4_answer_list = []
    for item in minimum_change_list:
        question = item['question']
        prompt = f"""Please paraphrase this question {question}. make sure the new question ask the same question but the language style is very different.

make sure to not change this part of the question 'Please provide the final answer (a number) at the end, after 'Final Answer:''""" 
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        item['question'] = answer
        gpt4_answer_list.append(item)
    return gpt4_answer_list

def clean_string(s):
    for i in range(2):
        # Check and remove a leading single or double quote
        if s.startswith(("'", '"')):
            s = s[1:]

        # Check and remove a trailing single or double quote
        if s.endswith(("'", '"')):
            s = s[:-1]
    return s


def paraphrase_data(minimum_change_list, task_name = ''):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    gpt4_answer_list = []
    for i, item in enumerate(minimum_change_list):
        prediction = item['answer']
        question = item['question']
        temp = {}
        temp['input'] = ''
        if task_name != 'CODE':
            prompt = f"""We have the prediction: {prediction} 


1. please paraphrase it without changing the order of sentence struecture. 
2. do not change anymeaning even if the text is logically wrong. just paraphrase it
3. You must answer the question directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'""" 
        else:
            prompt = f"""We got a code problem and we have the prediction: {prediction} 


1. We wish to get a paraphrase the prediction. please paraphrase it without changing the order of sentence struecture. 
2. Please not change the code. you can only change the sentences.
3. do not change any meaning even if the text is logically wrong. just paraphrase it
4. You must answer the question directly without say anything else. Please not saying anything 'like sure I can help you with' or 'sure, i will not mention the gold label'
5. You still need to keep the FINAL ANSWER at the end.
6. Remember this is a code problem. if you change the code, the code will fail."""
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)

        temp['question'] = question
        temp['input'] = ''
        temp['answer'] = answer
        if task_name == 'CODE':
            temp = item
            temp['answer'] == answer
        gpt4_answer_list.append(temp)
    return gpt4_answer_list



def proof_read_data_creator(data_list, task_name = '', creation_num = 1):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    gpt4_answer_list = []
    if task_name == 'gsm8k':
        for item in data_list:
            prediction = item['answer']
            question = item['question']
            temp = {}
            temp['input'] = ''
            if creation_num == 1:
                prompt = f"""Given the question: {question} 
and the prediction {prediction}

Please states the prediction in your own words. The prediction is 100% correct. You should not change the problem solving logic of the prediction. just states it in your own words.

Please make sure the Final Answer (a number) is placed at the end of the modified prediction. """ 
            # elif creation_num == 2:
#                 prompt = f"""Given the question: {question} 
# and the prediction {prediction}

# please solve the problem.


# Please make sure the Final Answer (a number) is placed at the end of the modified prediction. """ 
                # initial_prediction = create_gpt_completion(prompt, MODEL_ENGINE, client)
#                 prompt = f"""

# please modify the text style of the solution such that it is speaking similar to the initial prediction. The modified solution should not change the problem solving logic of the solution. just modify the text style.

# For example, if the question is Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nPlease inference then provide the final answer at the end.
# The solution is Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72
# and the initial prediction is Natalia sold 48 clips to her friends in April. In May, she sold half as many, which is 48/2 = 24 clips. To find out how many clips she sold altogether in April and May, we add the two amounts together:\n\n48 (April) + 24 (May) = 72 clips.\n\nFinal Answer: 72.

# You should modify the solution to be similar(as similar as possible. Use the same words is a strategy to make it similar) to the initial prediction and keep the original logic. 

# So the modified version is Natalia sold 48 clips to her friends in April. In May, she sold half as many, which is 48/2 = 24 clips. To find out how many clips she sold altogether in April and May, we add the two amounts together:\n\n48 (April) + 24 (May) = 72 clips.\n\nFinal Answer: 72.


# Given the question: {question} 
# and the solution: {prediction}
# and the initial prediction: {initial_prediction}
# Please make sure the Final Answer (a number) is placed at the end of the modified prediction. """ 
            
#                 prompt = f"""
# Given the problem: {question} 
# and the hint: {prediction}
# please solve the problem. please pretend as you do not know the hint. 
# Please make sure the Final Answer (a number) is placed at the end of the modified prediction. """ 
            
            if creation_num == 3:
#                 prompt = f"""Given the question: {question} 

# # please solve the problem.


# # Please make sure the Final Answer (a number) is placed at the end of the modified prediction. """ 
#                 initial_prediction = create_gpt_completion(prompt, MODEL_ENGINE, client)
                prompt = f"""Given the question: {question} 
and the groundtruth {prediction}


Please transform groundtruth in a more natural language way. We provide 3 transform examples. Please transform the groundtruth into the similar styles. 
Please make sure the Final Answer (a number) is placed at the end of the modified prediction. 

Example 1: please notice that we slightly modify the solution so that it sounds more fluent and it seems more like a human written solution. We do not add extremly more information because human do not solve the problem that way. when a human is solving problem, he does not write down toooooo much information. please make sure you do not add extra information that the original groundtruth not talking about.
Original question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Original groundtruth: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72
Transformed groundtruth: In April, Natalia sold clips to 48 of her friends. In May, she sold half as many clips as in April. To find how many clips she sold in May, we divide the number of clips sold in April by 2.\n\nMay's sales: 48 / 2 = 24 clips.\n\nNow we add the clips sold in April to the clips sold in May to get the total.\n\nTotal clips sold in April and May: 48 (April) + 24 (May) = 72 clips.\n\Final Answer: 72

Example 2: please notice that we slightly modify the solution so that it sounds more fluent and it seems more like a human written solution. We do not add extremly more information because human do not solve the problem that way. when a human is solving problem, he does not write down toooooo much information. please make sure you do not add extra information that the original groundtruth not talking about.
Original question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Original groundtruth: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10
Transformed groundtruth: Weng earns $12 per hour and babysat for 50 minutes. An hour has 60 minutes. To find out Weng's earnings for 50 minutes, we perform the following calculation:\n\n(50 minutes / 60 minutes) * $12/hour = X\n\n(5/6) * $12 = X\n\n$10 = X\n\nFinal Answer: 10


Given the question: {question}
and the groundtruth: {prediction}
Transformed groundtruth: 

Please give me the transformed groundtruth directly without saying anything like sure i can help you with.
please make sure you do not add extra information that the original groundtruth not talking about.
Remember to make sure the Final Answer (a number) is placed at the end of the transformedd groundtruth. 
""" 
            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
            item['answer'] = answer
            item['previous_prediction'] = prediction
            gpt4_answer_list.append(item)
    if task_name == 'math_algebra':
        for item in data_list:
            prediction = item['answer']
            question = item['question']
            temp = {}
            temp['input'] = ''
            if creation_num == 1:
                prompt = f"""Given the question: {question} 
and the prediction {prediction}

Please states the prediction in your own words. The prediction is 100% correct. You should not change the problem solving logic of the prediction. just states it in your own words.

Please make sure the Final Answer (a number) is placed at the end of the modified prediction. """ 

            if creation_num == 3:
                prompt = f"""Given the question: {question} 
and the groundtruth {prediction}


Please transform groundtruth in a more natural language way. We provide 3 transform examples. Please transform the groundtruth into the similar styles. 
Please make sure the Final Answer (a number) is placed at the end of the modified prediction. 

Example 1: please notice that we slightly modify the solution so that it sounds more fluent and it seems more like a human written solution. We do not add extremly more information because human do not solve the problem that way. when a human is solving problem, he does not write down toooooo much information. please make sure you do not add extra information that the original groundtruth not talking about.
Original question: If $x + y = 16$ and $x-y = 2$, what is the value of $x^2 - y^2$?\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'.
Original groundtruth: $x^2 - y^2$ factors into $(x+y)(x-y)$, so, to obtain the value of $x^2 - y^2$, simply multiply $16 \\cdot 2$ to get $\\boxed{32}$.
Transformed groundtruth: To find the value of $x^2 - y^2$, we can use the difference of squares factorization, which states that:\n\n$x^2 - y^2 = (x + y)(x - y)$\n\nGiven that:\n\n$x + y = 16$\n$x - y = 2$\n\nWe can substitute these into the factorization:\n\n$x^2 - y^2 = (16)(2)$\n\nMultiplying these values together, we get:\n\n$x^2 - y^2 = 32$\n\nFinal Answer: 32

Example 2: please notice that we slightly modify the solution so that it sounds more fluent and it seems more like a human written solution. We do not add extremly more information because human do not solve the problem that way. when a human is solving problem, he does not write down toooooo much information. please make sure you do not add extra information that the original groundtruth not talking about.
Original question: The value of $\\log_{10}{28471}$ is between the consecutive integers $a$ and $b$.  Find $a+b$.\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'.
Original groundtruth: We can have $\\log_{10}10000=4$ and $\\log_{10}100000=5$.  Since $\\log_{10}x$ increases as $x$ increases, we know that $\\log_{10}10000<\\log_{10}28471<\\log_{10}100000$, meaning $4<\\log_{10}28471<5$.  Thus, the desired sum is $4+5=\\boxed{9}$.
Transformed groundtruth: To estimate the value of $\\log_{10}{28471}$, we need to find two consecutive powers of 10 that are close to 28471.\n\nWe know that:\n\n$10^4 = 10000$ and $10^5 = 100000$\n\nSince 28471 is between these two powers of 10, the logarithm in question must be between 4 and 5.\n\nThus, $a = 4$ and $b = 5$.\n\nThe sum of $a$ and $b$ is $4 + 5 = 9$.\n\nFinal Answer: 9


Given the question: {question}
and the groundtruth: {prediction}
Transformed groundtruth: 

Please give me the transformed groundtruth directly without saying anything like sure i can help you with.
please make sure you do not add extra information that the original groundtruth not talking about.
Remember to make sure the Final Answer (a number) is placed at the end of the transformedd groundtruth. 
""" 
            answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
            item['answer'] = answer
            gpt4_answer_list.append(item)
    return gpt4_answer_list



def proof_read_data_creator_1(data_list, task_name = '', creation_num = 1):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    gpt4_answer_list = []
    for item in data_list:
        prediction = item['answer']
        question = item['question']
        temp = {}
        temp['input'] = ''
        if creation_num == 1:
            prompt = f"""{question} 

Please make sure the Final Answer: (only a number, nothing else) is placed at the end. """ 
        answer = create_gpt_completion(prompt, MODEL_ENGINE, client)
        item['answer'] = answer
        gpt4_answer_list.append(item)
    return gpt4_answer_list
# If Natalia sold half as many clips in May as she did in April, that means she sold half of 48 clips in May.\n\n48 / 2 = 24\n\nSo, Natalia sold 24 clips in May.\n\nTo find out how many clips Natalia sold altogether in April and May, we just need to add the number of clips she sold in each month:\n\n48 + 24 = 72\n\nSo, Natalia sold a total of 72 clips in April and May.\n\nFINAL ANSWER: 72 clips
def clean_string(s):
    for i in range(2):
        # Check and remove a leading single or double quote
        if s.startswith(("'", '"')):
            s = s[1:]

        # Check and remove a trailing single or double quote
        if s.endswith(("'", '"')):
            s = s[:-1]
    return s


def extract_checkpoint_names(directory):
    # List to store checkpoint names
    checkpoint_names = []

    # Iterate over the files and directories in the given directory
    for item in os.listdir(directory):
        # Check if the item is a directory and starts with "checkpoint-"
        if os.path.isdir(os.path.join(directory, item)) and item.startswith("checkpoint-"):
            checkpoint_names.append(item)

    # Sort the checkpoint names by the number following "checkpoint-"
    checkpoint_names.sort(key=lambda name: int(name.split('-')[1]))

    return checkpoint_names




# example_analyzation_1 = 1
# example_q1 = 1
# example_a1 = 1
# example_p1 = 1
# example_analyzation_2 = 1
# example_q2 = 1
# example_p2 = 1
# example_a2 = 1
# example_analyzation_3 = 1
# example_q3 = 1
# example_p3 = 1
# example_a3 = 1
# original_question = 1
# groundtruth_answer = 1
# previous_prediction = 1
# minimum_change_prompt = \
# f"""We have a question and the prediction. 
# 1. Please perform minimum change on the prediction to make it correct. 
# 2. Please also remove the hallucination, create your own answer when the inference steps from the prediction is empty. \
# 3. Please put the final answer at the end after FINAL ANSWER:
# For example:
# MINIMUM_CHANGED_PREDICTION_HERE. FINAL ANSWER: 181
# 4. Directly give me the minimum changed version of prediction without mentioning other things such as "Sure, I can help \
# you with that !" or "Here is the modified answer."

# Example1:{example_analyzation_1} We add the FINAL ANSWER: at the end
# given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
# Example1-Question: {example_q1}
# Example1-Prediction: {example_p1}
# Example1-Minimum Changed Prediction: {example_a1}

# Example2: {example_analyzation_2} We add the FINAL ANSWER: at the end
# given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
# Example2-Question: {example_q2}
# Example2-Prediction: {example_p2}
# Example2-Minimum Changed Prediction: {example_a2}



# Example3: {example_analyzation_3} We add the FINAL ANSWER: at the end
# given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
# Example3-Question: {example_q3}
# Example3-Prediction: {example_p3}
# Example3-Minimum Changed Prediction: {example_a3}


# Question:

# given the question, groundtruth and prediction, please perform minimum change to make the prediction correct.
# question: {original_question}
# Groundtruth: {groundtruth_answer}
# Prediction: {previous_prediction}
# Minimum Changed Prediction: """




def minimum_change_data_creation_reformat_answer(data_list, task_name = ''):
    from openai import OpenAI
    import openai
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    client = OpenAI(api_key=GPT_API)
    @retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(10))
    def create_gpt_completion(qa_, model_engine, client):
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "provide answer"},
                {"role": "user", "content": f"{qa_}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer

    true_false_list = []
    for i, item in enumerate(data_list):        
        previous_prediction = item['answer']
        minimum_change_prompt = \
f"""Please check if the following prediction is using correct format. It suppose to answer a math question properly. It suppose to put Final Answer: a_number_here at the end. If the prediction made the formating mistakes, please correct it.

There might be(but there is unlikely be) formating issue. Please make sure the prediction is like a answer with correct format. 
1. Make sure the format is like: inference, then provide the Final Answer at the end. The thing follow the Final Answer has be a number.
2. You have to answer the prediction directly without saying anything else. Please not saying anything 'like sure I can help you with'.
3. You will only you one Final Answer, and make sure Final Answer is palced at the end follow by a numnber. 
4. If there is mothing wrong, please copy the original predicition directly without changing anything. If there is something wrong, please keep as much original inference steps as poosible. This requirement is important.

There is two examples format modification you have to correct.

Mistakes 1: The prediction Answer first, the provide the Final Answer at the end.
Example 1: Final Answer: 500\n\nExplanation:\nSara spent $50 on shoes and $200 on a dress, so she spent $250 in total. Rachel wants to spend twice as much as Sara did, so Rachel's budget for the pair of shoes and dress would be $250 * 2 = $500. There's no need to divide the budget between the shoes and dress since the question does not ask for the individual costs, but rather the total amount Rachel should budget.\n\nFinal Answer: 500
Corrected example: Sara spent $50 on shoes and $200 on a dress, so she spent $250 in total. Rachel wants to spend twice as much as Sara did, so Rachel's budget for the pair of shoes and dress would be $250 * 2 = $500.\n\nFinal Answer: 500

Mistakes 2: The prediction do not put only a number at the end.
Example 2: Let's denote the number of purple flowers as P. Since there are 80% more purple flowers than yellow ones, we have P = 10 * 1.8 = 18 flowers.\n\nThe number of green flowers is 25% of the total number of yellow and purple flowers combined. Since there are 10 yellow flowers and 18 purple flowers, the total number of yellow and purple flowers is 10 + 18 = 28 flowers. Therefore, the number of green flowers is 0.25 * 28 = 7 flowers.\n\nFinal Answer: Mark has a total of 10 (yellow) + 18 (purple) + 7 (green) = 35 flowers in his garden.
Corrected example: Let's denote the number of purple flowers as P. Since there are 80% more purple flowers than yellow ones, we have P = 10 * 1.8 = 18 flowers.\n\nThe number of green flowers is 25% of the total number of yellow and purple flowers combined. Since there are 10 yellow flowers and 18 purple flowers, the total number of yellow and purple flowers is 10 + 18 = 28 flowers. Therefore, the number of green flowers is 0.25 * 28 = 7 flowers.\n\nMark has a total of 10 (yellow) + 18 (purple) + 7 (green) = 35 flowers in his garden.\n\nFinal Answer: 35

Here is the prediction that you need to modify: {previous_prediction}
Corrected Prediction: """

        
        answer = create_gpt_completion(minimum_change_prompt, MODEL_ENGINE, client)
        # print('answer: ', answer)
        # print('pppppp: ', previous_prediction)
        item['answer'] = answer
        true_false_list.append(item)
        
    return data_list
