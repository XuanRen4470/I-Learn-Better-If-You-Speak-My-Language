import os
import json
import re
category = 'MBPP'
# path = f'/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/{category}/gpt4_generated_{category.lower()}_False_100.json'
# output_path = f'/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/{category}/gpt4_generated_{category.lower()}_False_100_march_27.json'


# data_list = []
# with open(path, 'r') as file:
#     data = json.load(file)
#     for line in data:
#         solution = line['answer']
#         modified_solution = solution.replace('FINAL ANSWER', 'Final Answer')
#         line['answer'] = modified_solution

#         question = line['question']
#         modified_question = question.replace("""Answer: PUT_YOUR_ANSWER_HERE""", """You do not need to run the test example. It is only used to instruct the input format and the function name. Directly generate the code with no explaination.""")
#         line['question'] = modified_question
#         # line['numerical_final_answer'] = str(evaluate_expression(num))
#         # # print('solution', solution)
#         # # print('modified_solution', mwodified_solution)
#         # # print()
#         # # print()
#         data_list.append(line)


path = f'/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/{category}/train.json'
output_path = f'/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/{category}/train_modified.json'

def extract_final_answer(text):

    # Splitting the text to find the portion after "FINAL ANSWER:"
    parts = text.split("FINAL ANSWER:", 1)
    if len(parts) > 1:
        # If "FINAL ANSWER:" is found, include it in the extracted content and return
        return "FINAL ANSWER:" + parts[1].strip()
    else:
        # If "FINAL ANSWER:" is not found, return an empty string
        return text


data_list = []
with open(path, 'r') as file:
    data = json.load(file)
    for line in data:
        solution = line['answer']
        question = line['original_question']
        # extracted_content = extract_final_answer(solution)
        # extracted_content = extracted_content.replace('FINAL ANSWER', 'Final Answer')
        extracted_content = f"""Final Answer:
{solution}"""
        line['answer'] = extracted_content
        modified_question = \
f"""You are an expert Python programmer, and you have a computer task. For the task, you are given a test example to show you the input format and the function structure. Please be careful about whitespace between each line of the code. The test is only used to show you the input structure. You do not need to run the test.


Task: {question}
Test Example: {line['test'][0]}

Provide your answer directly without any explaination. Please provide the final answer (code) at the end, after 'Final Answer:'
"""
        

        line['question'] = modified_question
        data_list.append(line)

with open(output_path, 'w') as outfile:
    json.dump(data_list, outfile, indent=4)
a = 1
