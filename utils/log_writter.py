import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from config.config import HOME_DIRECTORY


def write_log(file_name, folder_name, log_line, accuracy = -1):
    print(log_line)
    with open(f"{HOME_DIRECTORY}/log/{folder_name}/{file_name}.txt", 'a') as f:
        f.write(f'{log_line}.\n')

    if accuracy != -1:
        with open(f"{HOME_DIRECTORY}/log_total/accuracy.txt", 'a') as f:
            f.write(f"""



{folder_name}
Accuracy: {accuracy}




""")

def write_log_dpo_accuracy_record(file_name, folder_name, log_line):
    with open(f"{HOME_DIRECTORY}/log/{folder_name}/{file_name}.txt", 'a') as f:
        f.write(f'{log_line}.\n')
