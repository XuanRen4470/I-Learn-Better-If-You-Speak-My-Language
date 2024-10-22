import os
import json

# Directory path containing JSON files
dir_path = "/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MATH/train/algebra"

# Initialize a list to hold all math problems
all_problems = []

# Traverse the directory and read all JSON files
for file in os.listdir(dir_path):
    if file.endswith(".json"):  # Ensures only JSON files are processed
        file_path = os.path.join(dir_path, file)
        with open(file_path, 'r') as json_file:
            problem = json.load(json_file)
            problem['input'] = ''
            if 'problem' in problem:
                # Assign the value of "category" to "new_category"
                problem['question'] = problem['problem']
                # Remove the old "category" key
                del problem['problem']
            if 'solution' in problem:
                # Assign the value of "category" to "new_category"
                problem['answer'] = problem['solution']
                # Remove the old "category" key
            all_problems.append(problem)

# Combine all problems into a single JSON object
combined_problems_json = json.dumps(all_problems, indent=4)

# Write the combined JSON to a file named 'train.json'
with open('/gpfs/users/a1796450/ACL_2024/Minimum_Change/dataset/MATH/train_algebra_total.json', 'w') as outfile:
    outfile.write(combined_problems_json)
