# I Learn Better If You Speak My Language: Understanding the Superior Performance of Fine-Tuning Large Language Models with LLM-Generated Responses

## Overview

This project includes training scripts, evaluation tools, and datasets for the paper I Learn Better If You Speak My Language: Understanding the Superior Performance of Fine-Tuning Large Language Models with LLM-Generated Responses. It builds upon the [LLAMA-FACTORY](https://github.com/hiyouga/LLaMA-Factory.git) project to train and test language models effectively.

**We strongly recommend reading our follow-up paper: Efficiently Selecting Response Generation Strategy by Self-Aligned Perplexity for Fine-Tuning LLMs.**
**This paper offers a deeper discussion on how to select better response-generation strategies and the role of familiarity in fine-tuning effectiveness.**

## Features

- **Training Scripts**: Customizable training scripts leveraging LLAMA-FACTORY.
- **Evaluation Tools**: Unified prediction functions for seamless evaluation.
- **Datasets Included**: All necessary datasets are provided within the project.
- **Perplexity Calculation**: (Coming soon) Scripts for perplexity calculations will be added.

## Getting Started

1. **Prerequisites**

   - Python 3.10

2. **Set Up LLAMA-FACTORY Directory**

   Modify the `LLAMA_FACTORY_DIRECTORY_new` variable in your scripts to point to your LLAMA-FACTORY directory:

   ```python
   LLAMA_FACTORY_DIRECTORY_new = '/path/to/your/LLAMA-FACTORY'
   ```

3. **Install Dependencies**

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```


### Training

To train the model:

1. **Modify the Training Script**

   In `utils/train.py`, locate the `train_llama_factory` function and update the model path:

### Evaluation

To evaluate the trained model:

1. **Modify the Evaluation Script**

   In `evaluation/eval.py`, find the `do_predict_llama_factory_unify` function and set your model path:


### Datasets

All datasets required for training and evaluation are available in the `datasets/` directory.

### Perplexity Calculation

*Note: The perplexity calculation scripts will be added soon.*

## Recommended Workflow

For the best experience:

1. **Create Custom Training Scripts**

   - Start by creating your own training scripts based on the provided templates.
   - Customize them according to your project's needs.

2. **Train Your Model**

   - Use the modified training scripts to train your model.
   - Ensure all paths and configurations point to your directories and models.

3. **Evaluate the Model**

   - After training, use the evaluation scripts to assess your model's performance.
   - Merge or adapt the evaluation scripts into your project as needed.

## Acknowledgments

- **[LLAMA-FACTORY](https://github.com/hiyouga/LLaMA-Factory.git)**: This project builds upon the excellent work done in the LLAMA-FACTORY project.
