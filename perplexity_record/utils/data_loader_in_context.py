import sys
import os
import json
import random
from torch.utils.data import DataLoader, Dataset

HOME_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.data_loader import load_GSM8K, load_ESNLI, load_BOOLQ, load_API_BANK_aug_2, load_ECQA, load_MATH, load_WINOGRANDE, load_MMLU


def in_context_learning_examples(question, original_question, prompt_style = 'gt_style', task = 'gsm8k', enforce_prompt_style = ''):
    front_prompt_math = \
"""
Requirement 1. Please inference first, then place the final result(a number) after the word Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Inference: INFERENCE_HERE
Final Answer: NUMBER_HERE
"""

    front_prompt_ecqa = \
"""
Requirement 1. Please inference first, then place the final result(1/2/3/4/5) after the word Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Inference: INFERENCE_HERE
Final Answer: (1/2/3/4/5)_HERE
"""

    front_prompt_api_bank = \
"""
Requirement 1. Please inference first, then place the API-REQUEST after the word Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Question:
Inference: INFERENCE_HERE
Final Answer: API-REQUEST_HERE
"""

    front_gt_prompt_api_bank = \
"""
Requirement 1. Please place the API-REQUEST after the word Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Question: 
Final Answer: API-REQUEST_HERE
"""

    front_prompt_boolq = \
"""
Requirement 1. Please inference first, then place the True or False after Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Question:
Inference: INFERENCE_HERE
Final Answer: True/False
"""

    front_gt_prompt_boolq = \
"""
Requirement 1. Please place the True or False after Final Answer: 
Requirement 2. Please solving the question with the similar styles as the examples below.

Format:
Question:
Final Answer: True/False
"""

    front_prompt_esnli = \
"""
We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.
Given a statement and a context, please determine whether the statement is entailment, contradiction, or neutral given the context. 

Requirement 1. Please inference first, then place the entailment, contradiction or neutral after Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Question:
Inference: INFERENCE_HERE
Final Answer: Entailment/Contradiction/Neutral
"""


    gsm8k_gpt4_style_prompt = \
"""
We have 3 examples below

Example 1
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Inference: Natalia sold 48 clips in April. In May, she sold half as many as she did in April, which means she sold \\( \\frac{1}{2} \\times 48 \\) clips in May. Half of 48 is 24. So in May, she sold 24 clips. To find the total amount of clips sold altogether in April and May, we sum the clips sold in April with those sold in May: 48 (April) + 24 (May) = 72 clips.\nFinal Answer: 72

Example 2
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Inference: Weng earns $12 per hour for babysitting. To find out how much she earns for 50 minutes, we first need to calculate what fraction of an hour 50 minutes is. Since there are 60 minutes in an hour, 50 minutes is 50/60 of an hour. \n\nWe can calculate her earnings for 50 minutes by multiplying her hourly rate by the fraction of the hour she worked, which is:\n\n$12/hour * (50 minutes / 60 minutes) = $12 * (5/6)\n\nMultiplying 12 by 5/6 (or 0.8333), we get:\n\n$12 * 0.8333 \u2248 $10\n\nTherefore, Weng earned approximately $10 for 50 minutes of babysitting.\nFinal Answer: 10

Example 3
Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Inference: Betty needs $100 for the wallet. She has half of that amount, which is $50. Her parents give her $15, and her grandparents give her twice what her parents gave her, which is $15 * 2 = $30. Now, Betty has her original $50 + $15 from her parents + $30 from her grandparents = $95. The wallet costs $100, so Betty still needs $100 - $95 = $5.\nFinal Answer: 5"

Now please solve the following question using the same inference style and format as the example above. 
Question: """



    gsm8k_rewrite_in_own_words_style_prompt = \
"""
We have 3 examples below

Example 1
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Inference: In May, Natalia managed to sell half the number of clips she sold in April, which means she sold 24 clips. Adding the 48 clips from April to the 24 clips from May gives us the total number of clips Natalia sold over the two months, which is 72 clips.\nFinal Answer: 72

Example 2
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Inference: Weng's hourly babysitting rate is $12, so to find out her earnings for a minute, we divide her hourly rate by the number of minutes in an hour. This gives us her rate per minute. After calculating that, we can multiply her per minute rate by the number of minutes she babysat yesterday, which is 50 minutes, to determine her total earnings for that duration. \n\nTo break it down systematically, we take her hourly rate of $12 and divide by 60, as there are 60 minutes in an hour. This calculation will give us her earnings per minute. Once we have this rate, we multiply it by the 50 minutes of babysitting she did. \n\nCompleting these calculations results in her total earnings for 50 minutes of babysitting.\nFinal Answer: 10

Example 3
Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Inference: Betty starts off with half the required amount for the wallet, which is $50. Her parents give her an additional $15, and her grandparents contribute double the amount given by her parents, adding up to $30. To find out how much more she needs, we subtract the total amount she has from the cost of the wallet: $100 - ($50 + $15 + $30) = $5.\nFinal Answer: 5"

Now please solve the following question using the same inference style and format as the example above. 
Question: """

   

    gsm8k_gt_style_prompt = \
"""
We have 3 examples below

Example 1
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Inference: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\nFinal Answer: 72

Example 2
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Inference: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\nFinal Answer: 10

Example 3
Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Inference: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\nFinal Answer: 5

Now please solve the following question using the same inference style and format as the example above. 
Question: """

    gsm8k_minimum_change_style_prompt = \
"""
We have 3 examples below

Example 1
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Inference: Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips.\n\nSo, Natalia sold a total of 48 (April) + 24 (May) = 72 clips in April and May.\n\nFinal Answer: 72

Example 2
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Inference: First, we need to convert the 50 minutes of babysitting into hours. We know that there are 60 minutes in an hour, so:\n\n50 minutes = 50 / 60 hours\n\nNext, we can calculate how much Weng earned by multiplying her hourly rate by the number of hours worked:\n\nEarnings = Hourly rate * Hours worked\nEarnings = $12/hour * (50 / 60) hours \n\nEarnings = $12 * 5/6\n\nEarnings = $10\n\nFinal Answer: 10

Example 3
Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Inference: Betty's parents gave her $15, and her grandparents gave her twice that amount. So, her grandparents gave her $15 * 2 = $30.\n\nBetty originally had $50, which is half of what she needs for the wallet.\n\nBetty has received a total of $15 (from her parents) + $30 (from her grandparents) + $50 (originally had) = $95.\n\nThe wallet costs $100, so Betty still needs $100 - $95 = $5 to buy the wallet.\n\nFinal Answer: 5

Now please solve the following question using the same inference style and format as the example above. 
Question: """

    

    gsm8k_mini_gpt4_prompt = \
"""
We have 3 examples below

Example 1
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Inference: Natalia sold 48 clips in April. In May, she sold half as many clips, which is 48 / 2 = 24 clips. Therefore, the total number of clips sold in April and May is 48 + 24 = 72.\n\nFinal Answer: 72

Example 2
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Inference: Weng earns $12 per hour, and she worked for 50 minutes. To calculate her earnings, we first convert her work time into hours: 50 minutes is equivalent to \\( \\frac{50}{60} \\) hours, which simplifies to \\( \\frac{5}{6} \\) hours. We then multiply her hourly rate by the number of hours worked: \n\n\\[\n12 \\times \\frac{5}{6} = 10\n\\]\n\nThus, Weng earned $10 for babysitting yesterday.\n\nFinal Answer: 10

Example 3
Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Inference: Betty needs $100 for the wallet, and she currently has half of this amount, which is $50. Her parents give her $15, and her grandparents give her twice that amount, which is $30 (2 x $15). \n\nNow, we calculate how much money she has after these contributions:\n- Initial amount: $50\n- From parents: $15\n- From grandparents: $30\n\nTotal money Betty has now:\n$50 + $15 + $30 = $95\n\nNow, to find out how much more she needs:\n$100 (cost of the wallet) - $95 (total money she has) = $5\n\nFinal Answer: 5

Now please solve the following question using the same inference style and format as the example above. 
Question: """

    gsm8k_anthropic_style_prompt = \
"""
We have 3 examples below

Example 1
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Inference: Natalia sold 48 clips in April.\nIn May, she sold half as many clips as in April.\nHalf of 48 is 24.\nSo, Natalia sold 24 clips in May.\nTo find the total number of clips sold in both months:\nApril clips + May clips = 48 + 24 = 72\n\nFinal Answer: 72

Example 2
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Inference: To calculate Weng's earnings, we need to:\n1. Convert her hourly rate to a per-minute rate\n2. Multiply that rate by the number of minutes she worked\n\n1. Hourly rate to per-minute rate:\n   $12 per hour = $12 \u00f7 60 minutes = $0.20 per minute\n\n2. Earnings for 50 minutes:\n   $0.20 per minute \u00d7 50 minutes = $10\n\nFinal Answer: 10

Example 3
Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Inference: Betty needs $100 for the wallet.\nShe has half of the money she needs, which is $50.\nHer parents give her $15.\nHer grandparents give her twice as much as her parents, which is $30.\n\nTotal money Betty has:\n$50 (initial) + $15 (parents) + $30 (grandparents) = $95\n\nMoney still needed:\n$100 (wallet cost) - $95 (money Betty has) = $5\n\nFinal Answer: 5

Now please solve the following question using the same inference style and format as the example above. 
Question: """







    if 'generated_xy_pairs' in task:
        task_name_ = 'gsm8k'
        

        if 'mini' in task:
            xy_pairs_model = 'gpt-4o-mini'
        if 'Breadth_Prompt' in prompt_style:
            full_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/{xy_pairs_model}_Breadth Prompt_{task_name_}_generated_xy_pairs_1000.json'
        if 'Reasoning_Prompt' in prompt_style:
            full_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/{xy_pairs_model}_Reasoning Prompt_{task_name_}_generated_xy_pairs_1000.json'
        if 'Concretizing_Prompt' in prompt_style:
            full_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/{xy_pairs_model}_Concretizing Prompt_{task_name_}_generated_xy_pairs_1000.json'
        if 'Deepen_Prompt' in prompt_style:
            full_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/{xy_pairs_model}_Deepen Prompt_{task_name_}_generated_xy_pairs_1000.json'
        if 'Constraints_Prompt' in prompt_style:
            full_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/{xy_pairs_model}_Constraints Prompt_{task_name_}_generated_xy_pairs_1000.json'
        with open(full_path, 'r') as file:
            generated_xy_pairs_list = json.load(file)
        generated_xy_pairs_list = generated_xy_pairs_list[:3]

        q_1 = generated_xy_pairs_list[0]['question']
        q_1 = q_1.replace("\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n", '')
        a_1 = generated_xy_pairs_list[0]['answer']

        q_2 = generated_xy_pairs_list[1]['question']
        q_2 = q_2.replace("\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n", '')
        a_2 = generated_xy_pairs_list[1]['answer']

        q_3 = generated_xy_pairs_list[2]['question']
        q_3 = q_3.replace("\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n", '')
        a_3 = generated_xy_pairs_list[2]['answer']

        gsm8k_xy_pairs_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q_1}
Inference: {a_1}

Example 2
Question: {q_2}
Inference: {a_2}

Example 3
Question: {q_3}
Inference: {a_3}

Now please solve the following question using the same inference style and format as the example above. 
Question: """







    math_gt_style_prompt = \
"""
We have 3 examples below

Example 1
Question: Find the distance between the points (0,15) and (8,0).
Inference: We use the distance formula: $$\\sqrt{(8 - 0)^2 + (0 - 15)^2} = \\sqrt{64 + 225} = \\boxed {17}.$$- OR -\n\nWe note that the points $(0, 15)$, $(8, 0)$, and $(0, 0)$ form a right triangle with legs of length 8 and 15. This is a Pythagorean triple, so the hypotenuse must have length $\\boxed{17}$. Final Answer: 17

Example 2
Question: What value of $x$ will give the minimum value for $x^2- 10x + 24$?
Inference: We start by completing the square.  \\[x^2-10x+24=(x-5)^2-1.\\] Since the square of a real number is at least 0, $(x-5)^2\\ge 0$ and $(x-5)^2-1 \\ge -1.$ Thus, the minimum value of the quadratic is $-1,$ which occurs when $x=\\boxed{5}.$ Final Answer: 5

Example 3
Question: What is the minimum value of $y$ if $y=3x^2+6x+9?
Inference: First, complete square as follows: $$y=3x^2+6x+9=3\\left(x^2+2x\\right)+9.$$ To complete the square, we need to add $\\left(\\frac{2}{2}\\right)^2=1$ after the $2x.$ So we have $$y+3=3\\left(x^2+2x+1\\right)+9.$$ This gives $$y=3\\left(x+1\\right)^2+6.$$ Now, since $\\left(x+1\\right)^2\\ge0,$ the minimum value is when the squared term is equal to $0.$ So the minimum value is $$y=3\\left(x+1\\right)^2+6=3\\cdot0+6=\\boxed{6}.$$ Final Answer: 6

Now please solve the following question using the same inference style and format as the example above. 
Question: """




    math_gpt4_style_prompt = \
"""
We have 3 examples below

Example 1
Question: Find the distance between the points (0,15) and (8,0).
Inference: To find the distance between the two points, (0,15) and (8,0), you can use the distance formula derived from the Pythagorean Theorem:\n\nDistance = sqrt((x2 - x1)^2 + (y2 - y1)^2)\n\nWhere (x1, y1) = (0, 15) and (x2, y2) = (8, 0).\n\nPlugging these values into the formula gives us:\nDistance = sqrt((8 - 0)^2 + (0 - 15)^2)\nDistance = sqrt(8^2 + (-15)^2)\nDistance = sqrt(64 + 225)\nDistance = sqrt(289)\nDistance = 17\n\nFinal Answer: 17

Example 2
Question: What value of $x$ will give the minimum value for $x^2- 10x + 24$?
Inference: To find the value of $x$ that gives the minimum value for a quadratic function in standard form $ax^2 + bx + c$, you can use the vertex formula $x = -\\frac{b}{2a}$.\n\nFor the quadratic function $x^2 - 10x + 24$, we have $a = 1$ and $b = -10$. Plugging these values into the vertex formula:\n\n$x = -\\frac{-10}{2 \\cdot 1} = \\frac{10}{2} = 5$.\n\nSo, the value of $x$ that will give the minimum value for the function $x^2 - 10x + 24$ is $x = 5$.\n\nFinal Answer: 5

Example 3
Question: What is the minimum value of $y$ if $y=3x^2+6x+9?
Inference: To find the minimum value of the function $y=3x^2+6x+9$, we can complete the square or use the fact that for a quadratic function $y=ax^2+bx+c$ with $a > 0$, the vertex form is $y=a(x-h)^2+k$ where the vertex $(h, k)$ gives the minimum point of the function.\n\nTo complete the square, we can rewrite the function as follows:\n\n$y=3(x^2+2x+3)$\n$y=3[(x+1)^2+2]$\n$y=3(x+1)^2+6$\n\nRewrite in vertex form, we can see that:\n$h=-1$ and $k=6$\n\nSince the coefficient of $x^2$ is positive, this parabola opens upwards, which means that the vertex represents the minimum point.\n\nThus, the minimum value of $y$ is when $x$ is -1.\nSubstitute $x=-1$ into the original equation to find the minimum value of $y$:\n\n$y=3(-1)^2+6(-1)+9$\n$y=3(1)-6+9$\n$y=3-6+9$\n$y=6$\n\nFinal Answer: 6

Now please solve the following question using the same inference style and format as the example above. 
Question: """



    math_mini_gpt4_prompt = \
"""
We have 3 examples below

Example 1
Question: Find the distance between the points (0,15) and (8,0).
Inference: To find the distance between the points (0,15) and (8,0), we can use the distance formula, which is given by:\n\n\\[ d = \\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} \\]\n\nHere, \\((x_1, y_1) = (0, 15)\\) and \\((x_2, y_2) = (8, 0)\\).\n\nCalculating the differences:\n\n\\[\nx_2 - x_1 = 8 - 0 = 8\n\\]\n\\[\ny_2 - y_1 = 0 - 15 = -15\n\\]\n\nNow substitute these values into the distance formula:\n\n\\[\nd = \\sqrt{(8)^2 + (-15)^2}\n\\]\n\\[\nd = \\sqrt{64 + 225}\n\\]\n\\[\nd = \\sqrt{289}\n\\]\n\\[\nd = 17\n\\]\n\nFinal Answer: 17

Example 2
Question: What value of $x$ will give the minimum value for $x^2- 10x + 24$?
Inference: To find the value of \\( x \\) that gives the minimum value for the quadratic function \\( x^2 - 10x + 24 \\), we can use the vertex formula. The vertex \\( x \\) coordinate of a quadratic equation \\( ax^2 + bx + c \\) is given by \\( x = -\\frac{b}{2a} \\).\n\nHere, \\( a = 1 \\) and \\( b = -10 \\). Plugging these values into the formula:\n\n\\[\nx = -\\frac{-10}{2 \\cdot 1} = \\frac{10}{2} = 5\n\\]\n\nThus, the value of \\( x \\) that gives the minimum value is 5.\n\nFinal Answer: 5

Example 3
Question: What is the minimum value of $y$ if $y=3x^2+6x+9?
Inference: The expression for \\( y \\) is a quadratic function in the form \\( y = ax^2 + bx + c \\), where \\( a = 3 \\), \\( b = 6 \\), and \\( c = 9 \\). \n\nSince the coefficient of \\( x^2 \\) (which is \\( a = 3 \\)) is positive, the parabola opens upwards, which means that there is a minimum value. The \\( x \\)-coordinate of the vertex (which gives the minimum \\( y \\) value) can be found using the formula \\( x = -\\frac{b}{2a} \\).\n\nCalculating this, we have:\n\\[\nx = -\\frac{6}{2 \\cdot 3} = -\\frac{6}{6} = -1\n\\]\n\nSubstituting \\( x = -1 \\) back into the equation for \\( y \\):\n\\[\ny = 3(-1)^2 + 6(-1) + 9\n\\]\n\\[\n= 3(1) - 6 + 9\n\\]\n\\[\n= 3 - 6 + 9 = 6\n\\]\n\nThus, the minimum value of \\( y \\) is 6.\n\nFinal Answer: 6

Now please solve the following question using the same inference style and format as the example above. 
Question: """

    math_minimum_change_style_prompt = \
"""
We have 3 examples below

Example 1
Question: Find the distance between the points (0,15) and (8,0).
Inference: To find the distance between two points (x1, y1) and (x2, y2), use the distance formula:\n\nDistance = \u221a[(x2 - x1)\u00b2 + (y2 - y1)\u00b2]\n\nSubstituting the given values into the formula:\n\nDistance = \u221a[(8 - 0)\u00b2 + (0 - 15)\u00b2]\n         = \u221a[(8)\u00b2 + (-15)\u00b2]\n         = \u221a[64 + 225]\n         = \u221a289\n         = 17\n\nFinal Answer: 17

Example 2
Question: What value of $x$ will give the minimum value for $x^2- 10x + 24$?
Inference: To find the minimum value of the quadratic function $f(x) = x^2 - 10x + 24$, we need to find the vertex of its graph. The x-coordinate of the vertex can be found using the formula:\n\n$$x = \\dfrac{-b}{2a}$$\n\nwhere $a$ is the coefficient of $x^2$ and $b$ is the coefficient of $x$. In this case, $a = 1$ and $b = -10$, so:\n\n$$x = \\dfrac{-(-10)}{2(1)}$$\n\n$$x = \\dfrac{10}{2}$$\n\n$$x = 5$$\n\nTherefore, the value of $x$ that gives the minimum value for $x^2 - 10x + 24$ is 5. Final Answer: 5

Example 3
Question: What is the minimum value of $y$ if $y=3x^2+6x+9?
Inference: To find the minimum value of the quadratic function $y = 3x^2+6x+9$, we need to find the vertex of its graph. The x-coordinate of the vertex can be found using the formula:\n\n$$x = -\\frac{b}{2a}$$\n\nwhere $a$ is the coefficient of $x^2$ and $b$ is the coefficient of $x$. In this case, $a=3$ and $b=6$:\n\n$$x = -\\frac{6}{2\\cdot 3} = -\\frac{6}{6} = -1$$\n\nNow that we have the $x$-coordinate of the vertex, we can find the minimum value of $y$ by substituting $x=-1$ back into the equation:\n\n$$y = 3(-1)^2 + 6(-1) + 9 = 3(1) - 6 + 9 = 3 - 6 + 9 = 6$$\n\nFinal Answer: 6

Now please solve the following question using the same inference style and format as the example above. 
Question: """

    math_anthropic_style_prompt = \
"""
We have 3 examples below

Example 1
Question: Find the distance between the points (0,15) and (8,0).
Inference: To find the distance between two points, we can use the distance formula:\n\nd = \u221a[(x2 - x1)\u00b2 + (y2 - y1)\u00b2]\n\nWhere (x1, y1) is the first point and (x2, y2) is the second point.\n\nIn this case:\n(x1, y1) = (0, 15)\n(x2, y2) = (8, 0)\n\nPlugging these values into the formula:\n\nd = \u221a[(8 - 0)\u00b2 + (0 - 15)\u00b2]\nd = \u221a[8\u00b2 + (-15)\u00b2]\nd = \u221a[64 + 225]\nd = \u221a289\nd = 17\n\nFinal Answer: 17

Example 2
Question: What value of $x$ will give the minimum value for $x^2- 10x + 24$?
Inference: To find the value of x that gives the minimum value for the quadratic function f(x) = x^2 - 10x + 24, we can use the following method:\n\n1. The quadratic function is in the form f(x) = ax^2 + bx + c, where a = 1, b = -10, and c = 24.\n\n2. For a quadratic function, the x-coordinate of the vertex gives the value of x where the function reaches its minimum (when a > 0) or maximum (when a < 0).\n\n3. The formula for the x-coordinate of the vertex is: x = -b / (2a)\n\n4. Substituting our values:\n   x = -(-10) / (2(1))\n   x = 10 / 2\n   x = 5\n\nTherefore, the function f(x) = x^2 - 10x + 24 reaches its minimum value when x = 5.\n\nFinal Answer: 5

Example 3
Question: What is the minimum value of $y$ if $y=3x^2+6x+9?
Inference: To find the minimum value of y in the quadratic function y = 3x^2 + 6x + 9, we need to find the vertex of the parabola.\n\nFor a quadratic function in the form y = ax^2 + bx + c, the x-coordinate of the vertex is given by x = -b / (2a).\n\nIn this case, a = 3, b = 6, and c = 9.\n\nx = -6 / (2 * 3) = -6 / 6 = -1\n\nNow we can find the y-coordinate of the vertex by plugging x = -1 into the original equation:\n\ny = 3(-1)^2 + 6(-1) + 9\ny = 3 - 6 + 9\ny = 6\n\nThe vertex (-1, 6) represents the minimum point of the parabola since the coefficient of x^2 is positive (a > 0).\n\nFinal Answer: 6

Now please solve the following question using the same inference style and format as the example above. 
Question: """

    math_rewrite_in_own_words_style_prompt = \
"""
We have 3 examples below

Example 1
Question: Find the distance between the points (0,15) and (8,0).
Inference: To determine the distance between the points (0,15) and (8,0), we can apply the distance formula which is the square root of the sum of the squares of the differences in the x-coordinates and the y-coordinates of the two points. In this case, we calculate the square root of the difference in the x-coordinates, 8 minus 0, squared, plus the difference in the y-coordinates, 0 minus 15, squared. Working this out, it simplifies to the square root of 64 plus 225, which equals the square root of 289. When we take the square root of this number, we get 17.\n\nAlternatively, we can observe that the given points and the origin (0,0) create a right-angled triangle where the lengths of the two shorter sides correspond to the differences in their respective x and y coordinates, which are 8 and 15. Recognizing this as a Pythagorean triple, we can identify that the length of the hypotenuse, which is the distance between the two points, must be 17, consistent with the result obtained using the distance formula. \n\nFinal Answer: 17.0

Example 2
Question: What value of $x$ will give the minimum value for $x^2- 10x + 24$?
Inference: To find the value of $x$ that minimizes the quadratic function $x^2 - 10x + 24$, we can rewrite the expression in the form of a perfect square. We do this by expressing the quadratic as $(x - a)^2 + b$, where $a$ is half of the coefficient of $x$ and $b$ adjusts the constant term to match the original expression. \n\nThe coefficient of $x$ in the quadratic is $-10$, so half of this value is $-10/2 = -5$. Squaring $-5$ gives us $25$, which we subtract from $24$ to find the constant term $b$ that completes the square. This results in the following equivalent form of the quadratic:\n\n\\[ x^2 - 10x + 24 = (x - 5)^2 - 1 \\]\n\nGiven that the minimum value of a square of a real number is zero, $(x - 5)^2$ will be at its minimum when $x = 5$. Therefore, $(x - 5)^2 - 1$ achieves its minimum value of $-1$ when $x$ is equal to $5$. \n\nFinal Answer: 5

Example 3
Question: What is the minimum value of $y$ if $y=3x^2+6x+9?
Inference: To find the minimum value of the quadratic function $y = 3x^2 + 6x + 9$, we can complete the square or use the vertex form of a quadratic equation. To complete the square:\n\n$y = 3x^2 + 6x + 9$\n\nFirst, factor out the coefficient of $x^2$:\n\n$y = 3(x^2 + 2x) + 9$\n\nNext, complete the square inside the parentheses by adding and subtracting $(b/2a)^2$, where $b = 2$ and $a = 1$ (since we factored out the 3):\n\n$y = 3(x^2 + 2x + (2/2)^2 - (2/2)^2) + 9$\n$y = 3(x^2 + 2x + 1 - 1) + 9$\n$y = 3(x + 1)^2 - 3 + 9$\n$y = 3(x + 1)^2 + 6$\n\nThe vertex form of a parabola is $y = a(x - h)^2 + k$, where $(h, k)$ is the vertex. In the equation above, the vertex of the parabola is $(-1, 6)$.\n\nSince $a = 3$ (a positive number), the parabola opens upwards, meaning the vertex is at the minimum point of the parabola.\n\nThe minimum value of $y$ then corresponds to the $y$-coordinate of the vertex.\n\nFinal Answer: 6

Now please solve the following question using the same inference style and format as the example above. 
Question: """





#     q1 = \
# """We have the question: What might a person see at the scene of a brutal killing?\nand the options:\n(1): bloody mess\n(2): pleasure\n(3): being imprisoned\n(4): feeling of guilt\n(5): cake\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'\n\nFormat:\nInference: INFERENCE_HERE\nFinal Answer: (1/2/3/4/5)_HERE"""

#     q2 = \
# """We have the question: If you want to set a romantic atmosphere you might light a candle where?\nand the options:\n(1): dimly lit room\n(2): synagogue\n(3): bedroom\n(4): birthday cake\n(5): roses\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'\n\nFormat:\nInference: INFERENCE_HERE\nFinal Answer: (1/2/3/4/5)_HERE"""

#     q3 = \
# """We have the question: What might the result of unwanted flirting be?\nand the options:\n(1): attraction\n(2): problems\n(3): the gallows\n(4): being slapped\n(5): curiosity\n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'\n\nFormat:\nInference: INFERENCE_HERE\nFinal Answer: (1/2/3/4/5)_HERE"""

    q1 = \
"""What might a person see at the scene of a brutal killing?\nand the options:\n(1): bloody mess\n(2): pleasure\n(3): being imprisoned\n(4): feeling of guilt\n(5): cake\n\nwhat is the correct option?"""

    q2 = \
"""If you want to set a romantic atmosphere you might light a candle where?\nand the options:\n(1): dimly lit room\n(2): synagogue\n(3): bedroom\n(4): birthday cake\n(5): roses\n\nwhat is the correct option?"""

    q3 = \
"""What might the result of unwanted flirting be?\nand the options:\n(1): attraction\n(2): problems\n(3): the gallows\n(4): being slapped\n(5): curiosity\n\nwhat is the correct option?"""

# \n\nwhat is the correct option?\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'

    ecqa_gt_style_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q1}
Inference: Bloody mess is covered or stained with blood. A person might see a bloody mess at the scene of a brutal killing.  Pleasure is about what a person sees at the scene of a brutal killing and one cannot be happy to see such brutality. You can\u2019t see someone in jail at the brutal killing scene. Feeling of guilt doesn\u2019t come as the killing is brutal or merciless. Cake is baseless and weird to think as it is a brutal killing scene and not a bakery.\nFinal Answer: 1

Example 2
Question: {q2}
Inference: A romantic atmosphere can be set in bedroom and not in a synagogue. Bedroom is a place where one sleeps unlike a dimly lit room or a birthday cake. Candles can be lit in a bedroom and not in roses.\nFinal Answer: 3

Example 3
Question: {q3}
Inference: Person can be slapped if he does unwanted flirting to someone. Attraction cannot be result of unwanted flirting. Unwanted flirting doesn't always creates problems.\n            The gallows or curiosity is not something that can be result of unwanted flirting.\nFinal Answer: 4

Now please solve the following question using the same inference style and format as the example above. 
Question: """

    ecqa_gpt4_style_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q1}
Inference: When considering what someone might see at the scene of a brutal killing, the most likely observation would be a disturbing and horrific sight with evidence of violence such as blood. Therefore, the option that corresponds with this description is a \"bloody mess.\"\nFinal Answer: 1

Example 2
Question: {q2}
Inference: Lighting a candle to set a romantic atmosphere is typically associated with creating an intimate and personal space where the ambiance is quiet and private. A dimly lit room could certainly be made more romantic with a candle, but it lacks specificity. A synagogue is a place of worship and not typically associated with romantic settings. A birthday cake is a celebratory item and is lit for the purpose of making a wish and blowing out candles rather than creating a romantic mood. Roses are a symbol associated with romance, but they do not create an ambiance on their own and are not a place where you would light a candle.\n\nThe bedroom, however, is a private space often associated with intimacy and romance. Lighting a candle in the bedroom can enhance the atmosphere by providing soft, flattering light and possibly adding a soothing scent, making it the most appropriate choice for a romantic setting among the given options.\nFinal Answer: 3

Example 3
Question: {q3}
Inference: Unwanted flirting can lead to various negative outcomes due to the discomfort or offense it may cause the recipient. Possible adverse results include creating a tense or awkward situation, causing emotional or social difficulties, or even precipitating a conflict. In extreme cases, it could result in a strong negative reaction such as being slapped, as this physical action could be someone\u2019s reflex to unwanted advances. Attractions are ideally mutual, so an unwanted flirtation is unlikely to result in genuine attraction. The gallows is an archaic and hyperbolic reference to consequences for crimes, which doesn\u2019t align well with the typical response to flirting. Curiosity could be a result in some contexts, but in the case of unwanted flirting it typically isn\u2019t the primary response. Option (2), problems, is also a possible outcome but it's more general than option (4), which is more specific and indicative of a direct and immediate response.\nFinal Answer: 4

Now please solve the following question using the same inference style and format as the example above. 
Question: """
    
    ecqa_rewrite_minimum_change_style_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q1}
Inference: Based on the context of the question and the given options, it is likely that the scene of a brutal killing would be disturbing and, therefore, neither \"pleasure\" nor \"cake\" would be appropriate to the scene. Thus, we can eliminate options (2) and (5).\n\nGiven the nature of a brutal killing, it is reasonable to expect that the aftermath would be quite chaotic and messy, characterized by a significant amount of blood. Hence, the correct option is most likely to be (1) \"bloody mess,\" which corresponds with what one might typically see at such a tragic scene.\n\nFinal Answer: 1

Example 2
Question: {q2}
Inference: Based on the context of the question, the correct answer would be (3) the bedroom. This option sets the mood for a romantic atmosphere the best, as candles are often used in the bedroom to create a romantic and intimate setting. The other options do not directly relate to setting a romantic atmosphere with a candle.\n\nFinal Answer: 3

Example 3
Question: {q3}
Inference: Based on the context of the question and the given options, it is clear that unwanted flirting could lead to a negative reaction such as being slapped. Therefore, the correct answer is (4).\n\nFinal Answer: 4

Now please solve the following question using the same inference style and format as the example above. 
Question: """

    ecqa_rewrite_in_own_words_style_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q1}
Inference: At a brutal killing scene, one would likely encounter a disturbing sight of blood and violence, thus a bloody mess aptly describes such a scenario. The notion of pleasure is inconsistent with the grim reality of a killing scene. Imprisonment is a consequence that follows due process but is not something observed at the actual site of a killing. A feeling of guilt is an internal emotion and not a visible element at a crime scene. Lastly, a cake has no relevance to a brutal killing scene, as it is typically associated with celebrations.\nFinal Answer: 1

Example 2
Question: {q2}
Inference: To create a romantic setting, a bedroom is a common choice, considering it's a private and intimate space that's often associated with romance. This is unlike a synagogue, which is a place of worship, or a birthday cake, which is not a location but an item typically associated with celebrations. Lighting a candle in a dimly lit room may enhance the ambience, but the specificity of a bedroom underscores the romantic intent. Additionally, one cannot practically light a candle 'in roses' as roses are flowers and not a suitable place for lighting candles safely.\nFinal Answer: 3

Example 3
Question: {q3}
Inference: When a person engages in unwanted flirting, it may lead to negative reactions from the other person. While attraction is typically a response to welcomed advances, unwanted flirting is unlikely to elicit such a feeling. Problems can arise from unwanted flirting, but they are not a guaranteed consequence. Outcomes such as \"the gallows\" are excessively severe and unrelated to the context of flirting, and curiosity is not typically a reaction to uncomfortable situations. A physical response, such as being slapped, can occur if someone feels harassed or invaded by the unwanted attention.\nFinal Answer: 4

Now please solve the following question using the same inference style and format as the example above. 
Question: """
    
    
    ecqa_rewrite_in_gpt4_style_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q1}
Inference: To infer the correct option, let's break down the context of the question and each option:

1. Bloody mess: In a scene described as a "brutal killing," it is likely that there would be visible blood and signs of violence. This option aligns with the context of a violent crime scene.

2. Pleasure: This option is not consistent with the context of a brutal killing, which is generally associated with distressing and horrific conditions rather than pleasure.

3. Being imprisoned: While imprisonment might be a consequence of a crime, it is not something typically observed directly at the scene of the killing itself.

4. Feeling of guilt: This option refers to an emotional state rather than something seen at the crime scene.

5. Cake: This option is unrelated to the context of a brutal killing and is not something one would expect to see at such a scene.

Given the context of a brutal killing, the most appropriate option that describes what a person might see is a "bloody mess."

Final Answer: 1

Example 2
Question: {q2}
Inference: To infer the correct option, let's analyze the context of setting a romantic atmosphere and each option:

1. Dimly lit room: A dimly lit room can be enhanced with candles to create a romantic atmosphere. This option is relevant to creating a romantic setting.

2. Synagogue: While candles might be used in religious settings, this is not typically associated with setting a romantic atmosphere.

3. Bedroom: A bedroom is a common place where people might light candles to set a romantic mood. This option aligns well with the idea of creating a romantic atmosphere.

4. Birthday cake: Candles on a birthday cake are typically used for celebrations rather than setting a romantic atmosphere.

5. Roses: While roses are romantic, they do not involve lighting candles.

Given the context of setting a romantic atmosphere, the most appropriate place to light a candle is a "bedroom."

Final Answer: 3

Example 3
Question: {q3}
Inference: Unwanted flirting can lead to various outcomes depending on the context and the reactions of the individuals involved. Let’s examine each option:

1. Attraction: This is unlikely because unwanted flirting usually does not foster genuine attraction. Instead, it can create discomfort or aversion.

2. Problems: This is a plausible outcome. Unwanted flirting can lead to interpersonal issues, misunderstandings, or conflicts, especially if one party feels harassed or disrespected.

3. The gallows: This option is metaphorical and extreme. It does not realistically represent the outcome of unwanted flirting in a modern context.

4. Being slapped: This is a physical and immediate reaction that could occur if someone feels significantly disrespected or violated by unwanted flirting. It's a plausible response, though more extreme than simply experiencing problems.

5. Curiosity: This is less likely to be a result of unwanted flirting, as curiosity is not a typical reaction to someone making advances that are unwelcome.

Based on the analysis, the most possible answer is:

Final Answer: 4

Now please solve the following question using the same inference style and format as the example above. 
Question: """
    
    ecqa_mini_gpt4_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q1}
Inference: When considering what a person might see at the scene of a brutal killing, the options can be evaluated based on their relevance to such a distressing situation. \n\n1. A \"bloody mess\" is a highly likely interpretative choice. The aftermath of violence usually involves significant bloodshed and disorder.\n2. \"Pleasure\" is not an appropriate observation in the context of a crime scene, especially one described as brutal.\n3. \"Being imprisoned\" typically reflects the consequences for perpetrators, rather than a scene observer\u2019s experience.\n4. \"Feeling of guilt\" might be an emotional response of a person involved, but it does not describe a visual aspect of the scene.\n5. \"Cake\" is entirely irrelevant in this context and cannot be fittingly associated with a murder scene.\n\nTherefore, the most plausible and relevant option to what might be seen is clearly a \"bloody mess.\" \n\nFinal Answer: 1

Example 2
Question: {q2}
Inference: To create a romantic atmosphere, you typically want a setting that is intimate and personal. Among the options provided, a \"bedroom\" is most closely associated with romance and private moments between couples. While a \"dimly lit room\" could also invoke a romantic feeling, it is less specific compared to the personal and private nature of a bedroom. Other options, like a \"synagogue\" and \"birthday cake,\" do not align with a romantic setting, and \"roses,\" while romantic, are more of an accessory rather than a place to light a candle. Therefore, the best option for setting a romantic atmosphere is the bedroom.\n\nFinal Answer: 3

Example 3
Question: {q3}
Inference: Unwanted flirting can lead to various negative consequences. It often makes the person on the receiving end uncomfortable, which can escalate into more serious reactions such as being slapped, especially if the person feels threatened or violated. Other options like attraction, curiosity, or even problems can arise in different contexts, but the immediate and most direct result of unwanted flirting is likely to provoke a defensive or angry response from the person being flirted with. \n\nFinal Answer: 4

Now please solve the following question using the same inference style and format as the example above. 
Question: """
    
    ecqa_anthropic_style_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q1}
Inference: At the scene of a brutal killing, a person would likely encounter a disturbing and graphic environment. The most immediate and obvious sight would be blood and other physical evidence of violence. This aligns closely with option (1) \"bloody mess.\"\n\nThe other options are less likely or inappropriate:\n(2) Pleasure is not a typical reaction to witnessing a crime scene.\n(3) Being imprisoned is not something one would see at a crime scene.\n(4) While a perpetrator might feel guilt, this is not something an observer would see.\n(5) Cake is entirely unrelated to a crime scene.\n\nGiven the violent nature of a brutal killing, the most logical and realistic option is a bloody mess.\n\nFinal Answer: 1

Example 2
Question: {q2}
Inference: When setting a romantic atmosphere, lighting a candle is often associated with creating a soft, intimate ambiance. Among the given options, the most suitable place for this would be in a bedroom. A bedroom is typically a private, personal space where couples can relax and spend quality time together. The soft glow of candlelight in a bedroom can enhance the romantic mood and create a cozy atmosphere.\n\nA dimly lit room could also be a good option, but it's less specific than a bedroom. A synagogue is a place of worship and not typically associated with romantic settings. A birthday cake is festive but not inherently romantic. Roses are often associated with romance, but they're not a location where you would light a candle.\n\nFinal Answer: 3

Example 3
Question: {q3}
Inference: Unwanted flirting can lead to various negative consequences. The recipient of such attention may feel uncomfortable, harassed, or even threatened. This behavior can create tension in social or professional settings, potentially damaging relationships or reputations. In some cases, it might escalate to more serious issues like stalking or harassment complaints.\n\nAmong the given options, \"problems\" (2) is a broad term that could encompass many potential outcomes, but it's not specific enough. \"Attraction\" (1) is unlikely since the flirting is unwanted. \"The gallows\" (3) is an extreme and unrealistic consequence. \"Curiosity\" (5) doesn't fit as a result of unwanted flirting.\n\n\"Being slapped\" (4) is a more specific and plausible reaction to unwanted flirting. While not always a literal outcome, it represents a clear, immediate rejection of the unwanted advances and conveys the negative impact such behavior can have.\n\nFinal Answer: 4

Now please solve the following question using the same inference style and format as the example above. 
Question: """

    

# """We know the definetion of entailment, contradiction and neutral is\nEntailment: The statement is definitely true given the context.\nContradiction: The statement is definitely false given the context.\nNeutral: The truth of the statement is undetermined or irrelevant given the context.\n\nWe have \nContext: A person on a horse jumps over a broken down airplane.\nStatement: A person is training his horse for a competition.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context. \n\nPlease inference first, then provide the final answer (Entailment/Contradiction/Neutral) at the end, after 'Final Answer:'\n
    q1 = \
"""We have \nContext: A person on a horse jumps over a broken down airplane.\nStatement: A person is training his horse for a competition.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context."""
    q2 = \
"""We have \nContext: A person on a horse jumps over a broken down airplane.\nStatement: A person is outdoors, on a horse.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context."""
    q3 = \
"""We have \nContext: Children smiling and waving at camera\nStatement: The kids are frowning.\n\nDetermine whether the statement is entailment, contradiction, or neutral given the context."""
    
    esnli_gpt4_style_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q1}
Inference: The context describes a person on a horse jumping over a broken down airplane, which suggests a dynamic and possibly unusual activity involving horse riding. However, this doesn't necessarily indicate that the person is training the horse specifically for a competition. The action could be for a variety of reasons outside of competition training, such as a stunt, an emergency, or simply recreational riding.\n\nWhile the skill displayed might be similar to what is useful for competition, we can't definitively conclude from the context that the purpose of jumping over an airplane is competition training. There's no direct information supporting that the training is aimed specifically at preparing for a competition, nor is there information directly contradicting this possibility. The scenario remains open to various interpretations without further details. \n\nFinal Answer: Neutral

Example 2
Question: {q2}
Inference: In the context described, a person is on a horse, and they jump over a broken down airplane. Such an activity would necessarily take place outdoors, as airplanes and sufficient space for horse jumping are usually found outside. Therefore, the fact that the person is on a horse and that they are outdoors is supported directly by the context.\n\nFinal Answer: Entailment

Example 3
Question: {q3}
Inference: Given the context, where \"Children are smiling and waving at camera,\" we can infer what the physical expressions of the children are. Smiling is explicitly the opposite facial expression to frowning, characterized by an upturned mouth and a general expression of happiness, as opposed to the downturned mouth and expression of discontent or unhappiness in a frown.\n\nSince the statement suggests that \"The kids are frowning,\" this directly opposes the context that they are smiling. Smiling and frowning are mutually exclusive facial expressions\u2014one cannot simultaneously smile and frown. Therefore, given the context indicates smiling, the statement that the children are frowning is in direct conflict with the information provided.\n\nFinal Answer: Contradiction

Now please solve the following question using the same inference style and format as the example above. 
Question: """

    esnli_mini_gpt4_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q1}
Inference: The context describes a scene where a person on a horse is performing a jump over a broken-down airplane. This indicates that the person is likely engaged in an activity involving the horse, such as possibly training or practicing skills relevant to riding or jumping. The statement in question discusses a person training his horse for a competition, which suggests a context related to skill improvement and competitive preparation.\n\nHowever, the specific scenario described in the context (jumping over a broken-down airplane) does not definitively confirm that the training is specifically for a competition, as the type of training could vary. Therefore, while there may be a connection between the two, it does not guarantee the truth of the statement based solely on the provided context.\n\nThus, the truth of the statement is undetermined, making the relationship between the context and the statement neutral.\n\nFinal Answer: Neutral

Example 2
Question: {q2}
Inference: Given the context of a person on a horse jumping over a broken-down airplane, we can infer several facts. The statement \"A person is outdoors, on a horse\" provides information that is consistent with the context. Since a horse is typically used outdoors, and the act of jumping over an airplane implies an outdoor setting, we can conclude that the statement is definitely true in this context.\n\nFinal Answer: Entailment

Example 3
Question: {q3}
Inference: The context describes children who are smiling and waving at the camera, which clearly indicates a happy and cheerful demeanor. The statement claiming that \"The kids are frowning\" directly contradicts this joyful behavior portrayed in the context. Since a frown signifies sadness or displeasure, and the children in the context are smiling, it is impossible for both to be true at the same time. Therefore, the statement must be considered definitely false given the context.\n\nFinal Answer: Contradiction

Now please solve the following question using the same inference style and format as the example above. 
Question: """


    esnli_minimum_change_style_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q1}
Inference: The statement 'A person is training his horse for a competition' is neutral given the context. The context states that a person is on a horse jumping over a broken down airplane, but it does not provide any information about the purpose of the jump or the training status of the horse. Therefore, the truth of the statement is undetermined or irrelevant given the context.\n\nFinal Answer: Neutral.

Example 2
Question: {q2}
Inference: The statement 'A person is outdoors, on a horse' is an entailment given the context. The context clearly mentions a person on a horse jumping over a broken down airplane, which suggests the person is outdoors. Therefore, the statement directly follows from the provided context, making it an entailment. \n\nFinal Answer: Entailment.

Example 3
Question: {q3}
Inference: Based on the context provided, the statement 'The kids are frowning' is a contradiction. The context describes children smiling and waving at the camera, which is the opposite of frowning. Therefore, the statement is definitely false given the context.\n\nFinal Answer: Contradiction.

Now please solve the following question using the same inference style and format as the example above. 
Question: """
    

    esnli_mini_gpt4_style_prompt = \
f"""
We have 3 examples below

Example1
Question: {q1}
Inference: Let's analyze the given context and statement:\n\nContext: A person on a horse jumps over a broken down airplane.\n\nStatement: A person is training his horse for a competition.\n\nAnalysis:\n\n1. **Entailment**: For the statement to be an entailment, it must be definitely true based on the context provided. In this case, while it is possible that the person jumping over the airplane is training for a competition, the context does not specifically indicate that this is the case. The context could suggest other scenarios for the horse's action, such as recreation or a demonstration.\n\n2. **Contradiction**: For the statement to be a contradiction, it must be definitely false given the context. The context does not provide any explicit information that would confirm or deny the possibility of training for a competition. Therefore, it cannot be categorically regarded as false.\n\n3. **Neutral**: For the statement to be neutral, its truth would be undetermined or irrelevant based on the context. Given that the context allows for multiple interpretations and does not confirm the specific scenario of training for a competition, the truth of the statement remains undetermined.\n\nBased on the above analysis, the context of the person on the horse jumping over an airplane does not provide enough information to confirm that the person is training for a competition; thus, the statement is reasonable but not substantiated by the context.\n\nFinal Answer: Neutral

Example2
Question: {q2}
Inference: Let's analyze the given context and statement:\n\n**Context:** A person on a horse jumps over a broken down airplane.\n\n**Statement:** A person is outdoors, on a horse.\n\n**Analysis:**\n\n1. **Entailment:** For the statement to be an entailment, it must be definitely true based on the context. The context describes a situation where a person is riding a horse while interacting with an airplane. Given that airplanes are typically located outdoors, it is reasonable to conclude that the person is also outdoors. This direct connection between the context and the statement strongly supports that the statement is indeed true.\n\n2. **Contradiction:** For the statement to be a contradiction, it would have to be definitely false based on the context. However, there is no evidence in the context suggesting that the person cannot be outdoors. The presence of a broken down airplane, typically associated with outdoor environments, contradicts any notion that the person is indoors.\n\n3. **Neutral:** For the statement to be neutral, the truth of the statement would need to be undetermined or irrelevant based on the context. In this scenario, the evidence presented in the context directly indicates that the person is likely outdoors, making the statement not only relevant but also verifiably true.\n\nGiven the context of a person on a horse jumping over a broken-down airplane, it is clear that the statement \"a person is outdoors, on a horse\" is unequivocally true.\n\nFinal Answer: Entailment

Example3
Question: {q3}
Inference: Let's analyze the given context and statement: \n\nContext: Children smiling and waving at camera. \n\nStatement: The kids are frowning.\n\nAnalysis:\n\n1. **Entailment**: For the statement to be an entailment, it must be definitely true based on the context. In this case, the context clearly depicts children who are smiling and actively engaged in a pleasant behavior (waving at the camera). Thus, the statement that they are frowning cannot be true as it contradicts the emotions expressed in the context.\n\n2. **Contradiction**: For the statement to be a contradiction, it must be definitely false based on the context. Since the context shows the children smiling, their behavior directly opposes the claim that they are frowning. Therefore, the claim that the kids are frowning is definitively false given the context, fulfilling the criteria for contradiction.\n\n3. **Neutral**: For the statement to be neutral, the truth of the statement would be undetermined or irrelevant based on the context. However, in this case, the context provides a clear indication of the children's expressions. Since they are smiling, the truth of the statement about the kids frowning is not ambiguous\u2014it is concrete.\n\nGiven the context of children who are smiling and waving, the statement that \"the kids are frowning\" is clearly false.\n\nFinal Answer: Contradiction


Now please solve the following question using the same inference style and format as the 3 example above. 
Question: """
    
    esnli_anthropic_style_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q1}
Inference: The context describes a person on a horse jumping over a broken down airplane, which is an unusual and specific scenario. The statement suggests that the person is training their horse for a competition. While it's possible that this activity could be part of training for a competition, there's no explicit information in the context to confirm this.\n\nJumping over a broken down airplane is not a typical training exercise for horse competitions. However, it could potentially be a part of some specialized or extreme equestrian event. Without more information, we can't definitively say whether this is training for a competition or simply a unique activity.\n\nThe context doesn't provide any information about the person's intentions or the purpose of the horse jumping. It could be for training, for fun, or for some other reason entirely. Therefore, we don't have enough information to determine if the statement is true or false based on the given context.\n\nFinal Answer: Neutral

Example 2
Question: {q2}
Inference: The context states that \"A person on a horse jumps over a broken down airplane.\" This clearly indicates that there is a person riding a horse. The statement \"A person is outdoors, on a horse\" is directly supported by this context.\n\nThe context implies an outdoor setting, as jumping over a broken down airplane would typically occur outside. Horses are also generally ridden outdoors.\n\nThe statement doesn't include any information about jumping over an airplane, but it doesn't contradict this detail either. It simply focuses on two key elements that are definitely true based on the context: a person is on a horse, and this is happening outdoors.\n\nFinal Answer: Entailment

Example 3
Question: {q3}
Inference: The context states that children are smiling and waving at the camera. This implies a positive, cheerful demeanor. In contrast, the statement claims that the kids are frowning, which indicates a negative or unhappy expression. Smiling and frowning are opposite facial expressions that cannot occur simultaneously. Therefore, the statement directly contradicts the information provided in the context.\n\nFinal Answer: Contradiction

Now please solve the following question using the same inference style and format as the example above. 
Question: """

    esnli_gt_style_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q1}
Inference: Because the person is not necessarily training his horse, the answer is Neutral.
Final Answer: Neutral

Example 2
Question: {q2}
Inference: Because a broken down airplane is outdoors, the answer is Entailment.
Final Answer: Entailment

Example 3
Question: {q3}
Inference: Because One cannot be smiling and frowning at the same time, the answer is contradiction.
Final Answer: Contradiction

Now please solve the following question using the same inference style and format as the example above. 
Question: """
    
    esnli_rewrite_in_gpt4_style_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q1}
Inference: The context describes a person on a horse jumping over a broken down airplane. This indicates an action involving a horse, but it does not provide any specific information about the purpose of the action, such as training for a competition. The statement about training the horse for a competition is not directly supported or refuted by the context. Therefore, the truth of the statement is not determined based on the given context.

Final Answer: Neutral

Example 2
Question: {q2}
Inference: The context describes a person on a horse jumping over a broken down airplane. This scenario implies that the person is outside, as they are interacting with an airplane in a setting where such a situation is likely to occur (i.e., outdoors). Therefore, the statement that a person is outdoors on a horse is supported by the context.

Final Answer: Entailment

Example 3
Question: {q3}
Inference: The context specifies that children are smiling and waving at the camera, which directly contradicts the statement that the kids are frowning. Thus, the statement that the kids are frowning is definitely false given the context.

Final Answer: Contradiction

Now please solve the following question using the same inference style and format as the example above. 
Question: """

    esnli_rewrite_in_own_words_style_prompt = \
f"""
We have 3 examples below

Example 1
Question: {q1}
Inference: The context provided depicts a somewhat uncommon situation where an individual and a horse are involved in an action that involves jumping over an inanimate object\u2014a broken down airplane. This scenario does not inherently suggest that the individual is engaging in this activity for the purpose of training the horse for a competition. The act of jumping over an obstacle could be for a variety of reasons, such as recreation, an emergency, or even a staged performance. It is not indicated with certainty that the person's intent is to prepare the horse for competitive events. Consequently, the connection between the context and the statement about training for a competition cannot be definitively established. Therefore, the relationship between the context and the statement is one where the truth of the statement cannot be determined based on the information given and thus is deemed neutral.\n\nFinal Answer: Neutral

Example 2
Question: {q2}
Inference: In the given context, a person on a horse jumping over a broken down airplane suggests an outdoor setting because airplanes are typically found outside, and the act of jumping over one implies an open space that is characteristic of an outdoor environment. Additionally, horseback riding is an activity commonly associated with being outdoors. Hence, the statement that a person is outdoors, on a horse, logically follows from and is supported by the context provided.\n\nFinal Answer: Entailment

Example 3
Question: {q3}
Inference: Smiling and frowning are two expressions that cannot occur simultaneously on a person's face. Smiling denotes happiness or friendliness, while frowning is indicative of displeasure or sadness. Given the provided context, which describes the children as \"smiling and waving at the camera\", the statement \"The kids are frowning\" asserts the opposite of what has been established. Since the children cannot be both smiling and frowning at the same time, the statement contradicts the context.\n\nFinal Answer: Contradiction

Now please solve the following question using the same inference style and format as the example above. 
Question: """




    q1 = \
"""
User: Find a dentist in London who can perform a root canal treatment.
API descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}
Generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)]:
"""
# """\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a dentist in London who can perform a root canal treatment.TIME: 2023-10-15 10:00:00\nGenerate next API Request: """
# """\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nGenerate next API Request: """
    

    q2 = \
"""Hint: ('->'means received tool)
User: Find a dentist in London who can perform a root canal treatment.
Previous API-Request: [ToolSearcher(keywords='dentist root canal treatment London')]->{'name': 'FindDentist', 'description': 'API for finding a dentist based on location and services offered.', 'input_parameters': {'location': {'type': 'str', 'description': 'The city or area to find dentists.'}, 'service': {'type': 'str', 'description': 'The type of dental service required.'}}, 'output_parameters': {'dentist': {'type': 'str', 'description': 'The name of the dentist.'}}}
Generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)]:
"""
# """\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a dentist in London who can perform a root canal treatment.TIME: 2023-10-15 10:00:00\nAPI-Request: [ToolSearcher(keywords='dentist root canal treatment London')]->{'name': 'FindDentist', 'description': 'API for finding a dentist based on location and services offered.', 'input_parameters': {'location': {'type': 'str', 'description': 'The city or area to find dentists.'}, 'service': {'type': 'str', 'description': 'The type of dental service required.'}}, 'output_parameters': {'dentist': {'type': 'str', 'description': 'The name of the dentist.'}}}\nGenerate next API Request: """

# """\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nAPI-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]->{'name': 'HealthcareProviderAppointmentChecker', 'description': 'API for checking the availability of appointments with healthcare providers.', 'input_parameters': {'specialty': {'type': 'str', 'description': 'The specialty of the healthcare provider.'}, 'location': {'type': 'str', 'description': 'The city location.'}}, 'output_parameters': {'appointments': {'type': 'list', 'description': 'A list of available appointment slots.'}}}\nGenerate next API Request: """

    q3 = \
"""User: Find a dentist in London who can perform a root canal treatment.
Previous API-Request: [ToolSearcher(keywords='dentist root canal treatment London')]->{'tool name': 'FindDentist'}
Previous API-Request: [FindDentist(location=London, service=root canal treatment)]->{'dentist': 'Dr. John Smith'}
Generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)]:
"""
# """\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a dentist in London who can perform a root canal treatment.TIME: 2023-10-15 10:00:00\nAPI-Request: [ToolSearcher(keywords='dentist root canal treatment London')]->{'name': 'FindDentist', 'description': 'API for finding a dentist based on location and services offered.', 'input_parameters': {'location': {'type': 'str', 'description': 'The city or area to find dentists.'}, 'service': {'type': 'str', 'description': 'The type of dental service required.'}}, 'output_parameters': {'dentist': {'type': 'str', 'description': 'The name of the dentist.'}}}\nAPI-Request: [FindDentist(location=London, service=root canal treatment)]->{'dentist': 'Dr. John Smith'}\nGenerate next API Request: """

# """\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a cardiologist in Los Angeles for a check-up appointment.TIME: 2034-04-15 10:00:00\nAPI-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]->{'name': 'HealthcareProviderAppointmentChecker', 'description': 'API for checking the availability of appointments with healthcare providers.', 'input_parameters': {'specialty': {'type': 'str', 'description': 'The specialty of the healthcare provider.'}, 'location': {'type': 'str', 'description': 'The city location.'}}, 'output_parameters': {'appointments': {'type': 'list', 'description': 'A list of available appointment slots.'}}}\nAPI-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]->{'appointments': ['2034-04-18 14:30:00', '2034-04-19 11:00:00', '2034-04-20 09:45:00']}\nGenerate next API Request: """

    q4 = \
"""User: Find a dentist in London who can perform a root canal treatment.
Previous API-Request: [ToolSearcher(keywords='dentist root canal treatment London')]->{'tool name': 'FindDentist'}
Previous API-Request: [FindDentist(location=London, service=root canal treatment)]->{'dentist': 'Dr. John Smith'}
Previous API-Request: [ToolSearcher(keywords='dentist availability')]->{'name': 'DentistAvailability', 'description': 'API for checking the availability of dentists.', 'input_parameters': {'dentist': {'type': 'str', 'description': 'The name of the dentist.'}, 'date': {'type': 'str', 'description': 'The date for checking availability.'}}, 'output_parameters': {'availability': {'type': 'str', 'description': 'The availability of the dentist on the given date.'}}}
Generate the next API request in the format of [ApiName(key1='value1', key2='value2', ...)]:
"""
# """\nGenerate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the user's utterance and available API requests.\nThe current time is {{time}}.\nInput: \nUser: User's utterence\n\nExpected output:\nAPI-Request: [ApiName(key1='value1', key2='value2', ...)]\n\nAPI descriptions:\n{\"apiCode\": \"ToolSearcher\", \"description\": \"Searches for relevant tools in library based on the keywords.\", \"parameters\": {\"keywords\": {\"type\": \"str\", \"description\": \"The keyword to search for.\"}}, \"response\": {\"best_matchs\": {\"type\": \"Union[List[dict], dict]\", \"description\": \"The best match tool(s).\"}}}\nUser: Find a dentist in London who can perform a root canal treatment.TIME: 2023-10-15 10:00:00\nAPI-Request: [ToolSearcher(keywords='dentist root canal treatment London')]->{'name': 'FindDentist', 'description': 'API for finding a dentist based on location and services offered.', 'input_parameters': {'location': {'type': 'str', 'description': 'The city or area to find dentists.'}, 'service': {'type': 'str', 'description': 'The type of dental service required.'}}, 'output_parameters': {'dentist': {'type': 'str', 'description': 'The name of the dentist.'}}}\nAPI-Request: [FindDentist(location=London, service=root canal treatment)]->{'dentist': 'Dr. John Smith'}\nAPI-Request: [ToolSearcher(keywords='dentist availability')]->{'name': 'DentistAvailability', 'description': 'API for checking the availability of dentists.', 'input_parameters': {'dentist': {'type': 'str', 'description': 'The name of the dentist.'}, 'date': {'type': 'str', 'description': 'The date for checking availability.'}}, 'output_parameters': {'availability': {'type': 'str', 'description': 'The availability of the dentist on the given date.'}}}\nGenerate next API Request: """    

#openai_gpt4_generated_api_bank_False_1000_r1
    api_bank_gpt4_style_prompt = \
f"""
We offer you 4 examples to show how to generate an API request based on the user's utterance and available API requests. Please follow the same format to solve the question below. Remember You only need to generate the next api call
Example 1
Question: {q1}
Inference: To fulfill the user\u2019s request effectively, I will analyze the keywords and context provided by the user. The user is looking for a dentist based in London who specializes in root canal treatments. Therefore, the search will focus on those specific criteria: the profession (dentist), the procedure (root canal treatment), and the location (London).\n\nWith these components identified, I will structure the API request by combining the keywords into a suitable format for the `ToolSearcher` API.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='dentist root canal treatment London')]

Example 2
Question: {q2}
Inference: To find a dentist in London who can perform a root canal treatment, I will utilize the FindDentist API, which requires the location and the type of dental service needed. In this case, the location is \"London\" and the service required is \"root canal treatment\". Therefore, the correct parameters for the API request are:\n\n- location: London\n- service: root canal treatment\n\nPutting this together, the final API request is constructed as follows:\n\nFinal Answer: API-Request: [FindDentist(location=London, service=root canal treatment)]

Example 3
Question: {q3}
Inference: To address the user's request for finding a dentist in London who can perform a root canal treatment, I infer that the key focus is to determine the availability of dentists for that specific service. The search would involve using the keywords \"dentist availability\" as it encompasses the need for information on dentists who offer root canal treatments.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='dentist availability')]

Example 4
Question: {q4}
Inference: To find a dentist in London who can perform a root canal treatment, we start by searching for relevant dentists available for that specific service. The tool \"FindDentist\" could be used to gather the name of a suitable dentist based on the parameters provided in the user's request. \n\nNext, we will check the availability of the identified dentist for the specified date (2023-10-15). In this case, the dentist found is \"Dr. John Smith\". Hence, we need to make an API request to check his availability on that date.\n\nFinal Answer: API-Request: [DentistAvailability(dentist=Dr. John Smith, date=2023-10-15)]

Now please solve the following question using the same inference style and format as the example above.
Question: """
# f"""
# We offer you 3 examples to show how to generate an API request based on the user's utterance and available API requests. Please follow the same format to solve the question below. Remember You only need to generate the next api call
# Example 1
# Question: {q1}
# Inference: To find a cardiologist in Los Angeles for a check-up appointment, we need to search for relevant tools that can assist with checking the availability of healthcare providers. The keyword \u201chealthcare provider appointment availability checker\u201d captures the essence of the request since it pertains to looking for appointment availability for specialists in the medical field. \n\nBased on this reasoning, I will create the API request.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]

# Example 2
# Question: {q2}
# Inference: To find a cardiologist in Los Angeles for a check-up appointment, I will use the Healthcare Provider Appointment Checker API. This API focuses on checking the availability of appointments with healthcare providers based on their specialty and location. Here, the specialty is 'cardiologist' and the location is 'Los Angeles'.\n\nFinal Answer: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]

# Example 3
# Question: {q3}
# Inference: The user's request is to find a cardiologist in Los Angeles for a check-up appointment. This falls under the category of seeking healthcare-related services, specifically looking for tools that assist in scheduling or checking appointments with medical providers. The relevant API to fulfill this request is the ToolSearcher, which can be utilized with appropriate keywords to locate services related to scheduling healthcare appointments.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]

# Now please solve the following question using the same inference style and format as the example above.
# Question: """

# sft data




    
    api_bank_gt_style_prompt = \
f"""
We offer you 4 examples to show how to generate an API request based on the user's utterance and available API requests. Please follow the same format to solve the question below. Remember You only need to generate the next api call

Example 1
Question: {q1}
Final Answer: API-Request: [ToolSearcher(keywords='dentist root canal treatment London')]

Example 2
Question: {q2}
Final Answer: API-Request: [FindDentist(location=London, service=root canal treatment)]

Example 3
Question: {q3}
Final Answer: API-Request: [ToolSearcher(keywords='dentist root canal treatment London')]

Example 4
Question: {q4}
Final Answer: API-Request: [DentistAvailability(dentist=Dr. John Smith, date=2023-10-15)]

Now please solve the following question using the same format as the example above. 
Question: """
# F"""
# We offer you 3 examples to show how to generate an API request based on the user's utterance and available API requests. Please follow the same format to solve the question below. Remember You only need to generate the next api call

# Example 1
# Question: {q1}
# Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]

# Example 2
# Question: {q2}
# Final Answer: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]

# Example 3
# Question: {q3}
# Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]

# Now please solve the following question using the same format as the example above. 
# Question: """


#openai_gpt4_generated_new_rewrite_strategy
    api_bank_provide_gpt4_style_example_prompt = \
f"""We offer you 4 examples to show how to generate an API request based on the user's utterance and available API requests. Please follow the same format to solve the question below. Remember You only need to generate the next api call

Example 1
Question: {q1}
Inference: The user is requesting to find a dentist in London who can perform a specific procedure, which is a root canal treatment. The available API description indicates that the ToolSearcher API can be used to search for relevant tools based on keywords related to the user's request. To match the user's needs, the keywords for the API request should include \"dentist\", \"root canal treatment\", and \"London\" to ensure a precise search for the required service in the specified location.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='dentist root canal treatment London')]

Example 2
Question: {q2}
Inference: The user is requesting to find a dentist in London who can perform root canal treatment. Based on the available API descriptions, the appropriate API to find a dentist based on location and services offered is the FindDentist API. The inputs required for this API are the location (London) and the service (root canal treatment) as specified by the user.\n\nFinal Answer: API-Request: [FindDentist(location=London, service=root canal treatment)]

Example 3
Question: {q3}
Inference: The user is looking for a dentist in London who can perform root canal treatment. After the previous API request identified a dentist, the next logical step would be to check for the dentist\u2019s availability. Therefore, I will generate a request to search for tools related to dentist availability.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='dentist availability')]

Example 4
Question: {q4}
Inference: The user requested to find a dentist in London who can perform a root canal treatment. The initial API call was made with the ToolSearcher to locate relevant tools, which returned the FindDentist API. This API was subsequently used to identify a specific dentist, Dr. John Smith. To proceed, the user should now check the availability of Dr. John Smith on the date of interest (2023-10-15) using the DentistAvailability API.\n\nFinal Answer: API-Request: [DentistAvailability(dentist=Dr. John Smith, date=2023-10-15)]

Now please solve the following question using the same inference style and format as the example above. 
Question: """
# f"""
# We offer you 3 examples to show how to generate an API request based on the user's utterance and available API requests. Please follow the same format to solve the question below. Remember You only need to generate the next api call

# Example 1
# Question: {q1}
# Inference: The user is looking to find a cardiologist in Los Angeles for a check-up appointment, which indicates a health-related query. The ToolSearcher API can be utilized to search for relevant tools based on keywords related to healthcare provider appointment availability. Hence, we should request the ToolSearcher API to search for tools pertaining to checking the availability of healthcare appointments.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]

# Example 2
# Question: {q2}
# Inference: The user is looking for a cardiologist in Los Angeles to schedule a check-up appointment. The previous API request made use of the ToolSearcher API to find tools for checking the availability of healthcare provider appointments. The identified tool, HealthcareProviderAppointmentChecker, requires specific inputs: the specialty (cardiologist) and the location (Los Angeles). Thus, the next logical API request should be directed to this specific healthcare provider appointment checking tool with the relevant parameters.\n\nFinal Answer: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]

# Example 3
# Question: {q3}
# Inference: The user has already located available appointment slots for a cardiologist in Los Angeles. They are now looking to schedule an appointment with one of those slots. This requires finding a scheduling tool. The relevant API that can assist in this task is the ToolSearcher, and the keyword we will use to find this tool is 'healthcare provider appointment scheduler'.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]

# Now please solve the following question using the same inference style and format as the example above. 
# Question: """

#gpt4_generated_step_by_step_1000
    api_bank_step_by_step_style_prompt = \
f"""We offer you 4 examples to show how to generate an API request based on the user's utterance and available API requests. Please follow the same format to solve the question below. Remember You only need to generate the next api call

Example 1
Question: {q1}
Inference: To create the API request based on the user's utterance, we start by analyzing the key components of the request.\n\n1. **Identify Keywords**: The user is looking for a specific type of professional service. The keywords from the utterance are:\n   - \"dentist\" (the type of professional)\n   - \"root canal treatment\" (the specific service needed)\n   - \"London\" (the location)\n\n2. **Combine Keywords for the API Request**: To accurately convey the search intent in the API request, we will combine these keywords in a single string that describes what the user is looking for. Therefore, the combined keywords will be:\n   - \"dentist root canal treatment London\"\n\n3. **Format the API Request**: The desired API request format requires specifying the API name along with its parameters. Based on the API description provided, the API name is \"ToolSearcher\" and it accepts one parameter: \"keywords\". Therefore, the formatted request will look like:\n\n   API-Request: [ToolSearcher(keywords='dentist root canal treatment London')]\n\nPutting it all together, we construct the final answer.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='dentist root canal treatment London')]

Example 2
Question: {q2}
Inference: To find a dentist in London who can perform a root canal treatment, we can break down the user's utterance into specific components that the APIs can work with. \n\n1. Identify the main keywords in the user\u2019s request: \n   - The profession needed: \"dentist\"\n   - The specific service requested: \"root canal treatment\"\n   - The location specified: \"London\"\n\n2. Based on this breakdown, we can map these keywords to the parameters needed for an API request. The relevant API we are looking to use would involve finding a dentist based on location and services offered.\n\n3. From the API descriptions provided, we can see that the FindDentist API requires two input parameters: \n   - `location`: which corresponds to \"London\"\n   - `service`: which corresponds to \"root canal treatment\"\n\n4. Formulate the API request using the extracted information:\n\n   - For `location`, we will set it to \"London\".\n   - For `service`, we will set it to \"root canal treatment\".\n\n5. Construct the final API request with the corresponding values.\n\nFinal Answer: API-Request: [FindDentist(location=London, service=root canal treatment)]

Example 3
Question: {q3}
Inference: To address the user's utterance regarding finding a dentist who can perform a specific treatment, we start with the keywords mentioned in the utterance. The user asked for a dentist in London capable of performing a root canal treatment. \n\nNext, we analyze the relevant parameters for our API request. Based on the available API, we focus on the keywords that align with the search functionality provided by the `ToolSearcher` API. Specifically, we note the keywords: \"dentist,\" \"root canal treatment,\" and \"London.\" \n\nCombining these into a single search keyword string, we get \"dentist root canal treatment London.\" This string encapsulates the user's specific need for a dentist who can perform a root canal in a particular location. \n\nNow, following the inference process and using the information provided, the API request can be constructed as follows:\n\nFinal Answer: API-Request: [ToolSearcher(keywords='dentist root canal treatment London')]

Example 4
Question: {q4}
Inference: To generate the appropriate API request based on the user\u2019s utterance, I will follow these steps:\n\n1. **Identify the User's Request**: The user has asked to find a dentist in London who can perform a root canal treatment.\n\n2. **Extract Keywords**: Key phrases in the user's utterance include \"dentist,\" \"root canal treatment,\" and \"London.\" These will be combined into a search keyword for the API request.\n\n3. **Formulate the First API Request**: The first action is to search for relevant dentists. Therefore, the ToolSearcher API will be invoked with the keywords derived from the user\u2019s input:\n   - Keywords: `'dentist root canal treatment London'`\n\n4. **Determine the Dentist\u2019s Name**: After the ToolSearcher is completed, I will assume the response provides the best match dentist, which in this hypothetical case is \u2018Dr. John Smith.\u2019\n\n5. **Check Availability of the Dentist**: The next step is to check the availability of Dr. John Smith for the current date, which is specified as 2023-10-15.\n\n6. **Formulate the Second API Request**: Now, I'll create an API request to check Dr. John Smith's availability using the DentistAvailability API:\n   - Dentist: `Dr. John Smith`\n   - Date: `2023-10-15`\n\nPutting this all together, the final API request reflects the need to check the availability of the dentist on the specified date.\n\nFinal Answer: API-Request: [DentistAvailability(dentist=Dr. John Smith, date=2023-10-15)]

Now please solve the following question using the same inference style and format as the example above. 
Question: """
# f"""
# We offer you 3 examples to show how to generate an API request based on the user's utterance and available API requests. Please follow the same format to solve the question below. Remember You only need to generate the next api call

# Example 1
# Question: {q1}
# Inference: To generate the API request, let's break down the user's utterance and analyze the relevant keywords:\n\n1. **User's Utterance:** \"Find a cardiologist in Los Angeles for a check-up appointment.\"\n2. **Identify Keywords:**\n   - The key service being sought is \"cardiologist.\"\n   - The location specified is \"Los Angeles.\"\n   - The context implies a need for \"check-up appointment.\"\n\n3. **Relevant API:** According to the provided API descriptions, the suitable API is `ToolSearcher`, which searches for relevant tools based on keywords.\n\n4. **Formulating the API Request:**\n   - We need to focus on the main aspect of the user's request, which involves finding tools related to healthcare providers, specifically cardiologists, in a specific location for an appointment.\n   - The pertinent keyword that encompasses these details while staying within the capabilities of the API is: \"healthcare provider appointment availability checker.\"\n\n5. **Constructing the Final Request Format:**\n   - The API request must be formatted as per the standards, including the necessary key-value pair for keywords.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]

# Example 2
# Question: {q2}
# Inference: To generate the API request based on the user's utterance, let\u2019s analyze the given information step by step:\n\n1. **User\u2019s Utterance**: The user wants to find a cardiologist in Los Angeles for a check-up appointment. From this, we can extract important details:\n   - **Specialty**: cardiologist\n   - **Location**: Los Angeles\n\n2. **Identify Relevant API**: The task is to find a healthcare provider, specifically focusing on availability for a cardiologist in a specific location.\n\n3. **Available API**: We have the `HealthcareProviderAppointmentChecker` API, which checks the availability of appointments with healthcare providers. It requires:\n   - **specialty**: The specialty of the healthcare provider (in this case, cardiologist).\n   - **location**: The city location where the user is searching (in this case, Los Angeles).\n\n4. **Construct the API Request**: Based on the extracted details and the requirements of the API, we can construct the API request as follows:\n   - We will input the specialty as 'cardiologist' and the location as 'Los Angeles'.\n\nPutting all the pieces together:\n\nFinal Answer: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]

# Example 3
# Question: {q3}
# Inference: To address the user\u2019s request, we first identify the main objective from the user's utterance. The user is looking to find a cardiologist in Los Angeles for a check-up appointment. We need to consider that the action required is to search for relevant tools related to scheduling appointments with healthcare providers, specifically within the context of accessing cardiologists in the specified location.\n\n1. **Identify Keywords**: The primary keywords from the user\u2019s request would be \u201ccardiologist,\u201d \u201cLos Angeles,\u201d and \u201cappointment.\u201d\n  \n2. **Determine the Relevant API**: The given API descriptions suggest using the `ToolSearcher` API to find tools that pertain to searching for healthcare providers for appointments. The appropriate keyword that captures the essence of the user's request is \u201chealthcare provider appointment scheduler\u201d due to the focus on scheduling appointments with specific providers.\n\n3. **Formulate the API Request**: Based on the identified keywords and the relevant API, we will construct the `ToolSearcher` request with the appropriate keyword.\n\nThus, the final API request to handle the user\u2019s request about finding a cardiologist in Los Angeles for a check-up appointment can be formulated as follows:\n\nFinal Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]

# Now please solve the following question using the same inference style and format as the example above. 
# Question: """
    
#openai_human_written_examples.json
    api_bank_human_written_examples_style_prompt = \
f"""We offer you 4 examples to show how to generate an API request based on the user's utterance and available API requests. Please follow the same format to solve the question below. Remember You only need to generate the next api call

Example 1
Question: {q1}
Inference: The user is looking for a dentist in London who can specifically perform root canal treatment. To help find the right tool for this query, we will use the ToolSearcher API, which searches for relevant tools based on keywords. The appropriate keywords to encapsulate the user's request include 'dentist', 'root canal treatment', and 'London'. Therefore, the next API request will be structured with these keywords to ensure we can effectively find the right dental services.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='dentist root canal treatment London')]

Example 2
Question: {q2}
Inference: The user is requesting to find a dentist in London who can perform a root canal treatment. The appropriate next step is to use the FindDentist API, as it is designed to find dentists based on their location and the specific service required. The input parameters for this API will be filled with the user\u2019s specified location (London) and the service they require (root canal treatment). Therefore, the next API request should be:\n\nFinal Answer: API-Request: [FindDentist(location=London, service=root canal treatment)]

Example 3
Question: {q3}
Inference: The user is looking for a dentist in London who can perform a root canal treatment. They would need to check if there are available dentists for this specific service. In the previous API requests, the ToolSearcher API was used to find relevant tools based on keywords. For the next step, we should query the availability of dentists specifically for the service requested. The best approach here is to search for tools related to 'dentist availability'.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='dentist availability')]

Example 4
Question: {q4}
Inference: The user is seeking to find a dentist who can perform a root canal treatment in London. After searching for relevant tools using the ToolSearcher API, it was established that the FindDentist API can locate dentists based on the location and required service. A specific dentist, Dr. John Smith, was identified for the root canal treatment. The next logical step is to check Dr. John Smith's availability for an appointment on the requested date, which is October 15, 2023, using the DentistAvailability API. \n\nFinal Answer: API-Request: [DentistAvailability(dentist=Dr. John Smith, date=2023-10-15)]

Now please solve the following question using the same inference style and format as the example above. 
Question: """
# f"""
# We offer you 3 examples to show how to generate an API request based on the user's utterance and available API requests. Please follow the same format to solve the question below. Remember You only need to generate the next api call

# Example 1
# Question: {q1}
# Inference: The user is requesting to find a healthcare provider (specifically a cardiologist) in Los Angeles for a check-up appointment. The first step should be to search for a tool that can help find a healthcare provider appointment availability checker. To accomplish this, we choose the ToolSearcher API from the available APIs. The ToolSearcher API is used to search for relevant tools based on the provided keywords according to the description. We need to fill out the keywords according to the description. The keywords could be 'healthcare provider appointment availability checker.' Therefore, the next step (which is also the first step) is:

# Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment availability checker')]


# Example 2
# Question: {q2}
# Inference: According to the API call history, the user has called the ToolSearcher API and found the HealthcareProviderAppointmentChecker API. The next step is to fill out the input parameters for HealthcareProviderAppointmentChecker and use it to find healthcare provider appointment availability. The input parameters are specialty and location. The user wants to find a cardiologist in Los Angeles for a check-up appointment. Therefore, the next API request should be:

# Final Answer: API-Request: [HealthcareProviderAppointmentChecker(specialty='cardiologist', location='Los Angeles')]


# Example 3
# Question: {q3}
# Inference: The user previously called the HealthcareProviderAppointmentChecker API and found three appointment times, which are '2034-04-18 14:30:00', '2034-04-19 11:00:00', and '2034-04-20 09:45:00'. The next step is to find the scheduler for the appointment. Since there is no available tool, the user needs to search for a tool that can schedule healthcare provider appointments. The ToolSearcher API can be used to search for relevant tools based on the keywords according to the description. The keywords should be 'healthcare provider appointment scheduler'. Therefore, the answer is:

# Final Answer: API-Request: [ToolSearcher(keywords='healthcare provider appointment scheduler')]


# Now please solve the following question using the same inference style and format as the example above. 
# Question: """




    api_bank_anthropic_style_prompt = \
f"""We offer you 4 examples to show how to generate an API request based on the user's utterance and available API requests. Please follow the same format to solve the question below. Remember You only need to generate the next api call

Example 1
Question: {q1}
Inference: To find a dentist in London who can perform a root canal treatment, we need to use the ToolSearcher API. This API allows us to search for relevant tools based on keywords. In this case, our keywords should include the type of professional we're looking for (dentist), the specific treatment needed (root canal), and the location (London).\n\nThe ToolSearcher API takes a single parameter, \"keywords\", which should be a string containing the search terms. We'll combine the relevant keywords into a single string to perform the search.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='dentist root canal treatment London')]

Example 2
Question: {q2}
Inference: Based on the user's request to find a dentist in London for a root canal treatment, we need to use the FindDentist API. This API requires two parameters: location and service.\n\nFor the location parameter, we can use \"London\" as specified in the user's request.\n\nFor the service parameter, we can use \"root canal treatment\" which is the specific dental procedure mentioned by the user.\n\nCombining these parameters, we can construct the API request as follows:\n\nFinal Answer: API-Request: [FindDentist(location='London', service='root canal treatment')]

Example 3
Question: {q3}
Inference: Based on the user's request to find a dentist for a root canal treatment in London, and the subsequent FindDentist API call that returned Dr. John Smith, the next logical step would be to check the availability of this dentist. The user would likely want to know when they can schedule an appointment for their root canal treatment.\n\nTo accomplish this, we need to search for a tool that can help us check the dentist's availability. The ToolSearcher API is suitable for this purpose, as it can search for relevant tools based on keywords.\n\nThe most appropriate keywords for this search would be \"dentist availability\" as we're looking for a tool that can check the availability of the dentist we found.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='dentist availability')]

Example 4
Question: {q4}
Inference: Based on the previous API requests and responses, we have identified a suitable dentist, Dr. John Smith, who can perform a root canal treatment in London. The next logical step would be to check the availability of this dentist for an appointment.\n\nThe DentistAvailability API is designed for this purpose, requiring the dentist's name and the date for checking availability. We have the dentist's name from the previous API response, and we can use the current date provided in the context for checking availability.\n\nFinal Answer: API-Request: [DentistAvailability(dentist='Dr. John Smith', date='2023-10-15')]

Now please solve the following question using the same inference style and format as the example above. 
Question: """
    
    api_bank_mini_gpt4_prompt = \
f"""We offer you 4 examples to show how to generate an API request based on the user's utterance and available API requests. Please follow the same format to solve the question below. Remember You only need to generate the next api call

Example 1
Question: {q1}
Inference: To construct the API request based on the user's utterance, we need to extract the relevant keywords from the given statement. The user is looking for a dentist in London specifically for root canal treatment. Therefore, the keywords that encapsulate this search can be summarized as \"dentist root canal treatment London.\"\n\nNow, we can format this information into the required API request structure.\n\nFinal Answer: API-Request: [ToolSearcher(keywords='dentist root canal treatment London')]

Example 2
Question: {q2}
Inference: To address the user's request for finding a dentist in London who can perform a root canal treatment, we will utilize the API designed for finding dentists based on location and the services they offer. The necessary keywords from the user's utterance include location (\"London\") and the specific dental service (\"root canal treatment\"). This allows us to format the API request correctly.\n\nFinal Answer: API-Request: [FindDentist(location=London, service=root canal treatment)]

Example 3
Question: {q3}
Inference: To find out the availability of dentists, we need to seek information specifically about their schedule or any appointments they may have. The request for availability aligns with a general inquiry about dentists in our earlier interactions, making \"dentist availability\" a relevant keyword to search with. \n\nFinal Answer: API-Request: [ToolSearcher(keywords='dentist availability')]

Example 4
Question: {q4}
Inference: To fulfill the user's request of finding a dentist in London who can perform a root canal treatment and check the dentist's availability on the specified date, we first identify the dentistry service and location from the user's query. \n\nBased on the previous API requests, we already deduced the dentist's name as \"Dr. John Smith\" from the `FindDentist` API output. The date provided for checking availability is \"2023-10-15\".\n\nNow, I will create the API request to check the availability of Dr. John Smith for the specified date.\n\nFinal Answer: API-Request: [DentistAvailability(dentist=Dr. John Smith, date=2023-10-15)]

Now please solve the following question using the same inference style and format as the example above. 
Question: """


# Please inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'\n
    q1 = \
"""Given the context: Persian (/\u02c8p\u025c\u02d0r\u0292\u0259n, -\u0283\u0259n/), also known by its endonym Farsi (\u0641\u0627\u0631\u0633\u06cc f\u0101rsi (f\u0252\u02d0\u027e\u02c8si\u02d0) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.\n\ndo iran and afghanistan speak the same language?"""
    q2 = \
"""Given the context: Good Samaritan laws offer legal protection to people who give reasonable assistance to those who are, or who they believe to be, injured, ill, in peril, or otherwise incapacitated. The protection is intended to reduce bystanders' hesitation to assist, for fear of being sued or prosecuted for unintentional injury or wrongful death. An example of such a law in common-law areas of Canada: a good Samaritan doctrine is a legal principle that prevents a rescuer who has voluntarily helped a victim in distress from being successfully sued for wrongdoing. Its purpose is to keep people from being reluctant to help a stranger in need for fear of legal repercussions should they make some mistake in treatment. By contrast, a duty to rescue law requires people to offer assistance and holds those who fail to do so liable.\n\ndo good samaritan laws protect those who help at an accident?"""
    q3 = \
"""Given the context: Epsom railway station serves the town of Epsom in Surrey. It is located off Waterloo Road and is less than two minutes' walk from the High Street. It is not in the London Oyster card zone unlike Epsom Downs or Tattenham Corner stations. The station building was replaced in 2012/2013 with a new building with apartments above the station (see end of article).\n\ncan you use oyster card at epsom station?"""
    
    q4 = \
"""Given the context: An ABA routing transit number (ABA RTN) is a nine-digit code, used in the United States, which appears on the bottom of negotiable instruments such as checks to identify the financial institution on which it was drawn. The ABA RTN was originally designed to facilitate the sorting, bundling, and shipment of paper checks back to the drawer's (check writer's) account. As new payment methods were developed (ACH and Wire), the system was expanded to accommodate these payment methods.\n\nare bank transit number and routing number the same?\n\nPlease inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'\n"""

    q5 = \
"""Given the context: The Browns entered this game looking to win consecutive games for the first time since they won three in a row during Weeks 8 to 10 of the 2014 season. The Steelers entered the game locked into the AFC's #3 seed and rested most of their starters. However, the Browns would still fall to the Steelers, losing 27--24 in an overtime affair. With the loss, the Browns finished the season with a 1--15 record, their worst record in franchise history. The Browns also became the tenth team in NFL history, but the first since the 2009 Rams, to finish 1--15. They failed to win a division game for the first time since the 2011 season and extended their losing streak within the division to 11 games. The Browns also failed to win a road game in a season for only the second time in franchise history and the first since the 1975 season. They also extended their road losing streak to 13 games and failed to win a road game against the Steelers for the 13th consecutive season. The Browns finished with the worst record in the NFL and secured the #1 overall pick in the 2017 NFL Draft. This marked the beginning of a 17 game losing streak continuing through the 2017 season and ending with a tie in Week 1 of the 2018 season.\n\ndid the cleveland browns win any games last season?\n\nPlease inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'\n"""

    q6 = \
"""Given the context: On May 10, 2017, The CW renewed the series for a fifth season. On July 20, 2017, it was announced by series creator Julie Plec ahead of Comic Con that the series' fifth season would be its last. The final season debuted on April 18, 2018.\n\nis there a season 7 of the originals?\n\nPlease inference first, then provide the final answer (True/False) at the end, after 'Final Answer:'\n"""

    
#     boolq_gpt4_style_prompt = \
# f"""
# We have 3 examples below

# Example 1
# {q1}
# Inference: Iran and Afghanistan both speak Persian, also known as Farsi in Iran and Dari in Afghanistan. Despite the different names, the languages are mutually intelligible and are considered dialects of the same language, stemming from the Western Iranian languages group within the Indo-European language family.\n\nFinal Answer: True

# Example 2
# Question: {q2}
# Inference: Good Samaritan laws are designed to provide legal protection to individuals who assist others who are in danger or distress. These laws are in place to encourage bystanders to help those in need by offering reasonable assistance without the fear of legal repercussions. In the event of an accident, if a bystander intervenes to help the injured or ill person, Good Samaritan laws would typically protect that individual from being sued for unintentional harm caused while rendering aid, provided the assistance given was reasonable and given without gross negligence or willful misconduct.\n\nFinal Answer: True

# Example 3
# Question: {q3}
# Inference: Although Epsom station is mentioned in the context, it is specifically stated that it is not in the London Oyster card zone, unlike neighboring stations. Thus, Oyster cards cannot be used at Epsom station for payment.\n\nFinal Answer: False

# Now please solve the following question using the same inference style and format as the example above. 
# Question: """

    boolq_gpt4_style_prompt = \
f"""
We have 6 examples below

Example 1
{q1}
Inference: Iran and Afghanistan both speak Persian, also known as Farsi in Iran and Dari in Afghanistan. Despite the different names, the languages are mutually intelligible and are considered dialects of the same language, stemming from the Western Iranian languages group within the Indo-European language family.\n\nFinal Answer: True

Example 2
Question: {q2}
Inference: Good Samaritan laws are designed to provide legal protection to individuals who assist others who are in danger or distress. These laws are in place to encourage bystanders to help those in need by offering reasonable assistance without the fear of legal repercussions. In the event of an accident, if a bystander intervenes to help the injured or ill person, Good Samaritan laws would typically protect that individual from being sued for unintentional harm caused while rendering aid, provided the assistance given was reasonable and given without gross negligence or willful misconduct.\n\nFinal Answer: True

Example 3
Question: {q3}
Inference: Although Epsom station is mentioned in the context, it is specifically stated that it is not in the London Oyster card zone, unlike neighboring stations. Thus, Oyster cards cannot be used at Epsom station for payment.\n\nFinal Answer: False

Example 4
Question: {q4}
Inference: In the context of United States banking, the term \"bank transit number\" is often used interchangeably with \"routing number\" or \"ABA routing transit number.\" Both terms refer to the same nine-digit code used to identify the financial institution associated with a bank account. This code is essential for directing funds to and from bank accounts for activities like check processing, direct deposit, automatic bill payments, and electronic fund transfers such as ACH and wire transfers.\n\nFinal Answer: True

Example 5
Question: {q5}
Inference: The given context indicates that the Cleveland Browns finished the season with a 1--15 record, suggesting they only won a single game in that particular season. Based on that information, the Browns did not win any games during the last season described (the one in which they finished with a 1--15 record).\n\nFinal Answer: False

Example 6
Question: {q6}
Inference: The information provided states that \"The Originals\" was renewed for a fifth season on May 10, 2017, and on July 20, 2017, it was announced by the series creator Julie Plec that the fifth season would be its last. The final season, which is the fifth season, debuted on April 18, 2018. Based on this information, it is clear that the series concluded with its fifth season and therefore there would not be a season 7.\n\nFinal Answer: False

Now please solve the following question using the same inference style and format as the example above. 
Question: """


#     boolq_gt_style_prompt = \
# F"""
# We have 3 examples below

# Example 1
# Question: {q1}
# Final Answer: True

# Example 2
# Question: {q2}
# Final Answer: True

# Example 3
# Question: {q3}
# Final Answer: False

# Now please solve the following question using the same style and format as the example above. 
# Question: """

    boolq_gt_style_prompt = \
F"""
We have 6 examples below

Example 1
Question: {q1}
Final Answer: True

Example 2
Question: {q2}
Final Answer: True

Example 3
Question: {q3}
Final Answer: False

Example 4
Question: {q4}
Final Answer: True

Example 5
Question: {q5}
Final Answer: False

Example 6
Question: {q6}
Final Answer: False

Now please solve the following question using the same style and format as the example above. 
Question: """



#     boolq_minimum_change_style_prompt = \
# f"""
# We have 3 examples below

# Example 1
# Question: {q1}
# Inference: Persian, also known as Farsi, is the predominant language in Iran. Dari is the name used for the Persian language in Afghanistan since 1958. The two are dialects of the same language, and while there are some differences between them, they are mutually intelligible. This means that speakers of both can generally understand each other without too much effort. There may be variations in vocabulary, pronunciation, and grammar, but Iranians and Afghans essentially speak the same language, Persian.\n\nFinal Answer: True

# Example 2
# Question: {q2}
# Inference: Based on the context provided, Good Samaritan laws do protect individuals who provide assistance at the scene of an accident, as long as their actions are considered reasonable.\n\nFinal Answer: True.

# Example 3
# Question: {q3}
# Inference: Although Epsom station is mentioned in the context, it is specifically stated that it is not in the London Oyster card zone, unlike neighboring stations. Thus, Oyster cards cannot be used at Epsom station for payment.\n\nFinal Answer: False


# Now please solve the following question using the same inference style and format as the example above. 
# Question: """
    
    boolq_minimum_change_style_prompt = \
f"""
We have 6 examples below

Example 1
Question: {q1}
Inference: Persian, also known as Farsi, is the predominant language in Iran. Dari is the name used for the Persian language in Afghanistan since 1958. The two are dialects of the same language, and while there are some differences between them, they are mutually intelligible. This means that speakers of both can generally understand each other without too much effort. There may be variations in vocabulary, pronunciation, and grammar, but Iranians and Afghans essentially speak the same language, Persian.\n\nFinal Answer: True

Example 2
Question: {q2}
Inference: Based on the context provided, Good Samaritan laws do protect individuals who provide assistance at the scene of an accident, as long as their actions are considered reasonable.\n\nFinal Answer: True.

Example 3
Question: {q3}
Inference: Although Epsom station is mentioned in the context, it is specifically stated that it is not in the London Oyster card zone, unlike neighboring stations. Thus, Oyster cards cannot be used at Epsom station for payment.\n\nFinal Answer: False

Example 4
Question: {q4}
Inference: Based on the context provided, the terms \"bank transit number\" and \"routing number\" refer to the same nine-digit code used to identify the financial institution on which a check was drawn. Therefore, the two terms are synonymous.\n\nFinal Answer: True

Example 5
Question: {q5}
Inference: The Cleveland Browns lost all of their games in the 2016 season, except for one against the San Diego Chargers. Therefore, they won only one game during that season. \n\nFinal Answer: False.

Example 6
Question: {q6}
Inference:  Based on the information provided, there is no indication of a seventh season for \"The Originals.\" The series was renewed for a fifth season in May 2017, and it was later announced in July 2017 by series creator Julie Plec that the fifth season would be its last. The final season debuted on April 18, 2018.\n\nFinal Answer: False.

Now please solve the following question using the same inference style and format as the example above. 
Question: """
    
#     boolq_rewrite_in_gpt4_style_prompt = \
# f"""
# We have 3 examples below

# Example 1
# Question: {q1}
# Inference: Persian, also known as Farsi, is a language that is spoken in Iran, Afghanistan, and Tajikistan. In Iran, it is referred to as Farsi, while in Afghanistan it is called Dari. Dari is a variety of Persian that is spoken in Afghanistan. Although Dari and Farsi are essentially the same language, they have some differences in pronunciation, vocabulary, and usage. Tajikistan also uses a variant of Persian known as Tajiki, which is written in the Cyrillic script.

# Final Answer: True

# Example 2
# Question: {q2}
# Inference: Based on the context, Good Samaritan laws provide legal protection to individuals who offer reasonable assistance to those who are injured, ill, or in distress. These laws are designed to encourage people to help others without fear of legal repercussions for unintended mistakes made during their assistance.

# Final Answer: True

# Example 3
# Question: {q3}
# Inference: Based on the context provided, Epsom railway station is not in the London Oyster card zone, while nearby stations such as Epsom Downs and Tattenham Corner are within this zone. Therefore, it is implied that Oyster cards cannot be used at Epsom railway station.

# Final Answer: False

# Now please solve the following question using the same inference style and format as the example above. 
# Question: """
    
#     boolq_mini_gpt4_style_prompt = \
# f"""
# We have 3 examples below

# Example 1
# Question: {q1}
# Inference: \n\nPersian, also referred to as Farsi, is indeed spoken in both Iran and Afghanistan. In Iran, it is commonly called Farsi, while in Afghanistan, the Persian language is recognized as Dari. Although Dari and Farsi are distinct in terms of pronunciation, vocabulary, and some aspects of usage, they are fundamentally variants of the same language, a Western Iranian language. This classification means that while there are differences in the way the language is spoken in each country, both countries are primarily speaking the same linguistic root.\n\nFinal Answer: True

# Example 2
# Question: {q2}
# Inference: \n\nGood Samaritan laws are specifically designed to offer legal protection to individuals who provide reasonable assistance to those in distress, such as those injured in an accident. The primary objective of these laws is to encourage bystanders to help victims without the fear of legal repercussions, should anything go wrong during their attempt to assist. Since the context mentions that Good Samaritan laws protect those who help individuals they believe to be incapacitated or injured, it directly supports the idea that such laws apply to those who assist at the scene of an accident.\n\nFinal Answer: True

# Example 3
# Question: {q3}
# Inference: \n\nEpsom railway station is located outside of the London Oyster card zone. This means that travelers using an Oyster card cannot use it at Epsom station. The context also mentions that Epsom Downs and Tattenham Corner stations are included in the Oyster card zone, but Epsom station is explicitly stated to be excluded from it.\n\nFinal Answer: False

# Now please solve the following question using the same inference style and format as the example above. 
# Question: """
    
    boolq_mini_gpt4_style_prompt = \
f"""
We have 6 examples below

Example 1
Question: {q1}
Inference: \n\nPersian, also referred to as Farsi, is indeed spoken in both Iran and Afghanistan. In Iran, it is commonly called Farsi, while in Afghanistan, the Persian language is recognized as Dari. Although Dari and Farsi are distinct in terms of pronunciation, vocabulary, and some aspects of usage, they are fundamentally variants of the same language, a Western Iranian language. This classification means that while there are differences in the way the language is spoken in each country, both countries are primarily speaking the same linguistic root.\n\nFinal Answer: True

Example 2
Question: {q2}
Inference: \n\nGood Samaritan laws are specifically designed to offer legal protection to individuals who provide reasonable assistance to those in distress, such as those injured in an accident. The primary objective of these laws is to encourage bystanders to help victims without the fear of legal repercussions, should anything go wrong during their attempt to assist. Since the context mentions that Good Samaritan laws protect those who help individuals they believe to be incapacitated or injured, it directly supports the idea that such laws apply to those who assist at the scene of an accident.\n\nFinal Answer: True

Example 3
Question: {q3}
Inference: \n\nEpsom railway station is located outside of the London Oyster card zone. This means that travelers using an Oyster card cannot use it at Epsom station. The context also mentions that Epsom Downs and Tattenham Corner stations are included in the Oyster card zone, but Epsom station is explicitly stated to be excluded from it.\n\nFinal Answer: False

Example 4
Question: {q4}
Inference:\n\nA bank transit number and a routing number refer to the same concept. The nine-digit ABA routing transit number (RTN) is a unique identifier used in the United States to designate a specific financial institution and appears on negotiable instruments like checks. It serves the function of facilitating the processing of these financial transactions, and it is commonly known as both a routing number and a transit number. Thus, they are interchangeable terms that denote the same numbering system used for identifying banks.\n\nFinal Answer: True

Example 5
Question: {q5}
Inference:\n\nThe context provided indicates that the Cleveland Browns finished the season with a 1\u201315 record, which means they only won one game throughout the entire season. Since the question asks if they won any games during the last season, the answer would logically be based on the record mentioned. With a single win in a sea of 15 losses, it is clear that the Browns did not achieve significant success that season.\n\nFinal Answer: False

Example 6
Question: {q6}
Inference:\n\nThe context provided indicates that The CW renewed the series \"The Originals\" for a fifth season, which was declared as the last season by series creator Julie Plec. Since the final season aired on April 18, 2018, it confirms that the series concluded after its fifth season. Therefore, there cannot be a seventh season as the show has officially ended after five seasons.\n\nFinal Answer: False

Now please solve the following question using the same inference style and format as the example above. 
Question: """
    

    boolq_anthropic_style_prompt = \
f"""
We have 6 examples below

Example 1
Question: {q1}
Inference: Based on the given context, Iran and Afghanistan do speak closely related languages, but they are not exactly the same. The text states that Persian, also known as Farsi, is primarily spoken in Iran. In Afghanistan, the language is officially known as Dari since 1958. While both Farsi and Dari are considered Western Iranian languages and part of the Persian language family, they are recognized as distinct official languages in their respective countries.\n\nThe fact that they are both described as Persian languages suggests a high degree of mutual intelligibility and shared linguistic heritage. However, the different official names (Farsi in Iran and Dari in Afghanistan) imply some level of distinction, whether it be in dialect, vocabulary, or official status.\n\nIt's important to note that languages often exist on a continuum, and the boundaries between closely related languages can be blurry. In this case, while the core language is essentially the same, there may be regional variations and differences in standardization that lead to their classification as separate official languages in Iran and Afghanistan.\n\nFINAL ANSWER: True

Example 2
Question: {q2}
Inference: Good Samaritan laws are specifically designed to protect individuals who provide assistance in emergency situations, including accidents. The context provided states that these laws offer legal protection to people who give reasonable assistance to those who are injured, ill, in peril, or otherwise incapacitated. This would certainly include helping at the scene of an accident.\n\nThe laws aim to reduce bystanders' hesitation to assist by protecting them from potential lawsuits or prosecution for unintentional injury or wrongful death. This encourages people to help others in need without fear of legal repercussions.\n\nThe example given for Canada further supports this, mentioning that the good Samaritan doctrine prevents a rescuer who has voluntarily helped a victim in distress from being successfully sued for wrongdoing. This would apply to accident scenarios where victims are likely to be in distress and in need of immediate assistance.\n\nGiven this information, it is reasonable to conclude that Good Samaritan laws do indeed protect those who help at an accident, as long as their assistance is reasonable and well-intentioned.\n\nFinal Answer: True

Example 3
Question: {q3}
Inference: Based on the given context, Epsom railway station is explicitly stated to not be in the London Oyster card zone. The context mentions that \"It is not in the London Oyster card zone unlike Epsom Downs or Tattenham Corner stations.\" This directly indicates that Oyster cards cannot be used at Epsom station.\n\nAdditionally, the context provides information about other nearby stations (Epsom Downs and Tattenham Corner) that are in the Oyster card zone, further emphasizing that Epsom station itself is not included in this system.\n\nGiven this information, we can conclude that Oyster cards are not accepted at Epsom railway station.\n\nFINAL ANSWER: False

Example 4
Question: {q4}
Inference: The context provides information about ABA routing transit numbers (ABA RTN), which are nine-digit codes used in the United States to identify financial institutions on negotiable instruments like checks. The term \"transit number\" is not explicitly mentioned in the given context, but we can infer that it likely refers to the same concept as the routing number.\n\nBoth \"bank transit number\" and \"routing number\" seem to serve the same purpose: identifying the financial institution associated with a particular transaction or account. The context mentions that the ABA RTN was originally designed for sorting and processing paper checks, but has since been expanded to accommodate other payment methods like ACH and Wire transfers.\n\nGiven that the terms appear to refer to the same nine-digit code used for identifying banks and facilitating transactions, it is reasonable to conclude that \"bank transit number\" and \"routing number\" are indeed the same thing, just different names for the same concept.\n\nFINAL ANSWER: True

Example 5
Question: {q5}
Inference: The context provides information about the Cleveland Browns\' performance in a specific season, mentioning that they finished with a 1-15 record. This indicates that they did win one game during that season. However, the question asks about "last season," which is not explicitly defined in the given context.\n\nThe context also mentions that this poor performance "marked the beginning of a 17 game losing streak continuing through the 2017 season and ending with a tie in Week 1 of the 2018 season." This suggests that the Browns did not win any games in the season following the 1-15 record, which would be the 2017 season.\n\nGiven this information, we can infer that if "last season" refers to the season immediately following the 1-15 record (likely the 2017 season), then the Browns did not win any games.\n\nFinal Answer: False

Example 6
Question: {q6}
Inference: Based on the given context, there is no indication of a seventh season of \"The Originals.\" The information provided states that the fifth season was announced as the final season by the series creator Julie Plec. The fifth and final season debuted on April 18, 2018. Since the show's conclusion was explicitly announced and the last season aired, it is highly unlikely that there would be a seventh season.\n\nFINAL ANSWER: False

Now please solve the following question using the same inference style and format as the example above. 
Question: """
    
    boolq_mini_gpt4_prompt = \
f"""
We have 6 examples below

Example 1
Question: {q1}
Inference: Persian, known as Farsi in Iran, is the predominant language spoken in both Iran and Afghanistan. In Afghanistan, the official variant of Persian is referred to as Dari. Both languages share a common linguistic heritage and are mutually intelligible, but they have some regional differences in pronunciation, vocabulary, and usage. Therefore, while they are different in terms of dialects, they are essentially the same language family.\n\nFinal Answer: True

Example 2
Question: {q2}
Inference: Good Samaritan laws are designed to protect individuals who voluntarily assist those in distress, such as victims of accidents, by preventing them from being sued or facing legal repercussions for their actions, even if those actions unintentionally cause harm. This legal protection is specifically intended to encourage bystanders to help without the fear of liability, promoting a culture of assistance in emergencies. Therefore, yes, Good Samaritan laws do protect those who help at an accident.\n\nFinal Answer: True

Example 3
Question: {q3}
Inference: The context indicates that Epsom railway station is not located within the London Oyster card zone, which means that Oyster cards cannot be used for travel at this station. This implies that travelers would need to use other forms of payment. Therefore, it can be concluded that Oyster cards are not accepted at Epsom station.\n\nFinal Answer: False

Example 4
Question: {q4}
Inference: The terms \"bank transit number\" and \"routing number\" refer to the same concept in the context of the U.S. banking system. The ABA routing transit number (ABA RTN) serves to identify financial institutions and facilitate transactions such as check processing, ACH, and wire transfers. Therefore, a bank transit number is another way to refer to a routing number.\n\nFinal Answer: True

Example 5
Question: {q5}
Inference: The context states that the Browns finished the season with a 1\u201315 record, which indicates they won only one game throughout the entire season. This also emphasizes that they failed to win any division games and did not secure any road victories. Therefore, based on this information, the Cleveland Browns did not win any games last season beyond that one victory.\n\nFinal Answer: False

Example 6
Question: {q6}
Inference: Based on the provided context, the series \"The Originals\" was renewed for a fifth season, which was confirmed to be its last season. Since it was explicitly stated that the fifth season would be the final season and it premiered on April 18, 2018, it indicates that no further seasons, including a seventh season, were produced after that.\n\nFinal Answer: False

Now please solve the following question using the same inference style and format as the example above. 
Question: """
    


    prompt = ''
    if enforce_prompt_style != '':
        if enforce_prompt_style == 'gsm8k_gt_style':
            prompt = gsm8k_gt_style_prompt
        if enforce_prompt_style == 'gsm8k_gpt4':
            prompt = gsm8k_gpt4_style_prompt
        if enforce_prompt_style == 'gsm8k_in_own_words':
            prompt = gsm8k_rewrite_in_own_words_style_prompt
    
        if enforce_prompt_style == 'math_algebra_gt_style':
            prompt = math_gt_style_prompt
        if enforce_prompt_style == 'math_algebra_gpt4':
            prompt = math_gpt4_style_prompt
        if enforce_prompt_style == 'math_algebra_in_own_words':
            prompt = math_rewrite_in_own_words_style_prompt

        if enforce_prompt_style == 'ecqa_gt_style':
            prompt = ecqa_gt_style_prompt
        if enforce_prompt_style == 'ecqa_gpt4':
            prompt = ecqa_gpt4_style_prompt
        if enforce_prompt_style == 'ecqa_in_own_words':
            prompt = ecqa_rewrite_in_own_words_style_prompt

    if  'gsm8k' in task:
        if enforce_prompt_style == '':
            if prompt_style == 'gt_style':
                prompt = gsm8k_gt_style_prompt
            if prompt_style == 'gpt4':
                prompt = gsm8k_gpt4_style_prompt
            if prompt_style == 'in_own_words':
                prompt = gsm8k_rewrite_in_own_words_style_prompt
            if prompt_style == 'minimum_change':
                prompt = gsm8k_minimum_change_style_prompt
            if prompt_style == 'mini_gpt4':
                prompt = gsm8k_mini_gpt4_prompt
            if prompt_style == 'anthropic':
                prompt = gsm8k_anthropic_style_prompt

            if 'generated_xy_pairs' in task:
                prompt = gsm8k_xy_pairs_prompt
            
                
            
#         format_prompt = \
# """Please put the final digital answer at the end after you finish inference in this format Final Answer: final neumerical answer

# Format:
# SOME_INFERENCE

# Final Answer: """
#         formated_question = question.replace(format_prompt, '')

        formated_question = prompt + original_question#formated_question
        formated_question = front_prompt_math + formated_question

    if task == 'math_algebra':
        if enforce_prompt_style == '':
            if prompt_style == 'gt_style':
                prompt = math_gt_style_prompt
            if prompt_style == 'gpt4':
                prompt = math_gpt4_style_prompt
            if prompt_style == 'in_own_words':
                prompt = math_rewrite_in_own_words_style_prompt
            if prompt_style == 'minimum_change':
                prompt = math_minimum_change_style_prompt
            if prompt_style == 'mini_gpt4':
                prompt = math_mini_gpt4_prompt
            if prompt_style == 'anthropic':
                prompt = math_anthropic_style_prompt

#         format_prompt = \
# """

# Please put the final digital answer at the end after you finish inference in this format FINAL ANSWER: final neumerical answer

# Format:
# SOME_INFERENCE

# FINAL ANSWER: """

#         formated_question = question.replace("Please provide the final answer (a number) at the end, after 'Final Answer:'", '')
#         formated_question = formated_question.replace(format_prompt, '')

        formated_question = prompt + original_question#formated_question
        formated_question = front_prompt_math + formated_question


    if task == 'ecqa':
        if enforce_prompt_style == '':
            if prompt_style == 'gt_style':
                prompt = ecqa_gt_style_prompt
            if prompt_style == 'gpt4':
                prompt = ecqa_gpt4_style_prompt
            if prompt_style == 'in_own_words':
                prompt = ecqa_rewrite_in_own_words_style_prompt
            if prompt_style == 'rewrite_in_gpt4_style':
                prompt = ecqa_rewrite_in_gpt4_style_prompt
            if prompt_style == 'minimum_change':
                prompt = ecqa_rewrite_minimum_change_style_prompt
            if prompt_style == 'mini_gpt4':
                prompt = ecqa_mini_gpt4_prompt
            if prompt_style == 'anthropic':
                prompt = ecqa_anthropic_style_prompt
            
#         format_prompt = ""
# # """\n\nPlease inference first, then provide the final answer (1/2/3/4/5) at the end, after 'Final Answer:'\n\nFormat:\nInference: INFERENCE_HERE\nFinal Answer: (1/2/3/4/5)_HERE"""
#         formated_question = question.replace(format_prompt, '')
        formated_question = prompt + original_question#formated_question
        formated_question = front_prompt_ecqa + formated_question
        
        
    if task == 'api_bank':
        if enforce_prompt_style == '':
            if prompt_style == 'step_by_step':
                prompt = api_bank_step_by_step_style_prompt
            if prompt_style == 'in_own_words':
                prompt = api_bank_provide_gpt4_style_example_prompt
            if prompt_style == 'gt_style':
                prompt = api_bank_gt_style_prompt
            if prompt_style == 'gpt4':
                prompt = api_bank_gpt4_style_prompt
            if prompt_style == 'human_written_examples':
                prompt = api_bank_human_written_examples_style_prompt
            if prompt_style == 'anthropic':
                prompt = api_bank_anthropic_style_prompt
            if prompt_style == 'mini_gpt4':
                prompt = api_bank_mini_gpt4_prompt



#         format_prompt = \
# """

# Please put the final digital answer at the end after you finish inference in this format FINAL ANSWER: final neumerical answer

# Format:
# SOME_INFERENCE

# FINAL ANSWER: """

#         formated_question = question.replace("Please provide the final answer (API-REQUEST) at the end, after 'Final Answer:'", '')
#         formated_question = formated_question.replace(format_prompt, '')

        formated_question = prompt + original_question#formated_question
        if prompt_style == 'gt_style':
            formated_question = front_gt_prompt_api_bank + formated_question
        else:
            formated_question = front_prompt_api_bank + formated_question

    if task == 'boolq':
        if enforce_prompt_style == '':
            if prompt_style == 'minimum_change':
                prompt = boolq_minimum_change_style_prompt
            if prompt_style == 'gt_style':
                prompt = boolq_gt_style_prompt
            if prompt_style == 'gpt4':
                prompt = boolq_gpt4_style_prompt
            # if prompt_style == 'rewrite_in_gpt4_style':
            #     prompt = boolq_rewrite_in_gpt4_style_prompt
            if prompt_style == 'mini_gpt4_style':
                prompt = boolq_mini_gpt4_style_prompt
            if prompt_style == 'anthropic':
                prompt = boolq_anthropic_style_prompt
            if prompt_style == 'mini_gpt4':
                prompt = boolq_mini_gpt4_prompt

            

        # formated_question = question
        # if prompt_style == 'gt_style':
        #     formated_question = formated_question.replace('Please inference first, then provide the final answer (True/False) at the end', 'Please provide the final answer (True/False) at the end')
        #     prompt = prompt.replace('Please inference first, then provide the final answer (True/False) at the end', 'Please provide the final answer (True/False) at the end')
        
        formated_question = prompt + original_question#formated_question

        if prompt_style == 'gt_style':
            formated_question = front_gt_prompt_boolq + formated_question
        else:
            formated_question = front_prompt_boolq + formated_question


    if task == 'esnli':
        if enforce_prompt_style == '':
            if prompt_style == 'minimum_change':
                prompt = esnli_minimum_change_style_prompt
            if prompt_style == 'gt_style':
                prompt = esnli_gt_style_prompt
            if prompt_style == 'gpt4':
                prompt = esnli_gpt4_style_prompt
            if prompt_style == 'rewrite_in_gpt4_style':
                prompt = esnli_rewrite_in_gpt4_style_prompt
            if prompt_style == 'mini_gpt4_style':
                prompt = esnli_mini_gpt4_style_prompt
            if prompt_style == 'anthropic':
                prompt = esnli_anthropic_style_prompt
            if prompt_style == 'mini_gpt4':
                prompt = esnli_mini_gpt4_prompt
            if prompt_style == 'in_own_words':
                prompt = esnli_rewrite_in_own_words_style_prompt

        # formated_question = question
        formated_question = prompt + original_question#formated_question
        formated_question = front_prompt_esnli + formated_question

    return formated_question



def load_dataset(task, prompt_style, use_original_question = False):
    if 'gsm8k' in task:
        task_name_ = 'gsm8k'
        if prompt_style == 'gt_style':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train_filtered.json'
        if prompt_style == 'minimum_change':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/mistral_minimum_change_1000_clean.json'
        if prompt_style == 'gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/gpt4_generated_gsm8k_False_999999_march_27.json'
        if prompt_style == 'anthropic':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/anthropic_gpt4_generated_gsm8k_False_1000_r1.json'
        if prompt_style == 'mini_gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/openai_mini_gpt4.json'


        if prompt_style == 'in_own_words':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
        if prompt_style == 'step_by_step':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_step_by_step_1000.json'
        if prompt_style == 'step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'detailed_step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/detailed_step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'paraphrase':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_direct_paraphrase_gpt4_answer.json'

        if prompt_style == 'anthropic_in_own_words':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
        if prompt_style == 'anthropic_step_by_step':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_gpt4_generated_step_by_step_1000.json'
        if prompt_style == 'anthropic_step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'anthropic_detailed_step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_detailed_step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'anthropic_paraphrase':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_gpt4_generated_direct_paraphrase_gpt4_answer.json'


  
        gt_data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train_filtered.json'
        gt_data_list = load_GSM8K(gt_data_path, 1000)
        original_question_list = []
        for gt_data in gt_data_list:
            if use_original_question:
                original_question_list.append(gt_data['original_question'])
            else:
                original_question_list.append(gt_data['question'])
    elif 'math_algebra' in task:
        task_name_ = 'math'
        if prompt_style == 'gt_style':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train_algebra_total_filtered.json'
        if prompt_style == 'minimum_change':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/math_algebra_total_mistral_minimum_change_1000.json'
        if prompt_style == 'gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/gpt4_generated_math_algebra_False_1000.json'
        if prompt_style == 'anthropic':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/anthropic_gpt4_generated_math_algebra_False_1000_r1.json'
        if prompt_style == 'mini_gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/openai_mini_gpt4.json'
        # if prompt_style == 'in_own_words':
        #     data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/rewirte_groundtruth_in_own_words.json'
        # if prompt_style == 'step_by_step':
        #     data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_step_by_step_1000.json'

        
        if prompt_style == 'in_own_words':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
        if prompt_style == 'step_by_step':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_step_by_step_1000.json'
        if prompt_style == 'step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'detailed_step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/detailed_step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'paraphrase':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_direct_paraphrase_gpt4_answer.json'

        if prompt_style == 'anthropic_in_own_words':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
        if prompt_style == 'anthropic_step_by_step':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_gpt4_generated_step_by_step_1000.json'
        if prompt_style == 'anthropic_step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'anthropic_detailed_step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_detailed_step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'anthropic_paraphrase':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_gpt4_generated_direct_paraphrase_gpt4_answer.json'


        gt_data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train_algebra_total_filtered.json'
        gt_data_list = load_MATH(gt_data_path, 1000, zeroshot = False)
        original_question_list = []
        for gt_data in gt_data_list:
            if use_original_question:
                original_question_list.append(gt_data['original_question'])
            else:
                original_question_list.append(gt_data['question'])
    elif 'ecqa' in task:
        task_name_ = 'ecqa'
        if prompt_style == 'gt_style':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train.json'
        if prompt_style == 'minimum_change':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/ecqa_mistral_minimum_change_1000_march_27.json'
        if prompt_style == 'gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/gpt4_generated_ecqa_False_1000_march_27.json'
        if prompt_style == 'anthropic':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/anthropic_gpt4_generated_ecqa_False_1000_r1.json'
        if prompt_style == 'mini_gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/openai_mini_gpt4.json'
        # if prompt_style == 'in_own_words':
        #     data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
        # if prompt_style == 'step_by_step':
        #     data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_step_by_step_1000.json'

        if prompt_style == 'in_own_words':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
        if prompt_style == 'step_by_step':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_step_by_step_1000.json'
        if prompt_style == 'step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'detailed_step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/detailed_step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'paraphrase':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_direct_paraphrase_gpt4_answer.json'

        if prompt_style == 'anthropic_in_own_words':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
        if prompt_style == 'anthropic_step_by_step':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_gpt4_generated_step_by_step_1000.json'
        if prompt_style == 'anthropic_step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'anthropic_detailed_step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_detailed_step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'anthropic_paraphrase':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_gpt4_generated_direct_paraphrase_gpt4_answer.json'
        if prompt_style == 'redundant':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/openai_gpt4_generated_ecqa_redundant_1000.json'


        gt_data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train.json'
        gt_data_list = load_ECQA(gt_data_path, 1000, finetune = True, use_gt_rationale = True, meta_math_template = False)
        original_question_list = []
        for gt_data in gt_data_list:
            if use_original_question:
                original_question_list.append(gt_data['original_question'])
            else:
                original_question_list.append(gt_data['question'])
    elif 'api_bank' in task:
        task_name_ = 'api_bank'
        if prompt_style == 'gt_style':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train/training-data_lv3-api-train.json'
        if prompt_style == 'gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/openai_gpt4_generated_api_bank_False_1000_r1.json'
        if prompt_style == 'anthropic':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/anthropic_gpt4_generated_api_bank_1000.json'
        if prompt_style == 'mini_gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/openai_mini_gpt4.json'
        if prompt_style == 'human_written_examples':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/openai_human_written_examples.json'
        if prompt_style == 'step_by_step':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_step_by_step_1000.json'
        if prompt_style == 'provide_gpt4_style_example':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/openai_gpt4_generated_new_rewrite_strategy.json'

        gt_data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train/training-data_lv3-api-train.json'
        gt_data_list = load_API_BANK_aug_2(gt_data_path, 1000)
        original_question_list = []
        for gt_data in gt_data_list:
            original_question_list.append(gt_data['question'])
    
    elif 'plan_bench' in task:
        task_name_ = 'plan_bench'
        if prompt_style == 'gt_style':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train_plan_generation.json'
        if prompt_style == 'gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/gpt4_generated_plan_bench_False_1000.json'
        if prompt_style == 'anthropic':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/anthropic_generated_plan_bench_False_1000.json'
        if prompt_style == 'mini_gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/openai_mini_gpt4.json'
        if prompt_style == 'human_written_examples':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/openai_human_written_examples.json'
        if prompt_style == 'step_by_step':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_step_by_step_1000.json'
        if prompt_style == 'provide_gpt4_style_example':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/write_in_gpt4_style.json'
        if prompt_style == 'in_own_words':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'

        gt_data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train_plan_generation.json'
        with open(gt_data_path, 'r') as file:
            gt_data_list = json.load(file)
        gt_data_list = gt_data_list[:1000]
        original_question_list = []
        for gt_data in gt_data_list:
            original_question_list.append(gt_data['question'])

    elif 'boolq' in task:
        task_name_ = 'boolq'
        if prompt_style == 'gt_style':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train.json'
        if prompt_style == 'minimum_change':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/boolq_mistral_minimum_change_1000_march_27.json'
        if prompt_style == 'gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/gpt4_generated_boolq_False_1000.json'
        if prompt_style == 'anthropic':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/anthropic_gpt4_generated_boolq_False_1000_r1.json'
        if prompt_style == 'mini_gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/openai_mini_gpt4.json'
        if prompt_style == 'mini_gpt4_style':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/write_in_gpt4_style.json'
        if prompt_style == 'step_by_step':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_step_by_step_1000.json'

        gt_data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train.json'
        gt_data_list = load_BOOLQ(gt_data_path, 1000, finetune = True)
        original_question_list = []
        for gt_data in gt_data_list:
            if use_original_question:
                original_question_list.append(gt_data['original_question'])
            else:
                original_question_list.append(gt_data['question'])
    
    elif 'winogrande' in task:
        task_name_ = 'winogrande'
        if prompt_style == 'gt_style':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train.json'
        if prompt_style == 'minimum_change':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/winogrande_mistral_minimum_change_1000_sep_9.json'
        if prompt_style == 'gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/openai_gpt4_generated_winogrande_False_1000_r1.json'
        if prompt_style == 'anthropic':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/anthropic_gpt4_generated_winogrande_False_1000_answer_without_groundtruth_False_enable_mini_gpt4_False.json'
        if prompt_style == 'mini_gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/openai_mini_gpt4.json'
        if prompt_style == 'step_by_step':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_step_by_step_1000.json'
        
        gt_data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train.json'
        gt_data_list = load_WINOGRANDE(gt_data_path, 1000, finetune = True)
        original_question_list = []
        for gt_data in gt_data_list:
            # if use_original_question:
            #     original_question_list.append(gt_data['original_question'])
            # else:
            #     original_question_list.append(gt_data['question'])
            original_question_list.append(gt_data['question'])

    
    elif 'mmlu' in task:
        task_name_ = 'mmlu'
        if prompt_style == 'gt_style':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/mmlu_train.json'
        if prompt_style == 'minimum_change':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/mmlu_mistral_minimum_change_1000_sep_19.json'
        if prompt_style == 'gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/mmlu_gpt-4o-2024-08-06_1000.json'
        if prompt_style == 'anthropic':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/anthropic_gpt4_generated_mmlu_False_1000.json'
        if prompt_style == 'mini_gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/openai_mini_gpt4.json'
        if prompt_style == 'step_by_step':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_step_by_step_1000.json'
        
        gt_data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/mmlu_train.json'
        gt_data_list = load_MMLU(gt_data_path, 1000, finetune = True)
        original_question_list = []
        for gt_data in gt_data_list:
            # if use_original_question:
            #     original_question_list.append(gt_data['original_question'])
            # else:
            #     original_question_list.append(gt_data['question'])
            original_question_list.append(gt_data['question'])

    elif 'esnli' in task:
        task_name_ = 'esnli'
        if prompt_style == 'gt_style':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train.json'
        if prompt_style == 'minimum_change':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/esnli_mistral_minimum_change_1000.json'
        if prompt_style == 'gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/gpt4_generated_esnli_False_1000.json'
        if prompt_style == 'anthropic':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/anthropic_gpt4_generated_esnli_False_1000_r1.json'
        if prompt_style == 'mini_gpt4':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/openai_mini_gpt4.json'
        if prompt_style == 'mini_gpt4_style':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/write_in_gpt4_style.json'
        # if prompt_style == 'in_own_words':
        #     data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
        # if prompt_style == 'step_by_step':
        #     data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_step_by_step_1000.json'

        if prompt_style == 'in_own_words':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
        if prompt_style == 'step_by_step':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_step_by_step_1000.json'
        if prompt_style == 'step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'detailed_step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/detailed_step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'paraphrase':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/gpt4_generated_direct_paraphrase_gpt4_answer.json'

        if prompt_style == 'anthropic_in_own_words':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_gpt4_generated_rewirte_groundtruth_in_own_words_1000.json'
        if prompt_style == 'anthropic_step_by_step':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_gpt4_generated_step_by_step_1000.json'
        if prompt_style == 'anthropic_step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'anthropic_detailed_step_by_step_gt':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_detailed_step_by_step_while_keep_original_distribution.json'
        if prompt_style == 'anthropic_paraphrase':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/anthropic_gpt4_generated_direct_paraphrase_gpt4_answer.json'
        if prompt_style == 'redundant':
            data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/openai_gpt4_generated_esnli_redundant_1000.json'


        gt_data_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/train.json'
        gt_data_list = load_ESNLI(gt_data_path, 1000, finetune = True, meta_math_template = False)
        original_question_list = []
        for gt_data in gt_data_list:
            if use_original_question:
                original_question_list.append(gt_data['original_question'])
            else:
                original_question_list.append(gt_data['question'])
    return gt_data_list, data_path, original_question_list


def random_select_in_context_learning_examples(question, original_question, prompt_style = 'gt_style', task = 'gsm8k', sub_samples_num_list = [0,1,2], end_template = """
Now please solve the following question using the same inference style and format as the examples above. 
Question: """, similarity_compare_to_in_context_mc = False):
    
    front_prompt_math = \
"""
Requirement 1. Please inference first, then place the final result(a number) after the word Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Inference: INFERENCE_HERE
Final Answer: NUMBER_HERE
"""

    front_prompt_ecqa = \
"""
Requirement 1. Please inference first, then place the final result(1/2/3/4/5) after the word Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Inference: INFERENCE_HERE
Final Answer: (1/2/3/4/5)_HERE
"""

    front_prompt_api_bank = \
"""
Requirement 1. Please inference first, then place the API-REQUEST after the word Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Question:
Inference: INFERENCE_HERE
Final Answer: API-REQUEST_HERE
"""

    front_gt_prompt_api_bank = \
"""
Requirement 1. Please place the API-REQUEST after the word Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Question: 
Final Answer: API-REQUEST_HERE
"""

    front_prompt_boolq = \
"""
Requirement 1. Please inference first, then place the True or False after Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Question:
Inference: INFERENCE_HERE
Final Answer: True/False
"""

    front_gt_prompt_boolq = \
"""
Requirement 1. Please place the True or False after Final Answer: 
Requirement 2. Please solving the question with the similar styles as the examples below.

Format:
Question:
Final Answer: True/False
"""

    front_prompt_esnli = \
"""
We know the definetion of entailment, contradiction and neutral is
Entailment: The statement is definitely true given the context.
Contradiction: The statement is definitely false given the context.
Neutral: The truth of the statement is undetermined or irrelevant given the context.
Given a statement and a context, please determine whether the statement is entailment, contradiction, or neutral given the context. 

Requirement 1. Please inference first, then place the entailment, contradiction or neutral after Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Question:
Inference: INFERENCE_HERE
Final Answer: Entailment/Contradiction/Neutral
"""

    front_prompt_winogrande = \
"""
Requirement 1. Please choose the correct option(1 or 2) and place it after Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Question:
Inference: INFERENCE_HERE
Final Answer: 1/2

"""

    front_gt_prompt_winogrande = \
"""
Requirement 1. Please choose the correct option(1 or 2) and place it after Final Answer: 
Requirement 2. Please solving the question with the similar styles as the examples below.

Format:
Question:
Final Answer: 1/2

"""

    front_prompt_mmlu = \
"""
Requirement 1. Please choose the correct option(A or B or C or D) and place it after Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Question:
Inference: INFERENCE_HERE
Final Answer: A/B/C/D

"""

    front_gt_prompt_mmlu = \
"""
Requirement 1. Please choose the correct option(A or B or C or D)) and place it after Final Answer: 
Requirement 2. Please solving the question with the similar styles as the examples below.

Format:
Question:
Final Answer: A/B/C/D

"""

    front_prompt_plan_bench = \
"""
Requirement 1. Please inference first, then place the plan after the word Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Question:
Inference: INFERENCE_HERE
Final Answer: PLAN_HERE
"""

    front_gt_prompt_plan_bench = \
"""
Requirement 1. Please place the plan after the word Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Question: 
Final Answer: PLAN_HERE
"""



    def prompt_generation_template(task, prompt_style, sub_samples_num_list, similarity_compare_to_in_context_mc = False):

        if 'gsm8k' in task:
            front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_prompt_math}
"""
        elif 'math_algebra' in task:
            front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_prompt_math}
"""
        elif 'ecqa' in task:
            front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_prompt_ecqa}
"""
        elif 'api_bank' in task:
            if prompt_style == 'gt_style':
                front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_gt_prompt_api_bank}
"""
            else:
                front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_prompt_api_bank}
"""
        elif 'plan_bench' in task:
            if prompt_style == 'gt_style':
                front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_gt_prompt_plan_bench}
"""
            else:
                front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_prompt_plan_bench}
"""
        elif 'boolq' in task:
            if prompt_style == 'gt_style':
                front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_gt_prompt_boolq}
"""
            else:
                front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_prompt_boolq}
"""
        elif 'winogrande' in task:
            if prompt_style == 'gt_style':
                front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_gt_prompt_winogrande}
"""
            else:
                front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_prompt_winogrande}
"""
        elif 'mmlu' in task:
            if prompt_style == 'gt_style':
                front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_gt_prompt_mmlu}
"""
            else:
                front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_prompt_mmlu}
"""
            
        elif 'esnli' in task:
            front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_prompt_esnli}
"""
        if similarity_compare_to_in_context_mc:
            gt_data_list, data_path, original_question_list = load_dataset(task, 'minimum_change')
        else:
            gt_data_list, data_path, original_question_list = load_dataset(task, prompt_style)
        if prompt_style == 'gt_style':
            data_list = gt_data_list
        else:
            with open(data_path, 'r') as file:
                data_list = json.load(file)
        
        in_context_prompt = ''
#         end_template = """
# Now please solve the following question using the same inference style and format as the examples above. 
# Question: """
        body_template = ''
        for i_, sub_sample_num in enumerate(sub_samples_num_list):
            question_item = original_question_list[sub_sample_num]
            answer_item = data_list[sub_sample_num]['answer']
            if (prompt_style == 'gt_style' and 'winogrande' in task) or (prompt_style == 'gt_style' and 'boolq' in task) or (prompt_style == 'gt_style' and 'api_bank' in task) or (prompt_style == 'gt_style' and 'plan_bench' in task):
                body_template += \
f"""Example {i_ + 1}
Question: {question_item}
Final Answer: {answer_item}

"""
            else:
                body_template += \
f"""Example {i_ + 1}
Question: {question_item}
Inference: {answer_item}

"""
        in_context_prompt = front_template + body_template + end_template

        return in_context_prompt
    
        
        
    if 'generated_xy_pairs' in task:
        task_name_ = 'gsm8k'
        
        if 'mini' in task:
            xy_pairs_model = 'gpt-4o-mini'
        if 'Breadth_Prompt' in prompt_style:
            full_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/{xy_pairs_model}_Breadth Prompt_{task_name_}_generated_xy_pairs_1000.json'
        if 'Reasoning_Prompt' in prompt_style:
            full_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/{xy_pairs_model}_Reasoning Prompt_{task_name_}_generated_xy_pairs_1000.json'
        if 'Concretizing_Prompt' in prompt_style:
            full_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/{xy_pairs_model}_Concretizing Prompt_{task_name_}_generated_xy_pairs_1000.json'
        if 'Deepen_Prompt' in prompt_style:
            full_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/{xy_pairs_model}_Deepen Prompt_{task_name_}_generated_xy_pairs_1000.json'
        if 'Constraints_Prompt' in prompt_style:
            full_path = f'{HOME_DIRECTORY}/dataset/{task_name_.upper()}/varient/{xy_pairs_model}_Constraints Prompt_{task_name_}_generated_xy_pairs_1000.json'
        with open(full_path, 'r') as file:
            generated_xy_pairs_list = json.load(file)
        generated_xy_pairs_list = generated_xy_pairs_list[:3]

        q_1 = generated_xy_pairs_list[0]['question']
        q_1 = q_1.replace("\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n", '')
        a_1 = generated_xy_pairs_list[0]['answer']

        q_2 = generated_xy_pairs_list[1]['question']
        q_2 = q_2.replace("\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n", '')
        a_2 = generated_xy_pairs_list[1]['answer']

        q_3 = generated_xy_pairs_list[2]['question']
        q_3 = q_3.replace("\n\nPlease provide the final answer (a number) at the end, after 'Final Answer:'\n", '')
        a_3 = generated_xy_pairs_list[2]['answer']

        formated_question = \
f"""
We have 3 examples below

Example 1
Question: {q_1}
Inference: {a_1}

Example 2
Question: {q_2}
Inference: {a_2}

Example 3
Question: {q_3}
Inference: {a_3}

Now please solve the following question using the same inference style and format as the example above. 
Question: """
    else:
        formated_question = prompt_generation_template(task, prompt_style, sub_samples_num_list, similarity_compare_to_in_context_mc = similarity_compare_to_in_context_mc)

    formated_question += question #original_question
    return formated_question



def create_random_incontext_training_samples(task, prompt_style, n_training_samples = 50, n_train_random_in_context_samples = 10, sample_num = 3, use_original_question = False, using_incontext_examples_in_question = True, qa_list = []):
    gt_data_list, data_path, original_question_list = load_dataset(task, prompt_style, use_original_question = use_original_question)
    if prompt_style == 'gt_style':
        data_list = gt_data_list
    else:
        with open(data_path, 'r') as file:
            data_list = json.load(file)
    
    def generate_random_lists(num_lists, sample_num=3, sub_sample_num = 10):
        value_range=range(0, sub_sample_num)
        random_lists = []
        for _ in range(num_lists):
            random_list = random.sample(value_range, sample_num)
            random_lists.append(random_list)
        return random_lists
    
    # if 'BOOLQ' in task.upper() or 'few_shot_in_context_samples_train_then_test' in task.lower():
    #     sample_num = 6
    # else:
    #     sample_num = 3
    # sample_num = 3
    sub_samples_num_list = generate_random_lists(n_training_samples, sample_num = sample_num, sub_sample_num = n_train_random_in_context_samples)


    train_data_list = data_list[:n_train_random_in_context_samples]
    train_data_list_temp = []

    for iii in range(n_training_samples):
        temp = {}
        index_of_sample = random.randint(0, len(train_data_list) - 1)
        random_sample = train_data_list[index_of_sample]

        answer_item = random_sample['answer']
        question_item = random_sample['question']
        original_question = original_question_list[index_of_sample]
        if using_incontext_examples_in_question:
            formated_question = random_select_in_context_learning_examples(question_item, original_question, prompt_style = prompt_style, task = task, sub_samples_num_list = sub_samples_num_list[iii])
        else:
            formated_question = question_item
        temp['question'] = formated_question
        temp['answer'] = answer_item
        temp['input'] = ''
        train_data_list_temp.append(temp)


    return train_data_list_temp


def put_to_json_file(data_list, file_name):
    dpo_path = f'{HOME_DIRECTORY}/dpo_data/GSM8K/_gpt4_data_dpo_5_mistral.json'
    intermediate_finetune_file_path = f"{HOME_DIRECTORY}/alpaca_data/few_shot_in_context_samples_train_then_test_{file_name}.json"
    train_data = []
    for item in data_list:
        temp = {}
        temp['instruction'] = item['question']
        temp['output'] = item['answer']
        temp['input'] = ''
        train_data.append(temp)
    with open(intermediate_finetune_file_path, 'w') as json_file:
        json.dump(train_data, json_file, indent=4)
    return intermediate_finetune_file_path, dpo_path





















def api_bank_random_select_in_context_learning_examples(question, original_question, prompt_style = 'gt_style', task = 'gsm8k', sub_samples_num_list = [0,1,2], qa_list = []):
 
    front_gt_prompt_api_bank = \
"""
Requirement 1. Please place the API-REQUEST after the word Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Question: 
Final Answer: API-REQUEST_HERE
"""

    front_prompt_api_bank = \
"""
Requirement 1. Please inference first, then place the API-REQUEST after the word Final Answer: 
Requirement 2. Please solving the question with the similar inference styles as the examples below.

Format:
Question:
Inference: INFERENCE_HERE
Final Answer: API-REQUEST_HERE
"""

    def prompt_generation_template(task, prompt_style, sub_samples_num_list):
        if prompt_style == 'gt_style':
            front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_gt_prompt_api_bank}
"""
        else:
            front_template = f"""We have {len(sub_samples_num_list)} examples below

{front_prompt_api_bank}
"""
        gt_data_list, data_path, original_question_list = load_dataset(task, prompt_style)
        original_question_list_temp = []
        for item in original_question_list:
            original_question_list_temp.append('A LONG QUESTION HERE')
        original_question_list = original_question_list_temp
        if prompt_style == 'gt_style':
            data_list = gt_data_list
        else:
            with open(data_path, 'r') as file:
                data_list = json.load(file)
        
        in_context_prompt = ''
        end_template = """




        
Now please solve the following question using the same inference style and format as the examples above. 
Question: """
        body_template = ''
        for i_, sub_sample_num in enumerate(sub_samples_num_list):
            if qa_list == []:
                question_item = original_question_list[sub_sample_num]
                answer_item = data_list[sub_sample_num]['answer']
            else:
                question_item = qa_list[0][sub_sample_num]
                answer_item = qa_list[1][sub_sample_num]
            if prompt_style == 'gt_style':
                body_template += \
f"""Example {i_ + 1}
Question: {question_item}
Final Answer: {answer_item}

"""
            else:
                body_template += \
f"""Example {i_ + 1}
Question: {question_item}
Inference: {answer_item}

"""
        in_context_prompt = front_template + body_template + end_template

        return in_context_prompt
    
    formated_question = prompt_generation_template(task, prompt_style, sub_samples_num_list)

    formated_question += question #original_question
    return formated_question



def api_bank_create_random_incontext_training_samples(task, prompt_style, n_training_samples = 50, n_train_random_in_context_samples = 10, sample_num = 3, use_original_question = False, using_incontext_examples_in_question = True, qa_list = []):
    gt_data_list, data_path, original_question_list = load_dataset(task, prompt_style, use_original_question = use_original_question)
    original_question_list_temp = []
    for item in original_question_list:
        original_question_list_temp.append('A LONG QUESTION HERE')
    original_question_list = original_question_list_temp
    if prompt_style == 'gt_style':
        data_list = gt_data_list
    else:
        with open(data_path, 'r') as file:
            data_list = json.load(file)
    
    def generate_random_lists(num_lists, sample_num=3, sub_sample_num = 10):
        value_range=range(0, sub_sample_num)
        random_lists = []
        for _ in range(num_lists):
            random_list = random.sample(value_range, sample_num)
            random_lists.append(random_list)
        return random_lists
    
    sub_samples_num_list = generate_random_lists(n_training_samples, sample_num = sample_num, sub_sample_num = n_train_random_in_context_samples)


    train_data_list = data_list[:n_train_random_in_context_samples]
    train_data_list_temp = []

    for iii in range(n_training_samples):
        temp = {}
        index_of_sample = random.randint(0, len(train_data_list) - 1)
        random_sample = train_data_list[index_of_sample]

        answer_item = random_sample['answer']
        question_item = random_sample['question']
        original_question = original_question_list[index_of_sample]
        if using_incontext_examples_in_question:
            formated_question = random_select_in_context_learning_examples(question_item, original_question, prompt_style = prompt_style, task = task, sub_samples_num_list = sub_samples_num_list[iii])
        else:
            formated_question = question_item
        temp['question'] = formated_question
        temp['answer'] = answer_item
        temp['input'] = ''
        train_data_list_temp.append(temp)


    return train_data_list_temp


def put_to_json_file(data_list, file_name):
    dpo_path = f'{HOME_DIRECTORY}/dpo_data/GSM8K/_gpt4_data_dpo_5_mistral.json'
    intermediate_finetune_file_path = f"{HOME_DIRECTORY}/alpaca_data/few_shot_in_context_samples_train_then_test_{file_name}.json"
    train_data = []
    for item in data_list:
        temp = {}
        temp['instruction'] = item['question']
        temp['output'] = item['answer']
        temp['input'] = ''
        train_data.append(temp)
    with open(intermediate_finetune_file_path, 'w') as json_file:
        json.dump(train_data, json_file, indent=4)
    return intermediate_finetune_file_path, dpo_path

