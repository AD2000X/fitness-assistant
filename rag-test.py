#!/usr/bin/env python
# coding: utf-8


import pandas as pd


# # Ingestion

# In[53]:


df = pd.read_csv('../data/data.csv')


# In[54]:


documents = df.to_dict(orient='records')


# In[55]:


get_ipython().system('wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py')


# In[56]:


import minsearch


# In[57]:


index = minsearch.Index(
    text_fields=["exercise_name", "type_of_activity", "type_of_equipment", "body_part",
                 "type", "muscle_groups_activated", "instructions"],
    keyword_fields=['id']
)


# In[58]:


index.fit(documents)


# # RAG Flow

# In[59]:


from openai import OpenAI


# In[60]:


from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
print("API Key loaded:", "Yes" if api_key else "No")


# In[61]:


from openai import OpenAI

client = OpenAI()


# In[62]:


def search(query):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[63]:


prompt_template = """
You're a fitness instructor. Answer the QUESTION based on the CONTEXT from our excercises database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

entry_template = """
exercise_name: {exercise_name}
type_of_activity: {type_of_activity}
type_of_equipment: {type_of_equipment}
body_part: {body_part}
type: {type}
muscle_groups_activated: {muscle_groups_activated}
instructions: {instructions}
""".strip()

def build_prompt(query, search_results):
    context = ""

    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


# In[64]:


def llm(prompt):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# In[65]:


def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


# In[69]:


answer = rag("I want some core calistenics that also help my back")
print(answer)


# ## Retrieveal Evaluation

# In[ ]:


df_question = pd.read_csv('../data/ground-truth-retrieval.csv')


# In[ ]:


df_question.head


# In[ ]:


ground_truth = df_question.to_dict(orient='records')


# In[ ]:


ground_truth[0]


# In[ ]:


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


# In[ ]:


def minsearch_search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[ ]:


def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['id']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'MRR': mrr(relevance_total),
    }


# In[ ]:


from tqdm.auto import tqdm


# In[ ]:


evaluate(ground_truth, lambda q: minsearch_search(q['question']))


# # Finding the best parameters

# In[ ]:


df_validation = df_question[:100]
df_test = df_question[100:]


# In[ ]:


import random

def simple_optimize(param_ranges, objective_function, n_iterations=10):
    best_params = None
    best_score = float('-inf')  # Assuming we're minimizing. Use float('-inf') if maximizing.

    for _ in range(n_iterations):
        # Generate random parameters
        current_params = {}
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                current_params[param] = random.randint(min_val, max_val)
            else:
                current_params[param] = random.uniform(min_val, max_val)
        
        # Evaluate the objective function
        current_score = objective_function(current_params)
        
        # Update best if current is better
        if current_score > best_score:  # Change to > if maximizing
            best_score = current_score
            best_params = current_params
    
    return best_params, best_score


# In[ ]:


gt_val = df_validation.to_dict(orient='records')


# In[ ]:


def minsearch_search(query, boost=None):
    if boost is None:
        boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[ ]:


param_ranges = {
    'exercise_name': (0.0, 3.0),
    'type_of_activity': (0.0, 3.0),
    'type_of_equipment': (0.0, 3.0),
    'body_part': (0.0, 3.0),
    'type': (0.0, 3.0),
    'muscle_groups_activated': (0.0, 3.0),
    'instructions': (0.0, 3.0),
}

def objective(boost_params):
    def search_function(q):
        return minsearch_search(q['question'], boost_params)

    results = evaluate(gt_val, search_function)
    return results['MRR']


# In[ ]:


simple_optimize(param_ranges, objective, n_iterations=10)


# In[ ]:


import optuna
# Silence Optuna's warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    # Define the parameter space
    boost_params = {
        'exercise_name': trial.suggest_float('exercise_name', 0.0, 3.0),
        'type_of_activity': trial.suggest_float('type_of_activity', 0.0, 3.0),
        'type_of_equipment': trial.suggest_float('type_of_equipment', 0.0, 3.0),
        'body_part': trial.suggest_float('body_part', 0.0, 3.0),
        'type': trial.suggest_float('type', 0.0, 3.0),
        'muscle_groups_activated': trial.suggest_float('muscle_groups', 0.0, 3.0),
        'instructions': trial.suggest_float('instructions', 0.0, 3.0)
    }
    
    def search_function(q):
        return minsearch_search(q['question'], boost_params)

    results = evaluate(gt_val, search_function)
    return results['MRR']

# Define the callback
def print_callback(study, trial):
    # Print information only when a new best value is found
    if study.best_trial == trial:
        print(f"Trial {trial.number} - New best MRR: {trial.value:.4f}")

# Create the study
study = optuna.create_study(direction='maximize')

# Run the optimization with the callback
print("Starting optimization...")
study.optimize(objective, n_trials=100, callbacks=[print_callback], show_progress_bar=False)

# Print the final results
print("\nOptimization finished!")
print(f"Best MRR: {study.best_value:.4f}")
print("\nBest parameters:")
for k, v in study.best_params.items():
    print(f"{k}: {v:.4f}")


# In[ ]:


def minsearch_improved(query):
    boost = {
        'exercise_name': 1.9777,
        'type_of_activity': 0.0102,
        'type_of_equipment': 0.0080,
        'body_part': 1.8424,
        'type': 0.8076,
        'muscle_groups': 2.9760,
        'instructions': 1.0320
    }

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results

evaluate(ground_truth, lambda q: minsearch_improved(q['question']))


# # RAG Evaluation

# In[ ]:


prompt2_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


# In[ ]:


len(ground_truth)


# In[ ]:


record = ground_truth[0]
question = record['question']
answer_llm = rag(question)


# In[ ]:


print(answer_llm)


# In[ ]:


prompt = prompt2_template.format(question=question, answer_llm=answer_llm)
print(prompt)


# In[ ]:


llm(prompt)


# In[70]:


import json


# In[72]:


evaluations = []


# In[ ]:


# for record in tqdm(ground_truth):
#     question = record['question']
#     answer_llm = rag(question) 

#     prompt = prompt2_template.format(
#         question=question,
#         answer_llm=answer_llm
#     )

#     evaluation = llm(prompt)
#     evaluation = json.loads(evaluation)

#     evaluations.append((record, answer_llm, evaluation))


# In[75]:


df_eval = pd.DataFrame(evaluations, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])

del df_eval['record']
del df_eval['evaluation']


# In[76]:


df_eval.relevance.value_counts(normalize=True)


# In[ ]:


# df_eval.to_csv('../data/rag-eval-gpt-4o-mini.csv', index=False)


# In[77]:


df_eval[df_eval.relevance == 'NON_RELEVANT']


# In[79]:


evaluations_gpt4o = []


# In[ ]:


# for record in tqdm(sample):
#     question = record['question']
#     answer_llm = rag(question, model='gpt-4o') 

#     prompt = prompt2_template.format(
#         question=question,
#         answer_llm=answer_llm
#     )

#     evaluation = llm(prompt)
#     evaluation = json.loads(evaluation)
    
#     evaluations_gpt4o.append((record, answer_llm, evaluation))


# In[80]:


df_eval = pd.DataFrame(evaluations_gpt4o, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])

del df_eval['record']
del df_eval['evaluation']


# In[81]:


df_eval.relevance.value_counts()


# In[82]:


df_eval.relevance.value_counts(normalize=True)


# In[83]:


df_eval.to_csv('../data/rag-eval-gpt-4o.csv', index=False)

