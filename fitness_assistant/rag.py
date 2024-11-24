import json

from time import time

import ingest

from openai import OpenAI


client = OpenAI()
index = ingest.load_index()


def search(query):
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


def llm(prompt):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)

    # answer_data["answer"],
    # answer_data["model_used"],
    # answer_data["response_time"],
    # answer_data["relevance"],
    # answer_data["relevance_explanation"],
    # answer_data["prompt_tokens"],
    # answer_data["completion_tokens"],
    # answer_data["total_tokens"],
    # answer_data["eval_prompt_tokens"],
    # answer_data["eval_completion_tokens"],
    # answer_data["eval_total_tokens"],
    # answer_data["openai_cost"],

    return answer