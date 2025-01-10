from typing import List
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

def get_chatgpt_results(claim:str, top_k_claims:List[str], model_choice:str, api_key:str):
    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert text analysis model. Your task is to determine whether a given statement (the unreviewed claim) matches or refers to the same topic as each one of the statements in a list. Perform a semantic and conceptual analysis to decide whether they are referring to the same thing.

                For each item in the list, provide two values:
                1. class: a number that can be 0 or 1.
                - 0 if it does NOT talk about the same topic
                - 1 if it DOES talk about the same topic
                2. probability: a value between 0 and 1 expressing your confidence in your classification.

                Make sure to:
                - Be concise and objective.
                - Base your response strictly on the provided content, without inferring any additional information.
                - Respect the requested output format.
                """
            ),
            (
                "human",
                """Given the following *unreviewed claim* and the list of statements, classify them according to the above criteria.

                Unreviewed claim:
                {unreviewed_claim}

                List of statements:
                {claims}

                Your answer must be an array (list) in JSON format where each element contains:
                - class: 0 or 1
                - probability: a number between 0 and 1

                For example:

                [
                {{
                    class: 1,
                    probability: 0.95
                }},
                {{
                    class: 0,
                    probability: 0.2
                }},
                ...
                ]

                Do not include any additional text or explanations outside the JSON.
                """
            ),
        ]
    )
    parsed_top_k_claims = "\n".join(f"Claim {i}: {k}" for i, k in enumerate(top_k_claims))
    # Generate the prompt
    prompt = template.invoke(
        {
            "unreviewed_claim": claim, 
            "claims": parsed_top_k_claims,
        }
    )
    # Invoke the LLM
    llm = ChatOpenAI(model=model_choice, api_key=api_key)
    result = llm.invoke(prompt)
    json_str = result.content.strip("```json").strip("```").strip()
    parsed_data = json.loads(json_str)
    return list(zip(top_k_claims, [k['probability'] for k in parsed_data]))
