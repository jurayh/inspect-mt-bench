"""Official MT-Bench single-answer judge prompts, ported verbatim.

Source: lm-sys/FastChat, fastchat/llm_judge/data/judge_prompts.jsonl,
pinned at commit 587d5cfa1609a43d192cedb8441cac3c17db105d (Apache-2.0).
Do not edit the template text: fidelity to the published MT-Bench judge
prompts is part of this eval's definition. Placeholders use {name} tokens
and are substituted by token replacement in scorer.py.
"""

from __future__ import annotations

SINGLE_V1 = {
    "system_prompt": "You are a helpful assistant.",
    "prompt_template": '[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{question}\n\n[The Start of Assistant\'s Answer]\n{answer}\n[The End of Assistant\'s Answer]',
}

SINGLE_MATH_V1 = {
    "system_prompt": "You are a helpful assistant.",
    "prompt_template": "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer_1}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]",
}

SINGLE_V1_MULTI_TURN = {
    "system_prompt": 'Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You evaluation should focus on the assistant\'s answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n',
    "prompt_template": "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n<|The End of Assistant A's Conversation with User|>",
}

SINGLE_MATH_V1_MULTI_TURN = {
    "system_prompt": 'Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant\'s answer. You evaluation should focus on the assistant\'s answer to the second question. Begin your evaluation by comparing the assistant\'s answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n',
    "prompt_template": "<|The Start of Reference Answer|>\n\n### User:\n{question_1}\n\n### Reference answer:\n{ref_answer_1}\n\n### User:\n{question_2}\n\n### Reference answer:\n{ref_answer_2}\n\n<|The End of Reference Answer|>\n\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n<|The End of Assistant A's Conversation with User|>",
}
