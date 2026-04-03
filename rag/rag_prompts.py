from langchain_core.prompts import PromptTemplate


def get_rag_prompt():
    template = """
You are a strict and reliable question-answering assistant.

Your task is to answer the question using ONLY the provided context.

=====================
CONTEXT:
{context}
=====================

QUESTION:
{question}

=====================
INSTRUCTIONS:

- Use ONLY the context above to answer the question.
- If the answer is NOT explicitly present in the context, respond EXACTLY with:
  "I don't know based on the provided documents."
- Do NOT use prior knowledge.
- Do NOT guess or infer.
- If the context is insufficient or unclear, say you don't know.
- Keep the answer concise, factual, and relevant.
- If multiple relevant points exist, summarize them clearly.
You must answer ONLY using the provided context.

If the context does not clearly identify the company,
respond with:
"I don’t have enough information to answer that."
Decide if the question is complete and answerable.
If the company is not स्पष्ट (clear), say "AMBIGUOUS".

Question: it is a multinational company
Answer: AMBIGUOUS
Do NOT guess or assume.
=====================

FINAL ANSWER:
"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )