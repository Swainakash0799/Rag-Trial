from langchain_core.output_parsers import (
    StrOutputParser
)

import config

from prompts import (
    retriever_prompt,
    answer_prompt
)


# --------------------------------
# Query Rewriter
# --------------------------------

def rewrite_query(question, chat_history):

    chain = retriever_prompt | config.llm| StrOutputParser()
    


    result = chain.invoke(
        {
            "question": question,
            "chat_history": chat_history
        }
    )


    return result


# --------------------------------
# Answer Generator
# --------------------------------

def generate_answer(question,context,chat_history):

    chain = answer_prompt| config.llm | StrOutputParser()
    


    result = chain.invoke(
        {
            "question": question,
            "context": context,
            "chat_history": chat_history
        }
    )


    return result


# --------------------------------
# Streaming Answer
# --------------------------------

def stream_answer(question,context,chat_history):

    chain = answer_prompt| config.llm| StrOutputParser()
    


    return chain.stream(
        {
            "question": question,
            "context": context,
            "chat_history": chat_history
        }
    )