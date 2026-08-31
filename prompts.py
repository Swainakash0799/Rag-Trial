from langchain_core.prompts import (
    ChatPromptTemplate
)


# --------------------------------
# Query Rewriting
# --------------------------------

retriever_prompt = ChatPromptTemplate.from_messages([

  ("system", """ You are a document search assistant.
    Rewrite the user's question into a short and clear search query.
    Use the conversation history when the user asks a follow-up question.
    Do not answer the question.Only return the search query."""),

  ("human", """ Conversation history: {chat_history}
   User question: {question}
   Search query:""")])


# --------------------------------
# Answer Prompt
# --------------------------------

answer_prompt = ChatPromptTemplate.from_messages([

    ("system",""" You are a helpful document assistant.
     Answer the question using ONLY the provided document context.
     Do not make up information.If the answer is not present in the documents, say:
    "I could not find this information in the uploaded documents."
     Use the provided source information when giving the answer."""),

    ("human","""Conversation history:{chat_history}
     Document context:{context}
     Question:{question}
     Answer:""")])