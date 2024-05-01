import datetime
import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from flask import Flask, request, jsonify

sys.path.append('../..')
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']
PORT = os.environ['QDRANT_PORT']


# LLM & Embeddings
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Vector DB
# Create a client instance connected to the local Qdrant server
qdrant_client = QdrantClient(host="localhost", port=PORT)

# FLASK
app = Flask(__name__)

def get_sys_prompt():
    # # Build prompt
    template = """Say I am a customer service assistant working for Smartpredict and invite him to ask about us if only we ask you about something that does not concern Smartpredict else don't say it.\
    Use the following pieces of context to answer the question. \
    Always Ensure your responses align with the values and goals of SmartPredict Services. \
    Alaways Maintain the same language as the follow up input message.
    Chat History:
    {chat_history}

    Follow Up Input: {question}
    Standalone question or instruction:"""
    QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=["chat_history", "question"])

    return QA_CHAIN_PROMPT

def db(qdrant_client, collection_name):
    embeddings = OpenAIEmbeddings()
    vectordb = Qdrant(
        client=qdrant_client, collection_name=collection_name, 
        embeddings=embeddings,
    )

    return vectordb 

def get_chain(vectordb, prompt):
    retriever=vectordb.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        # condense_question_prompt=prompt,
        # combine_docs_chain_kwargs={"prompt": prompt},
        # chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

@app.route('/query', methods=['POST'])
def query():
    chat_story=[]
    qa = get_chain(
        db(qdrant_client, "sp_documents"),
        get_sys_prompt()
    )
    data = request.json
    query = data.get('query')

    result = qa.invoke({"question": query, "chat_history": chat_story})

    return jsonify(result['answer']), 200

if __name__ == '__main__':
    app.run(debug=True)