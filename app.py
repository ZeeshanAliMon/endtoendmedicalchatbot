import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import PrivateAttr
from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from logging import Logger
from groq import Groq
from src.prompt import system_prompt
from langchain_core.language_models.llms import LLM
from dotenv import load_dotenv
load_dotenv()
# FastAPI app
app = FastAPI()

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
from typing import List, Dict
class GroqLLM(LLM):
    model_name: str = "llama-3.3-70b-versatile"

    _messages: List[Dict[str, str]] = PrivateAttr()
    _client: Groq = PrivateAttr()

    def __init__(self, system_prompt: str):
        super().__init__()
        self._messages = [
            {"role": "system", "content": system_prompt}
        ]
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        self._messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=self._messages,
        )
        return response.choices[0].message.content

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

    @property
    def _llm_type(self):
        return "groq"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
index_name = "medicalbot"
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


retriever = vectordb.as_retriever(search_kwargs={"k": 3})
# 3️⃣ Ask function
def ask(query: str):
    llm = GroqLLM(system_prompt)
    docs =  retriever.invoke(query)

    context = "\n\n".join(d.page_content for d in docs)
    # print(context)
    prompt = f"""

Context:
{context}

Question:
{query}

"""
    return  llm.invoke(prompt)


@app.post("/chat/")
async def chat_endpoint(request: Request):
    
    data =  await request.json()
    Logger.info(data "received")
    chat_input = data["chatInput"]
    if not chat_input:
        return {"reply": "No input provided"}
    
    reply = ask(chat_input)
    
    return {"reply": reply}
