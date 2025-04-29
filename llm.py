import anthropic
import os
from dotenv import load_dotenv
load_dotenv()

CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

client= anthropic.Anthropic(api_key=CLAUDE_API_KEY)

def llm_answer(context: str, question: str) -> str:
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1000,
        temperature=0.2,
        system="You are a helpful assistant that answers strictly from the given context. You will be given the context which is the result of a RAG system. It will also include the source information and the filename. Cite the sources in your final response as appropriate",
        messages=[
            {"role": "user","content": f"Context:\n{context}\n\nQ: {question}\nA:"}
        ]
    )
    return response.content[0].text.strip()