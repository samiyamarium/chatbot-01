#import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents import answer_question

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: QuestionRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        answer, sources = answer_question(request.question)
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#if __name__ == "__main__":
 #   port = int(os.environ.get("PORT", 8000))  # Railway sets PORT automatically
  #  import uvicorn
   # uvicorn.run(app, host="0.0.0.0", port=port)
