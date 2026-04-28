from fastapi import FastAPI
from pydantic import BaseModel
from recomender import recommend

app = FastAPI()

class UserRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def get_recommendations(request: UserRequest):

    results = recommend(request.text)

    return {
        "count": len(results),
        "recommendations": results.to_dict(orient="records")
    }