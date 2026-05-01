from fastapi import FastAPI
from pydantic import BaseModel
from app.recomender import recommend

app = FastAPI()

class UserRequest(BaseModel):
    text: str
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def get_recommendations(request: UserRequest):
    results = recommend(request.text)

    return {
        "session_id": request.session_id,
        "count": len(results),
        "recommendations": results[["Destination", "Country"]].to_dict(orient="records")
    }