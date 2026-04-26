from fastapi import FastAPI
from pydantic import BaseModel

from recomender import recommend

app = FastAPI()

class UserRequest(BaseModel):
    text: str

@app.post("/recommend")
def get_recommendations(request: UserRequest):

    results = recommend(request.text)

    response = []

    for _, row in results.iterrows():
        response.append({
            "destination": row["Destination"],
            "country": row["Country"],
            "cluster": int(row["Cluster"])
        })

    return {
        "count": len(response),
        "recommendations": response
    }