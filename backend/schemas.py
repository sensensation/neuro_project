from datetime import date
from pydantic import BaseModel

class PredictionData(BaseModel):
    query_date: date