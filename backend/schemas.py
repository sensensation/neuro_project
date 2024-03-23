from pydantic import BaseModel


class TrainResult(BaseModel):
    loss: float
    mae: float
