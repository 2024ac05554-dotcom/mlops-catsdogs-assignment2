from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    label: str = Field(..., description="Predicted class: 'cat' or 'dog'")
    probabilities: dict = Field(..., description="Class probabilities")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
