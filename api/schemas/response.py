from pydantic import BaseModel
from typing import Optional

class PredictResponse(BaseModel):
    action: str
    score: Optional[float] = None
    model_type: str
    model_dir: str
