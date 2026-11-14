from pydantic import BaseModel, Field
from typing import Optional
class PumpRecord(BaseModel):
    fPeak: float; pLow: float; pMid: float; pHigh: float; pKurtosis: float
    qMean: float; qVar: float; qSkewness: float; qKurtosis: float
    qPeak2Peak: float; qCrest: float; qRMS: float; qMAD: float; qCSRange: float
    failure: Optional[int] = Field(default=None, ge=0, le=1)
    scenario: Optional[str] = None
    severity: Optional[float] = None
    elapsed_seconds: Optional[int] = None
