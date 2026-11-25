from typing import Optional, Literal
from pydantic import BaseModel, Field

class TrafficState(BaseModel):
    north: int = Field(20, ge=0, le=1000)
    south: int = Field(20, ge=0, le=1000)
    east: int = Field(20, ge=0, le=1000)
    west: int = Field(20, ge=0, le=1000)
    ns_green: int = Field(30, ge=5, le=120)
    ew_green: int = Field(30, ge=5, le=120)
    yellow: int = Field(3, ge=2, le=10)
    duration: int = Field(180, ge=30, le=3600)
    start_phase: Literal["NS", "EW"] = "NS"
    closed_road: Literal["none", "N", "S", "E", "W"] = "none"
    emergency_approach: Literal["none", "N", "S", "E", "W"] = "none"

class SituationRequest(BaseModel):
    state: TrafficState
    description: str = ""

class Strategy(BaseModel):
    name: str
    description: str
    ns_green: int = Field(30, ge=5, le=120)
    ew_green: int = Field(30, ge=5, le=120)

class ExperimentRequest(BaseModel):
    state: TrafficState
    strategies: list[Strategy] = Field(default_factory=list)
    description: str = ""
    apply_strategy: Optional[str] = None

class ApplyStrategyRequest(BaseModel):
    strategy: Strategy
