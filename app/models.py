from pydantic import BaseModel, Field, field_validator
from typing import Optional


class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    test_type: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str = Field(..., min_length=1)
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = Field(default=False)

    @field_validator("recommendations")
    @classmethod
    def recommendations_max_ten(cls, v):
        if len(v) > 10:
            raise ValueError("recommendations must not exceed 10 items")
        return v
