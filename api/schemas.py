"""Request and response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    status: Literal["ok"]
    chunks_indexed: int


class DocumentResponse(BaseModel):
    filename: str = Field(description="Sanitized name the document is stored and cited under")
    chunks: int = Field(description="Number of chunks indexed")


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(
        default=None, ge=1, le=10, description="Chunks to retrieve (server default if omitted)"
    )
    use_rag: bool = Field(default=True, description="False answers without retrieval (baseline)")

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class Source(BaseModel):
    source: str
    chunk: int
    score: float = Field(description="Cosine similarity, higher is more relevant")
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(description="Retrieved chunks, most relevant first")
