class QueryRequest(BaseModel):
    """Request for RAG query."""

    question: str = Field(
        ...,
        description="Question to ask",
        min_length=1,
        max_length=1000,
    )
    include_sources: bool = Field(
        default=True,
        description="Include source documents in response",
    )
    enable_evaluation: bool = Field(
        default=False,
        description="Enable RAGAS evaluation (faithfulness, answer relevancy)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is RAG?",
                    "include_sources": True,
                    "enable_evaluation": False,
                }
            ]
        }
    }


class SourceDocument(BaseModel):
    """Source document information."""

    content: str = Field(..., description="Document content excerpt")
    metadata: dict[str, Any] = Field(..., description="Document metadata")


class QueryResponse(BaseModel):
    """Response for RAG query."""

    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: list[SourceDocument] | None = Field(
        None,
        description="Source documents used",
    )
    processing_time_ms: float = Field(
        ...,
        description="Query processing time in milliseconds",
    )
    evaluation: EvaluationScores | None = Field(
        None,
        description="RAGAS evaluation scores (if requested)",
    )

# ============== Error Schemas ==============


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""

    error: str = Field(default="Validation Error", description="Error type")
    message: str = Field(..., description="Error message")
    errors: list[dict] = Field(..., description="Validation errors")