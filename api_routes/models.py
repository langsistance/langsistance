#!/usr/bin/env python3

from pydantic import BaseModel
from typing import List, Optional
from sources.schemas import QueryRequest, QueryResponse
from sources.knowledge.knowledge import KnowledgeItem, ToolItem

# Knowledge Models
class KnowledgeCreateRequest(BaseModel):
    question: str
    answer: str
    public: int
    toolId: int
    params: str
    modelName: Optional[str] = None
    description: Optional[str] = None

class KnowledgeCreateResponse(BaseModel):
    success: bool
    message: str
    id: Optional[int] = None

class KnowledgeDeleteRequest(BaseModel):
    knowledgeId: int

class KnowledgeDeleteResponse(BaseModel):
    success: bool
    message: str

class KnowledgeUpdateRequest(BaseModel):
    knowledgeId: int
    question: Optional[str] = None
    description: Optional[str] = None
    answer: Optional[str] = None
    public: Optional[int] = None
    modelName: Optional[str] = None
    toolId: Optional[int] = None
    params: Optional[str] = None

class KnowledgeUpdateResponse(BaseModel):
    success: bool
    message: str

class KnowledgeQueryResponse(BaseModel):
    success: bool
    message: str
    data: List[KnowledgeItem]
    total: int

class KnowledgeCopyRequest(BaseModel):
    knowledgeId: int

class KnowledgeCopyResponse(BaseModel):
    success: bool
    message: str
    id: Optional[int] = None

# Tool Models
class ToolAndKnowledgeCreateRequest(BaseModel):
    # Tool fields
    tool_title: str
    tool_description: str
    tool_url: str
    tool_push: int
    tool_timeout: int
    tool_params: str
    # Knowledge fields
    knowledge_question: str
    knowledge_description: str
    knowledge_answer: str
    knowledge_public: int
    knowledge_embeddingId: int
    knowledge_model_name: str
    knowledge_params: str

class ToolAndKnowledgeCreateResponse(BaseModel):
    success: bool
    message: str
    tool_id: Optional[int] = None
    knowledge_id: Optional[int] = None

class ToolUpdateRequest(BaseModel):
    toolId: int
    title: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    params: Optional[str] = None

class ToolUpdateResponse(BaseModel):
    success: bool
    message: str

class ToolDeleteRequest(BaseModel):
    toolId: int

class ToolDeleteResponse(BaseModel):
    success: bool
    message: str

class ToolQueryResponse(BaseModel):
    success: bool
    message: str
    data: List[ToolItem]
    total: int

# Query Models
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    similarity_threshold: Optional[float] = 0.7

class KnowledgeToolResponse(BaseModel):
    success: bool
    message: str
    knowledge: Optional[KnowledgeItem] = None
    tool: Optional[ToolItem] = None
    similarity: Optional[float] = None

class ToolFetchRequest(BaseModel):
    query_id: str

class ToolFetchResponse(BaseModel):
    success: bool
    message: str
    tool: Optional[dict] = None

class ToolResponseRequest(BaseModel):
    query_id: str
    tool_response: dict

class ToolResponseResponse(BaseModel):
    success: bool
    message: str


class OpenAPISpecRequest(BaseModel):
    """
    OpenAPI规范配置请求模型
    """
    spec_format: str  # "json" 或 "yaml"
    spec_content: str  # OpenAPI规范内容


class OpenAPISpecResponse(BaseModel):
    """
    OpenAPI规范配置响应模型
    """
    success: bool
    message: str
    tool_id: Optional[int] = None

class ToolCreateRequest(BaseModel):
    # Tool fields
    tool_title: str
    tool_description: str
    tool_url: str
    tool_params: str
    tool_timeout: Optional[int] = None

class ToolCreateResponse(BaseModel):
    success: bool
    message: str
    tool_id: Optional[int] = None

class UrlRequest(BaseModel):
    """
    url请求模型
    """
    url: str  # "json" 或 "yaml"