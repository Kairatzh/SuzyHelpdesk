from typing import List, Dict, Optional
from pydantic import BaseModel, Field

"""Создать подсостояние и работать соединяя между собой при дальнейших улучшении агентную систему"""

# Global state
class State(BaseModel):
    query: Optional[str] = None
    context: List[Dict[str, str]] = Field(default_factory=list)
    answer: Optional[str] = None
    docs: Optional[str] = None
    user_id: Optional[str] = None
