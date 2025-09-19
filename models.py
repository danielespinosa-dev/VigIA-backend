from typing import Optional
from pydantic import BaseModel
from enum import Enum


class MsgPayload(BaseModel):
    msg_id: Optional[int]
    msg_name: str

class TipoAsistenteEnum(str, Enum):
    ambiental = "ambiental"
    social = "social"
    economica = "economica"