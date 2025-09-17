from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# --- Configuración MongoDB ---
MONGO_URL = "mongodb+srv://danielespinosa_db_user:JQjLwebrE1GcLcQa@cluster0.t0sfc3k.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = AsyncIOMotorClient(MONGO_URL)
db = client["VigIAHackathon"]

# --- Modelos Pydantic ---
class SolicitudModel(BaseModel):

    SolicitudID: Optional[str] = Field(default_factory=lambda: str(ObjectId()))
    CodigoProyecto: str
    ProveedorNombre: str
    ProveedorNIT: str
    FechaCreacion: datetime
    EstadoGeneral: str
    UsuarioSolicitante: str
    FuenteExcelPath: Optional[str]
    StorageFolderPath: Optional[str]
    PuntajeConsolidado: Optional[float]
    NivelGlobal: Optional[str]
    FechaFinalizacion: Optional[datetime]

    # Nuevo campo Estado con propiedades economica, social y ambiental
    Estado: dict = Field(default_factory=lambda: {"economica": "", "social": "", "ambiental": ""})

    class Config:
        orm_mode = True

# --- Router FastAPI ---
router = APIRouter(prefix="/vigia", tags=["Vigia"])

# --- CRUD Solicitud ---
@router.post("/solicitud", response_model=SolicitudModel)
async def create_solicitud(solicitud: SolicitudModel):
    data = solicitud.dict()
    await db.Solicitud.insert_one(data)
    return solicitud

@router.get("/solicitud/{solicitud_id}", response_model=SolicitudModel)
async def get_solicitud(solicitud_id: str):
    doc = await db.Solicitud.find_one({"SolicitudID": solicitud_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Solicitud not found")
    return SolicitudModel(**doc)

@router.get("/solicitudes", response_model=List[SolicitudModel])
async def list_solicitudes():
    solicitudes = []
    async for doc in db.Solicitud.find():
        solicitudes.append(SolicitudModel(**doc))
    return solicitudes

@router.put("/solicitud/{solicitud_id}", response_model=SolicitudModel)
async def update_solicitud(solicitud_id: str, solicitud: SolicitudModel):
    result = await db.Solicitud.replace_one({"SolicitudID": solicitud_id}, solicitud.dict())
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Solicitud not found")
    return solicitud

@router.delete("/solicitud/{solicitud_id}")
async def delete_solicitud(solicitud_id: str):
    result = await db.Solicitud.delete_one({"SolicitudID": solicitud_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Solicitud not found")
    return {"detail": "Solicitud deleted"}

# ... Puedes replicar la estructura para las otras entidades ...