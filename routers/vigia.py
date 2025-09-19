from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import os
import asyncio
import pandas as pd
from io import BytesIO, StringIO
from models import TipoAsistenteEnum
from services.openai_assistant import OpenAIAssistant
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# --- Configuración MongoDB ---
MONGO_URL = os.getenv("MONGO_URL")
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
    FuenteExcelPath: Optional[str] = None
    Anexos: List[dict] = Field(default_factory=list)
    StorageFolderPath: Optional[str] = None
    PuntajeConsolidado: Optional[float] = None
    NivelGlobal: Optional[str] = None
    FechaFinalizacion: Optional[datetime] = None
    Estado: dict = Field(default_factory=lambda: {"economica": "", "social": "", "ambiental": ""})
    EvaluacionAmbiental: Optional[List[dict]] = None
    EvaluacionSocial: Optional[List[dict]] = None
    EvaluacionEconomica: Optional[List[dict]] = None
    RespuestaAmbiental: Optional[str] = None
    RespuestaSocial: Optional[str] = None
    RespuestaEconomica: Optional[str] = None
    Cuestionario: Optional[str] = None
    Analisis: Optional[str] = None
    class Config:
        from_attributes = True  # Pydantic v2

# --- Utilidades ---
def extraer_cuestionario_csv(excel_file: UploadFile) -> Optional[str]:
    """
    Extrae el contenido de la hoja 'Cuestionario' de un archivo Excel recibido como UploadFile
    y lo retorna como un string en formato CSV. Retorna None si no existe la hoja.
    """
    try:
        contents = excel_file.file.read()
        df = pd.read_excel(BytesIO(contents), sheet_name="Cuestionario")
        output = StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()
    except Exception as e:
        print(f"[Vigia] Error extrayendo hoja 'Cuestionario' como CSV: {e}")
        return None

async def procesar_solicitud_con_assistant(
    solicitud: SolicitudModel,
    anexos_ids: list,
    assistant: OpenAIAssistant,
    tipo_asistente: TipoAsistenteEnum
):
    # Formatear anexos para el mensaje
    if anexos_ids:
        anexos_str = ', '.join([f"{a['filename']} (ID: {a['id']})" for a in anexos_ids])
    else:
        anexos_str = 'Ninguno'
    mensaje = (
        f"Solicitud creada para el proyecto {solicitud.CodigoProyecto}.\n"
        f"Proveedor: {solicitud.ProveedorNombre} (NIT: {solicitud.ProveedorNIT}).\n"
        f"Anexos: {anexos_str}.\n"
        f"Datos del formulario: {solicitud.Cuestionario if solicitud.Cuestionario else 'No hay datos de formulario.'}\n"
        f"Por favor, realiza la evaluación {tipo_asistente.value} correspondiente y responde con las observaciones y recomendaciones."
    )

    max_retries = 3
    retries = 0
    required_actions = []
    current_message = mensaje
    # Solo IDs para assistant
    current_file_ids = [a["id"] for a in anexos_ids]

    while retries < max_retries:
        required_action = await assistant.run_assistant_flow(
            current_message,
            file_ids=current_file_ids,
            tipo_asistente=tipo_asistente
        )
        if required_action:
            required_actions.append(required_action)
            break
        current_message = (
            f"{mensaje}\n\nPor favor, responde ejecutando la función configurada en el assistant. Intento {retries+2}."
        )
        retries += 1

    # Actualiza los campos según el tipo de asistente
    if tipo_asistente == TipoAsistenteEnum.ambiental:
        solicitud.EvaluacionAmbiental = required_actions
        solicitud.RespuestaAmbiental = next(
            (ra["assistant_response"] for ra in required_actions if isinstance(ra, dict) and ra.get("assistant_response")), ""
        )
        solicitud.Estado["ambiental"] = "done" if required_actions else "failed"
    elif tipo_asistente == TipoAsistenteEnum.social:
        solicitud.EvaluacionSocial = required_actions
        solicitud.RespuestaSocial = next(
            (ra["assistant_response"] for ra in required_actions if isinstance(ra, dict) and ra.get("assistant_response")), ""
        )
        solicitud.Estado["social"] = "done" if required_actions else "failed"
    elif tipo_asistente == TipoAsistenteEnum.economica:
        solicitud.EvaluacionEconomica = required_actions
        solicitud.RespuestaEconomica = next(
            (ra["assistant_response"] for ra in required_actions if isinstance(ra, dict) and ra.get("assistant_response")), ""
        )
        solicitud.Estado["economica"] = "done" if required_actions else "failed"

    await db.Solicitud.update_one({"SolicitudID": solicitud.SolicitudID}, {"$set": solicitud.dict()})
    doc = await db.Solicitud.find_one({"SolicitudID": solicitud.SolicitudID})
    solicitud = SolicitudModel(**doc) 
    if (
        solicitud.RespuestaAmbiental
        and solicitud.RespuestaSocial
        and solicitud.RespuestaEconomica
    ):
        await assistant.depureFiles()
        # analisis =await assistant.analizar_solicitud_completions(solicitud)
        # solicitud.Analisis=analisis
        await db.Solicitud.update_one({"SolicitudID": solicitud.SolicitudID}, {"$set": solicitud.dict()})
    
    print(f"[Vigia] Solicitud {solicitud.SolicitudID} actualizada tras evaluación {tipo_asistente.value}")

# --- Router FastAPI ---
router = APIRouter(prefix="/vigia", tags=["Vigia"])

@router.post("/solicitud", response_model=SolicitudModel)
async def create_solicitud(
    CodigoProyecto: str = Form(...),
    ProveedorNombre: str = Form(...),
    ProveedorNIT: str = Form(...),
    EstadoGeneral: str = Form(...),
    UsuarioSolicitante: str = Form(...),
    excel_file: UploadFile = File(...),
    anexos: List[UploadFile] = File(None)
):
    # Extraer cuestionario del Excel
    cuestionario_csv = extraer_cuestionario_csv(excel_file)

    # Configuración del assistant
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_ASSISTANT_ID_AMBIENTAL = os.getenv("OPENAI_ASSISTANT_ID_AMBIENTAL")
    OPENAI_ASSISTANT_ID_SOCIAL = os.getenv("OPENAI_ASSISTANT_ID_SOCIAL")
    OPENAI_ASSISTANT_ID_ECONOMICA = os.getenv("OPENAI_ASSISTANT_ID_ECONOMICA")

    assistant_ambiental = OpenAIAssistant(api_key=OPENAI_API_KEY, assistant_id=OPENAI_ASSISTANT_ID_AMBIENTAL)
    assistant_social = OpenAIAssistant(api_key=OPENAI_API_KEY, assistant_id=OPENAI_ASSISTANT_ID_SOCIAL)
    assistant_economica = OpenAIAssistant(api_key=OPENAI_API_KEY, assistant_id=OPENAI_ASSISTANT_ID_ECONOMICA)

    # Subir anexos y obtener sus IDs y nombres
    anexos_ids = []
    if anexos:
        for anexo in anexos:
            anexo_upload = await assistant_ambiental.upload_file_from_formdata(anexo, anexo.filename)
            if anexo_upload:
                anexos_ids.append({"id": anexo_upload["id"], "filename": anexo.filename})

    # Guardar la solicitud en la base de datos
    solicitud = SolicitudModel(
        CodigoProyecto=CodigoProyecto,
        ProveedorNombre=ProveedorNombre,
        ProveedorNIT=ProveedorNIT,
        FechaCreacion=datetime.utcnow(),
        EstadoGeneral=EstadoGeneral,
        UsuarioSolicitante=UsuarioSolicitante,
        FuenteExcelPath=excel_file.filename,
        Anexos=anexos_ids,
        Estado={"economica": "pending", "social": "pending", "ambiental": "pending"},
        Cuestionario=cuestionario_csv
    )
    await db.Solicitud.insert_one(solicitud.dict())
    print(f"[Vigia] Solicitud creada con ID: {solicitud.SolicitudID}")

    # Procesar los asistentes de forma asíncrona
    asyncio.create_task(procesar_solicitud_con_assistant(solicitud, anexos_ids, assistant_ambiental, TipoAsistenteEnum.ambiental))
    asyncio.create_task(procesar_solicitud_con_assistant(solicitud, anexos_ids, assistant_social, TipoAsistenteEnum.social))
    asyncio.create_task(procesar_solicitud_con_assistant(solicitud, anexos_ids, assistant_economica, TipoAsistenteEnum.economica))

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