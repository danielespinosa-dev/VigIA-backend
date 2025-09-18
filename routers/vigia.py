from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import os
import httpx
import asyncio
import pandas as pd
from io import BytesIO
from services.openai_assistant import OpenAIAssistant
from dotenv import load_dotenv
from enum import Enum

class TipoAsistenteEnum(str, Enum):
    ambiental = "ambiental"
    social = "social"
    economica = "economica"

load_dotenv()


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
    FuenteExcelPath: Optional[str] = None
    Anexos: List[str] = Field(default_factory=list)
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
    class Config:
        orm_mode = True

# --- Router FastAPI ---
router = APIRouter(prefix="/vigia", tags=["Vigia"])

# --- CRUD Solicitud ---
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
    print(f"Cuestionario extraído: {cuestionario_csv}")

    # Configuración del assistant
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_ASSISTANT_ID_AMBIENTAL = os.getenv("OPENAI_ASSISTANT_ID_AMBIENTAL")
    OPENAI_ASSISTANT_ID_SOCIAL = os.getenv("OPENAI_ASSISTANT_ID_SOCIAL")
    OPENAI_ASSISTANT_ID_ECONOMICA = os.getenv("OPENAI_ASSISTANT_ID_ECONOMICA")

    print(f"OPENAI_API_KEY: {OPENAI_API_KEY}, ASSISTANT_ID_AMBIENTAL: {OPENAI_ASSISTANT_ID_AMBIENTAL}, ASSISTANT_ID_SOCIAL: {OPENAI_ASSISTANT_ID_SOCIAL}, ASSISTANT_ID_ECONOMICA: {OPENAI_ASSISTANT_ID_ECONOMICA}")  # Log para depuración
    assistant_ambiental = OpenAIAssistant(api_key=OPENAI_API_KEY, assistant_id=OPENAI_ASSISTANT_ID_AMBIENTAL)
    assistant_social = OpenAIAssistant(api_key=OPENAI_API_KEY, assistant_id=OPENAI_ASSISTANT_ID_SOCIAL)
    assistant_economica = OpenAIAssistant(api_key=OPENAI_API_KEY, assistant_id=OPENAI_ASSISTANT_ID_ECONOMICA)

    # Subir archivo Excel
    #excel_file.file.seek(0)
    #excel_upload = await assistant_ambiental.upload_file_from_formdata(excel_file, excel_file.filename)
    #excel_file_id = excel_upload["id"] if excel_upload else None

    # Subir anexos y obtener sus IDs
    anexos_ids = []
    if anexos:
        for anexo in anexos:
            anexo_upload = await assistant_ambiental.upload_file_from_formdata(anexo, anexo.filename)
            if anexo_upload:
                anexos_ids.append(anexo_upload["id"])

    #print(f"excel_file_id: {excel_file_id}")
    print(f"anexos_ids: {anexos_ids}")

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
    print(f"Solicitud creada con ID: {solicitud.SolicitudID}")
    
    # Procesar el mensaje y assistant de forma asíncrona (no espera respuesta)
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

async def procesar_solicitud_con_assistant(solicitud: SolicitudModel, anexos_ids: list, assistant: OpenAIAssistant, tipo_asistente: TipoAsistenteEnum):
    mensaje = (
        f"Solicitud creada para el proyecto {solicitud.CodigoProyecto}.\n"
        f"Proveedor: {solicitud.ProveedorNombre} (NIT: {solicitud.ProveedorNIT}).\n"
        #f"Archivo Excel ID: {excel_file_id}\n"
        f"IDs de anexos: {', '.join(anexos_ids) if anexos_ids else 'Ninguno'}"
        f"Datos del formulario: {solicitud.Cuestionario}" if solicitud.Cuestionario else "No hay datos de formulario."
        f"\nPor favor, realiza la evaluación {tipo_asistente.value} correspondiente y responde con las observaciones y recomendaciones."
    )
    print(f"mensaje: {mensaje}")
    #file_ids = [excel_file_id] if excel_file_id else []
    #file_ids.extend(anexos_ids if anexos_ids else [])

    max_retries = 3
    retries = 0
    required_actions = []

    current_message = mensaje
    current_file_ids = anexos_ids

    while retries < max_retries:
        required_action = await assistant.run_assistant_flow(current_message, file_ids=current_file_ids)
        print(f"required_action (try {retries+1}): {required_action}")
        if required_action:
            required_actions.append(required_action)
            break
        # Si no hay required_action, preparar el siguiente intento
        current_message = (
            f"{mensaje}\n\nPor favor, responde ejecutando la función configurada en el assistant. Intento {retries+2}."
        )
        retries += 1

    # Actualiza los campos según el tipo de asistente
    if tipo_asistente == TipoAsistenteEnum.ambiental:
        solicitud.EvaluacionAmbiental = required_actions
        for ra in required_actions:
            if isinstance(ra, dict) and ra.get("assistant_response"):
                solicitud.RespuestaAmbiental = ra["assistant_response"]
                break
        else:
            solicitud.RespuestaAmbiental = ""
        solicitud.Estado["ambiental"] = "done" if required_actions else "failed"
    elif tipo_asistente == TipoAsistenteEnum.social:
        solicitud.EvaluacionSocial = required_actions
        for ra in required_actions:
            if isinstance(ra, dict) and ra.get("assistant_response"):
                solicitud.RespuestaSocial = ra["assistant_response"]
                break
        else:
            solicitud.RespuestaSocial = ""
        solicitud.Estado["social"] = "done" if required_actions else "failed"
    elif tipo_asistente == TipoAsistenteEnum.economica:
        solicitud.EvaluacionEconomica = required_actions
        for ra in required_actions:
            if isinstance(ra, dict) and ra.get("assistant_response"):
                solicitud.RespuestaEconomica = ra["assistant_response"]
                break
        else:
            solicitud.RespuestaEconomica = ""
        solicitud.Estado["economica"] = "done" if required_actions else "failed"

    await db.Solicitud.update_one({"SolicitudID": solicitud.SolicitudID}, {"$set": solicitud.dict()})
    
def extraer_cuestionario_excel(excel_file: UploadFile) -> Optional[list]:
    """
    Extrae el contenido de la hoja 'Cuestionario' de un archivo Excel recibido como UploadFile.
    Retorna una lista de diccionarios (JSON estructurado) o None si no existe la hoja.
    """
    try:
        contents = excel_file.file.read()
        df = pd.read_excel(BytesIO(contents), sheet_name="Cuestionario")
        return df.fillna("").to_dict(orient="records")
    except Exception as e:
        print(f"Error extrayendo hoja 'Cuestionario': {e}")
        return None
    
def extraer_cuestionario_csv(excel_file: UploadFile) -> Optional[str]:
    """
    Extrae el contenido de la hoja 'Cuestionario' de un archivo Excel recibido como UploadFile
    y lo retorna como un string en formato CSV. Retorna None si no existe la hoja.
    """
    try:
        contents = excel_file.file.read()
        df = pd.read_excel(BytesIO(contents), sheet_name="Cuestionario")
        from io import StringIO
        output = StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()
    except Exception as e:
        print(f"Error extrayendo hoja 'Cuestionario' como CSV: {e}")
        return None