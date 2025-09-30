from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from flask import json
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
    CuestionarioAmbiental: Optional[str] = None
    CuestionarioSocial: Optional[str] = None
    CuestionarioEconomica: Optional[str] = None
    Analisis: Optional[str] = None
    class Config:
        from_attributes = True  # Pydantic v2

# ...existing code...
def extraer_cuestionario_csv(excel_file: UploadFile) -> Optional[str]:
    """
    Extrae el contenido de la hoja 'Cuestionario' de un archivo Excel recibido como UploadFile,
    omite saltos de línea en los valores de las celdas y retorna un string CSV entendible para OpenAI.
    Retorna None si no existe la hoja.
    """
    try:
        contents = excel_file.file.read()
        # Leer la hoja 'Cuestionario' desde la fila 4 (skiprows=3)
        df = pd.read_excel(BytesIO(contents), sheet_name="Cuestionario", skiprows=3)
        # Seleccionar solo las columnas A a P (índices 0 a 15)
        df = df.iloc[:, 0:16]
        # Reemplazar saltos de línea en todas las celdas por espacios
        df = df.applymap(lambda x: str(x).replace('\n', ' ').replace('\r', ' ') if pd.notnull(x) else "")
        output = StringIO()
        # Generar CSV sin index y con separador coma
        df.to_csv(output, index=False, lineterminator='\n')
        return output.getvalue()
    except Exception as e:
        print(f"[Vigia] Error extrayendo hoja 'Cuestionario' como CSV: {e}")
        return None
# ...existing code...
# ...existing code...

# ...existing code...

def extraer_cuestionario_json_str(excel_file: UploadFile) -> Optional[list]:
    """
    Extrae el contenido de la hoja 'Cuestionario' de un archivo Excel recibido como UploadFile,
    omite y escapa saltos de línea en los nombres de los campos y en los valores de las celdas,
    y retorna un string JSON agrupando las filas que pertenecen al mismo grupo (por ejemplo, misma dimensión).
    Retorna None si no existe la hoja.
    """
    try:
        contents = excel_file.file.read()
        df = pd.read_excel(BytesIO(contents), sheet_name="Cuestionario", skiprows=3)
        df = df.iloc[:, 0:16]
        # Normaliza y escapa saltos de línea en los nombres de las columnas
        df.columns = [
            str(col).replace('\n', ' ').replace('\r', ' ').strip().lower().replace(" ", "_")
            for col in df.columns
        ]
        # Escapa saltos de línea en los valores de las celdas
        df = df.applymap(lambda x: str(x).replace('\n', ' ').replace('\r', ' ') if pd.notnull(x) else "")
        # Agrupar por 'dimensión' (columna A normalizada)
        agrupado = {}
        for _, row in df.iterrows():
            dimension = row.get('dimensión', 'Sin dimensión')
            if dimension not in agrupado:
                agrupado[dimension] = []
            fila = {k: v for k, v in row.items() if k != 'dimensión'}
            agrupado[dimension].append(fila)
        resultado = [
            {"dimension": dimension, "items": items}
            for dimension, items in agrupado.items()
        ]
        # return json.dumps(resultado, ensure_ascii=False)
        return depurarPreguntas(resultado)
    except Exception as e:
        print(f"[Vigia] Error extrayendo hoja 'Cuestionario' como JSON agrupado: {e}")
        return None
# ...existing code...
# ...existing code...
def depurarPreguntas(data: list) -> list:
    """
    Recibe una lista de objetos con estructura [{"dimension": str, "items": [dict, ...]}, ...]
    Elimina de cada item los campos indicados y renombra los campos de calificación, soportes y justificación.
    Retorna la lista depurada.
    """
    campos_excluir = {
        "opciones_de_respuesta",
        "puntaje_respuesta",
        "peso_criterio",
        "puntaje_del_criterio",
        "puntaje_de_la_dimensión",
        "peso_dimensión",
        "puntaje_final"
    }
    campo_renombrar = "calificación_asigne_en_la_columna_el_puntaje_de_la_respuesta_que_más_se_ajusta_a_la_realidad_de_tu_empresa."
    nuevo_nombre = "calificacion_por_proveedor"

    campo_renombrar1 = "soportes_aplicables_para_justificar_respuesta_estos_son_algunos_ejemplos_de_los_soportes_que_puedes_anexar_para_comprobar_la_respuesta_seleccionada."
    nuevo_nombre1 = "soportes"

    campo_renombrar2 = "justificación_explica_brevemente_lo_que_la_empresa_realiza_acorde_a_la_respuesta_seleccionada."
    nuevo_nombre2 = "justificacion_por_proveedor"

    resultado = []
    for bloque in data:
        nueva_items = []
        for item in bloque.get("items", []):
            nuevo_item = {}
            for k, v in item.items():
                if k in campos_excluir:
                    continue
                if k == campo_renombrar:
                    nuevo_item[nuevo_nombre] = v
                elif k == campo_renombrar1:
                    nuevo_item[nuevo_nombre1] = v
                elif k == campo_renombrar2:
                    nuevo_item[nuevo_nombre2] = v
                else:
                    nuevo_item[k] = v
            nueva_items.append(nuevo_item)
        resultado.append({
            "dimension": bloque.get("dimension", ""),
            "items": nueva_items
        })
    return resultado
# ...existing code...

async def procesar_solicitud_con_assistant(
    solicitud: SolicitudModel,
    anexos_ids: list,
    assistant: OpenAIAssistant,
    tipo_asistente: TipoAsistenteEnum
):
    cuestionario=solicitud.CuestionarioAmbiental if tipo_asistente == TipoAsistenteEnum.ambiental else solicitud.CuestionarioSocial if tipo_asistente == TipoAsistenteEnum.social else solicitud.CuestionarioEconomica
    
    # Formatear anexos para el mensaje
    if anexos_ids:
        anexos_str = ', '.join([f"{a['filename']} (ID: {a['id']})" for a in anexos_ids])
    else:
        anexos_str = 'Ninguno'
    mensaje = (
        f"Solicitud creada para el proyecto {solicitud.CodigoProyecto}.\n"
        f"Proveedor: {solicitud.ProveedorNombre} (NIT: {solicitud.ProveedorNIT}).\n"
        f"Anexos: {anexos_str}.\n"
        f"Por favor, realiza la evaluación {tipo_asistente.value} correspondiente y responde con las observaciones y recomendaciones."
        f"Datos del formulario: {cuestionario if cuestionario else 'No hay datos de formulario.'}\n"  
    )
    print(f"[Vigia] Mensaje Assistant: {mensaje}")
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
        solicitud.FechaFinalizacion = datetime.utcnow()
        solicitud.EstadoGeneral = "completado"
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
    cuestionario_csv = extraer_cuestionario_json_str(excel_file)
    # ...existing code...
    cuestionario_ambiental = [item for item in cuestionario_csv if "ambiental" in item.get("dimension", "").lower()]
    cuestionario_social = [item for item in cuestionario_csv if "social" in item.get("dimension", "").lower()]
    cuestionario_economica = [item for item in cuestionario_csv if "económica" in item.get("dimension", "").lower() or "gobernanza" in item.get("dimension", "").lower()]
    # ...existing code... 

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
            anexo_upload = await assistant_ambiental.upload_file_from_formdata_v2(anexo, anexo.filename)
            if anexo_upload:
                anexos_ids.append({"id": anexo_upload["id"], "filename": anexo.filename})

    # Guardar la solicitud en la base de datos
    solicitud = SolicitudModel(
        CodigoProyecto=CodigoProyecto,
        ProveedorNombre=ProveedorNombre,
        ProveedorNIT=ProveedorNIT,
        FechaCreacion=datetime.utcnow(),
        EstadoGeneral="En progreso",
        UsuarioSolicitante=UsuarioSolicitante,
        FuenteExcelPath=excel_file.filename,
        Anexos=anexos_ids,
        Estado={"economica": "pending", "social": "pending", "ambiental": "pending"},
        Cuestionario=json.dumps(cuestionario_csv, ensure_ascii=False),
        CuestionarioAmbiental=json.dumps(cuestionario_ambiental, ensure_ascii=False),
        CuestionarioSocial=json.dumps(cuestionario_social, ensure_ascii=False),
        CuestionarioEconomica=json.dumps(cuestionario_economica, ensure_ascii=False)
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