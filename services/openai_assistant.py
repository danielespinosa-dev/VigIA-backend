from flask import json
import httpx
import asyncio
from typing import Optional, Dict, Any, List
from models import TipoAsistenteEnum

class OpenAIAssistant:
    def __init__(self, api_key: str, assistant_id: str):
        self.api_key = api_key
        self.assistant_id = assistant_id
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }

    async def create_thread(self) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/threads",
                headers=self.headers
            )
            response.raise_for_status()
            thread_id = response.json()["id"]
            print(f"[OpenAI] Thread creado: {thread_id}")
            return thread_id

    async def create_message(self, thread_id: str, content: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/threads/{thread_id}/messages",
                headers=self.headers,
                json={"role": "user", "content": content}
            )
            response.raise_for_status()
            message_id = response.json()["id"]
            print(f"[OpenAI] Mensaje creado en thread {thread_id}: {message_id}")
            return message_id

    async def create_message_with_files(self, thread_id: str, content: str, file_ids: Optional[List[str]]) -> Optional[str]:
        """
        Crea un mensaje en el hilo del asistente incluyendo archivos adjuntos.
        Captura y loguea cualquier excepción, devolviendo None en caso de error.
        """
        try:
            attachments = []
            if file_ids:
                for fid in file_ids:
                    attachments.append({
                        "file_id": fid,
                        "tools": [{"type": "code_interpreter"}]
                    })
            message_payload = {
                "role": "user",
                "content": content,
                "attachments": attachments
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/threads/{thread_id}/messages",
                    headers=self.headers,
                    json=message_payload
                )
                response.raise_for_status()
                message_id = response.json()["id"]
                print(f"[OpenAI] Mensaje con archivos creado en thread {thread_id}: {message_id}")
                return message_id
        except Exception as e:
            print(f"[OpenAI][ERROR] create_message_with_files Unexpected error: {str(e)}")
            return None

    async def create_run(self, thread_id: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/threads/{thread_id}/runs",
                headers=self.headers,
                json={"assistant_id": self.assistant_id}
            )
            response.raise_for_status()
            run_id = response.json()["id"]
            print(f"[OpenAI] Run creado en thread {thread_id}: {run_id}")
            return run_id

    async def get_run_status(self, thread_id: str, run_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/threads/{thread_id}/runs/{run_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()

    async def wait_for_required_action(
        self,
        thread_id: str,
        run_id: str,
        tipo_asistente: TipoAsistenteEnum,
        interval: float = 10.0,
        timeout: float = 10000.0
    ) -> Optional[Dict[str, Any]]:
        elapsed = 0
        required_action_detected = False
        required_action_response = None
        while elapsed < timeout:
            run_status = await self.get_run_status(thread_id, run_id)
            status = run_status.get("status")
            if status == "requires_action" and run_status.get("required_action"):
                required_action_detected = True
                required_action = run_status["required_action"]
                required_action_response = required_action
                tool_calls = required_action.get("submit_tool_outputs", {}).get("tool_calls", [])
                tool_outputs = [
                    {
                        "tool_call_id": call["id"],
                        "output": "Ok, función ejecutada correctamente."
                    }
                    for call in tool_calls
                ]
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/threads/{thread_id}/runs/{run_id}/submit_tool_outputs",
                        headers=self.headers,
                        json={"tool_outputs": tool_outputs}
                    )
                    response.raise_for_status()
                print(f"[OpenAI] Acción requerida completada en run {run_id} ({tipo_asistente.value})")
                #return {
                #    "required_action": required_action_response,
                #    "assistant_response": None
                #}
            if status == "completed":
                if not required_action_detected:
                    retry_message = (
                        "Por favor, ejecuta la función configurada en el assistant y entrega el resultado de la revisión."
                    )
                    await self.create_message(thread_id, retry_message)
                    new_run_id = await self.create_run(thread_id)
                    print(f"[OpenAI] Run adicional creado en thread {thread_id}: {new_run_id} (no hubo required_action inicial)")
                    return await self.wait_for_required_action(thread_id, new_run_id, tipo_asistente, interval, timeout)
                assistant_response = await self.get_completed_run_response(thread_id, run_id)
                print(f"[OpenAI] Run completado en thread {thread_id}: {run_id}")
                return {
                    "required_action": required_action_response,
                    "assistant_response": assistant_response
                }
            if status in ["failed", "cancelled"]:
                print(f"[OpenAI] Run {run_id} fallido o cancelado ({status})")
                return {
                    "required_action": required_action_response,
                    "assistant_response": status
                }
            await asyncio.sleep(interval)
            elapsed += interval
        print(f"[OpenAI] wait_for_required_action Timeout esperando required_action o completion en run {run_id}")
        raise TimeoutError("wait_for_required_action Run did not reach required_action or completed state in time.")

    async def get_completed_run_response(self, thread_id: str, run_id: str) -> Optional[str]:
        """
        Consulta la respuesta del asistente cuando el run está completado.
        Retorna el contenido de texto del último mensaje del asistente o None si no hay respuesta.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/threads/{thread_id}/messages",
                headers=self.headers
            )
            response.raise_for_status()
            messages = response.json().get("data", [])
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content")
                    if isinstance(content, list):
                        texts = []
                        for c in content:
                            if c.get("type") == "text":
                                text_obj = c.get("text")
                                if isinstance(text_obj, dict):
                                    texts.append(text_obj.get("value", ""))
                                elif isinstance(text_obj, str):
                                    texts.append(text_obj)
                        return " ".join(texts)
                    elif isinstance(content, str):
                        return content
            return None

    async def run_assistant_flow(
        self,
        user_message: str,
        tipo_asistente: TipoAsistenteEnum,
        file_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Ejecuta el flujo completo: crea hilo, mensaje (con archivos si hay), run y espera el llamado a función.
        Retorna el required_action si se dispara, None si termina sin requerir acción.
        """
        try:
            thread_id = await self.create_thread()
            if file_ids:
                await self.create_message_with_files(thread_id, "Estos son los archivos que debes revisar", file_ids)
            await self.create_message(thread_id, user_message)
            run_id = await self.create_run(thread_id)
            result = await self.wait_for_required_action(thread_id, run_id, tipo_asistente=tipo_asistente)
            return result
        except httpx.HTTPStatusError as e:
            print(f"[OpenAI][ERROR] run_assistant_flow {tipo_asistente.value} {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"[OpenAI][ERROR] run_assistant_flow Unexpected error: {str(e)}")
            return None

    async def upload_file_from_formdata(self, file, filename: str, purpose: str = "assistants") -> Optional[Dict[str, Any]]:
        """
        Sube un archivo recibido como FormData (por ejemplo, desde FastAPI) al API de OpenAI.
        """
        try:
            async with httpx.AsyncClient() as client:
                files = {"file": (filename, await file.read(), "application/octet-stream")}
                data = {"purpose": purpose}
                response = await client.post(
                    f"{self.base_url}/files",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "OpenAI-Beta": "assistants=v2"
                    },
                    data=data,
                    files=files
                )
                response.raise_for_status()
                file_id = response.json().get("id")
                print(f"[OpenAI] Archivo subido: {file_id} ({filename})")
                return response.json()
        except httpx.HTTPStatusError as e:
            print(f"[OpenAI][ERROR] Upload file:{filename} {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"[OpenAI][ERROR] {filename} Unexpected error al subir archivo: {str(e)}")
            return None
        
    async def depureFiles(self):
        """
        Consulta todos los archivos en OpenAI y los elimina uno por uno.
        """
        try:
            async with httpx.AsyncClient() as client:
                # Obtener la lista de archivos
                response = await client.get(
                    f"{self.base_url}/files",
                    headers=self.headers
                )
                response.raise_for_status()
                files = response.json().get("data", [])
                print(f"[OpenAI] Archivos encontrados: {len(files)}")
                # Eliminar cada archivo
                for file in files:
                    file_id = file.get("id")
                    if file_id:
                        del_response = await client.delete(
                            f"{self.base_url}/files/{file_id}",
                            headers=self.headers
                        )
                        if del_response.status_code == 204:
                            print(f"[OpenAI] Archivo eliminado: {file_id}")
                        else:
                            print(f"[OpenAI][ERROR] No se pudo eliminar archivo: {file_id} - {del_response.status_code}")
        except Exception as e:
            print(f"[OpenAI][ERROR] depureFiles Unexpected error: {str(e)}")
    
    async def analizar_solicitud_completions(self, solicitud: Any) -> Optional[str]:
        """
        Consume el endpoint de completions de OpenAI para analizar la solicitud y sus evaluaciones.
        Recibe la solicitud (con evaluaciones ambiental, social y económica) y retorna un análisis detallado.
        """
        # Serializa los campos complejos a texto
        eval_ambiental = json.dumps(solicitud.EvaluacionAmbiental, ensure_ascii=False, indent=2) if solicitud.EvaluacionAmbiental else "Sin evaluación"
        eval_social = json.dumps(solicitud.EvaluacionSocial, ensure_ascii=False, indent=2) if solicitud.EvaluacionSocial else "Sin evaluación"
        eval_economica = json.dumps(solicitud.EvaluacionEconomica, ensure_ascii=False, indent=2) if solicitud.EvaluacionEconomica else "Sin evaluación"

        prompt = (
            f"Analiza detalladamente la siguiente solicitud y sus resultados de evaluación.\n"
            f"Proyecto: {solicitud.CodigoProyecto}\n"
            f"Proveedor: {solicitud.ProveedorNombre} (NIT: {solicitud.ProveedorNIT})\n"
            f"Estado General: {solicitud.EstadoGeneral}\n"
            f"Evaluación Ambiental: {eval_ambiental}\n"
            f"Evaluación Social: {eval_social}\n"
            f"Evaluación Económica: {eval_economica}\n"
            f"Cuestionario: {solicitud.Cuestionario}\n"
            "Por favor, realiza un análisis integral, identifica riesgos, oportunidades y recomendaciones para el proyecto."
        )
        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {"role": "system", "content": "Eres un experto en análisis de proyectos."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.7
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            print(f"[OpenAI][ERROR] analizar_solicitud_completions: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"[OpenAI][ERROR] analizar_solicitud_completions: {str(e)}")
            return None