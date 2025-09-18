import httpx
import asyncio
from typing import Optional, Dict, Any

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
            return response.json()["id"]

    async def create_message(self, thread_id: str, content: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/threads/{thread_id}/messages",
                headers=self.headers,
                json={"role": "user", "content": content}
            )
            response.raise_for_status()
            return response.json()["id"]

    async def create_message_with_files(self, thread_id: str, content: str, file_ids: list) -> str:
        """
        Crea un mensaje en el hilo del asistente incluyendo archivos adjuntos.
        :param thread_id: ID del hilo.
        :param content: Texto del mensaje.
        :param file_ids: Lista de IDs de archivos subidos a OpenAI.
        :return: ID del mensaje creado.
        """
        message_payload = {
            "role": "user",
            "content": content,
            "attachments": [
                {
                    "file_id": fid,
                    "tools": [{"type": "code_interpreter"}]
                }
                for fid in file_ids
            ] if file_ids else []
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/threads/{thread_id}/messages",
                headers=self.headers,
                json=message_payload
            )
            response.raise_for_status()
            return response.json()["id"]

    async def create_run(self, thread_id: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/threads/{thread_id}/runs",
                headers=self.headers,
                json={"assistant_id": self.assistant_id}
            )
            response.raise_for_status()
            return response.json()["id"]

    async def get_run_status(self, thread_id: str, run_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/threads/{thread_id}/runs/{run_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()

    async def wait_for_required_action(
        self, thread_id: str, run_id: str, interval: float = 2.0, timeout: float = 120.0
    ) -> Optional[Dict[str, Any]]:
        elapsed = 0
        required_action_detected = False
        required_action_response = None
        while elapsed < timeout:
            run_status = await self.get_run_status(thread_id, run_id)
            print(f"get_run_status: {run_status.get('status')}")
            if run_status.get("status") == "requires_action" and run_status.get("required_action"):
                required_action_detected = True
                # Completar el llamado a función con una respuesta genérica
                required_action = run_status["required_action"]
                required_action_response = required_action
                print(f"required_action: {required_action}")
                tool_calls = required_action.get("submit_tool_outputs", {}).get("tool_calls", [])
                tool_outputs = []
                for call in tool_calls:
                    tool_outputs.append({
                        "tool_call_id": call["id"],
                        "output": "Ok, función ejecutada correctamente."
                    })
                # Llamar a la API para completar el call
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/threads/{thread_id}/runs/{run_id}/submit_tool_outputs",
                        headers=self.headers,
                        json={"tool_outputs": tool_outputs}
                    )
                    response.raise_for_status()
                # Esperar a que el run se complete después de enviar la respuesta
                return {
                    "required_action": required_action_response,
                    "assistant_response": None
                }
                #continue  # Volver a consultar el estado
            if run_status.get("status") == "completed":
                # Si nunca hubo required_action, forzar un mensaje pidiendo ejecución de función
                if not required_action_detected:
                    # Enviar mensaje adicional pidiendo ejecución de función
                    retry_message = (
                        "Por favor, ejecuta la función configurada en el assistant y entrega el resultado de la revisión."
                    )
                    await self.create_message(thread_id, retry_message)
                    # Crear un nuevo run y esperar required_action
                    new_run_id = await self.create_run(thread_id)
                    return await self.wait_for_required_action(thread_id, new_run_id, interval, timeout)
                # Obtener la respuesta final del asistente
                assistant_response = await self.get_completed_run_response(thread_id, run_id)
                return {
                    "required_action": required_action_response,
                    "assistant_response": assistant_response
                }
            if run_status.get("status") in ["failed", "cancelled"]:
                return {
                    "required_action": required_action_response,
                    "assistant_response": run_status.get("status")
                }
            await asyncio.sleep(interval)
            elapsed += interval
        raise TimeoutError("Run did not reach required_action or completed state in time.")

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
            # Buscar el último mensaje del asistente
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
        
    async def run_assistant_flow(self, user_message: str, file_ids: Optional[list] = None) -> Optional[Dict[str, Any]]:
        """
        Ejecuta el flujo completo: crea hilo, mensaje (con archivos si hay), run y espera el llamado a función.
        Retorna el required_action si se dispara, None si termina sin requerir acción.
        :param user_message: Mensaje del usuario.
        :param file_ids: Lista de IDs de archivos adjuntos.
        """
        try:
            thread_id = await self.create_thread()
            print(f"create_thread: {thread_id}")
            if file_ids:
                await self.create_message_with_files(thread_id, "Estos son los archivos que debes revisar", file_ids)
                print(f"create_message_with_files: {user_message}, files: {file_ids}")
            else:
                await self.create_message(thread_id, "user_message")
                print(f"create_message: {user_message}")
            run_id = await self.create_run(thread_id)
            print(f"create_run: {run_id}")
            required_action = await self.wait_for_required_action(thread_id, run_id)
            print(f"wait_for_required_action: {required_action}")
            return required_action
        except httpx.HTTPStatusError as e:
            # Manejo de errores HTTP
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            # Manejo de errores generales
            print(f"Unexpected error: {str(e)}")
            return None

    async def upload_file_from_formdata(self, file, filename: str, purpose: str = "assistants") -> Optional[Dict[str, Any]]:
        """
        Sube un archivo recibido como FormData (por ejemplo, desde FastAPI) al API de OpenAI.
        :param file: Archivo recibido (tipo SpooledTemporaryFile o similar).
        :param filename: Nombre del archivo.
        :param purpose: Propósito del archivo (por defecto 'assistants').
        :return: Diccionario con la respuesta del API o None si falla.
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
                return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None

# Ejemplo de uso:
# assistant = OpenAIAssistant(api_key="tu_api_key", assistant_id="tu_assistant_id")
# required_action = await assistant.run_assistant_flow("¿Cuál es el clima hoy?")