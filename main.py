from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import MsgPayload
from routers.vigia import router as vigia_router
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

# Habilitar CORS para todos los orígenes (puedes personalizar los parámetros)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia esto si quieres restringir orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(vigia_router)
messages_list: dict[int, MsgPayload] = {}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hello"}


# About page route
@app.get("/about")
def about() -> dict[str, str]:
    return {"message": "This is the about page."}


# Route to add a message
@app.post("/messages/{msg_name}/")
def add_msg(msg_name: str) -> dict[str, MsgPayload]:
    # Generate an ID for the item based on the highest ID in the messages_list
    msg_id = max(messages_list.keys()) + 1 if messages_list else 0
    messages_list[msg_id] = MsgPayload(msg_id=msg_id, msg_name=msg_name)

    return {"message": messages_list[msg_id]}


# Route to list all messages
@app.get("/messages")
def message_items() -> dict[str, dict[int, MsgPayload]]:
    return {"messages:": messages_list}
