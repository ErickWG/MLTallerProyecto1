from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv

from .database import oracle
from .models import ml
from .routes.api import router as api_router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Detección de Fraude Telefónico",
    description="Sistema de detección de anomalías en llamadas telefónicas usando Machine Learning",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando API...")
    ml.cargar_modelo()
    try:
        if oracle.inicializar_oracle_pool():
            oracle.crear_tablas_oracle()
    except Exception as e:
        logger.error(f"Error inicializando Oracle: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Apagando API...")


app.include_router(api_router)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
