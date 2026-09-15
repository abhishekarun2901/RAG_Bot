from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
 
app = FastAPI(
    title="Enterprise Document RAG API",
    version="1.0.0",
    description="Hybrid Search & Grounded Generation RAG Pipeline"
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
app.include_router(api_router, prefix="/api/v1")
 
 
@app.get("/health")
async def health_check():
    return {"status": "healthy"}