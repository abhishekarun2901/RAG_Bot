# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.mcp_server import mcp

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

# Core API routes
app.include_router(api_router, prefix="/api/v1")

# Mount MCP ASGI sub-application (handles callable or property)
mcp_sub_app = mcp.http_app() if callable(mcp.http_app) else mcp.http_app
app.mount("/mcp", mcp_sub_app)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}