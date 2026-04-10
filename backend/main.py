from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import predict

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing sentiment of movie reviews",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/api/v1", tags=["predictions"])

# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint - returns API info."""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs"
    }

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
