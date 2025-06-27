
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from models import SearchRequest, SearchResponse, CyberCrimeSection
from vector_store import CyberCrimeVectorStore
from typing import List
import uvicorn

# Global variable to store vector store instance
vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global vector_store
    print("🚀 Starting Cyber Crime API...")
    print("📊 Initializing vector store...")
    
    vector_store = CyberCrimeVectorStore()
    vector_store.load_data_to_vector_store()
    
    print("✅ Vector store initialized successfully!")
    print("🔍 API ready for cyber crime section searches")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Cyber Crime API...")

app = FastAPI(
    title="Cyber Crime IPC Sections API",
    description="API for searching relevant IPC sections for cyber crimes using vector similarity",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Cyber Crime IPC Sections API",
        "description": "Search for relevant IPC sections based on crime descriptions",
        "endpoints": {
            "search": "/search",
            "all_sections": "/sections",
            "health": "/health"
        }
    }

@app.post("/search", response_model=SearchResponse)
async def search_sections(request: SearchRequest):
    """
    Search for cyber crime sections similar to the given query
    
    Example queries:
    - "instagram lottery fraud"
    - "hacking someone's computer"
    - "sending threatening messages"
    - "identity theft online"
    """
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        results = vector_store.search_similar_sections(request.query, request.limit)
        
        return SearchResponse(
            sections=[CyberCrimeSection(**section) for section in results["sections"]],
            similarity_scores=results["similarity_scores"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/sections", response_model=List[CyberCrimeSection])
async def get_all_sections():
    """Get all available cyber crime sections"""
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        # Get all documents from collection
        results = vector_store.collection.get(include=['metadatas'])
        
        sections = []
        for metadata in results['metadatas']:
            section_data = {
                "section": metadata['section'],
                "act": metadata['act'],
                "description": metadata['description'],
                "punishment": metadata['punishment'],
                "crime_type": metadata['crime_type'],
                "keywords": eval(metadata['keywords'])  # Convert string back to list
            }
            sections.append(CyberCrimeSection(**section_data))
        
        return sections
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sections: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_store_initialized": vector_store is not None,
        "total_sections": vector_store.collection.count() if vector_store else 0
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
