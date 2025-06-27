
import chromadb
from chromadb.config import Settings
import pandas as pd
from constants import DATA
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import json
import os

class CyberCrimeVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "cyber_crimes"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Cyber crime IPC sections"}
            )
            print(f"Created new collection: {self.collection_name}")
    
    def create_cyber_crimes_data(self) -> pd.DataFrame:
        """Create comprehensive cyber crimes dataset"""
        data = DATA
        
        return pd.DataFrame(data)
    
    def load_data_to_vector_store(self):
        """Load cyber crimes data into vector store"""
        df = self.create_cyber_crimes_data()
        
        # Check if collection already has data
        if self.collection.count() > 0:
            print(f"Collection already contains {self.collection.count()} documents")
            return
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            # Create searchable text combining all relevant fields
            searchable_text = f"{row['description']} {' '.join(row['keywords'])} {row['crime_type']} {row['act']}"
            documents.append(searchable_text)
            
            # Store metadata
            metadata = {
                "section": row['section'],
                "act": row['act'],
                "description": row['description'],
                "punishment": row['punishment'],
                "crime_type": row['crime_type'],
                "keywords": json.dumps(row['keywords'])
            }
            metadatas.append(metadata)
            ids.append(f"section_{idx}")
        
        # Generate embeddings
        embeddings = self.model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Loaded {len(documents)} cyber crime sections into vector store")
    
    def search_similar_sections(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search for similar cyber crime sections"""
        # Generate query embedding
        query_embedding = self.model.encode([query]).tolist()[0]
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=['metadatas', 'documents', 'distances']
        )
        
        sections = []
        scores = []
        
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                section_data = {
                    "section": metadata['section'],
                    "act": metadata['act'],
                    "description": metadata['description'],
                    "punishment": metadata['punishment'],
                    "crime_type": metadata['crime_type'],
                    "keywords": json.loads(metadata['keywords'])
                }
                sections.append(section_data)
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity_score = 1 / (1 + distance)
                scores.append(round(similarity_score, 4))
        
        return {
            "sections": sections,
            "similarity_scores": scores
        }
