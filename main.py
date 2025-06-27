import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

# Pydantic models
class SectionResponse(BaseModel):
    section: str
    act: str
    description: str
    similarity: float

class HealthResponse(BaseModel):
    status: str
    total_sections: int

# Global variables for vector store
vectorizer = None
X = None
ipc_df = None

def load_data():
    """Load and prepare the IPC sections data"""
    global ipc_df
    
    # Comprehensive IPC and IT Act sections data
    data = {
        'Section': ['65', '66', '66A', '66B', '66C', '66D', '66E', '66F', '67', '67A', '67B', '67C',
                   '69', '69A', '69B', '70', '71', '72', '73', '74', '75', '77', '77A', '77B', '78', '79',
                   '84B', '84C', '85', '378', '379', '411', '415', '419', '420', '425', '426', '463', '464',
                   '465', '468', '469', '292', '292A', '293', '294', '354C', '354D', '383', '499', '500',
                   '503', '506', '507', '509'],
        'Act': ['IT Act 2000'] * 29 + ['IPC'] * 26,
        'Description': [
            'Tampering with computer source documents',
            'Hacking with computer systems, data alteration',
            'Sending offensive messages through communication service',
            'Dishonestly receiving stolen computer resource or communication device',
            'Identity theft, digital signatures, password hacking',
            'Cheating by personation using computer resources',
            'Violation of privacy, taking pictures of private areas without consent',
            'Cyber terrorism',
            'Publishing or transmitting obscene material in electronic form',
            'Publishing or transmitting material containing sexually explicit acts',
            'Publishing or transmitting material depicting children in sexually explicit acts',
            'Preservation and retention of information by intermediaries',
            'Powers to issue directions for interception or monitoring or decryption',
            'Power to issue directions for blocking public access',
            'Power to authorize monitoring and collection of traffic data for cyber security',
            'Unauthorized access to protected system',
            'Penalty for misrepresentation',
            'Breach of confidentiality and privacy',
            'Publishing false digital signature certificates',
            'Publication for fraudulent purpose',
            'Act to apply for offenses committed outside India',
            'Compensation, penalties not to interfere with other punishment',
            'Compounding of offenses',
            'Offenses with three years imprisonment to be cognizable',
            'Empowers Police Inspector to investigate IT Act cases',
            'Exemption from liability of intermediary in certain cases',
            'Punishment for abetment of offenses',
            'Punishment for attempt to commit offenses',
            'Offenses by companies',
            'Theft of computer hardware',
            'Punishment for theft, stolen data, hijacked electronic devices',
            'Receiving stolen property',
            'Cheating',
            'Punishment for cheating by personation',
            'Cheating and dishonestly inducing delivery of property, bogus websites, cyber frauds',
            'Mischief',
            'Mischief by injury to work of irrigation or by wrongfully diverting water',
            'Forgery, email spoofing',
            'Making a false document',
            'Punishment for forgery, email spoofing',
            'Forgery for purpose of cheating',
            'Forgery for purpose of harming reputation',
            'Sale of obscene materials, publishing sexually explicit content electronically',
            'Printing grossly indecent matter for blackmail',
            'Sale of obscene objects to young person',
            'Obscene acts and songs',
            'Voyeurism, taking or publishing pictures of private parts without consent',
            'Stalking, including cyberstalking',
            'Web-jacking',
            'Sending defamatory messages by email',
            'Email abuse',
            'Sending threatening messages by email',
            'Punishment for criminal intimidation',
            'Criminal intimidation by anonymous communication',
            'Word, gesture or act intended to insult modesty of a woman'
        ],
        'Punishment': [
            'Up to 3 years imprisonment and/or fine up to Rs. 2 lakh',
            'Up to 3 years imprisonment or fine up to Rs. 5 lakh',
            'Up to 3 years imprisonment and/or fine',
            'Up to 3 years imprisonment and fine up to Rs. 1 lakh',
            'Up to 3 years imprisonment and fine up to Rs. 1 lakh',
            'Up to 3 years imprisonment and/or fine up to Rs. 1 lakh',
            'Up to 3 years imprisonment and/or fine up to Rs. 2 lakh',
            'Up to life imprisonment',
            'Up to 5 years imprisonment and fine up to Rs. 10 lakh',
            'Up to 7 years imprisonment and fine up to Rs. 10 lakh',
            'Up to 7 years imprisonment and fine up to Rs. 10 lakh',
            'Up to 3 years imprisonment and/or fine',
            'Up to 7 years imprisonment',
            'Up to 7 years imprisonment',
            'Up to 3 years imprisonment and/or fine',
            'Up to 10 years imprisonment and fine',
            'Up to 2 years imprisonment or fine up to Rs. 1 lakh',
            'Up to 2 years imprisonment or fine up to Rs. 1 lakh',
            'Up to 2 years imprisonment and fine',
            'Up to 3 years imprisonment and fine',
            'Applicable for offenses outside India',
            'Does not interfere with other punishments',
            'Allows compounding of certain offenses',
            'Makes certain offenses cognizable',
            'Investigative powers',
            'Safe harbor provisions',
            'Same as principal offense',
            'Up to half of principal offense punishment',
            'Fine and imprisonment for company officers',
            'Up to 3 years imprisonment and fine',
            'Up to 3 years imprisonment and fine',
            'Up to 3 years imprisonment and/or fine',
            'Up to 1 year imprisonment and/or fine',
            'Up to 3 years imprisonment or fine',
            'Up to 7 years imprisonment and fine',
            'Up to 3 months imprisonment and/or fine up to Rs. 500',
            'Up to 1 year imprisonment and/or fine up to Rs. 1000',
            'Up to 7 years imprisonment and/or fine',
            'Up to 3 years imprisonment and/or fine',
            'Up to 2 years imprisonment or fine',
            'Up to 7 years imprisonment and fine',
            'Up to 3 years imprisonment and/or fine',
            'Up to 2 years imprisonment and Rs. 2,000 fine',
            'Up to 2 years imprisonment and/or fine',
            'Up to 1 year imprisonment and/or fine',
            'Up to 3 months imprisonment and/or fine',
            'Up to 3 years for first offense, up to 7 years for repeat',
            'Up to 3 years for first offense, up to 5 years for repeat',
            'Up to 2 years imprisonment and/or fine',
            'Up to 2 years imprisonment and/or fine',
            'Up to 1 year imprisonment and/or fine',
            'Up to 2 years imprisonment and/or fine',
            'Up to 7 years imprisonment and/or fine',
            'Up to 1 year imprisonment and/or fine',
            'Up to 1 year imprisonment and/or fine'
        ]
    }
    
    ipc_df = pd.DataFrame(data)
    return ipc_df

def initialize_vector_store():
    """Initialize the TF-IDF vectorizer and create embeddings"""
    global vectorizer, X, ipc_df
    
    # Load data
    ipc_df = load_data()
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2),
        lowercase=True
    )
    
    # Combine description and section for better matching
    combined_text = ipc_df['Description'] + ' ' + ipc_df['Section'] + ' ' + ipc_df['Act']
    X = vectorizer.fit_transform(combined_text)
    
    print(f"Vector store initialized with {len(ipc_df)} sections")

def find_similar_sections(text: str, top_n: int = 5, threshold: float = 0.1):
    """Find similar IPC sections based on input text"""
    global vectorizer, X, ipc_df
    
    if vectorizer is None or X is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    # Transform input text
    text_vec = vectorizer.transform([text.lower()])
    
    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(text_vec, X).flatten()
    
    # Get top indices
    top_indices = np.argsort(cosine_similarities)[::-1][:top_n]
    
    # Filter by threshold
    filtered_indices = [idx for idx in top_indices if cosine_similarities[idx] >= threshold]
    
    if not filtered_indices:
        return []
    
    # Prepare results
    results = []
    for idx in filtered_indices:
        results.append({
            'section': ipc_df.iloc[idx]['Section'],
            'act': ipc_df.iloc[idx]['Act'],
            'description': ipc_df.iloc[idx]['Description'],
            'similarity': float(cosine_similarities[idx])
        })
    
    return results

# Initialize FastAPI app
app = FastAPI(
    title="IPC Section Finder API",
    description="Find relevant IPC and IT Act sections based on crime description",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize vector store when server starts"""
    initialize_vector_store()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        total_sections=len(ipc_df) if ipc_df is not None else 0
    )

@app.get("/sections", response_model=List[SectionResponse])
async def get_similar_sections(
    text: str = Query(..., description="Crime description text to find similar IPC sections"),
    top_n: int = Query(5, ge=1, le=20, description="Number of top results to return"),
    threshold: float = Query(0.1, ge=0.0, le=1.0, description="Minimum similarity threshold")
):
    """
    Find IPC sections similar to the provided crime description text.
    
    Example queries:
    - "instagram lottery fraud"
    - "hacking computer system"
    - "sending threatening messages"
    - "identity theft online"
    """
    try:
        results = find_similar_sections(text, top_n, threshold)
        
        if not results:
            raise HTTPException(
                status_code=404, 
                detail=f"No similar sections found for '{text}' with threshold {threshold}"
            )
        
        return [SectionResponse(**result) for result in results]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sections/all")
async def get_all_sections():
    """Get all available IPC and IT Act sections"""
    if ipc_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    return ipc_df.to_dict(orient='records')

@app.get("/sections/by-act/{act_name}")
async def get_sections_by_act(act_name: str):
    """Get sections filtered by act name (IPC or IT Act 2000)"""
    if ipc_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    filtered_df = ipc_df[ipc_df['Act'].str.contains(act_name, case=False)]
    
    if filtered_df.empty:
        raise HTTPException(status_code=404, detail=f"No sections found for act: {act_name}")
    
    return filtered_df.to_dict(orient='records')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
