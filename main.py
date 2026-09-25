import uvicorn
from interface import app

if __name__ == "__main__":
    # Jalankan server uvicorn dengan aplikasi FastAPI yang telah diimpor dari interface
    uvicorn.run(
        "interface:app", 
        host="0.0.0.0", 
        port=8000
    )