import os
import uvicorn
from app.web import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except OSError:
        uvicorn.run(app, host="0.0.0.0", port=8001)
