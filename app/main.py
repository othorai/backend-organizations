from fastapi import FastAPI
from app.routers import users, organizations
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include your routers
app.include_router(users.router, prefix="/authorization", tags=["Login & Signup"])
app.include_router(organizations.router, prefix="/api/v1", tags=["organizations"])

@app.get("/")
async def root():
    return {"message": "Welcome to Othor API"}