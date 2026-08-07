from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles 
from pathlib import Path
from app.database import Base, engine
from app.routers import auth_router, users_router, analysis_router, products_router, recommendations_router
from app.routers.misc_routers import wishlist_router, weather_router, chatbot_router, admin_router

app = FastAPI(title="DermaNova AI API", version="1.0.0")

# Mount static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(users_router.router)
app.include_router(analysis_router.router)
app.include_router(products_router.router)
app.include_router(recommendations_router.router)
app.include_router(wishlist_router)
app.include_router(weather_router)
app.include_router(chatbot_router)
app.include_router(admin_router)


@app.on_event("startup")
def on_startup():
    # Creates tables if they do not exist. For production, use Alembic migrations instead.
    Base.metadata.create_all(bind=engine)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
