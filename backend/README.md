# DermaSense AI — Backend (FastAPI + MySQL + JWT)

## Setup

1. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create the MySQL database and tables:
   ```
   mysql -u root -p < database/schema.sql
   ```
   Or let SQLAlchemy create tables automatically on first run (`Base.metadata.create_all`
   in `app/main.py`) — the `schema.sql` file is provided for reference / manual setup.

3. Set environment variables (or create a `.env` and load it however you prefer):
   ```
   export DERMASENSE_DATABASE_URL="mysql+pymysql://dermasense_user:password@localhost:3306/dermasense"
   export DERMASENSE_JWT_SECRET="replace-with-a-long-random-secret"
   ```

4. Seed the product catalog:
   ```
   python -m database.seed_products
   ```

5. Run the API:
   ```
   uvicorn app.main:app --reload --port 8000
   ```
   Docs available at `http://localhost:8000/docs`.

## Where to plug in a real AI model

`app/services/ai_service.py` contains three functions — `detect_face`,
`analyze_skin`, `analyze_hair` — that currently return deterministic mock
predictions derived from a hash of the uploaded image bytes. Replace their
internals with real inference (TensorFlow / PyTorch / YOLO / a hosted
API) while keeping the same function signatures and return dictionaries;
no router or frontend code needs to change.

## Where to plug in real image storage

`app/routers/users_router.py` (`upload_profile_picture`) and the analysis
routers currently do not persist the raw image file. Wire in Cloudinary
(or S3) by uploading the bytes and storing the returned secure URL on the
`Analysis.image_url` / `User.profile_picture_url` columns.

## Where to plug in real weather

`app/routers/misc_routers.py` (`get_weather`) returns simulated seasonal
data. Replace the body with a call to a weather provider (OpenWeatherMap,
WeatherAPI, Open-Meteo) using `lat`/`lon` and an API key from an
environment variable.

## Authentication

JWT bearer tokens issued at `/api/auth/login` and `/api/auth/register`.
Send `Authorization: Bearer <token>` on subsequent requests. Admin-only
routes are protected by `require_admin` in `app/auth.py`, which checks the
`is_admin` flag on the `users` table — set this manually in the database
for your admin account(s).
