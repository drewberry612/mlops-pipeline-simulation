from fastapi import FastAPI, UploadFile, File
from apscheduler.schedulers.background import BackgroundScheduler
import shutil
from datetime import datetime
import sqlite3

from orchestrator.retrain import retrain_model
from orchestrator.monitor import start_monitoring

app = FastAPI()
scheduler = BackgroundScheduler()

@app.on_event("startup")
def startup_event():
    scheduler.add_job(retrain_model, 'interval', minutes=60, id='retrain_job')
    scheduler.add_job(start_monitoring, 'interval', minutes=5, id='monitoring_job')
    scheduler.start()

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):

    with open(f"../object_storage/raw/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    conn = sqlite3.connect('../data_warehouse/warehouse.db')
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO images (filename, upload_time)
        VALUES (?,?)
        """,
        (file.filename, datetime.now())
    )
    image_id = cursor.lastrowid
    conn.commit()
    cursor.close()
    conn.close()

    with open("../object_storage/file_log.csv", "a") as log:
        log.write(f"{file.filename},{image_id},{datetime.now()},upload\n")

    return {"filename": file.filename, "status": "stored"}

@app.get("/db-data/")
def get_db_data():
    # TODO: Query and return database/logging/monitoring data
    return {"data": "sample database data"}

# Optionally, add shutdown event to stop scheduler
@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()