# test_server.py
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from sqlmodel import create_engine, Session

from models import EmptyTrayPayload, InspectionStage, UserData
from main import HandleEmptyTray, HandleTray
from seed_db import SessionModel # Import exact structural definition created above

app = FastAPI(title="Tray Processing API Production Test Service Gateway")

POSTGRES_URL = "postgresql://postgres:password@localhost:5432/tray_db"
engine = create_engine(POSTGRES_URL, echo=False)

def get_db_session():
    with Session(engine) as session:
        yield session

class LocalYoloClient:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n-seg.pt")

    async def get_tray_detections(self, frame: np.ndarray) -> list:
        results = self.model(frame, verbose=False)[0]
        detections = []
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                name = results.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                
                mask_arr = None
                if results.masks is not None:
                    mask_arr = results.masks[i].data[0].cpu().numpy().astype(np.uint8)
                    mask_arr = cv2.resize(mask_arr, (frame.shape[1], frame.shape[0]))
                
                detections.append({
                    "name": name, "confidence": conf, "box": xyxy, "mask_array": mask_arr
                })
        return detections

client_instance = LocalYoloClient()

class SessionRepoShim:
    def __init__(self, session: Session):
        self.session = session
        self.session_instance = None  # To mock production property injection safely

    def create(self, data_dict: dict):
        record = SessionModel(**data_dict)
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def update(self, id, data: dict):
        record = self.session.get(SessionModel, id)
        if record:
            for k, v in data.items():
                setattr(record, k, v)
            self.session.add(record)
            self.session.commit()

@app.post("/empty-tray")
async def process_empty_tray(
    file: UploadFile = File(...),
    part_number: str = Form(...),
    tray_id: str = Form(...),
    inspection_stage: InspectionStage = Form(...),
    inspection_type: str = Form(...),
    engine_no: str = Form(...),
    engine_hours: str = Form(...),
    component_hours: str = Form(...),
    product_id: str = Form(...),
    part_nomenclature: str = Form(...),
    work_order_no: str = Form(...),
    total_blades_quantity: int = Form(...),
    shop_order_no: str = Form(...),
    sop: str = Form(...),
    phase_number: str = Form(...),
    phase_quantity: int = Form(...),
    db: Session = Depends(get_db_session)
):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Corrupted image stream matrix payload.")

    payload = EmptyTrayPayload(
        part_number=part_number, tray_id=tray_id, inspection_stage=inspection_stage,
        inspection_type=inspection_type, engine_no=engine_no, engine_hours=engine_hours,
        component_hours=component_hours, product_id=product_id, part_nomenclature=part_nomenclature,
        work_order_no=work_order_no, total_blades_quantity=total_blades_quantity,
        shop_order_no=shop_order_no, sop=sop, phase_number=phase_number, phase_quantity=phase_quantity
    )

    repo_shim = SessionRepoShim(db)
    # Inject active engine dependency into repository property safely matching main.py requirements
    repo_shim.session = db
    
    handler = HandleEmptyTray(
        part_number=part_number, session_repo=repo_shim, data=payload,
        client=client_instance, minio=None, frame=frame
    )
    return await handler.inspect()

@app.post("/filled-tray")
async def process_filled_tray(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    part_number: str = Form(...),
    user_slots: str = Form("[]"),
    db: Session = Depends(get_db_session)
):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Corrupted image matrix.")

    import json
    user_payload = UserData(tray_id="", slots=json.loads(user_slots))

    repo_shim = SessionRepoShim(db)
    repo_shim.session = db
    
    handler = HandleTray(
        session_id=session_id, part_number=part_number, session_repo=repo_shim,
        config_repo=None, data_repo=None, minio=None, client=client_instance,
        user_data=user_payload, image=frame
    )
    return await handler.handle()

if __name__ == "__main__":
    uvicorn.run("test_server:app", host="127.0.0.1", port=8000, reload=True)
