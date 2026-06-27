# test_server.py
import uvicorn
import cv2
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from sqlmodel import create_engine, Session

from models import EmptyTrayPayload, InspectionStage, UserData, SessionModel
from main import HandleEmptyTray, HandleTray

app = FastAPI(title="Production-Aligned Tray API Router Test Service")

POSTGRES_URL = "postgresql://postgres:123456@localhost:5432/tray_db"
engine = create_engine(POSTGRES_URL, echo=False)

def get_db_session():
    with Session(engine) as session:
        yield session

# Centralized Local YOLO Framework Analytics Client
class LocalYoloClient:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO("/home/mdl/Projects/tray_algo/weights/best.pt")

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

# Repository Shim wrapping operations directly to matching columns
class ProductionSessionRepoShim:
    def __init__(self, session: Session):
        self.session = session

    def create(self, data_dict: dict):
        """Unpacks fields to match the exact database column structure."""
        # Split out structural payload details from analytical responses
        payload_data = data_dict.get("data", {})
        
        record = SessionModel(
            status="PENDING",
            part_number=data_dict.get("part_number", "PN-UNKNOWN"),
            inspection_stage=getattr(payload_data, 'inspection_stage', InspectionStage.VIEWING).value,
            type=getattr(payload_data, 'inspection_type', 'ROTOR'),
            engine_no=getattr(payload_data, 'engine_no', ''),
            engine_hours=getattr(payload_data, 'engine_hours', ''),
            component_hours=getattr(payload_data, 'component_hours', ''),
            product_id=getattr(payload_data, 'product_id', ''),
            part_nomenclature=getattr(payload_data, 'part_nomenclature', ''),
            work_order_no=getattr(payload_data, 'work_order_no', ''),
            total_blades_quantity=getattr(payload_data, 'total_blades_quantity', 0),
            shop_order_no=getattr(payload_data, 'shop_order_no', ''),
            sop=getattr(payload_data, 'sop', ''),
            phase_number=getattr(payload_data, 'phase_number', '1'),
            phase_quantity=getattr(payload_data, 'phase_quantity', 0),
            empty_tray_data=data_dict.get("empty_tray_data", {}) # Vision results JSON block
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def update(self, id, data: dict):
        record = self.session.get(SessionModel, id)
        if record:
            if "filled_tray_data" in data:
                record.filled_tray_data = data["filled_tray_data"]
                record.status = "COMPLETED"
            self.session.add(record)
            self.session.commit()

# ----------------------------------------------------
# MATCHING TRAYS ROUTING MODULE BOUNDARIES
# ----------------------------------------------------
# Create this small helper shim inside test_server.py if it isn't defined
class ConfigRepoShim:
    def __init__(self, session: Session):
        self.session = session
        
    # Optional wrapper if your production code invokes a separate engine helper method
    def get_session(self):
        return self.session

# ----------------------------------------------------
# UPDATE THE ENDPOINT CORRESPONDING ROUTES
# ----------------------------------------------------

@app.post("/tray/empty-tray")
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
        raise HTTPException(status_code=400, detail="Corrupted image.")

    payload = EmptyTrayPayload(
        part_number=part_number, tray_id=tray_id, inspection_stage=inspection_stage,
        inspection_type=inspection_type, engine_no=engine_no, engine_hours=engine_hours,
        component_hours=component_hours, product_id=product_id, part_nomenclature=part_nomenclature,
        work_order_no=work_order_no, total_blades_quantity=total_blades_quantity,
        shop_order_no=shop_order_no, sop=sop, phase_number=phase_number, phase_quantity=phase_quantity
    )

    repo_shim = ProductionSessionRepoShim(db)
    
    # CRITICAL: We pass db directly to HandleEmptyTray if it utilizes background lookups
    handler = HandleEmptyTray(
        part_number=part_number, session_repo=repo_shim, data=payload,
        client=client_instance, minio=None, frame=frame
    )
    # If your HandleEmptyTray uses internal async session mappings, pass 'db' here
    return await handler.inspect()


@app.post("/tray/filled-tray")
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

    user_payload = UserData(tray_id="", slots=json.loads(user_slots))

    session_repo = ProductionSessionRepoShim(db)
    
    # CRITICAL FIX: Pass the session directly into the configuration repo parameter
    # instead of 'None' so it can execute database queries downstream
    handler = HandleTray(
        session_id=session_id, part_number=part_number, session_repo=session_repo,
        config_repo=db, data_repo=db, minio=None, client=client_instance,
        user_data=user_payload, image=frame
    )
    return await handler.handle()


if __name__ == "__main__":
    uvicorn.run("test_server:app", host="127.0.0.1", port=8000, reload=True)
