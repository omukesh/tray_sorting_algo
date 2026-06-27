# models.py
import uuid
from enum import IntEnum, Enum
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel
from sqlmodel import SQLModel, Field

class TrayValidationStatus(IntEnum):
    OK = 1
    REJECT = 2
    ERROR = 3
    MISSING = 4
    MISSING_INFERENCE = 5
    CAMERA_NOT_INITIALIZED = 6

class InspectionStage(str, Enum):
    VIEWING = "VIEWING"
    INTERMEDIATE = "INTERMEDIATE"
    FINAL = "FINAL"

class EmptyTrayPayload(BaseModel):
    part_number: str
    tray_id: str
    inspection_stage: InspectionStage
    inspection_type: str
    engine_no: str
    engine_hours: str
    component_hours: str
    product_id: str
    part_nomenclature: str
    work_order_no: str
    total_blades_quantity: int
    shop_order_no: str
    sop: str
    phase_number: str
    phase_quantity: int

class UserData(BaseModel):
    tray_id: str
    slots: list[int] = []

class EmptyTrayError(IntEnum):
    TRAY_NOT_EMPTY = 1
    ARUCO_MISSING = 2
    CAMERA_NOT_INITIALIZED = 3
    TRAY_ID_MISMATCH = 4
    INTERNAL_ERROR = 5

class EmptyTrayValidationResponse(BaseModel):
    session_id: uuid.UUID | None = None
    status: TrayValidationStatus
    error: EmptyTrayError | None = None

class FilledTrayValidationResponse(BaseModel):
    status: TrayValidationStatus
    session_id: uuid.UUID
    message: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    blade_count: Optional[int] = None
    blade_matrix: Optional[List[List[int]]] = None

# Custom structural exceptions
class TrayNotEmptyError(Exception): pass
class TrayEmptyError(Exception): pass
class TrayArUcoMissingError(Exception): pass

class Data(SQLModel, table=False):  # table=False for simplified mockup setup
    session_id: uuid.UUID
    slot_id: int
    top_view_width: float
    orientation: float
    distance_from_camera: float
    blade_type_id: int
    blade_id: str
    swath_plan_images_count: Optional[Dict] = None
    tray_process_time: datetime
    inspection_passed: bool
    status: str
