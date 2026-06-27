# models.py
import uuid
from enum import IntEnum, Enum
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel
from sqlmodel import SQLModel, Field
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

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

class Data(SQLModel, table=False):  
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
    
class TrayConfig(SQLModel, table=True):
    __tablename__ = "tray_configs"

    aruco_id: int = Field(primary_key=True, description="physical ArUco identifier number")
    part_number: str = Field(unique=True, nullable=False, index=True)
    class_name: str = Field(nullable=False)
    et_rows: int = Field(nullable=False)
    et_cols: int = Field(nullable=False)
    ft_rows: int = Field(nullable=False)
    ft_cols: int = Field(nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    
# Add this right below the TrayConfig class in models.py
class SessionModel(SQLModel, table=True):
    __tablename__ = "sessions"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    status: str = Field(max_length=255)
    part_number: str = Field(max_length=255)
    inspection_stage: str = Field(max_length=255)
    type: str = Field(max_length=255)
    engine_no: str = Field(max_length=255)
    engine_hours: str = Field(max_length=255)
    component_hours: str = Field(max_length=255)
    product_id: str = Field(max_length=255)
    part_nomenclature: str = Field(max_length=255)
    work_order_no: str = Field(max_length=255)
    total_blades_quantity: int
    shop_order_no: str = Field(max_length=255)
    sop: str = Field(max_length=255)
    phase_number: str = Field(max_length=255)
    phase_quantity: int
    # Inside models.py, change those two fields to this clean format:
    empty_tray_data: Optional[Dict] = Field(default=None, sa_type=JSONB)
    filled_tray_data: Optional[Dict] = Field(default=None, sa_type=JSONB)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
