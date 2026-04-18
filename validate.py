# backend/app/processor/tray/validate.py

from app.processor.database import SessionRepo, ConfigRepo, DataRepo
from app.api.deps import SessionDep, MinioDep
from pydantic import BaseModel
from dataclasses import asdict
from app.models import Data
from app.core.logger import AppLogger
from datetime import datetime, timezone
import uuid

logger = AppLogger.get_logger()

class UserData(BaseModel):
    tray_id: str
    tray_color: str
    slots: list[int]

class TrayData(BaseModel):
    tray_id: str
    tray_color: str
    slots: list[int]
    top_view_width: list[float]
    orientation: list[float]
    distance_from_camera: list[float]
    image: str

class Validate():
    def __init__(self, session_repo: SessionRepo, config_repo: ConfigRepo, data_repo: DataRepo, session_id: str, minio: MinioDep) -> None:

        self.session_id = session_id
        self.session_repo = session_repo
        self.config_repo = config_repo
        self.data_repo = data_repo
        self.minio = minio

    def update_user_data(self, user_data: UserData, session_id: str) -> dict:
        user_config = {"user_data": user_data.model_dump()}
        self.session_repo.update(id=session_id, data=user_config)
        return user_config
         
    def update_tray_data(self, tray_data: dict):
        tray_data.pop("overlay_image", None)
        tray_config = {"filled_tray_data": tray_data, "tray_id": tray_data.get("tray_id")}
        self.session_repo.update(id=self.session_id, data=tray_config)
        return tray_config

    def compare(self, session_id, part_number: str) -> tuple[bool, bool]:
        tray_data = self.session_repo.get_column_value(column="filled_tray_data", id=session_id)
        rows = tray_data.get("rows")
        cols = tray_data.get("cols")

        top_view_width = self.config_repo.get_column_value(column="top_view_width", part_number=part_number)
        part_number_equal = tray_data.get("top_view_width") == top_view_width[0]

        return part_number_equal
    
    def _get_num_patches(self, part_number: str)-> dict:
        swath_plan = self.config_repo.get_column_value(column="swath_plan", part_number=part_number)
        return swath_plan
    
    def update_data_table(self, part_number: str, data: dict, slots: list)-> list[uuid.UUID]:

        blade_type_id = self.config_repo.get_column_value(column="id", part_number=part_number)
        data_ids = {}

        for slot in slots:
            row = Data(
                session_id=self.session_id,
                slot_id=slot,
                top_view_width=data.get("top_view_width"),
                orientation=data.get("top_view_width"),
                distance_from_camera=data.get("top_view_width"),
                blade_type_id=blade_type_id,
                blade_id=slot,
                swath_plan_images_count=self._get_num_patches(part_number=part_number),
                tray_process_time=datetime.now(timezone.utc)
            )
            row_dict = row.model_dump() 
            self.data_repo.create(row_dict)
            data_ids[row.slot_id] = row.id

        return data_ids
    
    def _get_tray_data(self, session_id):
        tray_data = self.session_repo.get_column_value(column="filled_tray_data", id=session_id)
        return tray_data
    
    def update_user_validated_data(self, session_id, data: UserData, part_number:str):
        tray_data = self._get_tray_data(session_id=session_id)
        blade_type_id = self.config_repo.get_column_value(column="id", part_number=part_number)
        data_ids = {}
        for slot, width, orient, distance in zip(
            data.slots, 
            tray_data.get("top_view_width"), 
            tray_data.get("orientation"), 
            tray_data.get("distance_from_camera")
        ):
            row = Data(
                session_id=self.session_id,
                slot_id=slot,
                top_view_width=width,
                orientation=orient,
                distance_from_camera=distance,
                blade_type_id=blade_type_id,
                tray_process_time=datetime.now(timezone.utc)
            )
            row_dict = row.model_dump() 
            self.data_repo.create(row_dict)
            data_ids[row.slot_id] = row.id

        return data_ids