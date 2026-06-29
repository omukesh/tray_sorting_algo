# main.py
from __future__ import annotations
import os
import cv2
import uuid
import numpy as np
from models import EmptyTrayError, EmptyTrayValidationResponse, FilledTrayValidationResponse, TrayValidationStatus

from tray_analyzer import TrayAnalyzer
from validate import TrayIdMismatchError, Validate

class HandleEmptyTray:
    def __init__(self, part_number: str | None, session_repo: any, data: any, client: any, minio: any, frame: np.ndarray = None):
        self.part_number = part_number or "PN-MOCK"
        self.session_repo = session_repo
        self.client = client
        self.minio = minio
        self.data = data
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8) if frame is None else frame
        self.validate = Validate(session_repo, None, None, "", minio)
        self.analyzer = TrayAnalyzer(client=self.client)

    async def inspect(self) -> EmptyTrayValidationResponse:
        try:
            detections = await self.client.get_tray_detections(self.frame)
            db_session = getattr(self.session_repo, 'session', None)
            response_data, overlay_image = await self.analyzer.analyze_frame(session=db_session, frame=self.frame, detections=detections, expected_part_number=self.part_number)
            await self.validate.validate_empty_tray(response_data, self.part_number)
            
            session_id = uuid.uuid4()
            os.makedirs("./mock_minio", exist_ok=True)
            cv2.imwrite(f"./mock_minio/empty_tray_{session_id}.jpg", self.frame)
            cv2.imwrite(f"./mock_minio/empty_tray_overlay_{session_id}.jpg", overlay_image)
            
            return EmptyTrayValidationResponse(session_id=session_id, status=TrayValidationStatus.OK, error=None)

        except TrayIdMismatchError:
            # Intercept and map out the TRAY_ID_MISMATCH enum cleanly!
            return EmptyTrayValidationResponse(session_id=None, status=TrayValidationStatus.REJECT, error=EmptyTrayError.TRAY_ID_MISMATCH)
        except Exception as e:
            return EmptyTrayValidationResponse(session_id=None, status=TrayValidationStatus.ERROR, error=EmptyTrayError.INTERNAL_ERROR)

class HandleTray:
    def __init__(self, session_id: str, part_number: str | None, session_repo: any, config_repo: any, data_repo: any, minio: any, client: any, user_data: any, image: np.ndarray = None):
        self.session_id = session_id
        self.part_number = part_number or "PN-MOCK"
        self.session_repo = session_repo
        self.config_repo = config_repo
        self.data_repo = data_repo
        self.minio = minio
        self.client = client
        self.user_data = user_data
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8) if image is None else image
        self.validate = Validate(session_repo, config_repo, data_repo, session_id, minio)
        self.analyzer = TrayAnalyzer(client=self.client)

    async def handle(self) -> FilledTrayValidationResponse:
        
        try:
            detections = await self.client.get_tray_detections(self.frame)
            db_session = getattr(self.session_repo, 'session', None)
            
            response_data, overlay_image = await self.analyzer.analyze_frame(
                session=db_session, 
                frame=self.frame, 
                detections=detections,
                expected_part_number=self.part_number
            )
            await self.validate.validate_filled_tray(response_data,self.part_number)
            
            self.validate.update_tray_data(response_data)
            os.makedirs("./mock_minio", exist_ok=True)
            cv2.imwrite(f"./mock_minio/filled_tray_{self.session_id}.jpg", self.frame)
            cv2.imwrite(f"./mock_minio/filled_tray_overlay_{self.session_id}.jpg", overlay_image)
            
            active_slots = sorted(set(response_data.get("blade_elements", [])))
            self.validate.update_data_table(self.part_number, active_slots)
            
            return FilledTrayValidationResponse(
                status=TrayValidationStatus.OK, session_id=uuid.UUID(self.session_id),
                message="Tray validation successful.", rows=response_data.get("rows"),
                columns=response_data.get("cols"), blade_count=response_data.get("count"),
                blade_matrix=response_data.get("occupancy_grid")
            )
        except TrayIdMismatchError:
            return FilledTrayValidationResponse(
                status=TrayValidationStatus.REJECT, session_id=uuid.UUID(self.session_id),
                message="Relational validation error: SKU attributes mismatch.",
                rows=None, columns=None, blade_count=0, blade_matrix=None
            )
        except Exception as e:
            return FilledTrayValidationResponse(
                status=TrayValidationStatus.ERROR, session_id=uuid.UUID(self.session_id),
                message=f"Internal handler exception error: {str(e)}",
                rows=None, columns=None, blade_count=0, blade_matrix=None
            )
