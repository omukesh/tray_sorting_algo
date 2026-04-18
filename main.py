import uuid
from app.processor.database import SessionRepo, ConfigRepo, DataRepo
from app.api.deps import MinioDep
from datetime import datetime, timezone
from typing import Optional, List
from .validate import Validate, UserData
from app.core.logger import AppLogger
from .tray_analyzer import TrayAnalyzer
import pprint
from ultralytics import YOLO
from enum import IntEnum
from app.utils import write_to_minio, get_dummy_data
from cv2.typing import MatLike
from app.models import TrayArUcoMissingError, EmptyTrayError, TrayNotEmptyError, TrayStatus, FilledTrayValidationResponse, BladeDetectionStatus, TrayValidationStatus, EmptyTrayValidationResponse, TrayEmptyError
from app.model_client import ModelClient

MINIO_BUCKET = "tray"
logger = AppLogger.get_logger()

class HandleTray:

    def __init__(self, session_id: str, part_number: str | None,
                 session_repo: SessionRepo, config_repo: ConfigRepo,
                 data_repo: DataRepo, minio: MinioDep,
                 client:ModelClient, user_data: UserData, image: MatLike| None = None):

        self.part_number = part_number
        self.session_repo = session_repo
        self.config_repo = config_repo
        self.data_repo = data_repo
        self.session_id = session_id
        self.minio = minio
        self.client = client
        self.image = image
        self.user_data = user_data
        self.validate = Validate(
            session_repo=session_repo,
            config_repo=config_repo,
            data_repo=data_repo,
            session_id=self.session_id,
            minio=minio,
        )

    async def handle(self) -> FilledTrayValidationResponse:
        try:
            if self.user_data.slots is not None:
                user_config = self.validate.update_user_data(
                    user_data=self.user_data,
                    session_id=self.session_id,
                )
                logger.info("User Data:\n%s", pprint.pformat(user_config))

            now = datetime.now(tz=timezone.utc)
            date_path = now.strftime('%Y-%m-%d')

            tray_data = await TrayAnalyzer(
                client=self.client,
                image=self.image,
            ).analyze_frame()
            
            logger.info(tray_data)
            if tray_data.get("tray_id") != self.user_data.tray_id:
                logger.info(f"tray_data tray_id: {tray_data.get('tray_id')} ({type(tray_data.get('tray_id'))})")
                logger.info(f"user_data tray_id: {self.user_data.tray_id} ({type(self.user_data.tray_id)})")

                return FilledTrayValidationResponse( 
                    status=TrayStatus.ID_MISMATCH.value,
                    session_id=self.session_id,
                    message="ID Mismatch",
                    rows=tray_data.get("rows", 0),
                    columns=tray_data.get("cols", 0),
                    blade_count=tray_data.get("count", 0),
                    blade_matrix=None,
                )

            if tray_data.get("tray_fill_status") == TrayStatus.EMPTY.value:
                logger.info(f"tray_fill_status: {tray_data.get('tray_fill_status')}")
                logger.info(f"Comparing with EMPTY: {TrayStatus.EMPTY.value}")
                return FilledTrayValidationResponse(
                    status=tray_data.get("tray_fill_status"),
                    session_id=self.session_id,
                    message="Tray is empty",
                    rows=tray_data.get("rows", 0),
                    columns=tray_data.get("cols", 0),
                    blade_count=tray_data.get("count", 0),
                    blade_matrix=None,
                )
            
            elif tray_data.get("tray_fill_status") == TrayStatus.MISSING.value:
                return FilledTrayValidationResponse(
                    status=tray_data.get("tray_fill_status"),
                    session_id=self.session_id,
                    message="Aruco Missing",
                    rows=tray_data.get("rows", 0),
                    columns=tray_data.get("cols", 0),
                    blade_count=tray_data.get("count", 0),
                    blade_matrix=None,
                )
            
            await write_to_minio(
                self.minio,
                tray_data.get("overlay_image"),
                "tray",
                output_path=f"{date_path}/{self.part_number}/{self.session_id}/filled_tray.jpg",
            )

            tray_data.pop("overlay_image", None)
            logger.info("Tray Data:\n%s", pprint.pformat(tray_data))

            _ = self.validate.update_tray_data(tray_data=tray_data)
            logger.info("Tray Data added to DB")

            part_number_equal = self.validate.compare(
                session_id=self.session_id,
                part_number=self.part_number,
            )

            matrix = tray_data.get("occupancy_grid")
            rows = tray_data.get("rows")
            cols = tray_data.get("cols")

            slots = []
            slot_n = 0
            for r in range(rows):
                for c in range(cols):
                    slot_n += 1
                    if matrix[r][c] == 1:
                        slots.append(slot_n)

            if  part_number_equal:
                _ = self.validate.update_data_table(
                    part_number=self.part_number,
                    data=tray_data,
                    slots=slots
                )
                logger.info("validation success")
                return FilledTrayValidationResponse(
                    status=tray_data.get("tray_fill_status"),
                    session_id=self.session_id,
                    message="Tray validation successfull",
                    rows=tray_data.get("rows", 0),
                    columns=tray_data.get("cols", 0),
                    blade_count=tray_data.get("count", 0),
                    blade_matrix=tray_data.get("occupancy_grid"),
                )

            if not part_number_equal:
                logger.info("top view width validation failed")
                return FilledTrayValidationResponse(
                    status=tray_data.get("tray_fill_status"),
                    session_id=self.session_id,
                    message="Top view validation failed",
                    rows=tray_data.get("rows", 0),
                    columns=tray_data.get("cols", 0),
                    blade_count=tray_data.get("count", 0),
                    blade_matrix=tray_data.get("occupancy_grid"),
                )
            blade_count, blade_matrix = get_dummy_data(session_id=self.session_id)
            return FilledTrayValidationResponse(
                status=1,
                session_id=self.session_id,
                message="Validation Successful",
                rows=8,
                columns=5,
                blade_count=blade_count,
                blade_matrix=blade_matrix
            )

        except Exception as e:
            logger.exception("Unhandled error in HandleTray.handle")
            return FilledTrayValidationResponse(
                status=TrayValidationStatus.ERROR,
                session_id=self.session_id,
                message=str(e),
            )

    def user_valid(self, user_data: UserData, part_number: str) -> list[uuid.UUID] | None:
        try:
            data_ids = self.validate.update_user_validated_data(
                session_id=self.session_id,
                data=user_data,
                part_number=part_number,
            )
            return data_ids
        except Exception:
            logger.exception("Unhandled error in user_valid")
            return None

class HandleEmptyTray:

    def __init__(self, part_number: str | None,
                 session_repo: SessionRepo,
                 tray_id: str,
                 camera,
                 client: ModelClient,
                 minio=MinioDep,
                 frame: MatLike | None = None,
                 ):
        self.part_number = part_number
        self.session_repo = session_repo
        self.tray_id = tray_id
        self.client = client
        self.camera = camera
        self.minio = minio
        self.frame = frame

    async def inspect(self):
        try:

            if self.frame is not None:
                tray_data = await TrayAnalyzer(
                    client=self.client,
                    camera=self.camera,
                ).capture_and_analyze()

            now = datetime.now(tz=timezone.utc)
            date_path = now.strftime('%Y-%m-%d')

            overlay_image = tray_data.get("overlay_image")

            await write_to_minio(
                self.minio,
                overlay_image,
                "tray",
                output_path=f"{date_path}/{self.part_number}/test/empty_tray.jpg",
            )

            if tray_data.get("tray_fill_status") == TrayStatus.FILLED.value:
                return EmptyTrayValidationResponse(
                    error=EmptyTrayError.TRAY_NOT_EMPTY.value
                )

            elif tray_data.get("tray_fill_status") == TrayStatus.MISSING.value:
                return EmptyTrayValidationResponse(
                    error=EmptyTrayError.ARUCO_MISSING.value
                )
            
            elif tray_data.get("tray_id") != self.tray_id:
                logger.info(f"{tray_data.get("tray_id")}, {self.tray_id}")
                return EmptyTrayValidationResponse(
                    error=EmptyTrayError.TRAY_ID_MISMATCH.value
                )

            tray_data.pop("overlay_image", None)

            session = self.session_repo.create({
                "status": "IN PROGRESS",
                "empty_tray_data": tray_data,
            })

            session_id = session.id
            logger.info(f"session_id in main function: {session_id}")
            await write_to_minio(
                self.minio,
                overlay_image,
                "tray",
                output_path=f"{date_path}/{self.part_number}/{session_id}/empty_tray.jpg",
            )
            return EmptyTrayValidationResponse(
                    session_id=session_id
                )

        except (TrayNotEmptyError, TrayArUcoMissingError):
            logger.exception("Known tray validation error")
            raise

        except Exception as e:
            logger.exception("Unhandled error in HandleEmptyTray.inspect")
            raise e
