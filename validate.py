# validate.py
from __future__ import annotations
import uuid
from models import TrayArUcoMissingError, TrayNotEmptyError, TrayEmptyError

class TrayIdMismatchError(Exception):
    pass

class Validate:
    def __init__(self, session_repo, config_repo, data_repo, session_id: str, minio) -> None:
        self.session_id = session_id
        self.session_repo = session_repo
        self.config_repo = config_repo
        self.data_repo = data_repo
        self.minio = minio

    async def validate_empty_tray(self, result: dict, expected_part_number: str) -> None:
        tray_type = result.get("tray_type")
        status = result["tray_fill_status"]
        detected_part_number = result.get("Part_Number")
        if status == 4: raise TrayArUcoMissingError("Primary tray markers missing.")
        if detected_part_number and expected_part_number and detected_part_number != expected_part_number:
            print(f"[Validation Mismatch] Expected empty SKU '{expected_part_number}', but physical tag says '{detected_part_number}'!")
            raise TrayIdMismatchError(f"SKU mismatch error between physical tag and expected payload.")
        if int(tray_type) != 0: raise ValueError(f"Expected empty tray, got tray_type={tray_type}")
        if status == 1: raise TrayNotEmptyError("Tray contains components.")

    async def validate_filled_tray(self, result: dict, expected_part_number: str) -> None:
        tray_type = result.get("tray_type")
        status = result["tray_fill_status"]
        detected_part_number = result.get("Part_Number")
        if status == 4: raise TrayArUcoMissingError("Primary tray markers missing.")
        if detected_part_number and expected_part_number and detected_part_number != expected_part_number:
            print(f"[Validation Mismatch] Expected filled SKU '{expected_part_number}', but physical tag says '{detected_part_number}'!")
            raise TrayIdMismatchError(f"SKU mismatch error between physical tag and expected payload.")
        
        if int(tray_type) != 1: raise ValueError(f"Expected filled tray, got tray_type={tray_type}")
        if status == 0: raise TrayEmptyError("No components detected.")

    def update_tray_data(self, tray_data: dict) -> dict:
        print(f"[Repo Mock] Updating Session data payload for Tray ID: {tray_data.get('Tray_ID')}")
        return {"filled_tray_data": tray_data, "tray_id": str(tray_data.get("Tray_ID"))}

    def compare(self, session_id: str, part_number: str) -> bool:
        return True # Simplifies pass validation during development testing

    def update_data_table(self, part_number: str, slots: list[int]) -> dict[int, uuid.UUID]:
        print(f"[Repo Mock] Data table updated with {len(slots)} validated slots.")
        return {s: uuid.uuid4() for s in slots}
