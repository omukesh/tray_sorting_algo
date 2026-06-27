# seed_db.py
import csv
import uuid
from datetime import datetime
from typing import Optional, Dict
from sqlmodel import SQLModel, create_engine, Session, Field
from sqlalchemy.dialects.postgresql import JSONB

class TrayConfig(SQLModel, table=True):
    __tablename__ = "tray_configs"

    aruco_id: int = Field(primary_key=True)
    part_number: str = Field(unique=True, nullable=False, index=True)
    class_name: str = Field(nullable=False)
    et_rows: int = Field(nullable=False)
    et_cols: int = Field(nullable=False)
    ft_rows: int = Field(nullable=False)
    ft_cols: int = Field(nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

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
    empty_tray_data: Optional[Dict] = Field(default=None, sa_type=JSONB, nullable=True)
    filled_tray_data: Optional[Dict] = Field(default=None, sa_type=JSONB, nullable=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# Database Connection
POSTGRES_URL = "postgresql://postgres:password@localhost:5432/tray_db"
engine = create_engine(POSTGRES_URL, echo=False)

def seed_from_csv(csv_filepath: str):
    print("[INFO] Creating database schema tables...")
    SQLModel.metadata.create_all(engine)
    
    print(f"[INFO] Opening file with BOM handling: {csv_filepath}")
    # Using 'utf-8-sig' to automatically clean the \ufeff artifact from the stream!
    with open(csv_filepath, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        # Strip any accidental spaces from keys just to be completely safe
        reader.fieldnames = [name.strip() for name in reader.fieldnames] if reader.fieldnames else []
        print(f"[INFO] Cleaned headers found: {reader.fieldnames}")
        
        with Session(engine) as session:
            for row in reader:
                config_row = TrayConfig(
                    aruco_id=int(row['Aruco ID']),
                    part_number=row['Part Number'].strip(),
                    class_name=row['Ground Truth Class Name'].strip(),
                    et_rows=int(row['ET Rows']),
                    et_cols=int(row['ET Cols']),
                    ft_rows=int(row['FT Rows']),
                    ft_cols=int(row['FT Cols'])
                )
                session.merge(config_row)
            session.commit()
    print("[SUCCESS] Postgres table successfully populated without BOM interference!")

if __name__ == "__main__":
    seed_from_csv("Blade_data_aruco_classes - Sheet2.csv")
