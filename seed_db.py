# seed_db.py
import csv
from datetime import datetime
from sqlmodel import SQLModel, create_engine, Session
from models import TrayConfig, SessionModel

# 2. LOCAL POSTGRES DEPLOYMENT Handshake
POSTGRES_URL = "postgresql://postgres:123456@localhost:5432/tray_db"
engine = create_engine(POSTGRES_URL, echo=False)

def seed_from_csv(csv_filepath: str):
    print("[INFO] Creating database schema tables...")
    # This reads all tables registered on SQLModel's metadata (TrayConfig and SessionModel)
    SQLModel.metadata.create_all(engine)
    
    print(f"[INFO] Opening file with BOM handling: {csv_filepath}")
    with open(csv_filepath, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        # Strip trailing/leading white-spaces from header elements dynamically
        reader.fieldnames = [name.strip() for name in reader.fieldnames] if reader.fieldnames else []
        print(f"[INFO] Synchronized headers active: {reader.fieldnames}")
        
        with Session(engine) as session:
            for row in reader:
                # Aligned mapping utilizing direct model class mappings
                config_row = TrayConfig(
                    aruco_id=int(row['Aruco ID:']),
                    part_number=row['Part No.:'].strip(),
                    class_name=row['Class Name:'].strip(),
                    et_rows=int(row['Empty Tray R:']),
                    et_cols=int(row['Empty Tray C:']),
                    ft_rows=int(row['Filling Tray R:']),
                    ft_cols=int(row['Filling Tray C:'])
                )
                session.merge(config_row)
            session.commit()
    print("[SUCCESS] Postgres table successfully populated with zero column mismatch errors!")

if __name__ == "__main__":
    seed_from_csv("Blade_data_aruco_classes - Sheet2.csv")
