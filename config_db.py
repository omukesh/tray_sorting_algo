# config_db.py
import csv
import os
import sqlite3

DB_PATH = "tray_config.db"
CSV_FILE_PATH = "Blade_data_aruco_classes - Sheet2.csv"

def init_and_seed_sqlite():
    if not os.path.exists(CSV_FILE_PATH):
        print(f" Error: Targeted CSV file '{CSV_FILE_PATH}' not found in active directory.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 1. FORCE DROP THE OLD SCHEMA TO RE-INITIALIZE FRESH COLUMNS CLEANLY
        cursor.execute("DROP TABLE IF EXISTS tray_configs;")
        
        # 2. CREATE FRESH SCHEMA ENFORCING PART NUMBERS
        cursor.execute("""
            CREATE TABLE tray_configs (
                aruco_id INTEGER PRIMARY KEY,
                part_number TEXT UNIQUE NOT NULL,
                class_name TEXT NOT NULL,
                et_rows INTEGER NOT NULL,
                et_cols INTEGER NOT NULL,
                ft_rows INTEGER NOT NULL,
                ft_cols INTEGER NOT NULL
            );
        """)
        
        print(" Old table dropped and fresh SQLite table schema initialized successfully.")

        # 3. PARSE DATA ROWS FROM CSV
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            rows_inserted = 0
            for row in reader:
                part_no = row.get("Part No.:", "").strip()
                class_name = row.get("Class Name:", "").strip()
                
                raw_aid = row.get("Aruco ID:")
                aruco_id = int(float(raw_aid.strip())) if raw_aid and raw_aid.strip() else 0
                
                raw_et_r = row.get("Empty Tray R:")
                et_r = int(float(raw_et_r.strip())) if raw_et_r and raw_et_r.strip() else 6
                
                raw_et_c = row.get("Empty Tray C:")
                et_c = int(float(raw_et_c.strip())) if raw_et_c and raw_et_c.strip() else 9
                
                raw_ft_r = row.get("Filling Tray R:")
                ft_r = int(float(raw_ft_r.strip())) if raw_ft_r and raw_ft_r.strip() else 5
                
                raw_ft_c = row.get("Filling Tray C:")
                ft_c = int(float(raw_ft_c.strip())) if raw_ft_c and raw_ft_c.strip() else 8

                if aruco_id == 0 or not part_no or not class_name:
                    continue

                # AUTO-ORIENTATION AXIS STABILIZATION
                if et_r > et_c: et_r, et_c = et_c, et_r
                if ft_r > ft_c: ft_r, ft_c = ft_c, ft_r

                cursor.execute("""
                    INSERT INTO tray_configs 
                    (aruco_id, part_number, class_name, et_rows, et_cols, ft_rows, ft_cols)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                """, (aruco_id, part_no, class_name, et_r, et_c, ft_r, ft_c))
                rows_inserted += 1

        conn.commit()
        conn.close()
        print(f" Successfully seeded {rows_inserted} fresh records into local '{DB_PATH}' file.")
    except Exception as e:
        print(f" Failed to execute database synchronization sequence: {str(e)}")

if __name__ == "__main__":
    init_and_seed_sqlite()