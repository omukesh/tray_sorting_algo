import sqlite3

def init_configuration_database(db_path: str = "tray_config.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Create the persistent configuration scheme
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tray_configs (
            aruco_id INTEGER PRIMARY KEY,
            sku_name TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            ft_rows INTEGER NOT NULL,
            ft_cols INTEGER NOT NULL,
            et_rows INTEGER NOT NULL,
            et_cols INTEGER NOT NULL
        )
    """)
    
    # 2. Complete, aligned production config matrix (Renamed 22-class setup)
    production_matrix = [
        (1,  "blade_hpcr012",    3,  8, 5, 6, 9),
        (2,  "blade_hpcr022",    4,  8, 5, 6, 9),
        (3,  "blade_hpcr032",    5,  8, 5, 6, 9),
        (4,  "blade_hpcr042",    5,  8, 5, 6, 9),
        (5,  "blade_hpcr052",    6,  8, 5, 6, 9),
        (7,  "blade_hpcr072",    7,  8, 5, 6, 9),
        (10, "blade_hpcs001",    8,  8, 5, 6, 9),
        (11, "blade_hpcs002",    9,  8, 5, 6, 9),
        (13, "blade_hpcs004",    10, 8, 5, 6, 9),
        (14, "blade_hpcs005",    11, 8, 5, 6, 9),
        (15, "blade_hpcs006",    12, 8, 5, 6, 9),
        (16, "blade_hpcs007",    13, 8, 5, 6, 9),
        (18, "blade_hpcs008",    14, 8, 5, 6, 9),
        (19, "blade_hpcs009",    15, 8, 5, 6, 9),
        (20, "blade_hpcs011",    16, 8, 5, 6, 9),
        (30, "blade_hptr020",    17, 4, 7, 5, 7),
        (22, "blade_lpcr046",    18, 4, 7, 6, 9),
        (23, "blade_lpcr35032",  19, 8, 5, 6, 9),
        (24, "blade_lpcr35046",  20, 8, 5, 6, 9),
        (31, "blade_lptr050",    21, 4, 7, 5, 7)
    ]
    
    cursor.executemany("""
        INSERT OR REPLACE INTO tray_configs 
        (aruco_id, sku_name, class_id, ft_rows, ft_cols, et_rows, et_cols)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, production_matrix)
    
    conn.commit()
    conn.close()
    print("SQLite Config Matrix initialized securely with zero drift bounds.")

if __name__ == "__main__":
    init_configuration_database()