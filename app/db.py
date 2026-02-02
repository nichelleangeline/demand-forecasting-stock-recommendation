from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DB_USER = "tktw_app"
DB_PASS = "tktw123"
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_NAME = "tktw_db"

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def test_connection():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DATABASE()"))
        db_name = result.scalar()
        print("Connected to:", db_name)

        tables = conn.execute(text(
            "SHOW TABLES"
        )).fetchall()
        print("Tables:", [row[0] for row in tables])


if __name__ == "__main__":
    test_connection()
