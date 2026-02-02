import hashlib
from sqlalchemy import text
from app.db import engine

def hash_password(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main():
    admin_email = "admin@tktw.local"
    admin_name = "Admin TKTW"
    raw_password = "admin123" 

    pwd_hash = hash_password(raw_password)

    with engine.begin() as conn:
        res = conn.execute(
            text(
                "SELECT user_id FROM user_account WHERE email = :email"
            ),
            {"email": admin_email},
        ).fetchone()

        if res:
            print("Admin sudah ada, user_id:", res[0])
            return

        conn.execute(
            text(
                """
                INSERT INTO user_account (full_name, email, password_hash, role, is_active)
                VALUES (:full_name, :email, :password_hash, :role, :is_active)
                """
            ),
            {
                "full_name": admin_name,
                "email": admin_email,
                "password_hash": pwd_hash,
                "role": "admin",
                "is_active": 1,
            },
        )

        print("Admin created.")
        print("Email   :", admin_email)
        print("Password:", raw_password)


if __name__ == "__main__":
    main()
