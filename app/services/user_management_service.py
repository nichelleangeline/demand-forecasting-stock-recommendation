# app/services/user_management_service.py

from typing import List, Dict, Optional
import secrets
import string

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.db import engine
from app.services.auth_service import hash_password  # ← pakai ini, bukan bikin sendiri


# =========================
# Helper: password
# =========================
def _generate_temp_password(length: int = 10) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


# =========================
# Query user
# =========================
def get_all_users() -> List[Dict]:
    """
    Ambil semua user untuk halaman admin.
    """
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    user_id,
                    full_name,
                    email,
                    role,
                    is_active
                FROM user_account
                ORDER BY full_name ASC
                """
            )
        ).mappings().all()

    return [dict(r) for r in rows]


# =========================
# Create user
# =========================
def create_user(
    full_name: str,
    email: str,
    role: str = "user",
    temp_password: Optional[str] = None,
) -> Dict:
    """
    Buat user baru.

    - Kalau temp_password None -> generate password acak.
    - Password selalu di-hash sebelum masuk DB.
    - Set must_change_password = 1 supaya dipaksa ganti di login pertama.
    """

    raw_password = (temp_password or _generate_temp_password()).strip()
    if not raw_password:
        raise ValueError("Password tidak boleh kosong setelah trimming.")

    password_hash = hash_password(raw_password)  # <- bcrypt dari auth_service

    try:
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    INSERT INTO user_account (
                        full_name,
                        email,
                        password_hash,
                        role,
                        is_active,
                        must_change_password
                    )
                    VALUES (
                        :full_name,
                        :email,
                        :password_hash,
                        :role,
                        1,
                        1
                    )
                    """
                ),
                {
                    "full_name": full_name.strip(),
                    "email": email.strip().lower(),
                    "password_hash": password_hash,
                    "role": role,
                },
            )
            user_id = result.lastrowid

    except IntegrityError as e:
        raise ValueError("Email sudah terdaftar, gunakan email lain.") from e

    return {
        "user_id": user_id,
        "email": email.strip().lower(),
        "temp_password": raw_password,
    }


# =========================
# Update data dasar user
# =========================
def update_user_basic(
    user_id: int,
    full_name: str,
    role: str,
    is_active: int = 1,
) -> None:
    """
    Update nama, role, dan status aktif user.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE user_account
                SET
                    full_name = :full_name,
                    role      = :role,
                    is_active = :is_active
                WHERE user_id = :user_id
                """
            ),
            {
                "full_name": full_name,
                "role": role,
                "is_active": int(is_active),
                "user_id": user_id,
            },
        )


def delete_user_account(user_id: int):
    """
    Hapus permanen akun dari user_account.
    Hanya boleh untuk akun yang sudah nonaktif (is_active = 0).
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM user_account
                WHERE user_id = :uid
                  AND is_active = 0
                """
            ),
            {"uid": user_id},
        )
