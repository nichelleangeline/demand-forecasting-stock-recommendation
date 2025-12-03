# app/services/auth_service.py

import hashlib
import secrets
import string
from datetime import datetime, timedelta

from sqlalchemy import text
from app.db import engine


# =====================================================
# 1. PASSWORD HASHING & USER LOOKUP
# =====================================================

def hash_password(password: str) -> str:
    """
    Hash password pakai SHA256.
    Sesuaikan kalau nanti mau ganti algo.
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(plain_password: str, stored_hash: str | None) -> bool:
    """
    Verifikasi password dengan hash yang tersimpan.
    """
    if not stored_hash:
        return False
    return hash_password(plain_password) == stored_hash


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def get_user_by_email(email: str):
    """
    Ambil satu user dari user_account berdasarkan email.
    """
    email_norm = _normalize_email(email)

    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT
                    user_id,
                    full_name,
                    email,
                    password_hash,
                    role,
                    is_active,
                    created_at,
                    must_change_password,
                    reset_token,
                    reset_expiry,
                    reset_code_used,
                    reset_requested
                FROM user_account
                WHERE email = :email
                """
            ),
            {"email": email_norm},
        ).mappings().first()

    if not row:
        return None
    return dict(row)


# =====================================================
# 2. RESET FLOW (REQUEST → APPROVE → CODE)
# =====================================================

def generate_reset_code(length: int = 6) -> str:
    """
    Bikin kode numeric, misalnya 6 digit.
    """
    return "".join(secrets.choice("0123456789") for _ in range(length))


def mark_reset_requested(email: str) -> None:
    """
    User mengajukan permintaan reset password.
    Di sini BELUM bikin kode. Hanya tandai bahwa user minta reset.

    - reset_requested = 1
    - reset_token / reset_expiry dibersihkan (biar bersih)
    """
    email_norm = _normalize_email(email)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE user_account
                SET reset_requested = 1,
                    reset_token     = NULL,
                    reset_expiry    = NULL,
                    reset_code_used = 0
                WHERE email = :email
                """
            ),
            {"email": email_norm},
        )


def set_reset_code(email: str, ttl_minutes: int = 30) -> str:
    """
    Generate & simpan kode reset untuk user_account.

    Disimpan di:
      - reset_token
      - reset_expiry
      - reset_code_used = 0
      - reset_requested = 0 (karena sudah di-approve)
    """
    email_norm = _normalize_email(email)

    code = generate_reset_code(6)
    expiry = datetime.now() + timedelta(minutes=ttl_minutes)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE user_account
                SET reset_token     = :code,
                    reset_expiry    = :exp,
                    reset_code_used = 0,
                    reset_requested = 0
                WHERE email = :email
                """
            ),
            {"code": code, "exp": expiry, "email": email_norm},
        )
    return code


def get_pending_reset_requests():
    """
    Untuk halaman admin:
    Ambil semua akun yang SUDAH minta reset namun BELUM dibuatkan kode.

    Kriteria:
      - reset_requested = 1
    """
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    user_id,
                    full_name,
                    email,
                    created_at
                FROM user_account
                WHERE reset_requested = 1
                ORDER BY created_at DESC
                """
            )
        ).mappings().all()

    return [dict(r) for r in rows]


def admin_approve_reset(email: str, ttl_minutes: int = 30) -> str:
    """
    Dipanggil admin ketika menyetujui permintaan reset.
    """
    return set_reset_code(email, ttl_minutes=ttl_minutes)


# =====================================================
# 3. VERIFIKASI & CLEANUP RESET CODE
# =====================================================

def verify_reset_code(email: str, code: str) -> bool:
    """
    Cek apakah kode reset valid:
      - email cocok
      - reset_token sama
      - belum expired
      - belum dipakai
    """
    email_norm = _normalize_email(email)

    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT reset_token, reset_expiry, reset_code_used
                FROM user_account
                WHERE email = :email
                """
            ),
            {"email": email_norm},
        ).mappings().first()

    if not row:
        return False

    if row["reset_code_used"]:
        return False

    if not row["reset_token"] or row["reset_token"] != code:
        return False

    if row["reset_expiry"] is None or row["reset_expiry"] < datetime.now():
        return False

    return True


def mark_reset_code_used_and_flag_change(email: str) -> None:
    """
    Kode dipakai:
      - paksa user ganti password (must_change_password = 1)
      - bersihkan reset_*
    """
    email_norm = _normalize_email(email)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE user_account
                SET must_change_password = 1,
                    reset_token          = NULL,
                    reset_expiry         = NULL,
                    reset_code_used      = 0,
                    reset_requested      = 0
                WHERE email = :email
                """
            ),
            {"email": email_norm},
        )


def get_pending_reset_codes():
    """
    Untuk halaman admin:
    Ambil semua akun yang punya reset_token AKTIF (belum expired).
    """
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    email,
                    reset_token,
                    reset_expiry,
                    reset_code_used
                FROM user_account
                WHERE reset_token IS NOT NULL
                  AND reset_expiry IS NOT NULL
                  AND reset_expiry >= NOW()
                ORDER BY reset_expiry DESC
                """
            )
        ).mappings().all()

    return [dict(r) for r in rows]


def cleanup_reset_codes():
    """
    Bersihkan semua reset code yang sudah lewat masa berlakunya.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE user_account
                SET reset_token     = NULL,
                    reset_expiry    = NULL,
                    reset_code_used = 0
                WHERE reset_expiry IS NOT NULL
                  AND reset_expiry < NOW()
                """
            )
        )


# =====================================================
# 4. UPDATE PASSWORD FINAL
# =====================================================

def update_password_and_clear_flag(email: str, new_password: str) -> None:
    """
    Ganti password permanen & bersihkan flag reset.
    """
    email_norm = _normalize_email(email)
    pwd_hash = hash_password(new_password)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE user_account
                SET password_hash        = :pwd,
                    must_change_password = 0,
                    reset_token          = NULL,
                    reset_expiry         = NULL,
                    reset_code_used      = 0,
                    reset_requested      = 0
                WHERE email = :email
                """
            ),
            {"pwd": pwd_hash, "email": email_norm},
        )


# =====================================================
# 5. CREATE USER (ADMIN)
# =====================================================

def _generate_temp_password(length: int = 10) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))


def create_user(
    full_name: str,
    email: str,
    role: str = "user",
    temp_password: str | None = None,
):
    """
    Buat akun baru:
      - kalau admin isi password → pakai itu
      - kalau kosong → generate password random
      - simpan hash ke user_account.password_hash
      - set must_change_password = 1 (wajib ganti di login pertama)
    """
    email_norm = _normalize_email(email)
    name_norm = full_name.strip()

    if not temp_password:
        temp_password = _generate_temp_password(10)

    pwd_hash = hash_password(temp_password)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO user_account (
                    full_name,
                    email,
                    password_hash,
                    role,
                    is_active,
                    must_change_password,
                    reset_token,
                    reset_expiry,
                    reset_code_used,
                    reset_requested,
                    created_at
                )
                VALUES (
                    :full_name,
                    :email,
                    :password_hash,
                    :role,
                    1,
                    1,
                    NULL,
                    NULL,
                    0,
                    0,
                    NOW()
                )
                """
            ),
            {
                "full_name": name_norm,
                "email": email_norm,
                "password_hash": pwd_hash,
                "role": role,
            },
        )

    return {
        "email": email_norm,
        "temp_password": temp_password,
    }
