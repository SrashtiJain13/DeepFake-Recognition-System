import bcrypt
from db import get_connection


def _normalize_credentials(username, password):
    username = (username or "").strip()
    password = password or ""
    return username, password


def register_user(username, password):
    username, password = _normalize_credentials(username, password)
    if not username or not password:
        return False, "Username and password are required"

    conn = get_connection()
    cursor = conn.cursor()

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, hashed.decode("utf-8"))
        )
        conn.commit()
        return True, "Registration successful"
    except Exception:
        conn.rollback()
        return False, "Unable to register. The username may already exist."

    finally:
        cursor.close()
        conn.close()


def login_user(username, password):
    username, password = _normalize_credentials(username, password)
    if not username or not password:
        return False

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT password FROM users WHERE username=%s",
        (username,)
    )

    result = cursor.fetchone()

    cursor.close()
    conn.close()

    if result is None:
        return False

    stored_password = str(result[0]).encode("utf-8")

    if bcrypt.checkpw(password.encode("utf-8"), stored_password):
        return True

    return False


def reset_password(username, new_password):
    username, new_password = _normalize_credentials(username, new_password)
    if not username or not new_password:
        return False, "Username and new password are required"

    conn = get_connection()
    cursor = conn.cursor()

    hashed = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    try:
        cursor.execute(
            "UPDATE users SET password=%s WHERE username=%s",
            (hashed, username)
        )
        conn.commit()

        if cursor.rowcount == 0:
            return False, "User not found"

        return True, "Password updated successfully"
    finally:
        cursor.close()
        conn.close()
