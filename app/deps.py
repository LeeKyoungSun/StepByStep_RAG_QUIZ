# app/deps.py
import os, jwt
from typing import Optional
from fastapi import Header, HTTPException
from app.schemas.common import User

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")   # HS256 or RS256
JWT_SECRET    = os.getenv("JWT_SECRET")               # HS256
JWT_PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY")          # RS256 (PEM)

def _decode_jwt(token: str) -> dict:
    try:
        if JWT_ALGORITHM.startswith("HS"):
            if not JWT_SECRET:
                raise HTTPException(500, detail={"error": {"message": "JWT_SECRET not set"}})
            return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if JWT_ALGORITHM.startswith("RS"):
            if not JWT_PUBLIC_KEY:
                raise HTTPException(500, detail={"error": {"message": "JWT_PUBLIC_KEY not set"}})
            return jwt.decode(token, JWT_PUBLIC_KEY, algorithms=[JWT_ALGORITHM])
        raise HTTPException(500, detail={"error": {"message": f"Unsupported alg: {JWT_ALGORITHM}"}})
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, detail={"error": {"message": "TOKEN_EXPIRED"}})
    except jwt.InvalidTokenError:
        raise HTTPException(401, detail={"error": {"message": "INVALID_TOKEN"}})

def get_current_user(
    Authorization: Optional[str] = Header(None, alias="Authorization"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
) -> User:
    """
    운영: 기본은 JWT를 요구. (FE -> BE -> AI)
    BE가 게이트웨이로서 X-User-Id/X-User-Email을 넣어줄 수도 있게 허용.
    """
    if Authorization:
        if not Authorization.startswith("Bearer "):
            raise HTTPException(401, detail={"error": {"message": "UNAUTHORIZED"}})
        payload = _decode_jwt(Authorization.split(" ", 1)[1].strip())
        try:
            return User(userId=int(payload["userId"]), email=payload["email"])
        except Exception:
            raise HTTPException(401, detail={"error": {"message": "INVALID_CLAIMS"}})

    # 게이트웨이(백엔드)가 전달한 헤더 허용
    if x_user_id:
        try:
            uid = int(x_user_id)
        except ValueError:
            raise HTTPException(401, detail={"error": {"message": "INVALID_X_USER_ID"}})
        return User(userId=uid, email=x_user_email or "gateway@local")

    raise HTTPException(401, detail={"error": {"message": "UNAUTHORIZED"}})
