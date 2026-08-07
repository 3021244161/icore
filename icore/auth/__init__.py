"""
icore.auth - v0.6 鉴权与授权模块。

提供 API Key 鉴权、JWT 鉴权、RBAC 角色权限控制，以及 FastAPI
依赖注入工厂。全部基于 Python 标准库 + Pydantic 实现，**不引入
python-jose / bcrypt / passlib** 等外部鉴权库：

    1. **APIKeyAuth** — 从 ``X-API-Key`` 请求头验证，支持 SHA-256
       哈希存储与 ``config/users.yaml`` 配置加载。
    2. **JWTAuth** — 基于 ``hmac`` + ``hashlib`` + ``base64`` 的
       HS256 JWT 签发与验证（自签发，dev/test 友好；生产可替换为
       外部 JWKS 校验）。
    3. **RBAC** — admin / developer / viewer 三角色权限矩阵。
    4. **create_auth_dependency** — FastAPI 依赖工厂，支持豁免路径、
       API Key / JWT 复合鉴权、RBAC 权限检查。

模块依赖：
    仅依赖标准库（``hmac`` / ``hashlib`` / ``base64`` / ``json`` /
    ``time`` / ``logging``）+ Pydantic v2 + ``icore.exceptions``。
    YAML 加载走懒导入（``yaml`` 未安装时返回空配置）。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import Request
from pydantic import BaseModel, Field

from icore.exceptions import AuthenticationError, AuthorizationError

__all__ = [
    "Principal",
    "APIKeyAuth",
    "JWTAuth",
    "RBAC",
    "create_auth_dependency",
    "create_permission_dependency",
    "AuthBundle",
    "load_users_yaml",
]

logger = logging.getLogger(__name__)


# ============================================================================
# Principal — authenticated identity
# ============================================================================


class Principal(BaseModel):
    """An authenticated identity produced by APIKeyAuth / JWTAuth.

    Attributes:
        user_id:      Unique user identifier (e.g. ``"admin"``, ``"alice"``).
        role:         Role name (``"admin"`` / ``"developer"`` / ``"viewer"``
                      or a custom role with an entry in
                      :attr:`RBAC.PERMISSIONS`).
        api_key:      The API key used for authentication (``None`` for JWT).
        permissions:  Resolved permission list for the role. Empty when the
                      role is unknown; callers can also rely on
                      :meth:`RBAC.has_permission` which checks the role
                      table directly.
    """

    user_id: str
    role: str
    api_key: Optional[str] = None
    permissions: list[str] = Field(default_factory=list)


# ============================================================================
# RBAC — Role-Based Access Control
# ============================================================================


class RBAC:
    """Role-based access control with a static permission matrix.

    Permission strings are free-form identifiers (e.g. ``"invoke"``,
    ``"history"``, ``"health"``, ``"config"``, ``"admin"``). The
    :attr:`PERMISSIONS` dict maps each role to its allowed permissions.
    """

    PERMISSIONS: dict[str, list[str]] = {
        "admin": ["invoke", "history", "health", "config", "admin"],
        "developer": ["invoke", "history", "health"],
        "viewer": ["health", "history"],
    }

    @classmethod
    def permissions_for(cls, role: str) -> list[str]:
        """Return the permission list for ``role`` (empty if unknown)."""
        return list(cls.PERMISSIONS.get(role, []))

    def has_permission(self, principal: Principal, required: str) -> bool:
        """Check whether ``principal`` has the ``required`` permission.

        Returns ``False`` if ``principal`` is ``None``. Checks both the
        principal's own ``permissions`` list and the role's permission
        table (so principals created without explicit permissions still
        work).
        """
        if principal is None:
            return False
        if required in principal.permissions:
            return True
        role_perms = self.PERMISSIONS.get(principal.role, [])
        return required in role_perms

    def require(self, principal: Principal, required: str) -> None:
        """Raise ``AuthenticationError`` or ``AuthorizationError``.

        - ``principal is None`` → :class:`AuthenticationError` (not
          authenticated).
        - ``principal`` lacks ``required`` → :class:`AuthorizationError`
          (authenticated but insufficient privileges).
        """
        if principal is None:
            raise AuthenticationError("Authentication required")
        if not self.has_permission(principal, required):
            raise AuthorizationError(
                f"Role '{principal.role}' lacks required permission "
                f"'{required}'"
            )


# ============================================================================
# API Key Authentication
# ============================================================================


def _sha256_hash(value: str) -> str:
    """Return the hex SHA-256 digest of ``value`` (UTF-8 encoded)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_users_yaml(path: str | Path) -> dict[str, Any]:
    """Load a ``users.yaml`` file and return the parsed dict.

    Returns ``{}`` if the file does not exist or PyYAML is not installed.
    Does NOT interpolate ``${VAR}`` placeholders (API keys should be
    pre-resolved or stored directly).
    """
    p = Path(path)
    if not p.exists():
        logger.debug("Users config file not found, skipping: %s", p)
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover
        logger.warning("PyYAML not installed; cannot load %s", p)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


class APIKeyAuth:
    """API Key authentication backed by a ``{api_key: user_id}`` mapping.

    Args:
        users:        Mapping of ``api_key → user_id``. When
                      ``hashed_keys`` is ``True``, the keys are treated as
                      SHA-256 hashes and incoming keys are hashed before
                      lookup.
        user_roles:   Optional mapping of ``user_id → role``. When a user
                      is not in this mapping, the default role
                      ``"developer"`` is assigned.
        hashed_keys:  When ``True``, stored keys are SHA-256 hashes;
                      incoming API keys are hashed before comparison.
    """

    DEFAULT_ROLE: str = "developer"

    def __init__(
        self,
        users: dict[str, str],
        user_roles: Optional[dict[str, str]] = None,
        hashed_keys: bool = False,
    ) -> None:
        # Store a copy so external mutation doesn't affect the auth.
        self._users: dict[str, str] = dict(users)
        self._user_roles: dict[str, str] = dict(user_roles) if user_roles else {}
        self._hashed: bool = hashed_keys
        if hashed_keys:
            # Pre-hash all stored keys so lookup is O(1).
            self._users = {
                _sha256_hash(k): v for k, v in self._users.items()
            }

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        hashed_keys: bool = False,
    ) -> "APIKeyAuth":
        """Build an :class:`APIKeyAuth` from a ``users.yaml`` file.

        Expected YAML format::

            users:
              admin:
                api_key: "icore-admin-xxx"
                role: admin
              developer:
                api_key: "icore-dev-yyy"
                role: developer

        Returns an empty auth (no users) if the file is absent.
        """
        data = load_users_yaml(path)
        users: dict[str, str] = {}
        user_roles: dict[str, str] = {}
        for user_id, info in (data.get("users") or {}).items():
            if not isinstance(info, dict):
                continue
            api_key = info.get("api_key")
            role = info.get("role", cls.DEFAULT_ROLE)
            if api_key:
                users[api_key] = user_id
                user_roles[user_id] = role
        return cls(users, user_roles=user_roles, hashed_keys=hashed_keys)

    async def authenticate(self, api_key: str) -> Optional[Principal]:
        """Authenticate an API key, returning a :class:`Principal` or ``None``.

        Returns ``None`` for empty or unknown keys.
        """
        if not api_key:
            return None
        lookup_key = _sha256_hash(api_key) if self._hashed else api_key
        user_id = self._users.get(lookup_key)
        if user_id is None:
            return None
        role = self._user_roles.get(user_id, self.DEFAULT_ROLE)
        permissions = RBAC.permissions_for(role)
        return Principal(
            user_id=user_id,
            role=role,
            api_key=api_key,
            permissions=permissions,
        )


# ============================================================================
# JWT Authentication (HS256, stdlib-only)
# ============================================================================


def _b64url_encode(data: bytes) -> str:
    """Base64url-encode ``data`` without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    """Base64url-decode ``s``, adding padding as needed."""
    padding = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)


def _hmac_sha256_sign(secret: str, message: str) -> str:
    """Return the base64url-encoded HMAC-SHA256 of ``message``."""
    sig = hmac.new(
        secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
    ).digest()
    return _b64url_encode(sig)


class JWTAuth:
    """HS256 JWT issuer and verifier (stdlib-only, no python-jose).

    Args:
        secret:     Shared secret for signing and verification.
        algorithm:  JWT algorithm. Only ``"HS256"`` is supported (raises
                    ``ValueError`` otherwise).
    """

    def __init__(self, secret: str, algorithm: str = "HS256") -> None:
        if algorithm != "HS256":
            raise ValueError(
                f"Unsupported algorithm: {algorithm!r}. Only 'HS256' is supported."
            )
        self._secret = secret
        self._algorithm = algorithm

    def issue_token(
        self, user_id: str, role: str, expires_in: int = 3600
    ) -> str:
        """Issue a signed JWT for ``(user_id, role)`` valid for ``expires_in`` seconds.

        The token payload contains:
            - ``sub``:  user_id
            - ``role``: role
            - ``iat``:  issued-at (unix timestamp)
            - ``exp``:  expiration (unix timestamp)
        """
        header = {"alg": self._algorithm, "typ": "JWT"}
        now = int(time.time())
        payload = {
            "sub": user_id,
            "role": role,
            "iat": now,
            "exp": now + expires_in,
        }
        header_b64 = _b64url_encode(
            json.dumps(header, separators=(",", ":")).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        signing_input = f"{header_b64}.{payload_b64}"
        signature = _hmac_sha256_sign(self._secret, signing_input)
        return f"{signing_input}.{signature}"

    async def verify_token(self, token: str) -> Optional[Principal]:
        """Verify a JWT and return a :class:`Principal` or ``None``.

        Returns ``None`` for: malformed tokens, signature mismatch,
        wrong algorithm, expired tokens, or missing required claims.
        """
        if not token:
            return None
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            header_b64, payload_b64, signature = parts

            # Verify signature using constant-time comparison.
            signing_input = f"{header_b64}.{payload_b64}"
            expected_sig = _hmac_sha256_sign(self._secret, signing_input)
            if not hmac.compare_digest(signature, expected_sig):
                return None

            # Verify header algorithm.
            header = json.loads(_b64url_decode(header_b64))
            if header.get("alg") != self._algorithm:
                return None

            # Decode and validate payload.
            payload = json.loads(_b64url_decode(payload_b64))
            now = int(time.time())
            exp = payload.get("exp", 0)
            if exp < now:
                return None  # expired

            user_id = payload.get("sub")
            role = payload.get("role")
            if not user_id or not role:
                return None

            permissions = RBAC.permissions_for(role)
            return Principal(
                user_id=str(user_id),
                role=str(role),
                permissions=permissions,
            )
        except Exception:
            logger.debug("JWT verification failed", exc_info=True)
            return None


# ============================================================================
# FastAPI Dependency Factory
# ============================================================================

#: Default paths exempt from authentication.
_DEFAULT_EXEMPT_PATHS: frozenset[str] = frozenset(
    {"/health", "/docs", "/openapi.json", "/redoc"}
)


def create_auth_dependency(
    api_key_auth: Optional[APIKeyAuth] = None,
    jwt_auth: Optional[JWTAuth] = None,
    rbac: Optional[RBAC] = None,
    required_permission: str = "invoke",
    exempt_paths: Optional[set[str]] = None,
) -> Callable[..., Any]:
    """Create a FastAPI dependency callable for authentication + authorization.

    The returned dependency:

        1. **Exempt paths**: requests to paths in ``exempt_paths`` return a
           read-only anonymous principal without any header checks.
        2. **API Key**: if ``X-API-Key`` header is present and
           ``api_key_auth`` is configured, authenticate via API key.
        3. **JWT**: if ``Authorization: Bearer <token>`` header is present
           and ``jwt_auth`` is configured, authenticate via JWT.
        4. **Fallback**: if neither header is present or both fail, raise
           :class:`AuthenticationError` (HTTP 401).
        5. **RBAC**: if ``rbac`` is configured, call ``rbac.require()`` to
           enforce ``required_permission`` (raises
           :class:`AuthorizationError` / HTTP 403 on failure).

    Either API Key or JWT can satisfy authentication (composite mode).
    Both authenticators can be configured simultaneously.

    Args:
        api_key_auth:        Optional :class:`APIKeyAuth` instance.
        jwt_auth:            Optional :class:`JWTAuth` instance.
        rbac:                Optional :class:`RBAC` instance. Defaults to a
                             new ``RBAC()`` when ``None``.
        required_permission: Permission required for non-exempt paths.
        exempt_paths:        Paths that skip authentication. Defaults to
                             ``{"/health", "/docs", "/openapi.json", "/redoc"}``.

    Returns:
        An async callable suitable for ``Depends()`` that returns a
        :class:`Principal`.
    """
    if exempt_paths is None:
        exempt_paths = set(_DEFAULT_EXEMPT_PATHS)
    effective_rbac = rbac if rbac is not None else RBAC()

    # Capture in closure.
    _api_key_auth = api_key_auth
    _jwt_auth = jwt_auth
    _rbac = effective_rbac
    _required = required_permission
    _exempt = exempt_paths

    async def _auth_dependency(request: Request) -> Principal:
        """FastAPI dependency: authenticate and authorize the request."""
        # 1. Exempt path — return anonymous viewer principal.
        path = request.url.path
        if path in _exempt:
            return Principal(
                user_id="anonymous",
                role="viewer",
                permissions=RBAC.permissions_for("viewer"),
            )

        # 2. Extract credentials from headers.
        x_api_key = request.headers.get("X-API-Key") or request.headers.get(
            "x-api-key"
        )
        authorization = request.headers.get("Authorization") or request.headers.get(
            "authorization"
        )

        principal: Optional[Principal] = None

        # 3. Try API Key authentication.
        if _api_key_auth is not None and x_api_key:
            principal = await _api_key_auth.authenticate(x_api_key)

        # 4. Try JWT authentication (if API key didn't succeed).
        if principal is None and _jwt_auth is not None and authorization:
            token = _extract_bearer_token(authorization)
            if token:
                principal = await _jwt_auth.verify_token(token)

        # 5. No valid credentials.
        if principal is None:
            raise AuthenticationError(
                "Valid API key (X-API-Key) or JWT (Authorization: Bearer) required"
            )

        # 6. RBAC permission check.
        _rbac.require(principal, _required)

        return principal

    return _auth_dependency


def _extract_bearer_token(authorization: str) -> Optional[str]:
    """Extract the token from an ``Authorization: Bearer <token>`` header.

    Returns ``None`` if the header is not a Bearer token.
    """
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


# ============================================================================
# Permission-scoped dependency factory + AuthBundle
# ============================================================================


@dataclass
class AuthBundle:
    """打包鉴权组件，方便 bootstrap 一次性传递。

    Attributes:
        api_key_auth:  API Key 鉴权器（或 None）
        jwt_auth:      JWT 鉴权器（或 None）
        rbac:          RBAC 权限矩阵
        exempt_paths:  免鉴权路径集合
    """

    api_key_auth: Optional["APIKeyAuth"] = None
    jwt_auth: Optional["JWTAuth"] = None
    rbac: Optional["RBAC"] = None
    exempt_paths: Optional[set[str]] = None

    def dependency(
        self, required_permission: str = "invoke"
    ) -> Callable[..., Any]:
        """生成一个指定权限的 FastAPI dependency。"""
        return create_auth_dependency(
            api_key_auth=self.api_key_auth,
            jwt_auth=self.jwt_auth,
            rbac=self.rbac,
            required_permission=required_permission,
            exempt_paths=self.exempt_paths,
        )


def create_permission_dependency(
    bundle: "AuthBundle",
    required_permission: str,
) -> Callable[..., Any]:
    """基于已有的 AuthBundle 生成一个指定权限的 dependency。

    用于运维端点按角色鉴权（如 /history 需 ``history`` 权限，
    /admin/dlq/* 需 ``admin`` 权限）。

    Args:
        bundle:             已构建的 AuthBundle
        required_permission: 该端点所需的权限字符串

    Returns:
        async dependency callable for ``Depends()``.
    """
    return bundle.dependency(required_permission=required_permission)
