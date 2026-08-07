"""
Tests for icore.auth (v0.6 鉴权与授权模块).

Covers:
    - APIKeyAuth: valid/invalid/empty keys, role resolution, hashed storage,
      YAML loading.
    - JWTAuth: issue + verify, expired token, tampered token, wrong secret,
      malformed token, unsupported algorithm.
    - RBAC: permission matrix for admin/developer/viewer, has_permission,
      require raises AuthenticationError / AuthorizationError.
    - create_auth_dependency: exempt paths, API Key flow, JWT flow,
      missing auth (401), insufficient permissions (403), composite auth
      (either API Key or JWT), custom exempt paths, config-level endpoint.
"""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from icore.auth import (
    APIKeyAuth,
    JWTAuth,
    Principal,
    RBAC,
    create_auth_dependency,
    load_users_yaml,
)
from icore.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ICoreError,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def rbac() -> RBAC:
    return RBAC()


@pytest.fixture
def api_key_auth() -> APIKeyAuth:
    """API Key auth with three roles: admin, developer, viewer."""
    return APIKeyAuth(
        users={
            "key-admin-123": "admin",
            "key-dev-456": "devuser",
            "key-viewer-789": "viewer1",
        },
        user_roles={
            "admin": "admin",
            "devuser": "developer",
            "viewer1": "viewer",
        },
    )


@pytest.fixture
def jwt_auth() -> JWTAuth:
    return JWTAuth(secret="test-secret-key-2026")


@pytest.fixture
def composite_auth_app(
    api_key_auth: APIKeyAuth,
    jwt_auth: JWTAuth,
    rbac: RBAC,
) -> FastAPI:
    """A test FastAPI app with both API Key and JWT auth, plus RBAC.

    Endpoints:
        - GET  /health  (exempt, no auth needed)
        - POST /invoke  (requires "invoke" permission)
        - GET  /config  (requires "config" permission — admin only)
    """
    app = FastAPI()

    @app.exception_handler(ICoreError)
    async def handle_icore_error(request, exc: ICoreError):
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.to_dict(),
        )

    health_dep = create_auth_dependency(
        api_key_auth=api_key_auth,
        jwt_auth=jwt_auth,
        rbac=rbac,
        required_permission="health",
    )
    invoke_dep = create_auth_dependency(
        api_key_auth=api_key_auth,
        jwt_auth=jwt_auth,
        rbac=rbac,
        required_permission="invoke",
    )
    config_dep = create_auth_dependency(
        api_key_auth=api_key_auth,
        jwt_auth=jwt_auth,
        rbac=rbac,
        required_permission="config",
    )

    @app.get("/health")
    async def health(principal: Principal = Depends(health_dep)):
        return {"status": "ok", "user": principal.user_id}

    @app.post("/invoke")
    async def invoke(principal: Principal = Depends(invoke_dep)):
        return {
            "status": "invoked",
            "user": principal.user_id,
            "role": principal.role,
        }

    @app.get("/config")
    async def config(principal: Principal = Depends(config_dep)):
        return {"config": "value", "user": principal.user_id}

    return app


# ===========================================================================
# APIKeyAuth
# ===========================================================================


class TestAPIKeyAuth:
    """API Key authentication."""

    @pytest.mark.asyncio
    async def test_valid_api_key_returns_principal(self, api_key_auth: APIKeyAuth):
        principal = await api_key_auth.authenticate("key-admin-123")
        assert principal is not None
        assert principal.user_id == "admin"
        assert principal.role == "admin"
        assert principal.api_key == "key-admin-123"
        assert "invoke" in principal.permissions
        assert "config" in principal.permissions

    @pytest.mark.asyncio
    async def test_invalid_api_key_returns_none(self, api_key_auth: APIKeyAuth):
        principal = await api_key_auth.authenticate("wrong-key")
        assert principal is None

    @pytest.mark.asyncio
    async def test_empty_api_key_returns_none(self, api_key_auth: APIKeyAuth):
        principal = await api_key_auth.authenticate("")
        assert principal is None

    @pytest.mark.asyncio
    async def test_developer_role_permissions(self, api_key_auth: APIKeyAuth):
        principal = await api_key_auth.authenticate("key-dev-456")
        assert principal is not None
        assert principal.role == "developer"
        assert "invoke" in principal.permissions
        assert "config" not in principal.permissions

    @pytest.mark.asyncio
    async def test_hashed_key_storage_works(self):
        """When hashed_keys=True, stored keys are SHA-256 hashes."""
        auth = APIKeyAuth(
            users={"my-secret-key": "user1"},
            user_roles={"user1": "developer"},
            hashed_keys=True,
        )
        principal = await auth.authenticate("my-secret-key")
        assert principal is not None
        assert principal.user_id == "user1"

    @pytest.mark.asyncio
    async def test_hashed_key_wrong_key_returns_none(self):
        auth = APIKeyAuth(
            users={"my-secret-key": "user1"},
            hashed_keys=True,
        )
        principal = await auth.authenticate("wrong-key")
        assert principal is None

    @pytest.mark.asyncio
    async def test_default_role_when_no_role_mapping(self):
        """Users without a role mapping get the default 'developer' role."""
        auth = APIKeyAuth(users={"key-no-role": "user1"})
        principal = await auth.authenticate("key-no-role")
        assert principal is not None
        assert principal.role == "developer"

    def test_from_yaml_loads_users(self):
        """APIKeyAuth.from_yaml parses a users.yaml file."""
        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "users.yaml"
            yaml_path.write_text(
                """
users:
  admin:
    api_key: "icore-admin-xxx"
    role: admin
  devuser:
    api_key: "icore-dev-yyy"
    role: developer
""",
                encoding="utf-8",
            )
            auth = APIKeyAuth.from_yaml(yaml_path)
            assert len(auth._users) == 2

    def test_from_yaml_missing_file_returns_empty(self):
        auth = APIKeyAuth.from_yaml("/nonexistent/path/users.yaml")
        assert len(auth._users) == 0


# ===========================================================================
# JWTAuth
# ===========================================================================


class TestJWTAuth:
    """JWT issue + verify."""

    @pytest.mark.asyncio
    async def test_issue_and_verify_token(self, jwt_auth: JWTAuth):
        token = jwt_auth.issue_token("alice", "admin", expires_in=3600)
        principal = await jwt_auth.verify_token(token)
        assert principal is not None
        assert principal.user_id == "alice"
        assert principal.role == "admin"
        assert "invoke" in principal.permissions
        assert "config" in principal.permissions

    @pytest.mark.asyncio
    async def test_expired_token_returns_none(self, jwt_auth: JWTAuth):
        """Issue a token that's already expired."""
        token = jwt_auth.issue_token("bob", "developer", expires_in=-10)
        principal = await jwt_auth.verify_token(token)
        assert principal is None

    @pytest.mark.asyncio
    async def test_tampered_token_returns_none(self, jwt_auth: JWTAuth):
        token = jwt_auth.issue_token("alice", "admin")
        # Tamper with the payload section.
        parts = token.split(".")
        tampered = f"{parts[0]}.eyJzdWIiOiJoYWNrZXIiLCJyb2xlIjoiYWRtaW4ifQ.{parts[2]}"
        principal = await jwt_auth.verify_token(tampered)
        assert principal is None

    @pytest.mark.asyncio
    async def test_wrong_secret_returns_none(self):
        """Token signed with one secret fails verification with another."""
        issuer = JWTAuth(secret="secret-A")
        verifier = JWTAuth(secret="secret-B")
        token = issuer.issue_token("alice", "admin")
        principal = await verifier.verify_token(token)
        assert principal is None

    @pytest.mark.asyncio
    async def test_malformed_token_returns_none(self, jwt_auth: JWTAuth):
        principal = await jwt_auth.verify_token("not.a.valid.jwt.token")
        assert principal is None

    @pytest.mark.asyncio
    async def test_empty_token_returns_none(self, jwt_auth: JWTAuth):
        principal = await jwt_auth.verify_token("")
        assert principal is None

    @pytest.mark.asyncio
    async def test_developer_role_permissions_in_jwt(self, jwt_auth: JWTAuth):
        token = jwt_auth.issue_token("dev", "developer")
        principal = await jwt_auth.verify_token(token)
        assert principal is not None
        assert principal.role == "developer"
        assert "invoke" in principal.permissions
        assert "config" not in principal.permissions

    def test_unsupported_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            JWTAuth(secret="key", algorithm="RS256")

    def test_token_has_three_parts(self, jwt_auth: JWTAuth):
        token = jwt_auth.issue_token("alice", "admin")
        assert len(token.split(".")) == 3


# ===========================================================================
# RBAC
# ===========================================================================


class TestRBAC:
    """Role-based access control permission matrix."""

    def test_admin_has_all_permissions(self, rbac: RBAC):
        principal = Principal(
            user_id="root",
            role="admin",
            permissions=RBAC.permissions_for("admin"),
        )
        for perm in ["invoke", "history", "health", "config", "admin"]:
            assert rbac.has_permission(principal, perm)

    def test_developer_has_invoke_but_not_config(self, rbac: RBAC):
        principal = Principal(
            user_id="dev",
            role="developer",
            permissions=RBAC.permissions_for("developer"),
        )
        assert rbac.has_permission(principal, "invoke")
        assert rbac.has_permission(principal, "health")
        assert not rbac.has_permission(principal, "config")
        assert not rbac.has_permission(principal, "admin")

    def test_viewer_has_health_but_not_invoke(self, rbac: RBAC):
        principal = Principal(
            user_id="viewer",
            role="viewer",
            permissions=RBAC.permissions_for("viewer"),
        )
        assert rbac.has_permission(principal, "health")
        assert rbac.has_permission(principal, "history")
        assert not rbac.has_permission(principal, "invoke")
        assert not rbac.has_permission(principal, "config")

    def test_require_raises_authorization_error_for_viewer(self, rbac: RBAC):
        principal = Principal(
            user_id="viewer",
            role="viewer",
            permissions=RBAC.permissions_for("viewer"),
        )
        with pytest.raises(AuthorizationError):
            rbac.require(principal, "invoke")

    def test_require_raises_authentication_error_for_none(self, rbac: RBAC):
        with pytest.raises(AuthenticationError):
            rbac.require(None, "invoke")

    def test_require_passes_for_authorized_principal(self, rbac: RBAC):
        principal = Principal(
            user_id="admin",
            role="admin",
            permissions=RBAC.permissions_for("admin"),
        )
        # Should not raise.
        rbac.require(principal, "config")

    def test_has_permission_false_for_none_principal(self, rbac: RBAC):
        assert rbac.has_permission(None, "invoke") is False

    def test_has_permission_checks_role_table(self, rbac: RBAC):
        """Even without explicit permissions list, role table is checked."""
        principal = Principal(user_id="dev", role="developer", permissions=[])
        assert rbac.has_permission(principal, "invoke") is True
        assert rbac.has_permission(principal, "config") is False

    def test_unknown_role_has_no_permissions(self, rbac: RBAC):
        principal = Principal(user_id="x", role="unknown_role", permissions=[])
        assert rbac.has_permission(principal, "invoke") is False


# ===========================================================================
# create_auth_dependency — integration tests with FastAPI TestClient
# ===========================================================================


class TestAuthDependencyExemptPaths:
    """Exempt path handling."""

    def test_health_endpoint_no_auth_needed(self, composite_auth_app: FastAPI):
        client = TestClient(composite_auth_app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_docs_endpoint_exempt_by_default(self, composite_auth_app: FastAPI):
        client = TestClient(composite_auth_app)
        resp = client.get("/docs")
        # /docs returns HTML (Swagger UI), not 401.
        assert resp.status_code == 200


class TestAuthDependencyAPIKey:
    """API Key authentication flow via HTTP."""

    def test_invoke_with_valid_admin_api_key(
        self, composite_auth_app: FastAPI
    ):
        client = TestClient(composite_auth_app)
        resp = client.post(
            "/invoke",
            headers={"X-API-Key": "key-admin-123"},
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"

    def test_invoke_with_valid_developer_api_key(
        self, composite_auth_app: FastAPI
    ):
        client = TestClient(composite_auth_app)
        resp = client.post(
            "/invoke",
            headers={"X-API-Key": "key-dev-456"},
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "developer"

    def test_config_endpoint_admin_allowed(self, composite_auth_app: FastAPI):
        client = TestClient(composite_auth_app)
        resp = client.get(
            "/config",
            headers={"X-API-Key": "key-admin-123"},
        )
        assert resp.status_code == 200

    def test_config_endpoint_viewer_forbidden(
        self, composite_auth_app: FastAPI
    ):
        """Viewer authenticates but lacks 'config' permission → 403."""
        client = TestClient(composite_auth_app)
        resp = client.get(
            "/config",
            headers={"X-API-Key": "key-viewer-789"},
        )
        assert resp.status_code == 403


class TestAuthDependencyJWT:
    """JWT authentication flow via HTTP."""

    def test_invoke_with_valid_jwt(
        self, composite_auth_app: FastAPI, jwt_auth: JWTAuth
    ):
        token = jwt_auth.issue_token("jwtuser", "developer")
        client = TestClient(composite_auth_app)
        resp = client.post(
            "/invoke",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["user"] == "jwtuser"

    def test_config_with_admin_jwt(
        self, composite_auth_app: FastAPI, jwt_auth: JWTAuth
    ):
        token = jwt_auth.issue_token("root", "admin")
        client = TestClient(composite_auth_app)
        resp = client.get(
            "/config",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_config_with_viewer_jwt_forbidden(
        self, composite_auth_app: FastAPI, jwt_auth: JWTAuth
    ):
        token = jwt_auth.issue_token("guest", "viewer")
        client = TestClient(composite_auth_app)
        resp = client.get(
            "/config",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403


class TestAuthDependencyFailureModes:
    """Missing / invalid auth → 401, insufficient → 403."""

    def test_invoke_without_auth_returns_401(self, composite_auth_app: FastAPI):
        client = TestClient(composite_auth_app)
        resp = client.post("/invoke")
        assert resp.status_code == 401

    def test_invoke_with_invalid_api_key_returns_401(
        self, composite_auth_app: FastAPI
    ):
        client = TestClient(composite_auth_app)
        resp = client.post(
            "/invoke",
            headers={"X-API-Key": "invalid-key"},
        )
        assert resp.status_code == 401

    def test_invoke_with_invalid_jwt_returns_401(
        self, composite_auth_app: FastAPI
    ):
        client = TestClient(composite_auth_app)
        resp = client.post(
            "/invoke",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert resp.status_code == 401

    def test_invoke_with_non_bearer_auth_returns_401(
        self, composite_auth_app: FastAPI
    ):
        client = TestClient(composite_auth_app)
        resp = client.post(
            "/invoke",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert resp.status_code == 401

    def test_viewer_invoke_returns_403(self, composite_auth_app: FastAPI):
        """Viewer authenticates but lacks 'invoke' → 403."""
        client = TestClient(composite_auth_app)
        resp = client.post(
            "/invoke",
            headers={"X-API-Key": "key-viewer-789"},
        )
        assert resp.status_code == 403


class TestAuthDependencyComposite:
    """Composite auth: either API Key or JWT satisfies authentication."""

    def test_api_key_works_when_jwt_also_configured(
        self, composite_auth_app: FastAPI
    ):
        client = TestClient(composite_auth_app)
        resp = client.post(
            "/invoke",
            headers={"X-API-Key": "key-dev-456"},
        )
        assert resp.status_code == 200

    def test_jwt_works_when_api_key_also_configured(
        self, composite_auth_app: FastAPI, jwt_auth: JWTAuth
    ):
        token = jwt_auth.issue_token("jwtuser", "developer")
        client = TestClient(composite_auth_app)
        resp = client.post(
            "/invoke",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_invalid_api_key_falls_through_to_jwt(
        self, composite_auth_app: FastAPI, jwt_auth: JWTAuth
    ):
        """Invalid API Key + valid JWT → authenticated via JWT."""
        token = jwt_auth.issue_token("jwtuser", "developer")
        client = TestClient(composite_auth_app)
        resp = client.post(
            "/invoke",
            headers={
                "X-API-Key": "wrong-key",
                "Authorization": f"Bearer {token}",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["user"] == "jwtuser"


class TestAuthDependencyCustomConfig:
    """Custom exempt paths and config-only dependency."""

    def test_custom_exempt_paths(self, api_key_auth: APIKeyAuth, rbac: RBAC):
        """Custom exempt paths include /public."""
        app = FastAPI()

        @app.exception_handler(ICoreError)
        async def handle_icore_error(request, exc: ICoreError):
            return JSONResponse(
                status_code=exc.http_status, content=exc.to_dict()
            )

        dep = create_auth_dependency(
            api_key_auth=api_key_auth,
            rbac=rbac,
            required_permission="invoke",
            exempt_paths={"/public"},
        )

        @app.get("/public")
        async def public_endpoint(principal: Principal = Depends(dep)):
            return {"user": principal.user_id}

        @app.get("/protected")
        async def protected_endpoint(principal: Principal = Depends(dep)):
            return {"user": principal.user_id}

        client = TestClient(app)
        # /public is exempt → 200 without auth.
        assert client.get("/public").status_code == 200
        # /protected requires auth → 401 without auth.
        assert client.get("/protected").status_code == 401

    def test_no_authenticators_configured_always_401(self, rbac: RBAC):
        """When neither API Key nor JWT auth is configured, all non-exempt
        paths return 401."""
        app = FastAPI()

        @app.exception_handler(ICoreError)
        async def handle_icore_error(request, exc: ICoreError):
            return JSONResponse(
                status_code=exc.http_status, content=exc.to_dict()
            )

        dep = create_auth_dependency(
            api_key_auth=None,
            jwt_auth=None,
            rbac=rbac,
            required_permission="invoke",
        )

        @app.post("/invoke")
        async def invoke(principal: Principal = Depends(dep)):
            return {"user": principal.user_id}

        client = TestClient(app)
        resp = client.post("/invoke")
        assert resp.status_code == 401


class TestLoadUsersYaml:
    """YAML loading helper."""

    def test_load_valid_yaml(self):
        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "users.yaml"
            yaml_path.write_text(
                """
users:
  admin:
    api_key: "key1"
    role: admin
""",
                encoding="utf-8",
            )
            data = load_users_yaml(yaml_path)
            assert "users" in data
            assert "admin" in data["users"]
            assert data["users"]["admin"]["api_key"] == "key1"

    def test_load_missing_file_returns_empty(self):
        data = load_users_yaml("/nonexistent/users.yaml")
        assert data == {}
