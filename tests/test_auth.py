import os
import sys

# Ensure src is on the path for imports
dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, "..", "src"))

import api_app.routes.auth as auth
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()
app.include_router(auth.router)


def test_register_login_and_me(monkeypatch):
    fake_db = {}

    def fake_obtener_usuario(username):
        user = fake_db.get(username)
        if user:
            return {"id": 1, "username": username, **user}
        return None

    def fake_crear_usuario(username, password_hash, role="user"):
        fake_db[username] = {"password_hash": password_hash, "role": role}

    monkeypatch.setattr(auth, "obtener_usuario", fake_obtener_usuario)
    monkeypatch.setattr(auth, "crear_usuario", fake_crear_usuario)

    client = TestClient(app)

    # Register user
    resp = client.post(
        "/auth/register",
        json={"username": "alice", "password": "secret", "role": "admin"},
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "admin"

    # Login user
    resp = client.post(
        "/auth/login", json={"username": "alice", "password": "secret"}
    )
    assert resp.status_code == 200
    token = resp.json()["access_token"]

    # Access protected route
    resp = client.get(
        "/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    assert resp.json()["username"] == "alice"
    assert resp.json()["role"] == "admin"
