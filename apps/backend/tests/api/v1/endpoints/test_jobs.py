import time

from app.services.simulation.run_service import SimulationRunService


class _FakeStream:
    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class _FakeProcess:
    def __init__(self):
        self.pid = 4242
        self.stdout = _FakeStream(
            [
                b"training started\n",
                "结果已保存到: ./results/fake.json\n".encode("utf-8"),
            ]
        )
        self.stderr = _FakeStream([])
        self.returncode = None

    async def wait(self) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = -15


async def _fake_spawn(self, command, env=None):  # noqa: ANN001
    return _FakeProcess()


def test_job_lifecycle_and_run(client, monkeypatch):
    monkeypatch.setattr(SimulationRunService, "_spawn_subprocess", _fake_spawn)

    create_payload = {
        "name": "exp-a",
        "description": "first experiment",
    }
    created = client.post("/api/v1/jobs", json=create_payload)
    assert created.status_code == 201
    job_id = created.json()["id"]
    assert isinstance(created.json().get("config_json"), dict)

    listed = client.get("/api/v1/jobs")
    assert listed.status_code == 200
    assert any(item["id"] == job_id for item in listed.json())

    updated = client.patch(f"/api/v1/jobs/{job_id}", json={"description": "updated"})
    assert updated.status_code == 200
    assert updated.json()["description"] == "updated"

    new_cfg = client.patch(
        f"/api/v1/jobs/{job_id}/config",
        json={"config": {"federated": {"num_rounds": 2}}},
    )
    assert new_cfg.status_code == 200
    assert new_cfg.json()["job_id"] == job_id
    assert new_cfg.json()["config_json"]["federated"]["num_rounds"] == 2

    listed_cfg = client.get(f"/api/v1/jobs/{job_id}/config")
    assert listed_cfg.status_code == 200
    assert listed_cfg.json()["job_id"] == job_id

    copied = client.post(f"/api/v1/jobs/{job_id}/copy", json={"name": "exp-a-copy"})
    assert copied.status_code == 201

    run_resp = client.post(f"/api/v1/jobs/{job_id}/runs")
    assert run_resp.status_code == 201
    run_id = run_resp.json()["id"]

    # Wait briefly for background watcher to persist logs and final status.
    time.sleep(0.1)

    run_status = client.get(f"/api/v1/runs/{run_id}")
    assert run_status.status_code == 200
    assert run_status.json()["status"] in {"running", "succeeded"}

    logs_resp = client.get(f"/api/v1/runs/{run_id}/logs")
    assert logs_resp.status_code == 200
    assert len(logs_resp.json()) >= 1

    results_resp = client.get(f"/api/v1/runs/{run_id}/results")
    assert results_resp.status_code == 200
    assert len(results_resp.json()) >= 1
