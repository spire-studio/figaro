def test_distributed_session_lifecycle(client):
    create_job_payload = {
        "name": "dist-exp-a",
        "description": "distributed test",
    }
    created_job = client.post("/api/v1/distributed/jobs", json=create_job_payload)
    assert created_job.status_code == 201
    job_id = created_job.json()["id"]
    assert created_job.json()["expected_clients"] >= 1

    updated_job = client.patch(
        f"/api/v1/distributed/jobs/{job_id}/config",
        json={
            "config": {
                "system": {"mode": "distributed", "node_role": "server"},
                "dataset": {"num_clients": 2},
                "federated": {"num_clients": 2, "num_rounds": 10},
            }
        },
    )
    assert updated_job.status_code == 200
    assert updated_job.json()["expected_clients"] == 2

    created_session = client.post(
        f"/api/v1/distributed/jobs/{job_id}/session",
        json={"server_ip": "127.0.0.1", "server_port": 50052},
    )
    assert created_session.status_code == 201
    session_id = created_session.json()["id"]
    assert created_session.json()["status"] == "waiting_clients"
    active_session = client.get(f"/api/v1/distributed/jobs/{job_id}/session")
    assert active_session.status_code == 200
    assert active_session.json()["id"] == session_id

    c0 = client.post(
        f"/api/v1/distributed/sessions/{session_id}/connect",
        json={"participant_name": "client-a", "metadata_json": {"device": "cpu"}},
    )
    c1 = client.post(
        f"/api/v1/distributed/sessions/{session_id}/connect",
        json={"participant_name": "client-b", "metadata_json": {"device": "cpu"}},
    )
    assert c0.status_code == 201
    assert c1.status_code == 201
    c0_id = c0.json()["id"]
    c1_id = c1.json()["id"]

    listed = client.get(f"/api/v1/distributed/sessions/{session_id}/participants")
    assert listed.status_code == 200
    assert len(listed.json()) == 2

    approved0 = client.post(f"/api/v1/distributed/participants/{c0_id}/approve")
    approved1 = client.post(f"/api/v1/distributed/participants/{c1_id}/approve")
    assert approved0.status_code == 200
    assert approved1.status_code == 200
    assert approved0.json()["status"] == "approved"
    assert approved1.json()["status"] == "approved"
    assert approved0.json()["assigned_participant_id"] in {0, 1}
    assert approved1.json()["assigned_participant_id"] in {0, 1}
    assert approved0.json()["assigned_participant_id"] != approved1.json()["assigned_participant_id"]

    ready0 = client.post(f"/api/v1/distributed/participants/{c0_id}/ready")
    ready1 = client.post(f"/api/v1/distributed/participants/{c1_id}/ready")
    assert ready0.status_code == 200
    assert ready1.status_code == 200
    assert ready0.json()["status"] == "ready"
    assert ready1.json()["status"] == "ready"

    session_ready = client.get(f"/api/v1/distributed/sessions/{session_id}")
    assert session_ready.status_code == 200
    assert session_ready.json()["status"] == "ready_to_start"

    started = client.post(f"/api/v1/distributed/sessions/{session_id}/start")
    assert started.status_code == 200
    assert started.json()["status"] == "running"


def test_distributed_start_requires_all_clients_ready(client):
    created_job = client.post(
        "/api/v1/distributed/jobs",
        json={
            "name": "dist-exp-b",
            "description": "distributed test",
        },
    )
    assert created_job.status_code == 201
    job_id = created_job.json()["id"]
    missing_active = client.get(f"/api/v1/distributed/jobs/{job_id}/session")
    assert missing_active.status_code == 404

    updated_job = client.patch(
        f"/api/v1/distributed/jobs/{job_id}/config",
        json={
            "config": {
                "system": {"mode": "distributed", "node_role": "server"},
                "dataset": {"num_clients": 1},
                "federated": {"num_clients": 1, "num_rounds": 5},
            }
        },
    )
    assert updated_job.status_code == 200

    created_session = client.post(
        f"/api/v1/distributed/jobs/{job_id}/session",
        json={"server_ip": "127.0.0.1", "server_port": 50052},
    )
    assert created_session.status_code == 201
    session_id = created_session.json()["id"]

    created_client = client.post(
        f"/api/v1/distributed/sessions/{session_id}/connect",
        json={"participant_name": "client-c", "metadata_json": {}},
    )
    assert created_client.status_code == 201
    client_id = created_client.json()["id"]

    approved = client.post(f"/api/v1/distributed/participants/{client_id}/approve")
    assert approved.status_code == 200

    started = client.post(f"/api/v1/distributed/sessions/{session_id}/start")
    assert started.status_code == 409
