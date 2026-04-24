def test_agent_models_endpoint_returns_registry_names(client):
    import app.api.v1.endpoints.agent as agent_module
    import app.services.llm as llm_module

    original = list(agent_module.LLMRegistry.LLMS)
    original_default = llm_module.settings.default_llm_model
    agent_module.LLMRegistry.LLMS = [
        {"name": "gpt-test-a", "llm": object()},
        {"name": "gpt-test-b", "llm": object()},
    ]
    llm_module.settings.default_llm_model = "gpt-test-b"
    try:
        response = client.get("/api/v1/agent/models")
    finally:
        agent_module.LLMRegistry.LLMS = original
        llm_module.settings.default_llm_model = original_default

    assert response.status_code == 200
    assert response.json() == {
        "models": ["gpt-test-a", "gpt-test-b"],
        "default_model": "gpt-test-b",
    }
