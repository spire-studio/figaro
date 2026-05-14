from __future__ import annotations

from app.services.simulation.run_metrics_service import SimulationRunMetricsService


def test_normalize_metrics_payload_preserves_llm_sections():
    service = SimulationRunMetricsService.__new__(SimulationRunMetricsService)

    normalized = service._normalize_metrics_payload(
        {
            "llm_results": {
                "rounds": [1, "2"],
                "train_loss": [1.5],
                "validation_loss": [1.2],
                "perplexity": [3.32],
                "token_throughput": [42],
                "adapter_size_bytes": [1024],
            },
            "llm_dataset": {"num_records": 12},
            "llm_runtime": {"status": "blocked"},
        }
    )

    assert normalized["llm_results"]["rounds"] == [1, 2]
    assert normalized["llm_results"]["token_throughput"] == [42.0]
    assert normalized["llm_dataset"]["num_records"] == 12
    assert normalized["llm_runtime"]["status"] == "blocked"

