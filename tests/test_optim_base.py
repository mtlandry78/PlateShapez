import json

import numpy as np
import pytest

from plateshapez.optim.base import (
    OptimizationResult,
    OracleQuery,
    OracleRecord,
    OracleResponse,
    PatternOptimizer,
    QueryLogger,
)


class TestOracleRecordSerialization:
    def test_to_json_line_round_trips(self):
        query = OracleQuery(genome=[0.1, -0.2, 0.3], query_index=7)
        response = OracleResponse(score=0.4, plate_class="B", detail={"reason": "misread"})
        record = OracleRecord(query=query, response=response)

        line = record.to_json_line()
        payload = json.loads(line)

        assert payload["query"]["genome"] == [0.1, -0.2, 0.3]
        assert payload["query"]["query_index"] == 7
        assert payload["response"]["score"] == 0.4
        assert payload["response"]["plate_class"] == "B"
        assert payload["response"]["detail"] == {"reason": "misread"}

    def test_response_detail_defaults_to_empty_dict(self):
        response = OracleResponse(score=1.0, plate_class="C")
        assert response.detail == {}


class TestQueryLogger:
    def test_creates_parent_directories(self, tmp_path):
        log_path = tmp_path / "nested" / "query_log.jsonl"
        logger = QueryLogger(log_path)
        logger.close()

        assert log_path.parent.exists()

    def test_log_appends_jsonl_lines(self, tmp_path):
        log_path = tmp_path / "query_log.jsonl"
        record_a = OracleRecord(
            query=OracleQuery(genome=[0.0], query_index=0),
            response=OracleResponse(score=0.0, plate_class="A"),
        )
        record_b = OracleRecord(
            query=OracleQuery(genome=[1.0], query_index=1),
            response=OracleResponse(score=1.0, plate_class="C"),
        )

        logger = QueryLogger(log_path)
        logger.log(record_a)
        logger.log(record_b)
        logger.close()

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["query"]["query_index"] == 0
        assert json.loads(lines[1])["query"]["query_index"] == 1

    def test_context_manager_closes_on_exit(self, tmp_path):
        log_path = tmp_path / "query_log.jsonl"
        with QueryLogger(log_path) as logger:
            logger.log(
                OracleRecord(
                    query=OracleQuery(genome=[0.0], query_index=0),
                    response=OracleResponse(score=0.0, plate_class="A"),
                )
            )

        with pytest.raises(RuntimeError, match="closed"):
            logger.log(
                OracleRecord(
                    query=OracleQuery(genome=[0.0], query_index=1),
                    response=OracleResponse(score=0.0, plate_class="A"),
                )
            )


class TestOptimizationResult:
    def test_fields_round_trip(self):
        result = OptimizationResult(
            best_genome=[0.1, 0.2], best_score=0.0, n_queries=10, history=[1.0, 0.5, 0.0]
        )
        assert result.best_genome == [0.1, 0.2]
        assert result.n_queries == 10
        assert result.history == [1.0, 0.5, 0.0]


class TestPatternOptimizerABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            PatternOptimizer()  # type: ignore[abstract]

    def test_incomplete_subclass_raises(self):
        class Incomplete(PatternOptimizer):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_used(self):
        class AlwaysZero(PatternOptimizer):
            def optimize(self, evaluate, genome_length, bounds, budget):
                genome = np.zeros(genome_length)
                score = evaluate(genome)
                return OptimizationResult(
                    best_genome=genome.tolist(), best_score=score, n_queries=1, history=[score]
                )

        optimizer = AlwaysZero()
        result = optimizer.optimize(lambda g: 0.5, 3, [(-1.0, 1.0)] * 3, budget=1)
        assert result.best_score == 0.5
        assert result.n_queries == 1
