"""Tests for planning.optic — focused on solvability classification."""
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import planning.optic as optic_module


class TestOpticTimeout:
    def test_timeout_emits_timeout_solvability(self, tmp_path):
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="optic", timeout=5)):
            result = optic_module.run(tmp_path, timeout=5)

        assert result["solvability"] == "timeout"
        assert result["success"] is False

    def test_timeout_message_mentions_duration(self, tmp_path):
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="optic", timeout=30)):
            result = optic_module.run(tmp_path, timeout=30)

        assert "30" in result["message"]

    def test_timeout_differs_from_unsolvable_resource(self, tmp_path):
        (tmp_path / "domain.pddl").write_text("(define (domain d))", encoding="utf-8")
        (tmp_path / "problem.pddl").write_text("(define (problem p) (:domain d))", encoding="utf-8")

        with patch.object(optic_module, "is_available", return_value=True), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="optic", timeout=1)):
            result = optic_module.run(tmp_path, timeout=1)

        assert result["solvability"] != "unsolvable_resource"
