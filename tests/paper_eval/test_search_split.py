"""search_split: the TPE search scores candidates on an explicit set of case
ids (paper_eval's temporal validation set) instead of a random split."""

from unittest.mock import MagicMock, patch

import pytest

from tests.helpers import make_event, make_log, make_trace

import core_utils
import pipeline as pipeline_module
from models import AnalysisConfig


def _log(*ids):
    return make_log(*[make_trace(make_event("A"), make_event("B", offset=1), case_id=i) for i in ids])


class TestSplitByCaseIds:
    def test_requested_ids_go_to_test_others_to_train(self, tmp_path):
        log = _log("c1", "c2", "c3", "c4")
        with patch.object(core_utils, "_load_log", return_value=log), \
                patch.object(core_utils, "write_traces_xes", side_effect=lambda l, t, p: p) as write:
            with core_utils.split_by_case_ids("log.xes", ["c2", "c4"]) as tts:
                assert [t.attributes["concept:name"] for t in tts.test_cases] == ["c2", "c4"]
                train_traces = write.call_args.args[1]
                assert [t.attributes["concept:name"] for t in train_traces] == ["c1", "c3"]
                assert (tts.n_train, tts.n_test) == (2, 2)

    def test_unknown_id_raises(self):
        with patch.object(core_utils, "_load_log", return_value=_log("c1", "c2")):
            with pytest.raises(ValueError, match="not found"):
                core_utils.split_by_case_ids("log.xes", ["c1", "missing"])


class TestBuildNetworkSearchSplit:
    def _run(self, **kwargs):
        tts = MagicMock()
        tts.__enter__.return_value = tts
        tts.train_path = "train.xes"
        tts.test_cases = ["val_trace"]
        with patch.object(pipeline_module.utils, "activity_names", return_value=set()), \
                patch.object(pipeline_module.utils, "split_by_case_ids", return_value=tts) as by_ids, \
                patch.object(pipeline_module.utils, "split", return_value=tts) as random_split, \
                patch.object(pipeline_module, "find_best_config", return_value=(AnalysisConfig(), [])) as fbc, \
                patch.object(pipeline_module, "Parser"), \
                patch.object(pipeline_module, "DomainBuilder"), \
                patch.object(pipeline_module, "build_domain_with_variant_map", return_value=(MagicMock(), {})):
            pipeline_module.Pipeline().build_network("log.xes", search=True, search_n_trials=1, **kwargs)
        return by_ids, random_split, fbc

    def test_explicit_ids_replace_random_split(self):
        by_ids, random_split, fbc = self._run(search_split=["v1", "v2"])
        assert by_ids.call_args.args[1] == ["v1", "v2"]
        random_split.assert_not_called()
        assert fbc.call_args.kwargs["test_traces"] == ["val_trace"]

    def test_default_keeps_random_split(self):
        by_ids, random_split, _ = self._run()
        by_ids.assert_not_called()
        random_split.assert_called_once()
