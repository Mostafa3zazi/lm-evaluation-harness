from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lm_eval import tasks
from lm_eval.api.instance import Instance


task_manager = tasks.TaskManager()


class _MetadataTokenizer:
    def __call__(self, text, **kwargs):
        token_map = {
            "</think_fast>": [99],
            "first thought </think> final thought ": [1, 2, 3, 4, 5],
            "answer words": [6, 7],
        }
        return SimpleNamespace(input_ids=token_map.get(text, text.split()))

    def decode(self, token_id):
        return "<eos>"


class TestVLLMValidation:
    """Tests for VLLM constructor validation."""

    vllm = pytest.importorskip("vllm")

    def test_data_parallel_with_expert_parallel_raises(self):
        """data_parallel_size > 1 with enable_expert_parallel=True must raise."""
        from lm_eval.models.vllm_causallms import VLLM

        with (
            patch.multiple(
                "lm_eval.models.vllm_causallms",
                find_spec=lambda name: None if name == "ray" else MagicMock(),
                LLM=MagicMock(),
                get_tokenizer=MagicMock(return_value=MagicMock()),
            ),
            patch("transformers.AutoConfig.from_pretrained", MagicMock()),
            pytest.raises(ValueError, match=r"data_parallel_size > 1.*expert_parallel"),
        ):
            VLLM(
                pretrained="mock-model",
                data_parallel_size=2,
                enable_expert_parallel=True,
            )

    def test_completion_metadata_uses_last_think_end_token(self):
        from lm_eval.models.vllm_causallms import VLLM

        lm = VLLM.__new__(VLLM)
        lm.think_end_token = r"</think>|</think_fast>|</think_faster>|<channel\|>"
        lm.tokenizer = _MetadataTokenizer()

        completion = SimpleNamespace(
            text="first thought </think> final thought </think_fast> answer words",
            finish_reason="stop",
            stop_reason="</think_fast>",
            token_ids=[1, 2, 3, 4, 5, 99, 6, 7],
        )

        metadata = lm._get_completion_metadata(completion)

        assert metadata == {
            "finish_reason": "stop",
            "stop_reason": "</think_fast>",
            "generated_token_count": 8,
            "word_count": 8,
            "has_think_end_token": True,
            "think_end_token": r"</think>|</think_fast>|</think_faster>|<channel\|>",
            "matched_think_end_token": "</think_fast>",
            "reasoning": "first thought </think> final thought ",
            "answer": "answer words",
            "reasoning_word_count": 5,
            "answer_word_count": 2,
            "reasoning_token_count": 5,
            "answer_token_count": 2,
        }

    def test_completion_metadata_handles_no_match_and_missing_attributes(self):
        from lm_eval.models.vllm_causallms import VLLM

        lm = VLLM.__new__(VLLM)
        lm.think_end_token = r"</think>|</think_fast>|</think_faster>|<channel\|>"

        metadata = lm._get_completion_metadata(
            SimpleNamespace(text="answer without thinking delimiter")
        )

        assert metadata == {
            "finish_reason": None,
            "stop_reason": None,
            "generated_token_count": 0,
            "word_count": 4,
            "has_think_end_token": False,
            "think_end_token": r"</think>|</think_fast>|</think_faster>|<channel\|>",
            "matched_think_end_token": None,
            "reasoning": "answer without thinking delimiter",
            "answer": "",
            "reasoning_word_count": 4,
            "answer_word_count": 0,
            "reasoning_token_count": 0,
            "answer_token_count": 0,
        }

    def test_generate_until_returns_answer_when_metadata_enabled(self):
        from lm_eval.models.vllm_causallms import VLLM

        lm = VLLM.__new__(VLLM)
        lm.tokenizer = _MetadataTokenizer()
        lm.think_end_token = r"</think>|</think_fast>|</think_faster>|<channel\|>"
        lm.log_completion_metadata = True
        lm.batch_size = 1
        lm.rank = 0
        lm.eot_token_id = 0
        lm._max_gen_toks = 10
        lm.max_length = 100
        lm.truncation_side = "left"
        lm.cache_hook = SimpleNamespace(add_partial=lambda *args: None)
        lm.tok_encode = lambda contexts: [[1] for _ in contexts]
        lm._model_generate = lambda **kwargs: [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="first thought </think> final thought </think_fast> answer words",
                        finish_reason="stop",
                        stop_reason="</think_fast>",
                        token_ids=[1, 2, 3, 4, 5, 99, 6, 7],
                    )
                ]
            )
        ]
        request = Instance(
            request_type="generate_until",
            doc={},
            arguments=("prompt", {"max_gen_toks": 10}),
            idx=0,
        )

        result = lm.generate_until([request], disable_tqdm=True)

        assert result == ["answer words"]
        assert request.generation_metadata[0]["reasoning"] == (
            "first thought </think> final thought "
        )
        assert request.generation_metadata[0]["answer"] == "answer words"

    def test_generate_until_returns_empty_answer_without_think_end_token(self):
        from lm_eval.models.vllm_causallms import VLLM

        lm = VLLM.__new__(VLLM)
        lm.tokenizer = _MetadataTokenizer()
        lm.think_end_token = r"</think>|</think_fast>|</think_faster>|<channel\|>"
        lm.log_completion_metadata = True
        lm.batch_size = 1
        lm.rank = 0
        lm.eot_token_id = 0
        lm._max_gen_toks = 10
        lm.max_length = 100
        lm.truncation_side = "left"
        lm.cache_hook = SimpleNamespace(add_partial=lambda *args: None)
        lm.tok_encode = lambda contexts: [[1] for _ in contexts]
        lm._model_generate = lambda **kwargs: [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="answer without thinking delimiter",
                        token_ids=[1, 2, 3, 4],
                    )
                ]
            )
        ]
        request = Instance(
            request_type="generate_until",
            doc={},
            arguments=("prompt", {"max_gen_toks": 10}),
            idx=0,
        )

        result = lm.generate_until([request], disable_tqdm=True)

        assert result == [""]
        assert request.generation_metadata[0]["reasoning"] == (
            "answer without thinking delimiter"
        )
        assert request.generation_metadata[0]["answer"] == ""
        assert request.generation_metadata[0]["reasoning_token_count"] == 4
        assert request.generation_metadata[0]["answer_token_count"] == 0


@pytest.mark.skip(reason="requires CUDA")
class Test_VLLM:
    vllm = pytest.importorskip("vllm")
    try:
        from lm_eval.models.vllm_causallms import VLLM

        LM = VLLM(pretrained="EleutherAI/pythia-70m")
    except ModuleNotFoundError:
        pass
    # torch.use_deterministic_algorithms(True)
    task_list = task_manager.load(["arc_easy", "gsm8k", "wikitext"])["tasks"]
    multiple_choice_task = task_list["arc_easy"]  # type: ignore
    multiple_choice_task.build_all_requests(limit=10, rank=0, world_size=1)
    MULTIPLE_CH: list[Instance] = multiple_choice_task.instances
    generate_until_task = task_list["gsm8k"]  # type: ignore
    generate_until_task._config.generation_kwargs["max_gen_toks"] = 10
    generate_until_task.build_all_requests(limit=10, rank=0, world_size=1)
    generate_until: list[Instance] = generate_until_task.instances
    rolling_task = task_list["wikitext"]  # type: ignore
    rolling_task.build_all_requests(limit=10, rank=0, world_size=1)
    ROLLING: list[Instance] = rolling_task.instances

    # TODO: make proper tests
    def test_logliklihood(self) -> None:
        res = self.LM.loglikelihood(self.MULTIPLE_CH)
        assert len(res) == len(self.MULTIPLE_CH)
        for x in res:
            assert isinstance(x[0], float)

    def test_generate_until(self) -> None:
        res = self.LM.generate_until(self.generate_until)
        assert len(res) == len(self.generate_until)
        for x in res:
            assert isinstance(x, str)

    def test_logliklihood_rolling(self) -> None:
        res = self.LM.loglikelihood_rolling(self.ROLLING)
        for x in res:
            assert isinstance(x, float)

    def test_loglikelihood_rejects_enable_thinking(self) -> None:
        with patch.object(self.LM, "enable_thinking", True):
            with pytest.raises(ValueError) as exc_info:
                self.LM.loglikelihood(self.MULTIPLE_CH)
            assert "arc_easy" in str(exc_info.value)
            assert "enable_thinking=True" in str(exc_info.value)
