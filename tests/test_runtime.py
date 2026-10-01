from hybrid_steering.judge.config import repo_root
from hybrid_steering.runtime import collect_steered_rows, import_path


class _Tokens:
    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return tokens

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    ):
        del tokenize, add_generation_prompt, enable_thinking
        return "|".join(message["content"] for message in messages)


class _Runner:
    def generate(self, texts, scale=1.0, prompt_position=-1, max_new_tokens=64):
        del max_new_tokens
        return [f"pos={prompt_position};scale={scale};{text}" for text in texts]


def test_collect_steered_rows_pairs_each_scale_with_its_baseline() -> None:
    rows = collect_steered_rows(
        _Runner(),
        _Tokens(),
        [{"question": "q", "source_id": "a"}],
        {0: "", 4: "filler"},
        [0.5],
        batch_size=2,
        token_budget=8,
        max_new_tokens=3,
        build_row=lambda length, example, base, scale, response: {
            "length": length,
            "id": example["source_id"],
            "base": base,
            "scale": scale,
            "response": response,
        },
    )
    assert [row["length"] for row in rows] == [0, 4]
    assert rows[0]["base"] == "pos=None;scale=1.0;q"
    assert rows[1]["response"] == "pos=0;scale=0.5;filler|q"
    assert rows[1]["base"].startswith("pos=None")


def test_import_path_loads_the_shared_question_bank() -> None:
    module = import_path(repo_root() / "experiments/forgetting/questions.py")
    rows = module.simple_questions(2, seed=0)
    assert len(rows) == 2
    assert rows[0]["question"]
