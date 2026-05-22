from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("models") / "llm" / "tiny-causal-lm"


def build_tokenizer(output_dir: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import PreTrainedTokenizerFast

    corpus = [
        "Question What is federated learning Answer Training a shared model across clients without centralizing their raw data",
        "Question What does LoRA train Answer Low rank adapter weights while keeping the base model frozen",
        "Question Why use a validation set Answer To estimate generalization with loss and perplexity after training rounds",
        "Question What is PEFT Answer Parameter efficient fine tuning that updates a small set of trainable parameters",
    ]
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=1,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    fast_tokenizer.save_pretrained(output_dir)
    return fast_tokenizer


def build_model(output_dir: Path) -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(output_dir)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(output_dir, safe_serialization=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tiny local causal LM fixture for Figaro LLM PEFT smoke runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Model output directory. Default: {DEFAULT_OUTPUT_DIR.as_posix()}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_model(args.output_dir)
    print(f"Tiny LLM fixture written to {args.output_dir.as_posix()}")


if __name__ == "__main__":
    main()
