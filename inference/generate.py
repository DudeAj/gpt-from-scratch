"""
generate.py
-----------
Load a trained GPT checkpoint and generate text autoregressively.

Usage:
    python -m inference.generate --prompt "To be or not" --max_new_tokens 200
    python -m inference.generate --prompt "What is" --strategy greedy
"""

import argparse
import torch

from model.gpt import GPT
from tokenizer.bpe_tokenizer import BPETokenizer
from configs.config import CHECKPOINT_DIR, DEVICE, MAX_NEW_TOKENS


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    """
    Always pick the single most probable next token.

    logits : (1, T, vocab_size)
    returns: (1, 1)  — the chosen token ID

    Simple and fast, but tends to produce repetitive loops because
    the model keeps reinforcing the same high-probability path.
    """
    # Take logits for the very last position only — that's the prediction
    # for the token that comes *after* the current sequence.
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def top_k_decode(
    logits: torch.Tensor,
    k: int = 50,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample from the top-k most probable tokens.

    logits      : (1, T, vocab_size)
    k           : how many candidates to keep
    temperature : > 1.0 → more random,  < 1.0 → more focused

    returns: (1, 1)

    Why temperature?
        logits / temperature before softmax reshapes the distribution.
        Dividing by a small number (e.g. 0.7) sharpens peaks — the model
        becomes more "decisive". Dividing by a large number (e.g. 1.5)
        flattens the distribution — more surprising/creative outputs.
    """
    # Focus on the last time step
    last_logits = logits[:, -1, :] / temperature   # (1, vocab_size)

    # Zero out every token outside the top-k so they can never be sampled.
    # torch.topk returns the k largest values; we find the threshold value
    # and mask everything below it to -inf.
    top_values, _ = torch.topk(last_logits, k)
    threshold = top_values[:, -1].unsqueeze(-1)           # k-th largest value
    filtered = last_logits.masked_fill(last_logits < threshold, float('-inf'))

    # Convert to probabilities and sample once
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)        # (1, 1)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    strategy: str = "top_k",
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    """
    Autoregressively generate text starting from `prompt`.

    Each iteration:
      1. Crop context to model's max sequence length (position embeddings
         only go up to max_seq_len — longer inputs would crash).
      2. Forward pass → logits for every position.
      3. Sample the *last* position's logits to get the next token.
      4. Append the new token and repeat.

    Parameters
    ----------
    model           : trained GPT instance (already on DEVICE)
    tokenizer       : matching BPETokenizer (same vocab as training)
    prompt          : seed text string
    max_new_tokens  : how many tokens to generate
    strategy        : "greedy" or "top_k"
    temperature     : sampling temperature (top_k only)
    top_k           : number of candidates to consider (top_k only)
    """
    model.eval()  # disables dropout — we want deterministic representations

    # Encode prompt → list of int token IDs → tensor (1, T)
    prompt_ids = tokenizer.encode(prompt)
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    max_seq_len = model.embedding.position_embedding.num_embeddings

    for _ in range(max_new_tokens):
        # --- 1. Crop context window ---
        # If our running sequence is longer than what the position
        # embeddings support, drop the oldest tokens from the left.
        context = tokens if tokens.size(1) <= max_seq_len \
                  else tokens[:, -max_seq_len:]

        # --- 2. Forward pass ---
        logits = model(context)       # (1, context_len, vocab_size)

        # --- 3. Sample next token ---
        if strategy == "greedy":
            next_token = greedy_decode(logits)
        else:
            next_token = top_k_decode(logits, k=top_k, temperature=temperature)

        # --- 4. Append and continue ---
        tokens = torch.cat([tokens, next_token], dim=1)   # (1, T+1)

    # Decode the generated portion only (skip the prompt)
    generated_ids = tokens[0, len(prompt_ids):].tolist()
    return tokenizer.decode(generated_ids)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained GPT.")
    parser.add_argument("--prompt",         type=str, default="To be or not to be",
                        help="Seed text for generation")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Number of new tokens to generate")
    parser.add_argument("--strategy",       type=str, default="top_k",
                        choices=["greedy", "top_k"],
                        help="Decoding strategy")
    parser.add_argument("--temperature",    type=float, default=1.0,
                        help="Sampling temperature (top_k only)")
    parser.add_argument("--top_k",          type=int, default=50,
                        help="Top-k candidates (top_k strategy only)")
    parser.add_argument("--checkpoint",     type=str,
                        default=str(CHECKPOINT_DIR / "best_model.pt"),
                        help="Path to model checkpoint")
    args = parser.parse_args()

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)

    cfg = ckpt["config"]
    model = GPT(
        vocab_size  = cfg["vocab_size"],
        max_seq_len = cfg["max_seq_len"],
        d_model     = cfg["d_model"],
        num_heads   = cfg["num_heads"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"])
    print(f"Model loaded (epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.4f})")

    # --- Load tokenizer ---
    tokenizer = BPETokenizer()
    tokenizer_path = str(CHECKPOINT_DIR / "tokenizer.json")
    tokenizer.load(tokenizer_path)

    # --- Generate ---
    print(f"\nPrompt : {args.prompt}")
    print(f"Strategy: {args.strategy}  |  Temperature: {args.temperature}  |  Top-k: {args.top_k}")
    print("-" * 60)

    output = generate(
        model          = model,
        tokenizer      = tokenizer,
        prompt         = args.prompt,
        max_new_tokens = args.max_new_tokens,
        strategy       = args.strategy,
        temperature    = args.temperature,
        top_k          = args.top_k,
    )

    print(args.prompt + output)


if __name__ == "__main__":
    main()
