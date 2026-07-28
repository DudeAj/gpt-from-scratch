"""
chat.py
-------
Interactive chat with the fine-tuned conversation model.

Usage:
    python -m inference.chat
    python -m inference.chat --checkpoint checkpoints/finetune_best.pt
    python -m inference.chat --temp 0.8 --top_k 40
"""

import argparse
import torch
from tokenizers import Tokenizer as HFTokenizer

from model.gpt import GPT
from dataset.dialog_dataset import load_dialog_tokenizer
from configs.config import FINETUNE_CKPT, DEVICE, MAX_NEW_TOKENS


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_reply(
    model: GPT,
    tokenizer: HFTokenizer,
    conversation: list[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """
    Given a list of conversation turns so far, generate the next
    assistant reply.

    The conversation is formatted exactly as it was during fine-tuning:
        <human> turn1 </s> <assistant> turn2 </s> <human> turn3 </s> <assistant>

    We feed this prefix in, then autoregressively sample until we hit
    </s> (end of reply) or max_new_tokens is reached.

    Parameters
    ----------
    conversation : alternating [human, assistant, human, ...] turns
                   (last entry should be the latest human message)
    """
    model.eval()

    human_token     = "<human>"
    assistant_token = "<assistant>"
    end_token       = "</s>"

    human_id     = tokenizer.token_to_id(human_token)
    assistant_id = tokenizer.token_to_id(assistant_token)
    end_id       = tokenizer.token_to_id(end_token)

    max_seq_len = model.embedding.position_embedding.num_embeddings

    # Build the prompt token sequence from conversation history
    prompt_ids = []
    for i, turn in enumerate(conversation):
        is_assistant = (i % 2 == 1)
        prefix_id    = assistant_id if is_assistant else human_id
        turn_ids     = tokenizer.encode(turn).ids
        prompt_ids  += [prefix_id] + turn_ids + [end_id]

    # Append <assistant> to signal the model to start its reply
    prompt_ids.append(assistant_id)

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    generated = []

    for _ in range(max_new_tokens):
        # Crop to max context window
        context = tokens if tokens.size(1) <= max_seq_len \
                  else tokens[:, -max_seq_len:]

        logits = model(context)                        # (1, T, vocab)
        next_logits = logits[:, -1, :] / temperature  # (1, vocab)

        # Top-k filtering
        top_vals, _ = torch.topk(next_logits, top_k)
        threshold   = top_vals[:, -1].unsqueeze(-1)
        next_logits = next_logits.masked_fill(next_logits < threshold, float('-inf'))

        probs      = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)   # (1, 1)

        token_id = next_token.item()

        # Stop when we hit the end-of-turn token
        if token_id == end_id:
            break

        generated.append(token_id)
        tokens = torch.cat([tokens, next_token], dim=1)

    # Decode only the newly generated tokens
    return tokenizer.decode(generated)


# ---------------------------------------------------------------------------
# CLI chat loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(FINETUNE_CKPT))
    parser.add_argument("--temp",  type=float, default=0.8)
    parser.add_argument("--top_k", type=int,   default=40)
    parser.add_argument("--max_turns", type=int, default=10,
                        help="Max conversation turns to keep in context")
    args = parser.parse_args()

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    cfg  = ckpt["config"]

    # --- Load tokenizer (with special tokens) ---
    tokenizer, vocab_size = load_dialog_tokenizer()

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
    print(f"Temperature: {args.temp}  |  Top-k: {args.top_k}")
    print("\nType your message. 'quit' or Ctrl-C to exit.\n")
    print("=" * 50)

    conversation = []   # alternating [human, assistant, ...]

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if not user_input:
            continue

        conversation.append(user_input)

        # Keep only the most recent N turns to stay within context window
        if len(conversation) > args.max_turns * 2:
            conversation = conversation[-(args.max_turns * 2):]

        reply = generate_reply(
            model        = model,
            tokenizer    = tokenizer,
            conversation = conversation,
            temperature  = args.temp,
            top_k        = args.top_k,
        )

        conversation.append(reply)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()
