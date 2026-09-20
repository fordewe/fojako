"""Token counting utility using tiktoken (OpenAI) and anthropic tokenizer estimation.

Provides accurate token counts for comparing full source vs structural summaries.
"""

import tiktoken


def count_tokens_tiktoken(text: str, model: str = "cl100k_base") -> int:
    """Count tokens using tiktoken encoding (used by OpenAI, good proxy for general LLM tokenization)."""
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def count_tokens_anthropic_estimate(text: str) -> int:
    """Estimate Anthropic token count.

    Anthropic uses a similar BPE tokenizer. A reasonable approximation is
    ~1.3 tokens per word or ~4 characters per token. For more accurate counts,
    use the Anthropic API's token counting endpoint.
    """
    return max(1, len(text) // 4)


def count_chars(text: str) -> int:
    return len(text)


def count_lines(text: str) -> int:
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


def compute_metrics(original: str, summary: str, structural_elements: int) -> dict:
    """Compute all compression metrics between original source and its summary.

    A structural element is one declaration the summary carries: a class, a
    method, or a property. Imports and the Package/File header lines are not
    counted — they are not declarations owned by the file, and their number
    would swamp the count in small files, making ID incomparable across files.
    The caller derives the count from the parsed FileSummary, not from the
    formatted text, so annotated declarations are not missed.

    Returns dict with:
        - original_chars, summary_chars
        - original_tokens, summary_tokens  (tiktoken)
        - original_lines, summary_lines
        - compression_ratio (CR): summary_tokens / original_tokens
        - token_reduction_rate (TRR): 1 - CR
        - structural_elements: as passed in
        - information_density (ID): structural_elements / summary_tokens
    """
    orig_tokens = count_tokens_tiktoken(original)
    summ_tokens = count_tokens_tiktoken(summary)

    cr = summ_tokens / orig_tokens if orig_tokens > 0 else 0.0
    trr = 1.0 - cr

    id_score = structural_elements / summ_tokens if summ_tokens > 0 else 0.0

    return {
        "original_chars": count_chars(original),
        "summary_chars": count_chars(summary),
        "original_lines": count_lines(original),
        "summary_lines": count_lines(summary),
        "original_tokens": orig_tokens,
        "summary_tokens": summ_tokens,
        "compression_ratio": round(cr, 4),
        "token_reduction_rate": round(trr, 4),
        "structural_elements": structural_elements,
        "information_density": round(id_score, 4),
    }
