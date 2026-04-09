"""
sliding_window.py
-----------------
Helper to split a long document into overlapping chunks.
We use overlapping windows to make sure we don't lose context right at the chunk boundaries—pretty standard for RAG pipelines.
"""

from typing import List


def sliding_window_chunks(
    text: str,
    window_size: int = 2000,
    overlap: int = 200
) -> List[str]:
    """
    Chunk input text into overlapping windows.

    Args:
        text        : The full document string to split.
        window_size : Max characters per chunk (default 2000).
        overlap     : Characters shared between consecutive chunks (default 200).

    Returns:
        A list of text chunk strings.

    Example:
        >>> chunks = sliding_window_chunks("hello world " * 200)
        >>> len(chunks) > 1
        True
    """
    if not text or window_size <= 0:
        return []

    # Ensure overlap is always smaller than the window size
    overlap = min(overlap, window_size - 1)

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + window_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Advance by (window_size - overlap) so next chunk shares 'overlap' chars
        step = window_size - overlap
        start += step

    return chunks


if __name__ == "__main__":
    # Quick smoke-test
    sample = "This is a sample sentence. " * 300
    chunks = sliding_window_chunks(sample)
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk length : {len(chunks[0])}")
    print(f"Second chunk length: {len(chunks[1])}")
    # Verify overlap: end of chunk[0] should match start of chunk[1]
    print(f"Overlap preserved  : {chunks[0][-200:] == chunks[1][:200]}")
