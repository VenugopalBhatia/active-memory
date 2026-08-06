from active_memory.context import ApproximateTokenCounter
from active_memory.ingestion.segmentation import segment_message


def test_segmentation_preserves_code_blocks_and_offsets() -> None:
    content = "Decision: use PostgreSQL for storage.\n\n```python\ndef connect():\n    return 'ok'\n```\n\nFollow up with migration tests."
    segments = segment_message(content, ApproximateTokenCounter(), minimum_tokens=3, maximum_tokens=20)
    code = next(segment for segment in segments if segment.metadata.get("code_block"))
    assert code.content.startswith("```python")
    assert content[code.start_offset:code.end_offset].strip() == code.content
    assert all(segment.content in content for segment in segments)


def test_long_prose_is_bounded_without_tiny_trailing_fragment() -> None:
    content = " ".join(f"word{i}" for i in range(100))
    counter = ApproximateTokenCounter(4.0)
    segments = segment_message(content, counter, minimum_tokens=5, maximum_tokens=25)
    assert len(segments) > 1
    assert all(counter.count(segment.content) <= 25 for segment in segments)

