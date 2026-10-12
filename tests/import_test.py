def test_import() -> None:
    try:
        import deep_tunnelling  # noqa: PLC0415
    except ImportError:
        deep_tunnelling = None  # ty:ignore[invalid-assignment]

    assert deep_tunnelling is not None, "deep_tunnelling module should not be None"
