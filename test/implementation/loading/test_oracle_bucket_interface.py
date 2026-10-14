from synthscript.loading import OnlineBucketInterface
from synthscript.shared import PathBundle


def test_obi_downloads_are_explicit(monkeypatch, paths: PathBundle):
    download_calls = []

    def fake_download(*args, **kwargs):
        download_calls.append((args, kwargs))
        return []

    monkeypatch.setattr(OnlineBucketInterface, "_download_image", fake_download)
    monkeypatch.setattr(
        OnlineBucketInterface, "test_connection_successful", lambda self: True
    )
    monkeypatch.setattr(
        OnlineBucketInterface,
        "_compute_pending_objects",
        lambda *args, **kwargs: {str(x): str(x) for x in range(5)},
    )
    interface = OnlineBucketInterface("https://example.test/bucket/")
    assert download_calls == []

    interface.setup(paths)
    assert len(download_calls) == 5
