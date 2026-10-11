import io
import json

import pytest

from astrotrain.publish import download, upload, version_of


class FakeS3:
    """Just enough of a boto3 S3 client, in memory"""

    def __init__(self):
        self.objects = {}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000):
        keys = sorted(
            k for b, k in self.objects if b == Bucket and k.startswith(Prefix)
        )
        return {"KeyCount": len(keys[:MaxKeys]), "Contents": [{"Key": k} for k in keys]}

    def get_paginator(self, name):
        client = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                yield client.list_objects_v2(Bucket=Bucket, Prefix=Prefix)

        return Paginator()

    def upload_file(self, filename, bucket, key):
        with open(filename, "rb") as f:
            self.objects[(bucket, key)] = f.read()

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def download_file(self, bucket, key, filename):
        with open(filename, "wb") as f:
            f.write(self.objects[(bucket, key)])


METADATA = {"saved_at": "2026-10-11T14:00:00+00:00", "commit": "73ebeb1aaaa"}


@pytest.fixture
def model_dir(tmp_path):
    folder = tmp_path / "countdown_gbm"
    (folder / "logs").mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps(METADATA))
    (folder / "gbm.txt").write_text("trees")
    (folder / "preprocessing.pkl").write_bytes(b"\x80pickle")
    (folder / "logs" / "train.log").write_text("a long log")
    return folder


def test_upload_publishes_a_dated_version_and_points_latest_at_it(model_dir):
    s3 = FakeS3()

    prefix = upload(model_dir, s3, "bucket")

    assert prefix == "models/countdown_gbm/2026-10-11-73ebeb1/"
    keys = {k for _, k in s3.objects}
    assert keys == {
        prefix + "metadata.json",
        prefix + "gbm.txt",
        prefix + "preprocessing.pkl",
        "models/countdown_gbm/latest.json",
    }  # logs stay local
    latest = json.loads(s3.objects[("bucket", "models/countdown_gbm/latest.json")])
    assert latest["version"] == "2026-10-11-73ebeb1"
    assert latest["metadata"] == METADATA


def test_upload_refuses_to_overwrite_a_published_version(model_dir):
    s3 = FakeS3()
    upload(model_dir, s3, "bucket")

    with pytest.raises(ValueError, match="already published"):
        upload(model_dir, s3, "bucket")


def test_upload_needs_metadata(model_dir):
    (model_dir / "metadata.json").unlink()

    with pytest.raises(ValueError, match="no metadata.json"):
        upload(model_dir, FakeS3(), "bucket")


def test_version_names():
    assert version_of({**METADATA, "uncommitted_changes": True}) == (
        "2026-10-11-73ebeb1-dirty"
    )
    assert version_of({**METADATA, "commit": None}) == "2026-10-11-nocommit"


def test_download_restores_the_latest_or_a_given_version(model_dir, tmp_path):
    s3 = FakeS3()
    upload(model_dir, s3, "bucket")
    newer = {**METADATA, "saved_at": "2026-10-12T09:00:00+00:00"}
    (model_dir / "metadata.json").write_text(json.dumps(newer))
    (model_dir / "gbm.txt").write_text("newer trees")
    upload(model_dir, s3, "bucket")

    latest = download("countdown_gbm", s3, "bucket", to=tmp_path / "latest")
    older = download(
        "countdown_gbm",
        s3,
        "bucket",
        version="2026-10-11-73ebeb1",
        to=tmp_path / "older",
    )

    assert (latest / "gbm.txt").read_text() == "newer trees"
    assert (older / "gbm.txt").read_text() == "trees"
    assert (older / "preprocessing.pkl").read_bytes() == b"\x80pickle"
    assert not (older / "logs").exists()


def test_download_of_an_unknown_version_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="Nothing published"):
        download("countdown_gbm", FakeS3(), "bucket", version="nope", to=tmp_path)
