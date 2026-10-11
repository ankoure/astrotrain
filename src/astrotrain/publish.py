"""
Publish saved models to S3, and download them again.

Each published model goes to s3://<bucket>/models/<name>/<version>/, where
<version> is the save date and commit (e.g. 2026-10-11-73ebeb1), and
models/<name>/latest.json points at the newest version. The bucket is set up by
terraform/models.tf; its name defaults to that Terraform default and can be
changed with ASTROTRAIN_MODELS_BUCKET. AWS credentials come from the usual
places, e.g. AWS_PROFILE=KourePowerUser.

usage: uv run python -m astrotrain.publish upload models/countdown_gbm
       uv run python -m astrotrain.publish download countdown_gbm [--version V]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

DEFAULT_BUCKET = "astrotrain-models"

# Training logs stay local; they're big and not needed to use a model
SKIPPED_FOLDERS = {"logs"}


def bucket_name() -> str:
    return os.environ.get("ASTROTRAIN_MODELS_BUCKET", DEFAULT_BUCKET)


def version_of(metadata: dict) -> str:
    """A published version's name: the save date and short commit"""
    commit = metadata.get("commit")
    commit = commit[:7] if commit else "nocommit"
    dirty = "-dirty" if metadata.get("uncommitted_changes") else ""
    return f"{metadata['saved_at'][:10]}-{commit}{dirty}"


def upload(model_dir: Path, client, bucket: str) -> str:
    """
    Upload a saved model folder (which must have a metadata.json) and point
    latest.json at it. Refuses to overwrite a version that's already published.
    Returns the version's S3 prefix.
    """
    model_dir = Path(model_dir)
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"{model_dir} has no metadata.json; save it again first")
    metadata = json.loads(metadata_path.read_text())
    name, version = model_dir.name, version_of(metadata)
    prefix = f"models/{name}/{version}/"

    existing = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    if existing.get("KeyCount", 0):
        raise ValueError(f"s3://{bucket}/{prefix} is already published")

    files = [
        path
        for path in sorted(model_dir.rglob("*"))
        if path.is_file()
        and not SKIPPED_FOLDERS & set(path.relative_to(model_dir).parts)
    ]
    for path in files:
        key = prefix + path.relative_to(model_dir).as_posix()
        client.upload_file(str(path), bucket, key)
        print(f"uploaded s3://{bucket}/{key}")

    latest = {"version": version, "prefix": prefix, "metadata": metadata}
    client.put_object(
        Bucket=bucket,
        Key=f"models/{name}/latest.json",
        Body=json.dumps(latest, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"models/{name}/latest.json now points at {version}")
    return prefix


def download(
    name: str,
    client,
    bucket: str,
    version: Optional[str] = None,
    to: Optional[Path] = None,
) -> Path:
    """Download a published model (the latest unless a version is given)"""
    if version is None:
        latest = client.get_object(Bucket=bucket, Key=f"models/{name}/latest.json")
        version = json.loads(latest["Body"].read())["version"]
    prefix = f"models/{name}/{version}/"
    to = Path(to or Path("models") / name)

    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        keys += [item["Key"] for item in page.get("Contents", [])]
    if not keys:
        raise ValueError(f"Nothing published at s3://{bucket}/{prefix}")

    for key in keys:
        destination = to / key[len(prefix) :]
        destination.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(destination))
    print(f"Downloaded {name} {version} to {to}")
    return to


def main():
    parser = argparse.ArgumentParser(description="Publish or download models in S3")
    commands = parser.add_subparsers(dest="command", required=True)
    up = commands.add_parser("upload", help="Publish a saved model folder")
    up.add_argument("model_dir", type=Path)
    down = commands.add_parser("download", help="Download a published model")
    down.add_argument("name", help="Model name, e.g. countdown_gbm")
    down.add_argument("--version", help="Version to download (default: latest)")
    down.add_argument("--to", type=Path, help="Folder (default: models/<name>)")
    args = parser.parse_args()

    import boto3

    client = boto3.client("s3")
    if args.command == "upload":
        upload(args.model_dir, client, bucket_name())
    else:
        download(args.name, client, bucket_name(), args.version, args.to)


if __name__ == "__main__":
    main()
