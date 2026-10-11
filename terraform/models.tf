# --- Trained model storage ------------------------------------------------ #
# astrotrain.publish uploads each published model to
#   models/<model name>/<YYYY-MM-DD>-<commit>/
# with a metadata.json describing how it was trained, and points
#   models/<model name>/latest.json
# at the newest version. Models are small (a few MB) and can be retrained from
# code + the public LAMP data, so this is for keeping track of and sharing the
# models we promote, not the only copy of anything irreplaceable.

resource "aws_s3_bucket" "models" {
  bucket = var.models_bucket
}

# Block all public access; models are shared deliberately, not by URL.
resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Versioning keeps the previous copy when an object is overwritten or deleted.
# Published versions go to new dated prefixes, so in practice this protects
# latest.json and any accidental re-upload over an existing version.
resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 encrypts new objects with SSE-S3 by default; stated here so it's explicit.
resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  # Overwritten/deleted versions (mostly old latest.json pointers) don't need
  # keeping forever. Current versions are never expired.
  rule {
    id     = "models-expire-noncurrent"
    status = "Enabled"
    filter {} # whole bucket
    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_days
    }
  }

  # Hygiene: clean up interrupted multipart uploads.
  rule {
    id     = "models-abort-mpu"
    status = "Enabled"
    filter {} # whole bucket
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }

  depends_on = [aws_s3_bucket_versioning.models]
}
