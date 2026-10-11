output "models_bucket" {
  description = "S3 bucket for trained models"
  value       = aws_s3_bucket.models.bucket
}

output "models_bucket_arn" {
  description = "ARN of the model bucket"
  value       = aws_s3_bucket.models.arn
}
