variable "region" {
  description = "AWS region for the model bucket"
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Named AWS CLI profile to authenticate with (e.g. KourePowerUser); null uses the default credential chain"
  type        = string
  default     = null
}

variable "models_bucket" {
  description = "S3 bucket for trained models. astrotrain.publish uses the same default name (override there with ASTROTRAIN_MODELS_BUCKET)."
  type        = string
  default     = "astrotrain-models"
}

variable "noncurrent_version_days" {
  description = "Days to keep an overwritten or deleted object version before it expires"
  type        = number
  default     = 90
}
