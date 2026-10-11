terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # State is local for now, as in us-rail-peformance-archiver. If this grows,
  # move to an S3 backend.
}

provider "aws" {
  region = var.region
  # Pin the named profile so creds resolve like the aws CLI does, independent of
  # AWS_PROFILE being exported. Set aws_profile in terraform.tfvars.
  profile = var.aws_profile

  default_tags {
    tags = {
      project   = "astrotrain"
      managedby = "terraform"
    }
  }
}
