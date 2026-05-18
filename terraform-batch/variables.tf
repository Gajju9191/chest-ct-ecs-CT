variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "VPC ID for Batch compute environment"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for Batch compute environment"
  type        = list(string)
}

variable "model_bucket" {
  description = "S3 bucket for models"
  default     = "chest-ct-models-155407238003"
}

variable "data_bucket" {
  description = "S3 bucket for training data"
  default     = "chest-ct-raw-data"
}

variable "job_vcpus" {
  description = "vCPUs for Batch job"
  default     = 4
}

variable "job_memory" {
  description = "Memory for Batch job (MB)"
  default     = 16384
}

variable "jenkins_url" {
  description = "Jenkins server URL"
  type        = string
  sensitive   = true
}

variable "jenkins_token" {
  description = "Jenkins webhook token"
  type        = string
  sensitive   = true
}