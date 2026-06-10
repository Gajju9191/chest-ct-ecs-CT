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

variable "models_bucket" {
  description = "S3 bucket for trained models"
  type        = string
  default     = "chest-ct-models-155407238004"
}

variable "raw_data_bucket" {
  description = "S3 bucket for raw training data"
  type        = string
  default     = "chest-models-gajju"
}

variable "job_vcpus" {
  description = "vCPUs for Batch job"
  type        = number
  default     = 4
}

variable "job_memory" {
  description = "Memory for Batch job (MB)"
  type        = number
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

variable "jenkins_username" {
  description = "Jenkins username"
  type        = string
  default     = "Gajanan Wagalgave"
}

variable "jenkins_api_token" {
  description = "Jenkins API token"
  type        = string
  sensitive   = true
}

variable "dagshub_token" {
  description = "DAGsHub access token for MLflow tracking"
  type        = string
  sensitive   = true
}