output "ecr_repository_url" {
  description = "ECR repository URL for the training image"
  value = aws_ecr_repository.training.repository_url
}

output "batch_job_queue" {
  description = "AWS Batch job queue name"
  value = aws_batch_job_queue.training.name
}

output "batch_job_definition" {
  description = "AWS Batch job definition name"
  value = aws_batch_job_definition.retraining.name
}

output "batch_job_definition_revision" {
  description = "AWS Batch job definition revision number"
  value = aws_batch_job_definition.retraining.revision
}