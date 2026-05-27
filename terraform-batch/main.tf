terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ECR Repository for training image
resource "aws_ecr_repository" "training" {
  name = "chest-ct-training"
  force_delete = true
}

# IAM Role for Batch Service (for Batch to call AWS services)
resource "aws_iam_role" "batch_service_role" {
  name = "chest-ct-batch-service-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "batch.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "batch_service_role_policy" {
  role       = aws_iam_role.batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

# ADDED: ECS Full Access for Batch service role
resource "aws_iam_role_policy_attachment" "batch_service_ecs_full" {
  role       = aws_iam_role.batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonECS_FullAccess"
}

# IAM Role for EC2 instances (for Batch compute resources)
# Updated with correct trust relationship for ECS tasks
resource "aws_iam_role" "batch_role" {
  name = "chest-ct-batch-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      },
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# ADDED: ECS Full Access for batch role
resource "aws_iam_role_policy_attachment" "batch_role_ecs_full" {
  role       = aws_iam_role.batch_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonECS_FullAccess"
}

# S3 Access for Batch
resource "aws_iam_role_policy_attachment" "batch_s3" {
  role       = aws_iam_role.batch_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# ECR Access for Batch
resource "aws_iam_role_policy_attachment" "batch_ecr" {
  role       = aws_iam_role.batch_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# CloudWatch Logs Access
resource "aws_iam_role_policy_attachment" "batch_logs" {
  role       = aws_iam_role.batch_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

# Instance Profile
resource "aws_iam_instance_profile" "batch_profile" {
  name = "chest-ct-batch-profile"
  role = aws_iam_role.batch_role.name
}

# Security Group
resource "aws_security_group" "batch" {
  name        = "chest-ct-batch-sg"
  description = "Security group for Batch compute"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Batch Compute Environment (On-Demand only - guaranteed capacity)
resource "aws_batch_compute_environment" "training" {
  compute_environment_name = "chest-ct-training-env"
  type                     = "MANAGED"
  service_role             = aws_iam_role.batch_service_role.arn

  compute_resources {
    type                = "EC2"
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    max_vcpus           = var.max_vcpus
    min_vcpus           = 0
    desired_vcpus       = 2
    instance_role       = aws_iam_instance_profile.batch_profile.arn
    instance_type       = var.instance_types
    subnets             = var.subnet_ids
    security_group_ids  = [aws_security_group.batch.id]
    
    # Explicitly set On-Demand (no Spot)
    bid_percentage = 0
  }
}

# Batch Job Queue
resource "aws_batch_job_queue" "training" {
  name     = "chest-ct-training-queue"
  state    = "ENABLED"
  priority = 1
  
  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.training.arn
  }
}

# Batch Job Definition (Updated with DAGsHub MLflow environment variables)
resource "aws_batch_job_definition" "retraining" {
  name = "chest-ct-retraining"
  type = "container"

  container_properties = jsonencode({
    image = "${aws_ecr_repository.training.repository_url}:latest"
    
    resourceRequirements = [
      { type = "VCPU", value = tostring(var.job_vcpus) },
      { type = "MEMORY", value = tostring(var.job_memory) }
    ]
    
    environment = [
      { name = "MODEL_BUCKET", value = var.model_bucket },
      { name = "DATA_BUCKET", value = var.data_bucket },
      { name = "AWS_REGION", value = var.aws_region },
      { name = "JENKINS_URL", value = var.jenkins_url },
      { name = "JENKINS_TOKEN", value = var.jenkins_token },
      { name = "JENKINS_USERNAME", value = var.jenkins_username },
      { name = "JENKINS_API_TOKEN", value = var.jenkins_api_token },
      # DAGsHub MLflow Configuration
      { name = "MLFLOW_TRACKING_URI", value = "https://dagshub.com/Gajju9191/chest-ct-ecs.mlflow" },
      { name = "MLFLOW_TRACKING_USERNAME", value = "Gajju9191" },
      { name = "MLFLOW_TRACKING_PASSWORD", value = var.dagshub_token }
    ]
    
    executionRoleArn = aws_iam_role.batch_role.arn
    
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group" = "/aws/batch/chest-ct-retraining"
        "awslogs-region" = var.aws_region
      }
    }
  })
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "batch" {
  name = "/aws/batch/chest-ct-retraining"
  retention_in_days = 30
}

# Daily Schedule at 2 PM IST (8:30 AM UTC)
resource "aws_cloudwatch_event_rule" "daily_retraining" {
  name                = "chest-ct-daily-retraining"
  description         = "Trigger retraining daily at 2 PM IST (8:30 AM UTC)"
  schedule_expression = "cron(30 8 * * ? *)"
}

# IAM Role for EventBridge
resource "aws_iam_role" "eventbridge_role" {
  name = "chest-ct-eventbridge-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "eventbridge_policy" {
  name = "chest-ct-eventbridge-policy"
  role = aws_iam_role.eventbridge_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "batch:SubmitJob"
        Resource = "*"
      }
    ]
  })
}

# EventBridge Target
resource "aws_cloudwatch_event_target" "batch_job" {
  rule      = aws_cloudwatch_event_rule.daily_retraining.name
  target_id = "SubmitBatchJob"
  arn       = aws_batch_job_queue.training.arn

  batch_target {
    job_name       = "chest-ct-retraining-${formatdate("YYYYMMDD-HHmm", timestamp())}"
    job_definition = aws_batch_job_definition.retraining.arn
  }

  role_arn = aws_iam_role.eventbridge_role.arn
}