
# ============================================
# CodeBuild IAM Role
# ============================================
resource "aws_iam_role" "codebuild_role" {
  name = "CodeBuild-CT-Role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "codebuild.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# ECR Push Policy for CodeBuild
resource "aws_iam_policy" "codebuild_ecr_push" {
  name        = "ECRPushPolicy"
  description = "Allow CodeBuild to push images to ECR"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = "*"
      }
    ]
  })
}

# Attach ECR Push Policy
resource "aws_iam_role_policy_attachment" "codebuild_ecr_push" {
  role       = aws_iam_role.codebuild_role.name
  policy_arn = aws_iam_policy.codebuild_ecr_push.arn
}

# Attach CodeBuild Developer Access
resource "aws_iam_role_policy_attachment" "codebuild_developer" {
  role       = aws_iam_role.codebuild_role.name
  policy_arn = "arn:aws:iam::aws:policy/AWSCodeBuildDeveloperAccess"
}

# Attach CloudWatch Logs
resource "aws_iam_role_policy_attachment" "codebuild_logs" {
  role       = aws_iam_role.codebuild_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

# ============================================
# CodeBuild Project
# ============================================
resource "aws_codebuild_project" "training" {
  name          = "chest-ct-training-build"
  description   = "Builds Docker image for chest CT retraining"
  build_timeout = 60
  service_role  = aws_iam_role.codebuild_role.arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_MEDIUM"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true  # Required for Docker builds

    environment_variable {
      name  = "AWS_ACCOUNT_ID"
      value = data.aws_caller_identity.current.account_id
    }
  }

  source {
    type            = "GITHUB"
    location        = "https://github.com/Gajju9191/chest-ct-ecs-CT.git"
    git_clone_depth = 1
    
    buildspec = <<-EOT
      version: 0.2
      phases:
        pre_build:
          commands:
            - echo Logging in to Amazon ECR...
            - aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${data.aws_caller_identity.current.account_id}.dkr.ecr.us-east-1.amazonaws.com
        build:
          commands:
            - echo Building Docker image...
            - docker build -f Dockerfile.train -t chest-ct-training .
            - docker tag chest-ct-training:latest ${data.aws_caller_identity.current.account_id}.dkr.ecr.us-east-1.amazonaws.com/chest-ct-training:latest
        post_build:
          commands:
            - echo Pushing Docker image...
            - docker push ${data.aws_caller_identity.current.account_id}.dkr.ecr.us-east-1.amazonaws.com/chest-ct-training:latest
            - echo Build completed on `date`
    EOT
  }

  cache {
    type = "NO_CACHE"
  }

  tags = {
    Name        = "chest-ct-training-build"
    Environment = var.environment
  }
}

# ============================================
# Data source for current account ID
# ============================================
data "aws_caller_identity" "current" {}