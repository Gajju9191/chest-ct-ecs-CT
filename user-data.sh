#!/bin/bash
echo ECS_CLUSTER=chest-ct-training-env_Batch_b0f0d84f-0373-3107-8a80-afeb1682a0af >> /etc/ecs/ecs.config
echo ECS_ENABLE_TASK_IAM_ROLE=true >> /etc/ecs/ecs.config
echo ECS_ENABLE_TASK_IAM_ROLE_NETWORK_HOST=true >> /etc/ecs/ecs.config
echo ECS_LOGLEVEL=info >> /etc/ecs/ecs.config

# Start ECS agent
systemctl enable ecs
systemctl start ecs

echo "ECS agent started for cluster: chest-ct-training-env_Batch_b0f0d84f-0373-3107-8a80-afeb1682a0af"
