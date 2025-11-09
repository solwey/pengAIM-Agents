#!/bin/bash
set -euo pipefail

# Fetch AWS secret values for container
aws secretsmanager get-secret-value --region "$AWS_REGION" --secret-id "$SECRET_NAME_ENV" --query SecretString --output text | jq -r 'keys[] as $k | "\($k)=\"\(.[$k])\""' > $DOTENV_FNAME

# Login into ECR and build the docker image
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
docker buildx build --platform=linux/amd64 --load --tag "$IMG_TAG" --build-arg API_BUILD_ENVIRONMENT="$API_BUILD_ENVIRONMENT" --build-arg ENV_FILE=$DOTENV_FNAME --file ./Dockerfile .
docker push $IMG_TAG

# Run migrations
./scripts/_run-migrations.sh

# Restart ECS services
aws ecs update-service --region "$AWS_REGION" --cluster "$CLUSTER_NAME" --service "$SERVICE_NAME" --no-cli-pager --force-new-deployment
aws ecs update-service --region "$AWS_REGION" --cluster "$CLUSTER_NAME" --service "$WORKER_NAME" --no-cli-pager --force-new-deployment
