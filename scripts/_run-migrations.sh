#!/bin/bash
set -euo pipefail

# Run migrations
echo "Running migration task..."
run_result=$(AWS_MAX_ATTEMPTS=1 aws ecs run-task --region "$AWS_REGION" --cluster "$CLUSTER_NAME" --capacity-provider-strategy "$CAPACITY_PROVIDER_STRATEGY" --task-definition "$TASK_NAME" --overrides file://scripts/ecs-overrides-migration-task.json)
container_arn=$(echo $run_result | jq '.tasks[0].taskArn' | sed -e 's/^"//' -e 's/"$//')
echo "Migration container ARN = $container_arn"
aws ecs wait tasks-stopped --region "$AWS_REGION" --cluster "$CLUSTER_NAME" --tasks "${container_arn}"
describe_result=$(aws ecs describe-tasks --region "$AWS_REGION" --cluster "$CLUSTER_NAME" --tasks "${container_arn}")
migration_status=$(echo $describe_result | jq '.tasks[0].containers[0].exitCode')

if [ $migration_status -ne 0 ]; then
  echo "Error: Migration task exited with non-zero status [$migration_status]"
  exit $migration_status
fi
