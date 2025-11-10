#!/bin/bash
set -euo pipefail

# Read ENV vars
ENV_FILE=./.env_deploy
export $(cat $ENV_FILE | xargs)

# Fetch AWS secret values for CI
export AWS_REGION=us-east-2
export SECRET_NAME_ENV=qa-peng-aim-agents
export SECRET_NAME_CI=qa-peng-aim-agents-ci

export $(aws secretsmanager get-secret-value --region "$AWS_REGION" --secret-id "$SECRET_NAME_CI" --query SecretString --output text | jq -r 'keys[] as $k | "\($k)=\"\(.[$k])\""' | xargs)

echo "Building STAGING AGENTS with image tag [$IMG_TAG]..."
./scripts/_deploy.sh

# Notify in Slack
export SLACK_TITLE="STAGING AGENTS"
export SLACK_MSG="New version is live."
./scripts/_send-slack-msg.sh
