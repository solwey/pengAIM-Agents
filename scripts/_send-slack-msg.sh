#!/bin/bash
set -euo pipefail

# Send Slack message
echo $SLACK_TITLE:$SLACK_MSG
curl --request POST 'https://slack.com/api/chat.postMessage' \
--header "Authorization: Bearer $SLACK_BOT_TOKEN" \
--header 'Content-Type: application/json' \
--data-raw "{ \"channel\": \"$CHANNEL_ID\", \"text\": \"*$SLACK_TITLE*: $SLACK_MSG\" }"
