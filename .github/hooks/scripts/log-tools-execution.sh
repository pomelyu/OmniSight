#!/bin/bash

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.toolName')
RESULT_TYPE=$(echo "$INPUT" | jq -r '.toolResult.resultType')

# Track statistics
echo "$(date),${TOOL_NAME},${RESULT_TYPE}" >> tool-stats.csv

# Alert on failures
if [ "$RESULT_TYPE" = "failure" ]; then
  RESULT_TEXT=$(echo "$INPUT" | jq -r '.toolResult.textResultForLlm')
  echo "FAILURE: $TOOL_NAME - $RESULT_TEXT" | mail -s "Agent Tool Failed" admin@example.com
fi
