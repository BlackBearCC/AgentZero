#!/bin/bash

if [ "$SERVICE_TYPE" = "api" ]; then
    exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
elif [ "$SERVICE_TYPE" = "telegram" ]; then
    echo "Starting Telegram Bot service..."
    exec python -m src.services.telegram_service
else
    echo "Unknown service type: $SERVICE_TYPE"
    exit 1
fi 