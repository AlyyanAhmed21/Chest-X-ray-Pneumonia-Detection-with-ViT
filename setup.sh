#!/bin/bash

# Create the directory DVC needs for credentials
mkdir -p .dvc/tmp

# Write the Google Drive credentials from the HF secret into the file
echo "$GDRIVE_CREDENTIALS_DATA" > .dvc/tmp/gdrive-user-credentials.json

# Pull the model file from the default remote (which we set to Google Drive)
dvc pull artifacts/model_training/model.dvc -f