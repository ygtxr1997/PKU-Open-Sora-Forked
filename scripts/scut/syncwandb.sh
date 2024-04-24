#!/bin/bash
while true; do
    echo "sync wandb"
    wandb sync wandb/latest-run
    sleep 300
done