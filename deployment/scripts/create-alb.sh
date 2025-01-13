#!/bin/bash

set -e  # Exit on error

# Load environment variables
source deployment/scripts/load-env.sh
load_env
MAX_LENGTH=28
TRIMMED_ORG_APP_NAME="${ORG_APP_NAME:0:$MAX_LENGTH}"
TRIMMED_TG_NAME="${ORG_APP_NAME:0:$MAX_LENGTH}"

# Create ALB
echo "Creating Application Load Balancer..."
ORG_ALB_ARN=$(aws elbv2 create-load-balancer \
    --name "${TRIMMED_ORG_APP_NAME}-alb" \
    --subnets "${VPC_SUBNET_1}" "${VPC_SUBNET_2}" \
    --security-groups "${SECURITY_GROUP}" \
    --scheme internet-facing \
    --type application \
    --region eu-north-1 \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text)

echo "ALB created with ARN: ${ORG_ALB_ARN}"

# Create Target Group
echo "Creating Target Group..."
ORG_TARGET_GROUP_ARN=$(aws elbv2 create-target-group \
    --name "${TRIMMED_TG_NAME}-tg" \
    --protocol HTTP \
    --port 8000 \
    --vpc-id "${VPC_ID}" \
    --target-type ip \
    --health-check-path "/docs" \
    --health-check-interval-seconds 30 \
    --region eu-north-1 \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text)

echo "Target Group created with ARN: ${ORG_TARGET_GROUP_ARN}"

# Create Listener
echo "Creating Listener..."
ORG_LISTENER_ARN=$(aws elbv2 create-listener \
    --load-balancer-arn "${ORG_ALB_ARN}" \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn="${ORG_TARGET_GROUP_ARN}" \
    --region eu-north-1 \
    --query 'Listeners[0].ListenerArn' \
    --output text)

echo "Listener created with ARN: ${ORG_LISTENER_ARN}"

# Save ARNs to environment
cat > .env.tmp << EOF
$(cat .env)
ORG_TARGET_GROUP_ARN="${ORG_TARGET_GROUP_ARN}"
ORG_ALB_ARN="${ORG_ALB_ARN}"
ORG_LISTENER_ARN="${ORG_LISTENER_ARN}"
EOF

mv .env.tmp .env

echo "Load balancer setup complete!"