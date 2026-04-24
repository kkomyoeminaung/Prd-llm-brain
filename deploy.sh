#!/bin/bash
# PRD-LLM Deployment Script

echo "🚀 Deploying PRD-LLM to Production..."

# 1. Build Image
docker build -t prd-llm:v2.0 .

# 2. Push to Registry (Simulated)
# docker push your-registry/prd-llm:v2.0

# 3. Apply K8s Configs
kubectl apply -f k8s/deployment.yaml

# 4. Success Message
echo "✅ PRD-LLM is now scaling across 3 nodes."
echo "🔗 Access your API at the LoadBalancer IP."
