#!/bin/bash

# Deploy LightRAG from private registry
# Usage: ./deploy_registry.sh [IMAGE_TAG]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
REGISTRY="reg.securityzone.vn"
PROJECT_NAME="lightrag"
IMAGE_NAME="lightrag_watsonx"
IMAGE_TAG="${1:-latest}"
FULL_IMAGE="$REGISTRY/$PROJECT_NAME/$IMAGE_NAME:$IMAGE_TAG"

echo -e "${GREEN}🚀 Starting LightRAG deployment from private registry...${NC}"
echo -e "${BLUE}📦 Image: $FULL_IMAGE${NC}"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ Error: .env file not found!${NC}"
    echo "Please create .env file with your WatsonX configuration"
    exit 1
fi

echo -e "${BLUE}📋 Checking Docker...${NC}"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running!${NC}"
    exit 1
fi

# Login to registry if needed
echo -e "${YELLOW}🔐 Checking registry access...${NC}"
if ! docker pull $FULL_IMAGE >/dev/null 2>&1; then
    echo -e "${YELLOW}🔑 Registry login required...${NC}"
    echo "Please login to registry:"
    docker login $REGISTRY
fi

echo -e "${YELLOW}📁 Creating data directories...${NC}"
mkdir -p data/rag_storage
mkdir -p data/inputs
mkdir -p data/tiktoken
chmod -R 755 data/

echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker-compose -f docker-compose.registry.yml down || true

echo -e "${YELLOW}📥 Pulling latest image...${NC}"
docker pull $FULL_IMAGE

echo -e "${YELLOW}🚀 Starting LightRAG services...${NC}"
IMAGE_TAG=$IMAGE_TAG docker-compose -f docker-compose.registry.yml up -d

# Wait for service to be ready
echo -e "${YELLOW}⏳ Waiting for service to be ready...${NC}"
sleep 15

# Check if service is running
if docker-compose -f docker-compose.registry.yml ps | grep -q "Up"; then
    echo -e "${GREEN}✅ LightRAG deployed successfully from registry!${NC}"
    
    # Get server IP
    SERVER_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
    
    echo -e "${GREEN}🌐 Web UI: http://$SERVER_IP:9621${NC}"
    echo -e "${GREEN}📚 API Docs: http://$SERVER_IP:9621/docs${NC}"
    echo -e "${GREEN}🔗 OpenAI API: http://$SERVER_IP:9621/api/v1/chat/completions${NC}"
else
    echo -e "${RED}❌ Deployment failed! Check logs with: docker-compose -f docker-compose.registry.yml logs${NC}"
    exit 1
fi

# Show recent logs
echo -e "${BLUE}📜 Recent logs:${NC}"
docker-compose -f docker-compose.registry.yml logs --tail=20

echo -e "${GREEN}🎉 Registry deployment completed successfully!${NC}"
echo ""
echo -e "${CYAN}🔧 Useful commands:${NC}"
echo -e "${NC}  • View logs: ${YELLOW}docker-compose -f docker-compose.registry.yml logs -f${NC}"
echo -e "${NC}  • Restart: ${YELLOW}docker-compose -f docker-compose.registry.yml restart${NC}"
echo -e "${NC}  • Stop: ${YELLOW}docker-compose -f docker-compose.registry.yml down${NC}"
echo -e "${NC}  • Update: ${YELLOW}./deploy_registry.sh latest${NC}"