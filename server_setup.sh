#!/bin/bash

# Server Setup Script - chạy trên server sau khi upload code
# Usage: ./server_setup.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}🛠 LightRAG Server Setup Script${NC}"

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo -e "${YELLOW}⚠️  Running as root. Consider using a regular user for deployment.${NC}"
fi

# Check Docker installation
echo -e "${BLUE}🐳 Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}📦 Docker not found. Installing Docker...${NC}"
    
    # Update package index
    sudo apt-get update
    
    # Install dependencies
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Add Docker repository
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    echo -e "${GREEN}✅ Docker installed successfully${NC}"
    echo -e "${YELLOW}⚠️  Please logout and login again for Docker group changes to take effect${NC}"
else
    echo -e "${GREEN}✅ Docker is already installed${NC}"
fi

# Check Docker Compose
echo -e "${BLUE}🔧 Checking Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}📦 Docker Compose not found. Installing...${NC}"
    
    # Install docker-compose
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    echo -e "${GREEN}✅ Docker Compose installed successfully${NC}"
else
    echo -e "${GREEN}✅ Docker Compose is already installed${NC}"
fi

# Check Docker service
echo -e "${BLUE}🚀 Checking Docker service...${NC}"
if ! sudo systemctl is-active --quiet docker; then
    echo -e "${YELLOW}🔄 Starting Docker service...${NC}"
    sudo systemctl start docker
    sudo systemctl enable docker
fi
echo -e "${GREEN}✅ Docker service is running${NC}"

# Install useful tools
echo -e "${BLUE}🛠 Installing additional tools...${NC}"
sudo apt-get update
sudo apt-get install -y \
    htop \
    curl \
    wget \
    unzip \
    nano \
    git \
    ufw \
    net-tools

# Configure firewall
echo -e "${BLUE}🛡 Configuring firewall...${NC}"
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 9621/tcp
echo -e "${GREEN}✅ Firewall configured (SSH and port 9621 allowed)${NC}"

# Set up project directory
PROJECT_DIR="/home/$USER/lightrag"
if [ -d "$PROJECT_DIR" ]; then
    echo -e "${BLUE}📁 Project directory exists: $PROJECT_DIR${NC}"
else
    echo -e "${YELLOW}📁 Creating project directory: $PROJECT_DIR${NC}"
    mkdir -p "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

# Create environment file if not exists
if [ ! -f ".env" ]; then
    if [ -f "production.env" ]; then
        echo -e "${YELLOW}📝 Copying production.env to .env${NC}"
        cp production.env .env
        echo -e "${CYAN}⚠️  IMPORTANT: Edit .env file with your actual configuration!${NC}"
        echo -e "${CYAN}   Run: nano .env${NC}"
    else
        echo -e "${RED}❌ production.env not found. Please upload the project files first.${NC}"
    fi
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

# Set up data directories
echo -e "${BLUE}📂 Setting up data directories...${NC}"
mkdir -p data/{rag_storage,inputs,tiktoken}
chmod -R 755 data/

# Make scripts executable
if [ -f "deploy.sh" ]; then
    chmod +x deploy.sh
    echo -e "${GREEN}✅ deploy.sh is executable${NC}"
fi

if [ -f "upload_to_server.sh" ]; then
    chmod +x upload_to_server.sh
fi

# Display system info
echo -e "${CYAN}📊 System Information:${NC}"
echo -e "${NC}  OS: $(lsb_release -d | cut -f2)${NC}"
echo -e "${NC}  Docker: $(docker --version)${NC}"
echo -e "${NC}  Docker Compose: $(docker-compose --version)${NC}"
echo -e "${NC}  Memory: $(free -h | grep ^Mem | awk '{print $2}')${NC}"
echo -e "${NC}  Disk Space: $(df -h / | tail -1 | awk '{print $4}') available${NC}"

echo ""
echo -e "${GREEN}🎉 Server setup completed!${NC}"
echo ""
echo -e "${CYAN}📋 Next steps:${NC}"
echo -e "${NC}1. Edit configuration: ${YELLOW}nano .env${NC}"
echo -e "${NC}2. Update with your WatsonX API credentials${NC}"
echo -e "${NC}3. Deploy the application: ${YELLOW}./deploy.sh${NC}"
echo ""
echo -e "${CYAN}📖 For detailed instructions, see: README_DEPLOYMENT.md${NC}"