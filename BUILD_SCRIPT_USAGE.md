# Build Script Usage Guide

## ✅ **PowerShell Script đã được fix**

Tất cả lỗi syntax đã được khắc phục:
- ❌ Ampersand (&) characters → ✅ Fixed
- ❌ DateTime calculation errors → ✅ Fixed  
- ❌ String multiplication issues → ✅ Fixed
- ❌ Emoji/special characters → ✅ Replaced with text

## 🚀 **Cách sử dụng build_and_push.ps1**

### **Basic Usage:**

```powershell
# Build và push với tag cụ thể
.\build_and_push.ps1 -Tag "v1.2.0"

# Build, login và push 
.\build_and_push.ps1 -Tag "v1.2.0" -Login

# Build với verbose output
.\build_and_push.ps1 -Tag "v1.2.0" -Verbose

# Clean build (no cache)
.\build_and_push.ps1 -Tag "v1.2.0" -NoCache
```

### **Advanced Usage:**

```powershell
# Multi-architecture build (experimental)
.\build_and_push.ps1 -Tag "v1.2.0" -MultiArch

# Skip build, only push existing image
.\build_and_push.ps1 -Tag "v1.2.0" -NoBuild

# Custom registry settings
.\build_and_push.ps1 -Registry "my-registry.com" -ProjectName "myproject" -ImageName "myimage" -Tag "v1.0.0"
```

### **Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-Registry` | string | `reg.securityzone.vn` | Docker registry URL |
| `-ProjectName` | string | `lightrag` | Project namespace |
| `-ImageName` | string | `lightrag_watsonx` | Image name |
| `-Tag` | string | `latest` | Image tag |
| `-Dockerfile` | string | `Dockerfile.optimized` | Dockerfile to use |
| `-NoBuild` | switch | `false` | Skip build, only push |
| `-Login` | switch | `false` | Login to registry first |
| `-Optimized` | switch | `true` | Use optimized Dockerfile |
| `-MultiArch` | switch | `false` | Multi-platform build |
| `-NoCache` | switch | `false` | Build without cache |
| `-Verbose` | switch | `false` | Verbose output |

## 📊 **Expected Output:**

```
Docker LightRAG WatsonX - Optimized Build and Push Script
============================================================
INFO: Registry: reg.securityzone.vn
INFO: Project: lightrag
INFO: Image: lightrag_watsonx
INFO: Tag: v1.2.0
INFO: Dockerfile: Dockerfile.optimized
INFO: Full Image: reg.securityzone.vn/lightrag/lightrag_watsonx:v1.2.0

STEP: Checking prerequisites...
SUCCESS: Docker is available
SUCCESS: Using optimized Dockerfile
INFO: Using optimized .dockerignore

STEP: Building Docker image with optimizations...
INFO: Build command: docker build -t reg.securityzone.vn/lightrag/lightrag_watsonx:v1.2.0 -f Dockerfile.optimized .
[Docker build output...]
SUCCESS: Build completed successfully in 180s!

STEP: Tagging as latest...
SUCCESS: Tagged as: reg.securityzone.vn/lightrag/lightrag_watsonx:latest

STEP: Pushing to registry...
[Docker push output...]
SUCCESS: Latest tag pushed successfully

INFO: Cleaned up temporary .dockerignore

BUILD AND PUSH COMPLETED SUCCESSFULLY!
============================================================
SUCCESS: Total time: 245s
SUCCESS: Image: reg.securityzone.vn/lightrag/lightrag_watsonx:v1.2.0

DEPLOYMENT INFORMATION:
Registry URL: https://reg.securityzone.vn
Image: reg.securityzone.vn/lightrag/lightrag_watsonx:v1.2.0

DEPLOYMENT COMMANDS:
Server deployment:
  ./deploy_registry.sh v1.2.0

Docker run (production):
  docker run -d --name lightrag-watsonx \
    -p 9621:9621 \
    -e WATSONX_API_KEY=your_key \
    -e WATSONX_PROJECT_ID=your_project \
    -e OPENAI_API_KEY=your_openai_key \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    reg.securityzone.vn/lightrag/lightrag_watsonx:v1.2.0

Docker Compose:
  # Update image in docker-compose.yml:
  # image: reg.securityzone.vn/lightrag/lightrag_watsonx:v1.2.0
  docker-compose up -d

Kubernetes:
  kubectl set image deployment/lightrag-watsonx lightrag-watsonx=reg.securityzone.vn/lightrag/lightrag_watsonx:v1.2.0

OPTIMIZATION TIPS:
  • Use -NoCache for clean builds
  • Use -MultiArch for multi-platform support
  • Use -Verbose for detailed build output

SECURITY REMINDERS:
  • Store API keys in environment variables or secrets
  • Use non-root user in production
  • Enable container scanning in registry
  • Regular security updates
```

## 🔍 **Troubleshooting:**

### **Common Issues:**

1. **"Docker is not running"**
   ```powershell
   # Start Docker Desktop
   # Wait for Docker to fully start
   # Then retry the script
   ```

2. **"Dockerfile not found"**
   ```powershell
   # Ensure you're in the LightRAG root directory
   # Check if Dockerfile.optimized exists
   ls Dockerfile*
   ```

3. **"Login failed"**
   ```powershell
   # Manual login first
   docker login reg.securityzone.vn
   
   # Then run script without -Login flag
   .\build_and_push.ps1 -Tag "v1.2.0"
   ```

4. **"Push failed"**
   ```powershell
   # Check network connection
   # Verify registry credentials
   # Ensure you have push permissions
   ```

## ✅ **Next Steps after Build:**

1. **Deploy on server:**
   ```bash
   ./deploy_registry.sh v1.2.0
   ```

2. **Verify deployment:**
   ```bash
   curl http://server:9621/health
   ```

3. **Check logs:**
   ```bash
   docker logs lightrag-watsonx
   ```

## 🎯 **Best Practices:**

- ✅ Always tag with specific versions (avoid `latest` in production)
- ✅ Use `-NoCache` for important releases
- ✅ Test locally before pushing to registry
- ✅ Keep build times fast with optimized Dockerfile
- ✅ Monitor registry storage usage

Script is now fully functional and ready for production use!