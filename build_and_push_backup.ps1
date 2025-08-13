# Optimized PowerShell script to build and push LightRAG with WatsonX to private registry
# Usage: .\build_and_push.ps1 [-Tag "v1.0.0"] [-Optimized] [-MultiArch]

param(
    [string]$Registry = "reg.securityzone.vn",
    [string]$ProjectName = "lightrag", 
    [string]$ImageName = "lightrag_watsonx", 
    [string]$Tag = "latest",
    [string]$Dockerfile = "Dockerfile.optimized",
    [switch]$NoBuild = $false,
    [switch]$Login = $false,
    [switch]$Optimized = $true,
    [switch]$MultiArch = $false,
    [switch]$NoCache = $false,
    [switch]$Verbose = $false
)

# Configuration
$FullImage = "$Registry/$ProjectName/$ImageName" + ":" + $Tag
$BuildStart = Get-Date

# Functions for colored output
function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Step {
    param([string]$Message)
    Write-Host "🔄 $Message" -ForegroundColor Cyan
}

# Header
Write-Host "🐳 LightRAG WatsonX - Optimized Build `& Push Script" -ForegroundColor Magenta
Write-Host "=" * 60
Write-Info "Registry: $Registry"
Write-Info "Project: $ProjectName"
Write-Info "Image: $ImageName"
Write-Info "Tag: $Tag"
Write-Info "Dockerfile: $Dockerfile"
Write-Info "Full Image: $FullImage"
Write-Host ""

# Check prerequisites
Write-Step "Checking prerequisites..."

# Check Docker
try {
    docker version | Out-Null
    Write-Success "Docker is available"
} catch {
    Write-Error "Docker is not running or not installed!"
    exit 1
}

# Check Dockerfile
if ($Optimized -and (Test-Path "Dockerfile.optimized")) {
    $Dockerfile = "Dockerfile.optimized"
    Write-Success "Using optimized Dockerfile"
} elseif (Test-Path $Dockerfile) {
    Write-Success "Using Dockerfile: $Dockerfile"
} else {
    Write-Error "Dockerfile '$Dockerfile' not found!"
    exit 1
}

# Check .dockerignore
if (Test-Path ".dockerignore.optimized") {
    Copy-Item ".dockerignore.optimized" ".dockerignore" -Force
    Write-Info "Using optimized .dockerignore"
}

# Docker login if requested
if ($Login) {
    Write-Step "Logging in to registry..."
    docker login $Registry
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker login failed!"
        exit 1
    }
    Write-Success "Successfully logged in to $Registry"
}

# Build image
if (-not $NoBuild) {
    Write-Step "Building Docker image with optimizations..."
    
    # Prepare build arguments
    $BuildArgs = @(
        "build"
        "-t", $FullImage
        "-f", $Dockerfile
    )
    
    # Add build optimizations
    if ($NoCache) {
        $BuildArgs += "--no-cache"
        Write-Info "No-cache build enabled"
    }
    
    if ($Verbose) {
        $BuildArgs += "--progress=plain"
        Write-Info "Verbose build output enabled"
    }
    
    # Multi-architecture build (experimental)
    if ($MultiArch) {
        Write-Warning "Multi-architecture build enabled (experimental)"
        $BuildArgs = @(
            "buildx", "build"
            "--platform", "linux/amd64,linux/arm64"
            "--push"
            "-t", $FullImage
            "-f", $Dockerfile
        )
        if ($NoCache) { $BuildArgs += "--no-cache" }
        if ($Verbose) { $BuildArgs += "--progress=plain" }
        $BuildArgs += "."
        
        Write-Info "Build command: docker $($BuildArgs -join ' ')"
        & docker @BuildArgs
    } else {
        # Standard build
        $BuildArgs += "."
        
        Write-Info "Build command: docker $($BuildArgs -join ' ')"
        & docker @BuildArgs
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed!"
        # Cleanup
        if (Test-Path ".dockerignore.temp") {
            Remove-Item ".dockerignore.temp" -Force
        }
        exit 1
    }
    
    $BuildEnd = Get-Date
    $BuildTime = [math]::Round(($BuildEnd - $BuildStart).TotalSeconds)
    Write-Success "Build completed successfully in ${BuildTime}s!"
    
    # Show image information
    $ImageInfo = docker images $FullImage --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    Write-Info "Image details:"
    Write-Host $ImageInfo
}

# Tag additional versions if needed
if ($Tag -ne "latest") {
    Write-Step "Tagging as latest..."
    $LatestImage = "$Registry/$ProjectName/$ImageName" + ":latest"
    docker tag $FullImage $LatestImage
    Write-Success "Tagged as: $LatestImage"
}

# Push to registry
if (-not $MultiArch) {  # MultiArch already pushes during build
    Write-Step "Pushing to registry..."
    docker push $FullImage
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Push failed!"
        exit 1
    }
    
    # Push latest tag if created
    if ($Tag -ne "latest") {
        Write-Step "Pushing latest tag..."
        $LatestImage = "$Registry/$ProjectName/$ImageName" + ":latest"
        docker push $LatestImage
        
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Failed to push latest tag, but main tag was pushed successfully"
        } else {
            Write-Success "Latest tag pushed successfully"
        }
    }
}

# Cleanup
if ((Test-Path ".dockerignore") -and (Test-Path ".dockerignore.optimized")) {
    Remove-Item ".dockerignore" -Force
    Write-Info "Cleaned up temporary .dockerignore"
}

# Final summary
$TotalTime = [math]::Round((Get-Date - $BuildStart).TotalSeconds)
Write-Host ""
Write-Host "🎉 BUILD `& PUSH COMPLETED SUCCESSFULLY! 🎉" -ForegroundColor Green
Write-Host "=" * 60
Write-Success "Total time: ${TotalTime}s"
Write-Success "Image: $FullImage"

# Show deployment information
Write-Host ""
Write-Host "📋 DEPLOYMENT INFORMATION:" -ForegroundColor Yellow
Write-Host "Registry URL: https://$Registry" -ForegroundColor Cyan
Write-Host "Image: $FullImage" -ForegroundColor Cyan
Write-Host ""

Write-Host "🚀 DEPLOYMENT COMMANDS:" -ForegroundColor Yellow
Write-Host "Server deployment:" -ForegroundColor White
Write-Host "  ./deploy_registry.sh $Tag" -ForegroundColor Gray
Write-Host ""
Write-Host "Docker run (production):" -ForegroundColor White
Write-Host "  docker run -d --name lightrag-watsonx \\" -ForegroundColor Gray
Write-Host "    -p 9621:9621 \\" -ForegroundColor Gray
Write-Host "    -e WATSONX_API_KEY=your_key \\" -ForegroundColor Gray
Write-Host "    -e WATSONX_PROJECT_ID=your_project \\" -ForegroundColor Gray
Write-Host "    -e OPENAI_API_KEY=your_openai_key \\" -ForegroundColor Gray
Write-Host "    -v `$(pwd)/data:/app/data \\" -ForegroundColor Gray
Write-Host "    -v `$(pwd)/logs:/app/logs \\" -ForegroundColor Gray
Write-Host "    $FullImage" -ForegroundColor Gray
Write-Host ""

Write-Host "Docker Compose:" -ForegroundColor White
Write-Host "  # Update image in docker-compose.yml:" -ForegroundColor Gray
Write-Host "  # image: $FullImage" -ForegroundColor Gray
Write-Host "  docker-compose up -d" -ForegroundColor Gray
Write-Host ""

Write-Host "Kubernetes:" -ForegroundColor White
Write-Host "  kubectl set image deployment/lightrag-watsonx lightrag-watsonx=$FullImage" -ForegroundColor Gray
Write-Host ""

# Show optimization suggestions
Write-Host "💡 OPTIMIZATION TIPS:" -ForegroundColor Yellow
if (-not $Optimized) {
    Write-Host "  • Use -Optimized flag for faster builds" -ForegroundColor Gray
}
if (-not $NoCache) {
    Write-Host "  • Use -NoCache for clean builds" -ForegroundColor Gray
}
Write-Host "  • Use -MultiArch for multi-platform support" -ForegroundColor Gray
Write-Host "  • Use -Verbose for detailed build output" -ForegroundColor Gray
Write-Host ""

# Security reminder
Write-Host "🔒 SECURITY REMINDERS:" -ForegroundColor Red
Write-Host "  • Store API keys in environment variables or secrets" -ForegroundColor Gray
Write-Host "  • Use non-root user in production" -ForegroundColor Gray
Write-Host "  • Enable container scanning in registry" -ForegroundColor Gray
Write-Host "  • Regular security updates" -ForegroundColor Gray