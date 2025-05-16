# AMAOS Build Guide

## Overview

This guide covers the build process, CI/CD pipeline configuration, and release procedures for AMAOS.

## Local Build

### Development Environment

1. **Setup Python Environment**
```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. **Install Development Tools**
```powershell
# Install pre-commit hooks
pre-commit install

# Install additional tools
pip install build twine pytest-cov black mypy ruff
```

### Building the Package

1. **Local Build**
```powershell
# Build package
python -m build

# Run tests
pytest tests/
```

2. **Local Docker Build**
```powershell
# Build Docker image
docker build -t amaos:local .

# Test Docker image
docker run amaos:local pytest
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
name: AMAOS CI/CD

on:
  push:
    branches: [ main, develop ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    services:
      redis:
        image: redis
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt -r requirements-dev.txt

    - name: Code quality checks
      run: |
        black . --check
        ruff check .
        mypy .

    - name: Run tests
      run: |
        pytest tests/ --cov=amaos --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  security:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Security scan
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        ignore-unfixed: true
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload security results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/v'))

    steps:
    - uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}

    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          amaos/core:latest
          amaos/core:${{ github.sha }}
        cache-from: type=registry,ref=amaos/core:buildcache
        cache-to: type=registry,ref=amaos/core:buildcache,mode=max

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest

    steps:
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2

    - name: Update EKS cluster
      run: |
        aws eks update-kubeconfig --name amaos-cluster
        helm upgrade --install amaos ./deploy/helm/amaos \
          --namespace amaos \
          --set image.tag=${{ github.sha }}
```

## Release Process

### Version Management

1. **Update Version**
```powershell
# Update version in pyproject.toml
$version = "1.2.3"
(Get-Content pyproject.toml) -replace 'version = ".*"', "version = `"$version`"" | Set-Content pyproject.toml
```

2. **Create Release Tag**
```powershell
# Create and push tag
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3
```

### Release Checklist

- [ ] Update version number
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release tag
- [ ] Monitor deployment

## Deployment Environments

### Development

```yaml
# dev-values.yaml
global:
  environment: development
debug:
  enabled: true
monitoring:
  verbose: true
```

### Staging

```yaml
# staging-values.yaml
global:
  environment: staging
security:
  strict: true
monitoring:
  alerts:
    enabled: true
```

### Production

```yaml
# prod-values.yaml
global:
  environment: production
security:
  strict: true
  networkPolicies:
    enabled: true
monitoring:
  highAvailability: true
```

## Quality Gates

### Code Quality

- Black formatting
- Ruff linting
- mypy type checking
- pytest coverage > 80%
- SonarQube analysis

### Security

- Trivy vulnerability scan
- SAST analysis
- Dependency check
- Container scan
- Secret detection

### Performance

- Load testing
- Resource monitoring
- Response times
- Error rates
- Database performance

## Additional Resources

- [Docker Hub Repository](https://hub.docker.com/r/amaos/core)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/)
- [Helm Charts Repository](https://github.com/amaos/helm-charts)
