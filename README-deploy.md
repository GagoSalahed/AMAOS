# AMAOS Deployment Guide

## Local Development Setup

1. Clone the repository and create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and configure your settings:
```bash
cp .env.example .env
```

3. Run the development server:
```bash
python -m amaos.entry
```

## Docker Deployment

### Local Docker Setup

1. Build and run with Docker Compose:
```bash
docker compose up --build -d
```

2. Monitor logs:
```bash
docker compose logs -f
```

3. Scale services if needed:
```bash
docker compose up -d --scale amaos=2
```

### Oracle VM Deployment

1. Setup Oracle Always Free VM:
   - Create Ubuntu 22.04 instance
   - Open ports 8000 (API) and 9090 (Prometheus)
   - Configure security rules

2. Install dependencies:
```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get install docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
```

3. Clone and deploy:
```bash
git clone <repository-url>
cd AMAOS
cp .env.example .env
# Edit .env with production settings
docker compose up --build -d
```

4. Setup monitoring:
   - Access Prometheus: http://your-vm-ip:9090
   - Configure alerting rules in prometheus_rules.yml
   - Optional: Add Grafana for visualization

## Health Checks

The system provides multiple health check endpoints:

- API Health: `GET /health`
- Prometheus Metrics: `GET /metrics`
- Agent Status: `GET /agents/{agent_id}/health`

## Production Checklist

- [ ] Configure proper API authentication
- [ ] Set up log rotation
- [ ] Configure backups for Redis data
- [ ] Set appropriate rate limits
- [ ] Configure SSL/TLS
- [ ] Set up monitoring alerts
- [ ] Configure error reporting

## Rollback Procedure

In case of issues:

1. Rollback to previous version:
```bash
docker compose down
git checkout <previous-tag>
docker compose up --build -d
```

2. Restore data if needed:
```bash
docker compose cp backup.rdb redis:/data/dump.rdb
docker compose restart redis
```
