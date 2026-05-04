AWS_ACCOUNT_ID = 458960552929
AWS_REGION     = eu-central-1
IMAGE          = $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/berlin-aqi-mlops:latest

EC2_HOST       = 18.195.97.53
EC2_USER       = ec2-user
SSH_KEY        = ~/.ssh/berlin-aqi-key.pem
CONTAINER_NAME = berlin-aqi

login:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

bundle:
	uv run python -m src.bundle

build:
	docker build --platform linux/amd64 -t berlin-aqi-mlops .

tag:
	docker tag berlin-aqi-mlops:latest $(IMAGE)

push:
	docker push $(IMAGE)

deploy: login bundle build tag push

ssh:
	ssh -i $(SSH_KEY) $(EC2_USER)@$(EC2_HOST)

frontend:
	uv run streamlit run frontend/app.py

# Remote deploy: SSH to EC2, pull the new image, swap containers, bootstrap
# the prediction cache. Intended to run straight after `make deploy`.
# Runs as a single piped script so we exit as soon as any step fails.
define EC2_DEPLOY_SCRIPT
set -euo pipefail
echo "==> ECR login"
aws ecr get-login-password --region $(AWS_REGION) \
  | sudo docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
echo "==> Pull latest image"
sudo docker pull $(IMAGE)
echo "==> Stop + remove existing container (if any)"
sudo docker stop $(CONTAINER_NAME) 2>/dev/null || true
sudo docker rm $(CONTAINER_NAME) 2>/dev/null || true
echo "==> Start new container"
sudo docker run -d --name $(CONTAINER_NAME) \
  --init --restart unless-stopped \
  -p 8000:8000 -p 8501:8501 \
  --env-file /opt/berlin-aqi/.env \
  -v /opt/berlin-aqi/data:/app/data \
  $(IMAGE)
echo "==> Bootstrap prediction cache (takes a few minutes)"
sudo docker exec $(CONTAINER_NAME) python -m src.refresh
echo "==> Health check"
curl -sS http://localhost:8000/health
echo
endef
export EC2_DEPLOY_SCRIPT

ec2-deploy:
	echo "$$EC2_DEPLOY_SCRIPT" | ssh -i $(SSH_KEY) $(EC2_USER)@$(EC2_HOST) bash -s
