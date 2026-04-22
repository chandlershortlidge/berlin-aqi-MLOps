AWS_ACCOUNT_ID = 458960552929
AWS_REGION = eu-central-1

login:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

build:
	docker build --platform linux/amd64 -t berlin-aqi-mlops .

tag:
	docker tag berlin-aqi-mlops:latest $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/berlin-aqi-mlops:latest

push:
	docker push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/berlin-aqi-mlops:latest

deploy: login build tag push

ssh:
	ssh -i ~/.ssh/berlin-aqi-key.pem ec2-user@3.71.44.98