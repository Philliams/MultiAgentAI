project_name ?= multi-agent-ai

dev_name = ${project_name}-dev
backend_name = ${project_name}-backend

# BUILD COMMANDS
build_dev:
	docker build -t ${dev_name} -f ./docker/dockerfile_dev .

# RUN COMMANDS
run_script: build_dev
	docker rm --force ${backend_name}
	docker run --name ${backend_name} \
	--mount type=bind,source=./data,target=/data \
	-it ${dev_name} \
	python /src/backend/main_backend.py