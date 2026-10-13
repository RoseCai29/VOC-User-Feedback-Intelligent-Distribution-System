.PHONY: install dev test server docker-build docker-up clean

install:
	pip install -r requirements.txt

dev:
	python -m src.main

server:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

test:
	python -m pytest tests/ -v

docker-build:
	docker build -t voc-agent .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	powershell -Command "Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force"
	cmd /c "if exist data rmdir /s /q data"
	cmd /c "if exist logs rmdir /s /q logs"