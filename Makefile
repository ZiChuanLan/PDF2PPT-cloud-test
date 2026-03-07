.PHONY: dev dev-local up down build logs clean test lint

# Development commands
dev: up logs

dev-local:
	bash scripts/dev/local_dev.sh

up:
	docker compose up --build -d

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

# Individual service logs
logs-web:
	docker compose logs -f web

logs-api:
	docker compose logs -f api

logs-worker:
	docker compose logs -f worker

logs-redis:
	docker compose logs -f redis

# Restart services
restart:
	docker compose restart

restart-api:
	docker compose restart api worker

restart-web:
	docker compose restart web

# Clean up
clean:
	docker compose down -v --rmi local

# Health check
health:
	curl -s http://localhost:8000/health | jq

# Redis CLI
redis-cli:
	docker compose exec redis redis-cli

# Shell access
shell-api:
	docker compose exec api /bin/bash

shell-web:
	docker compose exec web /bin/sh

# Status
ps:
	docker compose ps

# Validate compose file
validate:
	docker compose config

# Local QA
#
# Public repo keeps runtime code only. `test` stays as a placeholder so
# common workflows do not fail after test fixtures are removed.
# `lint` remains the cheap sanity check.

test:
	@echo "No repository tests are kept in this public branch."

lint: lint-api lint-web

lint-api:
	@set -eu; \
	if [ -x "api/.venv/bin/python" ]; then PYTHON="api/.venv/bin/python"; \
	elif [ -x ".venv/bin/python" ]; then PYTHON=".venv/bin/python"; \
	else PYTHON="python3"; fi; \
	"$${PYTHON}" -m compileall -q api

lint-web:
	@set -eu; \
	cd web; \
	npm run lint
