# LeChiffre — aliases for existing Docker Compose commands.
# No initialization, host directory creation, Git changes or implicit sudo.

.DEFAULT_GOAL := help

COMPOSE ?= docker compose
HANDS ?= 100
PLAYERS ?= 2
ARGS ?=
RUN_OPTIONS ?=

.PHONY: help build up down app arena poker behavioural test smoke ci

help:
	@printf '%s\n' \
	  'LeChiffre — Docker Compose' \
	  '  make build                         Build the application and poker engine' \
	  '  make up                            Run the default one-shot Compose jobs' \
	  '  make down                          Stop the Compose project' \
	  '  make app                           Run LeChiffre random simulation' \
	  '  make arena HANDS=100 PLAYERS=2     Compare existing Arena agents' \
	  '  make poker ARGS="holdem rank AcKdQhJsTs"  Run an rsp CLI command' \
	  '  make behavioural ARGS="..." RUN_OPTIONS="..."  Run the existing behavioural CLI' \
	  '  make test                          Build toolchain and execute Rust tests' \
	  '  make smoke                         Build and smoke-test both CLIs' \
	  '  make ci                            Run tests and CLI smoke tests' \
	  '' \
	  'If Docker requires sudo: make arena COMPOSE="sudo docker compose"' \
	  'No target runs sudo implicitly or prepares files/directories on the host.'

build:
	$(COMPOSE) build app poker

up:
	$(COMPOSE) up --abort-on-container-exit

down:
	$(COMPOSE) down

app:
	$(COMPOSE) run --rm -T app

arena:
	$(COMPOSE) run --rm poker arena compare /workspace/agents -n $(HANDS) -p $(PLAYERS)

poker:
	$(COMPOSE) run --rm poker $(ARGS)

behavioural:
	@test -n '$(strip $(ARGS))' || { echo 'Specify ARGS with the existing behavioural CLI command.' >&2; exit 2; }
	$(COMPOSE) run --rm -T $(RUN_OPTIONS) behavioural $(ARGS)

test:
	$(COMPOSE) --profile ci build test
	$(COMPOSE) --profile ci run --rm -T test

smoke: build
	$(COMPOSE) run --rm -T app --simulate-random
	$(COMPOSE) run --rm -T poker holdem rank AcKdQhJsTs

ci: test smoke
