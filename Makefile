.PHONY: install install-dev test scrape-once analyze-once run-scraper run-analyzer migrate compose-up compose-down options-pull

install:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt

install-dev: install

test:
	python -m pytest -v

scrape-once:            ## one scrape cycle, no scheduler
	python -m scraper.yahoo_news_scraper

analyze-once:           ## one analysis cycle, no scheduler
	python -m analyzer.pipeline

run-scraper:            ## long-running scraper service
	python -m scraper.scheduler

run-analyzer:           ## long-running analyzer service
	python -m analyzer.scheduler

options-pull:           ## e.g. make options-pull TICKERS="AAPL MSFT"
	python -m analyzer.options_data $(TICKERS)

migrate:
	alembic upgrade head

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down
