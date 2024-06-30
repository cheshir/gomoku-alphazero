.PHONY: test
test:
	pytest -v tests

.PHONY: serve
serve:
	python3.9 src/web_server.py
