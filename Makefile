.PHONY: test
test:
	pytest -v tests

.PHONY: serve
serve:
	python3 src/web_server.py

mdify:
	python3 ~/Documents/soft/scripts/files_bundler.py src templates tests Makefile requirements.txt -o context.md
