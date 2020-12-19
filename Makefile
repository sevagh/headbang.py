fmt: black test

mypy:
	mypy headbang.py

black:
	black headbang.py test_headbang.py

test:
	python test_headbang.py

.PHONY: fmt black mypy
