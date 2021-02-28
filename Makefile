fmt: black

black:
	black headbang/*.py tests/*.py

ghpages_dev:
	cd docs/ && bundle exec jekyll serve

lint:
	pyflakes *.py headbang/*.py | grep -v '__init__.py'

.PHONY: fmt black
