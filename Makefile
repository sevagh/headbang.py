fmt: black

black:
	black headbang/*.py bin/*.py tests/*.py

ghpages_dev:
	cd docs/ && bundle exec jekyll serve

.PHONY: fmt black
