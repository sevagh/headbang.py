fmt: black

black:
	black headbang.py headbang/*.py mir_beat_eval.py reference_beats.py annotate_beats.py

ghpages_dev:
	cd docs/ && bundle exec jekyll serve

.PHONY: fmt black
