fmt: black

black:
	black headbang.py headbang/*.py mir_beat_eval.py dbn_reference_beats.py

ghpages_dev:
	cd docs/ && bundle exec jekyll serve

.PHONY: fmt black
