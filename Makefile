fmt:
	black *.py beat_tracking/*.py

black: fmt

.PHONY: fmt black
