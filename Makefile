.DEFAULT_GOAL := all

ifeq ($(PYTHON),)
	ifeq ($(shell which python3),)
		PYTHON = python
	else
		PYTHON = python3
	endif
endif

.PHONY: install
install:
	$(PYTHON) setup.py install

.PHONY: dev
dev:
	$(PYTHON) setup.py develop

.PHONY: all
all: install clean

# test all
.PHONY: test_all
test_all: 
	coverage run --source=. -m unittest discover -v

# test a single file
.PHONY: test_file
test_file:
	coverage run --source=. -m unittest -v tests/$(FILE)

# show coverage report
.PHONY: test_coverage
test_coverage: 
	coverage report -m

.PHONY: clean
clean:
	rm -rf *.pyc
	rm -rf __pycache__/
	rm -rf .DS_Store
	rm -rf deepsnap.egg-info
	rm -rf dist
	rm -rf build
	rm -rf cora
	rm -rf enzymes
	rm -rf planetoid
