SPHINXBUILD   = sphinx-build
SPHINXPROJ    = deepsnap
SOURCEDIR     = source
BUILDDIR      = build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)"

.PHONY: clean
clean:
	rm -rf build
	rm -rf source/deepsnap.rst
	rm -rf source/modules.rst
