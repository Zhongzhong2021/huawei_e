.PHONY: all sync check clean
all: sync
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
	cp build/main.pdf paper.pdf
	python3 scripts/check_pdf.py
sync:
	python3 scripts/sync_chapters.py
check:
	python3 scripts/check_pdf.py
clean:
	latexmk -C -outdir=build main.tex
