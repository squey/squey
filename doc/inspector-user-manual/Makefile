PDFLATEX=pdflatex
RM=rm -f

all: manual

manual: manual.tex normalization.tex formats.tex
	${PDFLATEX} manual.tex

clean:
	${RM} index.aux  manual.aux  manual.idx  manual.log  manual.out manual.pdf manual.toc *~


