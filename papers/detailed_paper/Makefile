#
# $Header: /numrelcvs/NumrelGWDA/docs/TemplBank/Makefile,v 1.4 2007/09/23 10:57:22 ajith Exp $
#

DOCS = TestGR.pdf

BYPROD = TestGR.ps TestGR.blg

TEXS = TestGR.tex 

FIGS = 

default: ${DOCS}

#TestGR.pdf: TestGR.ps
#	ps2pdf TestGR.ps TestGR.pdf

#TestGR.ps: TestGR.dvi ${FIGS}
#	dvips -t a4 -P pdf -o TestGR.ps TestGR.dvi

TestGR.pdf: TestGR.tex ${FIGS} ${TEXS}
#	latex TestGR && latex TestGR && latex TestGR && bibtex TestGR
	pdflatex TestGR  && bibtex TestGR && bibtex TestGR && pdflatex TestGR && pdflatex TestGR

TestGR.ps: TestGR.pdf ${FIGS}
	pdftops TestGR.pdf

%.pdf: %.eps
	epstopdf --outfile=$@ ${@:.pdf=.eps}

clean:
	rm -f *.log *.out *.dvi *.aux *~ ${DOCS} ${BYPROD}
