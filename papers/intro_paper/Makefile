#
# $Header: /numrelcvs/NumrelGWDA/docs/TemplBank/Makefile,v 1.4 2007/09/23 10:57:22 ajith Exp $
#

DOCS = TGR_HM.pdf

BYPROD = TGR_HM.ps TGR_HM.blg

TEXS = TGR_HM.tex 

FIGS = 

default: ${DOCS}

#TGR_HM.pdf: TGR_HM.ps
#	ps2pdf TGR_HM.ps TGR_HM.pdf

#TGR_HM.ps: TGR_HM.dvi ${FIGS}
#	dvips -t a4 -P pdf -o TGR_HM.ps TGR_HM.dvi

TGR_HM.pdf: TGR_HM.tex ${FIGS} ${TEXS}
#	latex TGR_HM && latex TGR_HM && latex TGR_HM && bibtex TGR_HM
	pdflatex TGR_HM  && bibtex TGR_HM && bibtex TGR_HM && pdflatex TGR_HM && pdflatex TGR_HM

TGR_HM.ps: TGR_HM.pdf ${FIGS}
	pdftops TGR_HM.pdf

%.pdf: %.eps
	epstopdf --outfile=$@ ${@:.pdf=.eps}

clean:
	rm -f *.log *.out *.dvi *.aux *~ ${DOCS} ${BYPROD}
