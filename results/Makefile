MAIN = main
all: $(MAIN).pdf
$(MAIN).pdf: $(MAIN).tex figures/*.pdf
	pdflatex -interaction=nonstopmode $(MAIN).tex >/dev/null
	pdflatex -interaction=nonstopmode $(MAIN).tex >/dev/null
	@echo "errors: $$(grep -cE '^! ' $(MAIN).log || true)"
	@echo "undefined refs: $$(grep -c 'Warning: Reference' $(MAIN).log || true)"
	@echo "TODOs remaining: $$(grep -c '\\TODO' $(MAIN).tex || true)"
final: $(MAIN).tex
	@sed 's|\\newcommand{\\TODO}\[1\]{.*}|\\newcommand{\\TODO}[1]{}|; s|\\newcommand{\\DONE}\[1\]{.*}|\\newcommand{\\DONE}[1]{}|' $(MAIN).tex > $(MAIN)_final.tex
	pdflatex -interaction=nonstopmode $(MAIN)_final.tex >/dev/null
	pdflatex -interaction=nonstopmode $(MAIN)_final.tex >/dev/null
	@echo "wrote $(MAIN)_final.pdf (TODO/DONE annotations stripped)"
clean:
	rm -f *.aux *.log *.out *.toc $(MAIN)_final.*
.PHONY: all final clean
