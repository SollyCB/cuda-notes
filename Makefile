.PHONY: all clean

all:
	rst2man cuda.rst cuda.man
	rst2html5 cuda.rst cuda.html

clean:
	rm cuda.man
