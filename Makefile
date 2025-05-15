.PHONY: all clean

all:
	rst2man cuda.rst cuda.man

clean:
	rm cuda.man
