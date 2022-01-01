raybox.dylib: raybox.c
	cc -Wall -Werror -Wpedantic -shared -O3 -o $@ $<
