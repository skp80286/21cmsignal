CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lfftw3 -lm

ps_gk: ps_gk.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ps_gk.o: ps_gk.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f ps_gk *.o
