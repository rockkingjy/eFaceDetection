
VPATH=./src/
OBJDIR=./obj/
EXEC=npd.bin

CC=gcc
LDFLAGS=-lm
CFLAGS=-Wall -Wfatal-errors -Ofast

OBJ=common.o npd.o main.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) 

all: obj $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

