CC ?= clang #gcc
CFLAGS = -O3 -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
LDFLAGS =
LDLIBS = -lm
INCLUDES =
CFLAGS_COND = -march=native

OUTPUT_FILE = -o $@


train: train_llama.c
	$(CC)  $(CFLAGS)  -march=native $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) -g3 -o train_llama 

run: train
	./train_llama save    # runs a forward, backward and update weights pass

test:
	python test_train_llama.py

clean:
	rm -f train_llama
	rm -f state.bin