# LLaMA2 C Implementation

A C implementation of the LLaMA2 model with training capabilities, inspired by [Karpathy's llama2.c](https://github.com/karpathy/llama2.c).

## Files

### train_llama.c
- Custom implementation of LLaMA2 training in C
- Implements forward pass, backward propagation, and Adam weight updates
- Based on [llama2.c/run.c](https://github.com/karpathy/llama2.c/blob/master/run.c)
- Emphasis is on understanding rather than optimization


### test_train_llama.py
- Gradient computation test
- Steps to run test
  - `make train` : Build the train_llama binary
  - `make run`
    - Loads the model from a checkpoint 'stories15M.bin' 
    - Performs a single forward, backward and weight update pass
    - Saves the activations, gradients in 'state.bin'
  - `make test`
    - Load the activations and gradients in 'state.bin'
    - Load llama2 model (in in model.py) from the 'stories15M.pt' checkpoint
    - Run a single forward pass
    - Test tensor equality 
- Tests individual model components:
  - Attention mechanisms
  - Feed-forward networks
  - Layer normalization
  - Weight update procedures

## Implementation Details
This project reimplements the core LLaMA2 architecture in C with a focus on:
- Training capability demonstration

## Credits
Model architecture and approach based on:
- [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy
- Original [model.py](https://github.com/karpathy/llama2.c/blob/master/model.py) implementation
- 