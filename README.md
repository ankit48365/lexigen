
# LexiGen
## understand how to create a basic text generator from scratch

1. Tokenization ~ First, tokenize the input text. first split the text into individual words (or tokens) and create a vocabulary of unique words.

```
def tokenize(text):
    tokens = text.lower().split()
    return tokens

def build_vocab(tokens):
    vocab = list(set(tokens))
    return vocab

text = "Hello, this is a simple text generator example."
tokens = tokenize(text)
vocab = build_vocab(tokens)
print("Tokens:", tokens)
print("Vocabulary:", vocab)
```


2. Encoding ~ Encode the tokens into numerical vectors. This involves creating one-hot vectors for each word in the vocabulary.

```
import numpy as np

def one_hot_encode(tokens, vocab):
    vocab_size = len(vocab)
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    encoded_tokens = np.zeros((len(tokens), vocab_size))
    for i, token in enumerate(tokens):
        encoded_tokens[i, token_to_idx[token]] = 1
    return encoded_tokens

encoded_tokens = one_hot_encode(tokens, vocab)
print("Encoded Tokens:\n", encoded_tokens)
```

3. Training Model ~ Train a simple model to predict the next word given the current word. Using a single-layer neural network for this purpose.

Forward Propagation ~ By Multiplying the input vectors by the weight matrix and applying the activation function (softmax).

```
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def forward_propagation(encoded_tokens, weights):
    z = np.dot(encoded_tokens, weights)
    predictions = softmax(z.T).T
    return predictions
```

Loss Function ~  Use cross-entropy loss to measure the error between the predicted and actual outputs.

```
def cross_entropy_loss(predictions, targets):
    return -np.sum(targets * np.log(predictions))
```

Backward Propagation ~ By calculating the gradient of the loss with respect to the weights and updating the weights accordingly.

```
def backward_propagation(encoded_tokens, predictions, targets, learning_rate):
    dL_dz = predictions - targets
    dL_dw = np.dot(encoded_tokens.T, dL_dz)
    return dL_dw

def update_weights(weights, dL_dw, learning_rate):
    return weights - learning_rate * dL_dw

```

4. Training Loop ~ Combine the above functions into a training loop to train the model on the input text.


```
def train_model(tokens, vocab, learning_rate=0.1, epochs=1000):
    vocab_size = len(vocab)
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    encoded_tokens = one_hot_encode(tokens, vocab)
    weights = np.random.randn(vocab_size, vocab_size)

    for epoch in range(epochs):
        predictions = forward_propagation(encoded_tokens, weights)
        targets = np.roll(encoded_tokens, -1, axis=0)  # Shift tokens to create targets
        loss = cross_entropy_loss(predictions, targets)
        dL_dw = backward_propagation(encoded_tokens, predictions, targets, learning_rate)
        weights = update_weights(weights, dL_dw, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights

weights = train_model(tokens, vocab)
```

5. Generating Text ! Finally, use the trained model to generate text by predicting the next word given the current word.

```
def generate_text(weights, vocab, start_token, length=10):
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    current_token = start_token
    generated_text = [current_token]

    for _ in range(length - 1):
        current_idx = token_to_idx[current_token]
        predictions = forward_propagation(one_hot_encode([current_token], vocab), weights)
        next_idx = np.argmax(predictions[current_idx])
        current_token = idx_to_token[next_idx]
        generated_text.append(current_token)

    return " ".join(generated_text)

start_token = "hello"
generated_text = generate_text(weights, vocab, start_token)
print("Generated Text:", generated_text)
```


Mathematics Involved

    1. Matrices and Vectors:
        ○ We use matrices to represent the weights of the model and vectors to represent the one-hot encoded tokens.
        
    2. Matrix Multiplication:
        ○ In forward propagation, we multiply the input vectors by the weight matrix to compute the predictions.
        
    3. Softmax Function:
        ○ We apply the softmax function to convert the raw scores (logits) into probabilities.
        
    4. Cross-Entropy Loss:
        ○ We calculate the cross-entropy loss to measure the difference between the predicted and actual outputs.
        
    5. Gradient Descent:
        ○ We use gradient descent to update the weights by calculating the gradient of the loss with respect to the weights.
        
I hope this helps you ! Feel free to ask if you have any questions or need further explanations.

