# Neural Network in C - MNIST Digit Classifier

## Overview
This project implements a fully connected neural network from scratch in pure C (C11). The model is trained and evaluated on the MNIST handwritten digit dataset using manual forward propagation, backpropagation, and stochastic gradient descent.
Architecture:
- Input layer: 784 neurons (28 X 28 grayscale image)
- Hidden layer: 256 neurons (Sigmoid activation)
- Output layer: 10 neurons (Sigmoid activation, one per digit 0 - 9)

Training uses Xavier (Glorot) initialization and Mean Squared Error (MSE) loss.

---

## Build (On Windows)
Run: gcc -std=c11 -Wall -Wextra -O2 main.c -o nn.exe -lm

It will create nn.exe and then simply run this .exe file.
If you want to name it something else then replace "nn" with desired name.

---

## Complete program flow
```
Program Start
    ↓
Initialize Parameters
    → Seed random number generator (time-based seed)
    → Randomly initialize weight1 (784×256)
    → Randomly initialize weight2 (256×10)
    → Randomly initialize bias1 (256)
    → Randomly initialize bias2 (10)
    → Initialization uses Xavier/Glorot uniform distribution:
        limit = sqrt(6 / (fan_in + fan_out))
        weights ∈ [-limit, +limit]
    ↓
Load MNIST Dataset From Binary Files
    → Open training and testing IDX files
    → Skip IDX headers
        - Images: 16 bytes
        - Labels: 8 bytes
    → Read raw image bytes (unsigned char)
    → Normalize each pixel:
        pixel = pixel / 255.0
    → Read label bytes
    → Convert each label to one-hot encoding (size 10 vector)
    ↓
FOR epoch = 1 → NUMBER_OF_EPOCHS
    → Reset training accuracy counter

    FOR each Training Image (60000 samples)

        → Determine correct label
            (max_index from one-hot vector)

        Forward Pass:
            Input Layer (784 values)
                ↓
            Hidden Layer Computation:
                z = weight1 · input + bias1
                a = sigmoid(z)
                (256 activations)
                ↓
            Output Layer Computation:
                z = weight2 · hidden + bias2
                a = sigmoid(z)
                (10 output probabilities)

        Prediction:
            → Select index of maximum output activation

        Accuracy Update:
            → If prediction matches correct label,
              increment training accuracy counter

        Backpropagation:

            Output Layer:
                error = target - output
                delta_output = error * output * (1 - output)

            Hidden Layer:
                propagated_error = weight2ᵀ · delta_output
                delta_hidden = propagated_error * hidden * (1 - hidden)

        Parameter Update (Stochastic Gradient Descent):

            Update weight2:
                weight2 += learning_rate * (delta_output × hidden)

            Update bias2:
                bias2 += learning_rate * delta_output

            Update weight1:
                weight1 += learning_rate * (delta_hidden × input)

            Update bias1:
                bias1 += learning_rate * delta_hidden

    END Training Images Loop

    → Compute and print training accuracy for epoch

END Epoch Loop
    ↓
Save Model Parameters To "model.bin"
    → Write weight1 (784×256 doubles)
    → Write weight2 (256×10 doubles)
    → Write bias1 (256 doubles)
    → Write bias2 (10 doubles)
    ↓
Reset Test Accuracy Counter
    ↓
FOR each Test Image (10000 samples)

    → Determine correct label

    Forward Pass Only:
        Input → Hidden (sigmoid)
        Hidden → Output (sigmoid)

    → Predict digit (max output activation)

    → Compare prediction with true label
        If correct → increment test accuracy counter

END Test Loop
    ↓
Print Final Testing Accuracy
    ↓
Program Exit
```

(The above flow is made using ChatGPT, because I suck at explaining things. Also this readme is curated using ChatGPT in order for better communication)

---

## Training configuration
- Loss function: Mean Squared Error (MSE)
- Activation function: Sigmoid (hidden and output layer)
- Optimizer: Stochastic Gradient Descent (sample-wise updates)
- Learning rate: 0.1
- Epochs: 10

---

## Observed performance
Results:
- Epoch 1 training accuracy: 0.8969 
- Epoch 2 training accuracy: 0.9471 
- Epoch 3 training accuracy: 0.9625 
- Epoch 4 training accuracy: 0.9708 
- Epoch 5 training accuracy: 0.9760 
- Epoch 6 training accuracy: 0.9798 
- Epoch 7 training accuracy: 0.9825 
- Epoch 8 training accuracy: 0.9848 
- Epoch 9 training accuracy: 0.9864 
- Epoch 10 training accuracy: 0.9878 

- Testing accuracy: 0.978000

(I have tested this only for once, atleast for now. If you see any problems please let me know)

---

## Suggested improvements

- Replace output with softmax + cross-entropy
- Switch hidden activation to ReLU
- Implement mini-batch training
- Add learning rate scheduling
- Implement Momentum and Adam optimizer
- Improve cache locality for matrix operations
- Add model loading capability

P.S. These are the suggestions made my ChatGPT as it mocked my implementation. So, at the time of writing this I can't confirm some of these as possible improvements. I'll check as I improve myself.

---

## Mentions
- @tsoding for motivating me for developing whatever I liked
- https://github.com/mounirouadi/Deep-Neural-Network-in-C for the idea and data for my implementation (I studied the code and learnt a lot).
- AI coding agents (ChatGPT and Gemini) and Google for answering my dumbass questions.
