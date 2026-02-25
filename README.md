Program Start
↓
Randomly Initialize Weights (weight1, weight2)
Randomly Initialize Biases (bias1, bias2)
↓
Load MNIST Dataset From Binary Files
   → Normalize images (pixel / 255.0)
   → Convert labels → One Hot Encoding
↓
FOR epoch = 0 → NUMBER_OF_EPOCHS

    Reset Training Accuracy Counter (local variable — shadowed bug)

    FOR each Training Image (60000)

        Extract Correct Label (max_index from one-hot label)

        Forward Pass:
            Input → Hidden Layer (Sigmoid)
            Hidden → Output Layer (Sigmoid)

        Prediction:
            Choose Highest Probability Output

        Update Global Forward Accuracy Counter
            (does NOT match printed accuracy due to shadowing)

        Backpropagation:
            Compute Output Error
            Compute Hidden Layer Error
            Compute Input Layer Delta (extra propagation)

        Update Parameters (SGD):
            Update weight1
            Update weight2
            Update bias1
            Update bias2

    END Training Images Loop

    Print Training Accuracy
        (always incorrect due to variable shadowing)

END Epoch Loop
↓
Save Model Parameters To "model.bin"
    → weight1
    → weight2
    → bias1
    → bias2
↓
Reset Test Accuracy Counter (local shadowed variable)
↓
FOR each Test Image (10000)

    Extract Correct Label

    Forward Pass Only:
        Input → Hidden (Sigmoid)
        Hidden → Output (Sigmoid)

    Predict Digit (max probability)

    Print Prediction To Console

    Compare With True Label
        Increment Global Correct Prediction Counter

END Test Loop
↓
Print Testing Accuracy
    (incorrect due to shadowing)
↓
Program Exit