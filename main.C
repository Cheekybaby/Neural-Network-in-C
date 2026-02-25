/*
Headers:
    stdio.h: for input/output functions like printf and scanf.
    stdlib.h: for general utility functions like malloc, free, and exit.
    math.h: for mathematical functions like sqrt, pow, and trigonometric functions.
    time.h: for time (used for random seeding)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_NODES 784 // (28 * 28 pixels)
#define HIDDEN_NODES 256
#define OUTPUT_NODES 10 // For 0 - 9

#define NO_OF_TRAINING_IMGS 60000
#define NO_OF_TESTING_IMGS 10000

#define NO_OF_EPOCHS 10

double training_imgs[NO_OF_TRAINING_IMGS][INPUT_NODES];
double training_labels[NO_OF_TRAINING_IMGS][OUTPUT_NODES];
double testing_imgs[NO_OF_TESTING_IMGS][INPUT_NODES];
double testing_labels[NO_OF_TESTING_IMGS][OUTPUT_NODES];

double weight1[INPUT_NODES][HIDDEN_NODES];
double weight2[HIDDEN_NODES][OUTPUT_NODES];
double bias1[HIDDEN_NODES];
double bias2[OUTPUT_NODES];

int correct_predictions;
int forward_prob_op;

double random_weight(int fan_in, int fan_out)
{
    double limit = sqrt(6.0 / (fan_in + fan_out));
    double r = (double)rand() / RAND_MAX;

    return (2.0 * r * limit) - limit;
}
void initialize_parameters()
{
    // seed RNG once
    srand((unsigned int)time(NULL));

    // initialize weights
    for (int i = 0; i < INPUT_NODES; i++)
    {
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            weight1[i][j] = random_weight(INPUT_NODES, HIDDEN_NODES);
        }
    }
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            weight2[i][j] = random_weight(HIDDEN_NODES, OUTPUT_NODES);
        }
    }

    // initialize biases
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        bias1[i] = random_weight(1, HIDDEN_NODES);
    }
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        bias2[i] = random_weight(1, OUTPUT_NODES);
    }
}

void loadDataFromBin()
{
    // open all the files
    FILE *training_imgs_file = fopen("./train/mnist_train_images.bin", "rb");
    if (training_imgs_file == NULL)
    {
        printf("Error Opening training images file\n");
        exit(1);
    }
    fseek(training_imgs_file, 16, SEEK_SET);

    FILE *training_labels_file = fopen("./train/mnist_train_labels.bin", "rb");
    if (training_labels_file == NULL)
    {
        printf("Error Opening training labels file\n");
        exit(1);
    }
    fseek(training_labels_file, 8, SEEK_SET);

    FILE *testing_imgs_file = fopen("./test/mnist_test_images.bin", "rb");
    if (testing_imgs_file == NULL)
    {
        printf("Error Opening testing images file\n");
        exit(1);
    }
    fseek(testing_imgs_file, 16, SEEK_SET);

    FILE *testing_labels_file = fopen("./test/mnist_test_labels.bin", "rb");
    if (testing_labels_file == NULL)
    {
        printf("Error Opening testing labels file\n");
        exit(1);
    }
    fseek(testing_labels_file, 8, SEEK_SET);
    
    // read the training images
    unsigned char buffer[INPUT_NODES];

    for (int i = 0; i < NO_OF_TRAINING_IMGS; i++)
    {
        if ((fread(buffer, sizeof(unsigned char), INPUT_NODES, training_imgs_file)) != INPUT_NODES)
        {
            printf("Error reading from training images");
            exit(1);
        }

        for (int j = 0; j < INPUT_NODES; j++)
        {
            training_imgs[i][j] = (double)buffer[j] / 255.0;
        }
    }
    // read the training labels
    for (int i = 0; i < NO_OF_TRAINING_IMGS; i++)
    {
        unsigned char label;
        if ((fread(&label, sizeof(unsigned char), 1, training_labels_file)) != 1)
        {
            printf("Error reading training labels");
            exit(1);
        }
        if (label >= OUTPUT_NODES)
        {
            printf("Invalid training label\n");
            exit(1);
        }
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            training_labels[i][j] = (j == label) ? 1 : 0;
        }
    }
    // read the test images
    for (int i = 0; i < NO_OF_TESTING_IMGS; i++)
    {
        if ((fread(buffer, sizeof(unsigned char), INPUT_NODES, testing_imgs_file)) != INPUT_NODES)
        {
            printf("Error in reading testing images file");
            exit(1);
        }
        for (int j = 0; j < INPUT_NODES; j++)
        {
            testing_imgs[i][j] = (double)buffer[j] / 255.0;
        }
    }
    // read the test labels
    for (int i = 0; i < NO_OF_TESTING_IMGS; i++)
    {
        unsigned char label;
        if ((fread(&label, sizeof(unsigned char), 1, testing_labels_file)) != 1)
        {
            printf("Error in readding testing labels file");
            exit(1);
        }
        if (label >= OUTPUT_NODES)
        {
            printf("Invalid training label\n");
            exit(1);
        }
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            testing_labels[i][j] = (j == label) ? 1 : 0;
        }
    }

    // close all the files
    fclose(training_imgs_file);
    fclose(training_labels_file);
    fclose(testing_imgs_file);
    fclose(testing_labels_file);
}

int max_index(double arr[], int size)
{
    int max_i = 0;
    for (int i = 1; i < size; i++)
    {
        if (arr[i] > arr[max_i])
        {
            max_i = i;
        }
    }
    return max_i;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

void train(double input[INPUT_NODES], double output[OUTPUT_NODES], int correct_label)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];

    // feedforward
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        double sum = bias1[i];
        for (int j = 0; j < INPUT_NODES; j++)
        {
            sum += input[j] * weight1[j][i];
        }
        hidden[i] = sigmoid(sum);
    }
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        double sum = bias2[i];
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            sum += hidden[j] * weight2[j][i];
        }
        output_layer[i] = sigmoid(sum);
    }

    int index = max_index(output_layer, OUTPUT_NODES);

    if (index == correct_label)
    {
        forward_prob_op++;
    }

    // backpropagation
    double delta_op[OUTPUT_NODES];
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        double error = output[i] - output_layer[i];

        delta_op[i] = error * output_layer[i] * (1 - output_layer[i]);
    }
    double delta_hidden[HIDDEN_NODES];
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        double sum = 0;
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            sum += (delta_op[j] * weight2[i][j]);
        }
        delta_hidden[i] = sum * hidden[i] * (1 - hidden[i]);
    }

    // update weights and biases
    double learning_rate = 0.1;
    // update weight 2
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            weight2[i][j] += (learning_rate * delta_op[j] * hidden[i]);
        }
    }
    // update bias2
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        bias2[i] += (learning_rate * delta_op[i]);
    }
    // update weight 1
    for (int i = 0; i < INPUT_NODES; i++)
    {
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            weight1[i][j] += (learning_rate * delta_hidden[j] * input[i]);
        }
    }
    // update bias1
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        bias1[i] += (learning_rate * delta_hidden[i]);
    }
}

void runEpochs()
{
    for (int epoch = 0; epoch < NO_OF_EPOCHS; epoch++)
    {
        forward_prob_op = 0;

        for (int i = 0; i < NO_OF_TRAINING_IMGS; i++)
        {
            int correct_label = max_index(training_labels[i], OUTPUT_NODES);

            train(
                training_imgs[i],
                training_labels[i],
                correct_label);
        }

        printf("Epoch %d Training Accuracy: %.4f\n", epoch + 1, (double)forward_prob_op / NO_OF_TRAINING_IMGS);
    }
}

void save_weight_biases(char *file_name)
{
    FILE *file = fopen(file_name, "wb");
    if (file == NULL)
    {
        printf("Error opening file\n");
        exit(1);
    }
    fwrite(weight1, sizeof(double), HIDDEN_NODES * INPUT_NODES, file);
    fwrite(weight2, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
    fwrite(bias1, sizeof(double), HIDDEN_NODES, file);
    fwrite(bias2, sizeof(double), OUTPUT_NODES, file);

    fclose(file);
}

void test(const double input[INPUT_NODES], int correct_label)
{
    double hidden[HIDDEN_NODES];
    double output_layers[OUTPUT_NODES];

    // feedforward
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        double sum = bias1[i];
        for (int j = 0; j < INPUT_NODES; j++)
        {
            sum += (input[j] * weight1[j][i]);
        }
        hidden[i] = sigmoid(sum);
    }

    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        double sum = bias2[i];
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            sum += (hidden[j] * weight2[j][i]);
        }
        output_layers[i] = sigmoid(sum);
    }

    int index = max_index(output_layers, OUTPUT_NODES);
    if (index == correct_label)
    {
        correct_predictions++;
    }
}

int main()
{
    // Randomly initialize weights (weight1, weight2)
    // Randomly initialize biases (bias1, bias2)
    initialize_parameters();

    // Load MNIST dataset from binary files
    //  - Normalize images (pixel / 255.0)
    //  - Convert labels -> One-hot encoding
    loadDataFromBin();

    // Run for the number of epochs
    runEpochs();

    // save weight biases
    save_weight_biases("model.bin");

    // test the network
    correct_predictions = 0;
    for (int i = 0; i < NO_OF_TESTING_IMGS; i++)
    {
        int correct_label = max_index(testing_labels[i], OUTPUT_NODES);
        test(testing_imgs[i], correct_label);
    }

    printf("Testing accuracy: %f\n", (double)correct_predictions / NO_OF_TESTING_IMGS);

    return 0;
}
