//=======================================================================
// Copyright (c) 2019 Yanfei Tang
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <algorithm>
#include <iterator>
#include "mnist/mnist_reader.hpp"

using namespace std;

typedef vector<vector<double> > Matrix;
typedef vector<double> Vector;

// The sigmoid function
double sigmoid(double z){
    return 1.0/(1.0 + exp(-z));
}

Vector sigmoid(Vector z){
    Vector temp(z.size());
    for(size_t i = 0; i < z.size(); ++i){
	temp[i] = sigmoid(z[i]);
    }
    return temp;
}

// Derivative of the sigmoid function
double sigmoid_prime(double z){
    return sigmoid(z) * (1 - sigmoid(z));
}

Vector sigmoid_prime(Vector z){
    Vector temp(z.size());
    for(size_t i = 0; i < z.size(); ++i){
	temp[i] = sigmoid_prime(z[i]);
    }
    return temp;
}

// Normal distribution matrix with dimension x rows and y columns
Matrix randn(int x, int y){
    Matrix temp(x, vector<double>(y));
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);    
    for (int i = 0; i < x; i++){
	for (int j = 0; j < y; j++){
	    temp[i][j] = distribution(generator);
	}
    }
    return temp;
}

// Normal distribution vector with x elements
Vector randn(int x){
    Vector temp = vector<double>(x);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);    
    for (int i = 0; i < x; i++){
	temp[i] = distribution(generator);	
    }
    return temp;
}

// Matrix * vector multiplication
Vector dot(Matrix w, Vector a){
    if (w[0].size() != a.size()){
	exit(EXIT_FAILURE);
    }
    
    Vector out(w.size());
    for (size_t i = 0; i < w.size(); ++i){
	double sum = 0;
	for (size_t j = 0; j < w[i].size(); ++j){
	    sum += w[i][j] * a[j];
	}
	out[i] = sum;
    }

    return out;
}

// vector * vector^T multiplication
Matrix dot(Vector a, Vector b){
    Matrix out(a.size(), vector<double>(b.size()));
    for (size_t i = 0; i < a.size(); ++i){
	for (size_t j = 0; j < b.size(); ++j){
	    out[i][j] = a[i] * b[j];
	}
    }
    return out;
}

// Return a 10-dimensional unit Vector with a 1.0 in the jth
// position and zeroes elsewhere.
Vector vectorized_result(uint8_t j){
    Vector e(10, 0);
    e[int(j)] = 1.0;
    return e;
}

// Vector + Vector
Vector vec_sum(Vector a, Vector b){
    if (a.size() != b.size()){
	exit(EXIT_FAILURE);
    }

    Vector out(a.size());
    for (size_t i = 0; i < a.size(); ++i){
	out[i] = a[i] + b[i];
    }
    return out;
}

// Vector - Vector
Vector vec_minus(Vector a, Vector b){
    if (a.size() != b.size()){
	exit(EXIT_FAILURE);
    }

    Vector out(a.size());
    for (size_t i = 0; i < a.size(); ++i){
	out[i] = a[i] - b[i];
    }
    return out;
}

// Hadamard vector multiplication
Vector hadamard(Vector a, Vector b){
    if (a.size() != b.size()){
	exit(EXIT_FAILURE);
    }

    Vector out(a.size());
    for (size_t i = 0; i < a.size(); ++i){
	out[i] = a[i] * b[i];
    }
    return out;
}

// Transpose of a matrix
Matrix transpose(Matrix a){
    Matrix b(a[0].size(), vector<double>(a.size()));
    for (size_t i = 0; i < b.size(); ++i){
	for (size_t j = 0; j < b[0].size(); ++j){
	    b[i][j] = a[j][i];
	}
    }
    return b;
}

// sum of matrix
Matrix mat_sum(Matrix a, Matrix b){
    if (a.size() != b.size() or a[0].size() != b[0].size()){
	exit(EXIT_FAILURE);
    }

    Matrix out(a.size(), vector<double>(a[0].size(), 0.0));
    for (size_t i = 0; i < a.size(); ++i){
	for (size_t j = 0; j < a[0].size(); ++j){
	    out[i][j] = a[i][j] + b[i][j];
	}
    }
    return out;
}

//  Matrix - Matrix
Matrix mat_minus(Matrix a, Matrix b){
    if (a.size() != b.size() or a[0].size() != b[0].size()){
	exit(EXIT_FAILURE);
    }

    Matrix out(a.size(), vector<double>(a[0].size(), 0.0));
    for (size_t i = 0; i < a.size(); ++i){
	for (size_t j = 0; j < a[0].size(); ++j){
	    out[i][j] = a[i][j] - b[i][j];
	}
    }
    return out;
}

// scalar multiplication
Vector vec_multi(double scalar, Vector a){
    Vector out(a.size());
    for (size_t i = 0; i < a.size(); ++i){
	out[i] = scalar * a[i];
    }
    return out;
}

Matrix mat_multi(double scalar, Matrix a){
    Matrix out(a.size(), vector<double>(a[0].size()));
    for (size_t i = 0; i < a.size(); ++i){
	for (size_t j = 0; j < a[0].size(); ++j){
	    out[i][j] = scalar * a[i][j];
	}
    }
    return out;
}


void print_vec(Vector a){
    cout << endl;
    for (size_t i = 0; i < a.size(); ++i){
	cout << a[i] << " ";
    }
}

void print_mat(Matrix a){
    for (size_t i = 0; i < a.size(); ++i){
	cout << endl;
	for (size_t j = 0; j < a[0].size(); ++j){
	    cout << a[i][j] << " ";
	}
    }
}

void print_img(vector<uint8_t> a){
    for (size_t i = 0; i < 28; ++i){
	cout << "\n";
	for (size_t j = 0; j < 29; ++j){
	    if(size_t(a[28*i +j]) == 0){
		cout << "0 ";
	    }
	    else{
		cout << "* ";
	    }
	}
    }
    cout << "\n";				    
}

int max_index(vector<double> a){    
    return distance(a.begin(), max_element(a.begin(), a.end()));
}

class Network
{

public:

    int num_layers;
    vector<int> sizes;
    vector<Vector> biases;
    vector<Matrix> weights;

    
    Network(vector<int>);

    Vector feedforward(Vector);

    void update_mini_batch(Matrix, Matrix, double);

    void backprop(Vector, Vector, vector<Vector>&, vector<Matrix>&);

    Vector cost_derivative(Vector, Vector);

    void SGD(Matrix, Matrix, int, int, double, Matrix, Matrix);

    int evaluate(Matrix, Matrix);
    
};

Network::Network(vector<int> sizes_)
{
    sizes = sizes_;
    num_layers = sizes.size();

    biases.resize(num_layers - 1);
    weights.resize(num_layers - 1);

    for (size_t i = 0; i < biases.size(); ++i){
	biases[i] = randn(sizes[i+1]);
    }

    for (size_t i = 0; i < weights.size(); ++i){
	weights[i] = randn(sizes[i+1], sizes[i]);
    }
}


Vector Network::feedforward(Vector a)
{
    vector<Vector> neurons(num_layers - 1);
    
    for (int i = 0; i < num_layers - 1; ++i){
	neurons[i].resize(sizes[i+1]);
    }
    
    for(int i = 0; i < num_layers - 1; ++i){
	if(i == 0){
	    neurons[i] = sigmoid(vec_sum(dot(weights[i], a), biases[i]));
	}
	else{
	    neurons[i] = sigmoid(vec_sum(dot(weights[i], neurons[i-1]), biases[i]));
	}
    }

    return neurons[num_layers-2];
}

Vector Network::cost_derivative(Vector output_activations, Vector y){
    return vec_minus(output_activations, y);
}

void Network::backprop(Vector x, Vector y,
		       vector<Vector> & nabla_b,
		       vector<Matrix> & nabla_w){
    
    nabla_b.resize(num_layers - 1);
    nabla_w.resize(num_layers - 1);
    for (int i = 0; i < num_layers - 1; ++i){
	nabla_b[i].resize(biases[i].size());
	nabla_w[i].resize(weights[i].size(), vector<double>(weights[i][0].size()));
    }

    Vector activation = x;
    vector<Vector> activations;
    activations.push_back(activation);

    vector<Vector> zs;

    for (int i = 0; i < num_layers - 1; ++i){
	Vector z;
	z = vec_sum(dot(weights[i], activation), biases[i]);
	zs.push_back(z);
	activation = sigmoid(z);
	activations.push_back(activation);
    }

    Vector delta;
    delta = hadamard(cost_derivative(activations.back(), y), sigmoid_prime(zs.back()));
    
    nabla_b[num_layers - 2] = delta;
    nabla_w[num_layers - 2] = dot(delta, activations[num_layers - 2]);

    for (int l = num_layers - 2; l > 0; l--){
	Vector z, sp;
	z = zs[l-1];
	sp = sigmoid_prime(z);
	delta = hadamard(dot(transpose(weights[l]), delta), sp);
	nabla_b[l - 1] = delta;
	nabla_w[l - 1] = dot(delta, activations[l-1]);
    }
}


void Network::update_mini_batch(Matrix mini_batch_images, Matrix mini_batch_labels, double eta)
{
    if (mini_batch_images.size() != mini_batch_labels.size()){
	exit(EXIT_FAILURE);
    }

    double n_mini_batch = double(mini_batch_images.size());
    
    vector<Vector> nabla_b;
    vector<Matrix> nabla_w;
    nabla_b.resize(num_layers - 1);
    nabla_w.resize(num_layers - 1);
    for (int i = 0; i < num_layers - 1; ++i){
	nabla_b[i].resize(biases[i].size(), 0.0);
	nabla_w[i].resize(weights[i].size(), vector<double>(weights[i][0].size(), 0.0));
    }

    for (size_t i = 0; i < mini_batch_images.size(); ++i){
	vector<Vector> delta_nabla_b;
	vector<Matrix> delta_nabla_w;
	backprop(mini_batch_images[i], mini_batch_labels[i], delta_nabla_b, delta_nabla_w);
	for (int j = 0; j < num_layers - 1; ++j){
	    nabla_b[j] = vec_sum(nabla_b[j], delta_nabla_b[j]);
	    nabla_w[j] = mat_sum(nabla_w[j], delta_nabla_w[j]);
	}	
    }

    for (int i = 0; i < num_layers - 1; ++i){
	weights[i] = mat_minus(weights[i], mat_multi(eta/n_mini_batch, nabla_w[i]));
	biases[i]  = vec_minus(biases[i],  vec_multi(eta/n_mini_batch, nabla_b[i]));
    }
    
}


int Network::evaluate(Matrix test_images, Matrix test_labels)
{
    if (test_images.size() != test_labels.size()){
	exit(EXIT_FAILURE);
    }

    int sum, a, b;
    sum = 0;
    
    for (size_t i = 0; i < test_images.size(); ++i){
	a = max_index(feedforward(test_images[i]));
	b = max_index(test_labels[i]);
	if (a == b){
	    sum++;
	}
    }
    return sum;
}

void Network::SGD(Matrix train_images, Matrix train_labels,
		  int epochs, int mini_batch_size, double eta,
		  Matrix test_images, Matrix test_labels)
{
    int n = int(train_images.size());
    int n_test = int(test_images.size());


    
    for (int i = 0; i < epochs; ++i){
	vector<int> indexes;
	indexes.reserve(train_images.size());
	for (size_t i = 0; i < train_images.size(); ++i){
	    indexes.push_back(i);
	}
	random_shuffle(indexes.begin(), indexes.end());
	 
	
	for (int j = 0; j < n/mini_batch_size; ++j ){
	    Matrix mini_batch_images;
	    Matrix mini_batch_labels;
	    for (int k = 0; k < mini_batch_size; ++k){
		int idx = j * mini_batch_size + k;
		mini_batch_images.push_back(train_images[indexes[idx]]);
		mini_batch_labels.push_back(train_labels[indexes[idx]]);
	    }
	    
	    update_mini_batch(mini_batch_images, mini_batch_labels, eta);
	    	    
	}

	cout << "Epoch {" << i << "} : {" << evaluate(test_images, test_labels) << \
	    "} / {" << n_test << "}\n";

    }
}

    

int main(int argc, char* argv[]) {
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    int epochs = 30;
    int mini_batch_size = 10;
    double eta = 3.0;
    vector<int> sizes = {784, 30, 10};

    std::clog << "********************Neural Networks ***********************\n";
    std::clog << "**                                                       **\n";
    std::clog << "**         Copyright Yanfei Tang, 2019                   **\n";
    std::clog << "***********************************************************\n";
    std::clog << "\n";

    int ii = 0;
    while (++ii < argc){

	std::string str(argv[ii]);
	if (str == "-epochs" && ii < argc-1) epochs = atoi(argv[++ii]);
	if (str == "-mbs" && ii < argc-1) mini_batch_size = atoi(argv[++ii]);
	if (str == "-eta" && ii < argc-1) eta = atof(argv[++ii]);

	if (str == "-h" || str == "--help"){
	    std::clog << "DNN [Option]\n";
	    std::clog << "        -epochs   number of epochs\n";
	    std::clog << "        -mbs      mini batch size\n";
	    std::clog << "        -eta      learning rate\n";

	    return 0;
	}

    }


    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    // Convert uint8_t type to double
    int num_train, num_pixel, num_test;
    num_train = dataset.training_images.size();
    num_pixel = dataset.training_images[0].size();
    num_test  = dataset.test_images.size();
	
    Matrix train_images(num_train, vector<double>(num_pixel));
    Matrix  test_images(num_test,  vector<double>(num_pixel));
    Matrix train_labels(num_train, vector<double>(10));
    Matrix  test_labels(num_test,  vector<double>(10));

    for (int i = 0; i < num_train; ++i){
	for (int j = 0; j < num_pixel; ++j){
	    train_images[i][j] = double(dataset.training_images[i][j])/255.0;
	}
    }

    for (int i = 0; i < num_test; ++i){
	for (int j = 0; j < num_pixel; ++j){
	    test_images[i][j] = double(dataset.test_images[i][j])/255.0;
	}
    }

    for (int i = 0; i < num_train; ++i){
	train_labels[i] = vectorized_result(dataset.training_labels[i]);
    }

    for (int i = 0; i < num_test; ++i){
	test_labels[i] = vectorized_result(dataset.test_labels[i]);
    }

    std::cout << "Nbr of training images = " << train_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << train_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << test_labels.size() << std::endl;
    std::cout << "Nbr of pixels = " << train_images[0].size() << std::endl;
    // End of conversion
    
    Network network(sizes);

    
    network.SGD(train_images, train_labels,
		epochs, mini_batch_size, eta,
		test_images, test_labels);

    
    
    return 0;
}
