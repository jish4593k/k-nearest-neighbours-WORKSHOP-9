#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <random>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

// Function to generate synthetic data with 10 additional features
void generateSyntheticData(Mat& X, Mat& Y) {
    int n_samples = 400;
    Mat original_features(n_samples, 2, CV_32FC1);
    randn(original_features, Scalar(0), Scalar(1));

    Mat additional_features(n_samples, 10, CV_32FC1);
    randn(additional_features, Scalar(0), Scalar(1));

    hconcat(original_features, additional_features, X);

    Y.create(n_samples, 1, CV_32SC1);

    for (int i = 0; i < n_samples; i++) {
        Mat sample = X.row(i);
        float score = sum(sample.colRange(0, 3))[0];
        Y.at<int>(i, 0) = (score > 0) ? 1 : 0;
    }
}

int main() {
    // Generate synthetic data
    Mat X, Y;
    generateSyntheticData(X, Y);

    // Split the dataset into training and test sets
    int n_samples = X.rows;
    int n_train = static_cast<int>(n_samples * 0.75);
    int n_test = n_samples - n_train;

    Mat X_train = X.rowRange(0, n_train);
    Mat Y_train = Y.rowRange(0, n_train);

    Mat X_test = X.rowRange(n_train, n_samples);
    Mat Y_test = Y.rowRange(n_train, n_samples);

    // Feature scaling
    Ptr<ml::TrainData> trainData = ml::TrainData::create(X_train, ml::ROW_SAMPLE, Y_train);
    trainData->setTrainTestSplitRatio(1.0);
    Ptr<ml::StandardScaler> scaler = ml::createStandardScaler();
    scaler->setInput(trainData);
    Mat scaled_X_train = Mat(n_train, X_train.cols, X_train.type());
    scaler->transform(X_train, scaled_X_train);

    // Create and train the K-Nearest Neighbors classifier
    Ptr<ml::KNearest> knn = ml::KNearest::create();
    knn->setIsClassifier(true);
    knn->setDefaultK(5);
    knn->train(scaled_X_train, ml::ROW_SAMPLE, Y_train);

    // Feature scale the test set
    Mat scaled_X_test = Mat(n_test, X_test.cols, X_test.type());
    scaler->transform(X_test, scaled_X_test);

    // Predict the test set results
    Mat Y_pred;
    knn->findNearest(scaled_X_test, knn->getDefaultK(), Y_pred);

    // Evaluate the model
    Mat conf_matrix;
    int total_correct = 0;

    for (int i = 0; i < n_test; i++) {
        if (Y_pred.at<float>(i, 0) == Y_test.at<int>(i, 0)) {
            total_correct++;
        }
    }

    double accuracy = static_cast<double>(total_correct) / n_test;
    cout << "Accuracy: " << accuracy * 100 << "%" << endl;

    return 0;
}
