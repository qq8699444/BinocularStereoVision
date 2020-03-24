#include <cstdio>  
#include <vector>  
#include <opencv2/core/core.hpp>  
 
using namespace std;
using namespace cv;
 
const double DERIV_STEP = 1e-5;
const int MAX_ITER = 100;
 
void GaussNewton(double(*Func)(const Mat &input, const Mat params), 
	const Mat &inputs, const Mat &outputs, Mat params);
 
double Deriv(double(*Func)(const Mat &input, const Mat params),  
	const Mat &input, const Mat params, int n);
 
double Func(const Mat &input, const Mat params);
 
int main()
{
	// For this demo we're going to try and fit to the function  
	// F = A*sin(Bx) + C*cos(Dx)，There are 4 parameters: A, B, C, D  
	int num_params = 4;
	// Generate random data using these parameters  
	int total_data = 100;
 
	double A = 5;
	double B = 1;
	double C = 10;
	double D = 2;
 
	Mat inputs(total_data, 1, CV_64F);
	Mat outputs(total_data, 1, CV_64F);
 
	for (int i = 0; i < total_data; i++) {
		double x = -10.0 + 20.0* rand() / (1.0 + RAND_MAX); // random between [-10 and 10]  
		double y = A*sin(B*x) + C*cos(D*x);
		// Add some noise  
		// y += -1.0 + 2.0*rand() / (1.0 + RAND_MAX);  
		inputs.at<double>(i, 0) = x;
		outputs.at<double>(i, 0) = y;
	}
	// Guess the parameters, it should be close to the true value, else it can fail for very sensitive functions!  
    
	Mat params(num_params, 1, CV_64F);
	params.at<double>(0, 0) = 1;
	params.at<double>(1, 0) = 1;
	params.at<double>(2, 0) = 8; // changing to 1 will cause it not to find the solution, too far away  
	params.at<double>(3, 0) = 1;
	GaussNewton(Func, inputs, outputs, params);
	printf("True parameters: %f %f %f %f\n", A, B, C, D);
	printf("Parameters from GaussNewton: %f %f %f %f\n", params.at<double>(0, 0), params.at<double>(1, 0),
		params.at<double>(2, 0), params.at<double>(3, 0));
 
	return 0;
}
 
double Func(const Mat &input, const Mat params)
{
	// Assumes input is a single row matrix  
	// Assumes params is a column matrix  
	double A = params.at<double>(0, 0);
	double B = params.at<double>(1, 0);
	double C = params.at<double>(2, 0);
	double D = params.at<double>(3, 0);
	double x = input.at<double>(0, 0);
	return A*sin(B*x) + C*cos(D*x);
}
 
double Deriv(double(*Func)(const Mat &input, const Mat params), const Mat &input, const Mat params, int n)
{
	// Assumes input is a single row matrix  
	// Returns the derivative of the nth parameter  
	Mat params1 = params.clone();
	Mat params2 = params.clone();
	// Use central difference  to get derivative  
	params1.at<double>(n, 0) -= DERIV_STEP;
	params2.at<double>(n, 0) += DERIV_STEP;
	double p1 = Func(input, params1);
	double p2 = Func(input, params2);
	double d = (p2 - p1) / (2 * DERIV_STEP);
	return d;
}
 
void GaussNewton(double(*Func)(const Mat &input, const Mat params),
	const Mat &inputs, const Mat &outputs, Mat params)
{
	int m = inputs.rows;
	int n = inputs.cols;
	int num_params = params.rows;
	Mat r(m, 1, CV_64F); // residual matrix  
	Mat Jf(m, num_params, CV_64F); // Jacobian of Func()  
	Mat input(1, n, CV_64F); // single row input  
 
	double last_mse = 0;
 
	for (int i = 0; i < MAX_ITER; i++) 
	{
		double mse = 0;
		for (int j = 0; j < m; j++) 
		{
			for (int k = 0; k < n; k++) 
			{
				input.at<double>(0, k) = inputs.at<double>(j, k);
			}
			r.at<double>(j, 0) = outputs.at<double>(j, 0) - Func(input, params);
			mse += r.at<double>(j, 0)*r.at<double>(j, 0);
			for (int k = 0; k < num_params; k++) 
			{
				Jf.at<double>(j, k) = Deriv(Func, input, params, k);
			}
		}
		mse /= m;
		// The difference in mse is very small, so quit  
		if (fabs(mse - last_mse) < 1e-8) 
		{
			break;
		}
		Mat delta = ((Jf.t()*Jf)).inv() * Jf.t()*r;
		params += delta; 
		printf("%f\n", mse);
		last_mse = mse;
	}
}