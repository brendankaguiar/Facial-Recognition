/*
Version 0.1 Started working on code incldeing assign_values, print_matrix, and allocate_matrix - Julia Adamczyk
Version 0.2 Implemented deallocate_matrix, refactored c into c++, modified jacobi.c, tested code - Brendan Aguiar
Version 0.3 Implemented readAlgorithm. Tested code by writing and confirming two images in infranview - Brendan Aguiar
Version 0.4 Implemented toVector() and mean(). Included Julia_reading.cpp - Julia Adamczyk
Version 0.5
Version 0.6 Implemented the computation of A^T * A - Julia Adamczyk
Version 0.7 implemented printeigenFaces. mapped images to pgm format. Tested implementation with jacobi.c - Brendan Aguiar
Version 0.8 Implemented readegienfaces/readeigenvectors. Code runs, but reproduces the same 2 eigenfaces. - Brendan Aguiar
Version 0.9 Renamed variables in terms of slides. Corrected Calculations. Tested print of lambda, x_bar, and U - Brendan Aguiar
Version 1.0 Confirmed print of Average face and eigenfaces. Beginning Reconstruction check in testing.cpp- Brendan Aguiar
*/

#include <iostream>
#include <fstream>
#include "jacobi.c"
#include "image.h"
#include <string.h>
#define USER 2 //if user == 1, Julia's directory in use, else Brendan's directory in use
#define N_DATA 1204 //number of images used in data set

int width = 48;
int length = 60;
char JuliaDir[] = "C:\\Users\\Julia\\Desktop\\cs479\\Project3\\";
char BrendanDir[] = "C:\\Users\\aguia\\source\\repos\\Facial Recognition\\Facial Recognition\\";

void readImageHeader(char fname[], int& N, int& M, int& Q, bool& type);
void readImage(char fname[], ImageType& image);
void writeImage(char fname[], ImageType& image);
double** allocate_matrix(int N, int M);
void deallocate_matrix(double** matrix, int M);
void readAlgorithm(ImageType* images, char* dir);
void toVectors(double** image_stack, ImageType* images);
float* calculate_mean(double** image_stack);//also calculates max and min
void printAvgFace(float* mean);
void printeigenFaces(double** vectors, char* directory_names);
double** compute_Phi(double** image_stack, float* mean, int rows, int cols);
double** transpose(double** matrix, int row, int col);
double** multiply_matrices(double** first, double** second, int r1, int c1, int r2, int c2);
void printValues(double* w);
double** readeigenVectors();
double* readeigenValues();
void printVectors(double** V, int m, int n, char fname[]);
void normalizeMatrix(double** A, int m, int n);
void printmean(float* xbar);
int main() {

	cout << "Setting up inital variables and datatypes...\n";
	char directory_names[4][5] = { "fa_H", "fa_L", "fb_H", "fb_L" }; //choose the correct directory
	char dir_in_use[5];
	strcpy(dir_in_use, directory_names[0]);//change directory index to use the corresponding directory
	//image_stack represents N vectors of M features of variable x
	ImageType* images = new ImageType[N_DATA];
	double** image_stack = allocate_matrix(N_DATA, (width * length));

	cout << "Loading images...\n";
	readAlgorithm(images, dir_in_use);//reading images from fa_H. Change index to choose directory
	toVectors(image_stack, images);//convert each image matrix into a vector on image_stack (1204 x 2880 )
	cout << "Calculating x_bar...\n";
	float* x_bar = calculate_mean(image_stack); // x_bar (1 X 2880)

	cout << "Printing average face to average_face.pgm..." << endl;
	printAvgFace(x_bar); // print x_bar as average face image

	cout << "Printing mean.txt..." << endl;
	printmean(x_bar);
	cout << "Calculating Covariance Matrix...\n";

	double** Phi = compute_Phi(image_stack, x_bar, width * length, N_DATA);// Phi (2880 x 1204) //rows[0][#] and [#][0]

	char OName[] = "Omega_fa.txt";
	//printVectors(Phi, 2880, 1204, OName);
	double** PhiTranspose = transpose(Phi, width * length, N_DATA);// PhiTranspose (1204 x 2880)
	double** U = readeigenVectors();
	/*
	double** Cov = multiply_matrices(PhiTranspose, Phi, N_DATA, width * length, width * length, N_DATA);// Covariance Matrix (1204 x 1204)
	for (int i = 1; i <= 1204; i++)//Dividing Covariance by number of images
	{
		for (int j = 1; j <= 1204; j++)
			Cov[i][j] /= N_DATA;
	}
	cout << "Calculating lambda and U..\n";


	double** V = allocate_matrix(N_DATA, N_DATA);//allocating eigen vectors
	double* lambda = new double[N_DATA];//allocating eigen values
	int c = jacobi(Cov, (dimension)N_DATA, lambda, V);
	double** U = multiply_matrices(PhiTranspose, V, width * length, N_DATA, N_DATA, N_DATA); // U (2880x1204)
	normalizeMatrix(U, length * width, N_DATA);
	*/
	double** Omega_fa = multiply_matrices(PhiTranspose, U, N_DATA, width * length, width * length, N_DATA);
	//cout << "Printing lambda and U..\n";
	//printValues(lambda); //Print lambda values for testing phase
	//printeigenFaces(U, directory_names[0]);//Print U values as images
	//char UName[] = "U.txt";
	//printVectors(U, 2880, 1204, UName);
	cout << "Printing Omega for directory " << dir_in_use[0] << "...\n";
	//double** UTranspose = transpose(U, width * length, N_DATA);//UTranspose (1204 x 2880)
// Omega_i (1204 x 1204) eigencoefficients 

	char UName[] = "U.txt";
	printVectors(Omega_fa, N_DATA, N_DATA, OName);
	printVectors(U,2880, 1204, UName);
	cout << "Ending Program...\n";
	deallocate_matrix(Omega_fa, N_DATA);
	//deallocate_matrix(V, N_DATA);
	//deallocate_matrix(UTranspose, N_DATA);
	deallocate_matrix(Phi, (width * length));
	deallocate_matrix(U, (width * length));
	//delete[] lambda;
	//deallocate_matrix(Cov, N_DATA);
	deallocate_matrix(PhiTranspose, N_DATA);
	delete[] x_bar;
	deallocate_matrix(image_stack, N_DATA - 1);
	delete[] images;
	return 0;
}

double** allocate_matrix(int N, int M) {
	double** matrix;
	matrix = new double* [N];

	for (int i = 0; i < N; i++) {
		matrix[i] = new double[M];
	}
	return matrix;
}
void deallocate_matrix(double** matrix, int N) {
	for (int i = 0; i < N ; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}
void readImageHeader(char fname[], int& N, int& M, int& Q, bool& type)
{
	//int i, j;
	//unsigned char* charImage;
	char header[100], * ptr, * ptr2;
	ifstream ifp;

	ifp.open(fname, ios::in | ios::binary);

	if (!ifp) {
		cout << "Can't read image: " << fname << endl;
		exit(1);
	}
	// read header

	type = false; // PGM

	ifp.getline(header, 100, ' ');
	if ((header[0] == 80) &&  /* 'P' */
		(header[1] == 53)) {  /* '5' */
		type = false;
	}
	else if ((header[0] == 80) &&  /* 'P' */
		(header[1] == 54)) {        /* '6' */
		type = true;
	}
	else {
		cout << "Image " << fname << " is not PGM or PPM" << endl;
		exit(1);
	}

	ifp.getline(header, 100, '\n');
	M = strtol(header, &ptr, 0);
	N = strtol(ptr, &ptr2, 0);
	Q = atoi(ptr2);

	ifp.close();

}
void readImage(char fname[], ImageType& image)
{
	int i, j;
	int N, M, Q;
	unsigned char* charImage;
	char header[100], * ptr, * ptr2;
	ifstream ifp;

	ifp.open(fname, ios::in | ios::binary);

	if (!ifp) {
		cout << "Can't read image: " << fname << endl;
		exit(1);
	}
	// read header
	ifp.getline(header, 100, ' ');
	if ((header[0] != 80) ||    /* 'P' */
		(header[1] != 53)) {   /* '5' */
		cout << "Image " << fname << " is not PGM" << endl;
		exit(1);
	}
	ifp.getline(header, 100, '\n');
	M = strtol(header, &ptr, 0);
	N = strtol(ptr, &ptr2, 0);
	Q = atoi(ptr2);
	charImage = (unsigned char*) new unsigned char[M * N];
	ifp.read(reinterpret_cast<char*>(charImage), (M * N) * sizeof(unsigned char));
	if (ifp.fail()) {
		cout << "Image " << fname << " has wrong size" << endl;
		exit(1);
	}

	ifp.close();

	//
	// Convert the unsigned characters to integers
	//

	int val;

	for (i = 0; i < N; i++)
		for (j = 0; j < M; j++) {
			val = (int)charImage[i * M + j];
			image.setPixelVal(i, j, val);
		}
	delete[] charImage;

}
void writeImage(char fname[], ImageType& image)
{
	int i, j;
	int N, M, Q;
	unsigned char* charImage;
	ofstream ofp;

	image.getImageInfo(N, M, Q);

	charImage = (unsigned char*) new unsigned char[M * N];

	// convert the integer values to unsigned char

	int val;

	for (i = 0; i < N; i++)
		for (j = 0; j < M; j++) {
			image.getPixelVal(i, j, val);
			charImage[i * M + j] = (unsigned char)val;
		}

	ofp.open(fname, ios::out | ios::binary);

	if (!ofp) {
		cout << "Can't open file: " << fname << endl;
		exit(1);
	}

	ofp << "P5" << endl;
	ofp << M << " " << N << endl;
	ofp << Q << endl;

	ofp.write(reinterpret_cast<char*>(charImage), (M * N) * sizeof(unsigned char));

	if (ofp.fail()) {
		cout << "Can't write image " << fname << endl;
		exit(0);
	}

	ofp.close();

	delete[] charImage;

}
void readAlgorithm(ImageType* images, char* dir_name) {
	//Read pgm file names with text file. Match name with corresponding image and read
	char text_file[100];
	//int size = 0;
	if (USER == 1)//use Julia's directory
		strcpy(text_file, JuliaDir);
	else//use Brendan's directory
		strcpy(text_file, BrendanDir);
	char text_file3[100];
	char text_file2[100];
	strcpy(text_file2, text_file);
	strcat(text_file2, dir_name);
	strcpy(text_file3, text_file2);
	strcat(text_file, dir_name);
	strcat(text_file, "\\names.txt");
	//text file with all names (each folder has a separate names.txt with all file names in that directory)
	ifstream myFile;
	int n, m, q;
	bool t;
	char* str;
	str = new char[100];
	myFile.open(text_file);
	int i = 0;
	if (myFile.is_open()) {
		while (!myFile.eof())
		{
			myFile.getline(str, 22);
			strcat(text_file2, "\\");
			strcat(text_file2, str);//append text file with pgm file
			readImageHeader(text_file2, n, m, q, t);//read image from the file
			images[i].setImageInfo(n, m, q);
			readImage(text_file2, images[i]);
			images[i].getImageInfo(n, m, q);
			memset(text_file2, 0, strlen(text_file2));//reset text_file2 for next pgm file
			strcpy(text_file2, text_file3);
			i++;
		}
		delete[] str;
	}
	myFile.close();
}
void toVectors(double** image_stack, ImageType* images)
{
	int q;
	for (int i = 0; i < N_DATA; i++) {
		int h = 0;
		for (int j = 0; j < 60; j++) {
			for (int k = 0; k < 48; k++) {
				images[i].getPixelVal(j, k, q);
				image_stack[i][h] = (float)q;
				h++;
			}
		}
	}
}
float* calculate_mean(double** image_stack) {//returns Nx1 vector and fmin and fmax value of mean
	//add sums for each row of the vector
	float* sum = new float[48 * 60];
	float* mean = new float[48 * 60];
	//initialize each to 0
	for (int i = 0; i < 48 * 60; i++) {
		sum[i] = 0;
	}
	//loop M times
	for (int i = 0; i < 60 * 48; i++) {
		//add helper for the correct place in a vector xi (k will reset after each through a single vector
		for (int j = 0; j < N_DATA; j++) {
			sum[i] += (float)image_stack[j][i];//sum[2880][1204]
		}
	}
	for (int i = 0; i < 48 * 60; i++) {
		mean[i] = (float)sum[i] / (float)N_DATA;
	}
	delete[] sum;
	return mean;
}
void printAvgFace(float* mean) {
	ImageType avgFace;
	int rows = 60;
	int cols = 48;
	int levels = 255;
	int y = 0;
	avgFace.setImageInfo(rows, cols, levels);
	int h = 0;
	float min = mean[0];//set up initial min/max values
	float max = mean[0];
	for (int i = 0; i < length * width; i++)
	{
		if (mean[i] < min)
			min = mean[i];
		if (mean[i] > max)
			max = mean[i];
	}

	for (int j = 0; j < rows; j++)
	{
		for (int k = 0; k < cols; k++)
		{
			//cast it here
			y = (int)255 * ((mean[h] - min) / (max - min));
			avgFace.setPixelVal(j, k, y);
			h++;
		}
	}
	char fname[] = "average_face.pgm";
	writeImage(fname, avgFace);
}
double** compute_Phi(double** image, float* mean, int rows, int cols) {
	//make sure to allocate from i = 1 cause this is how jacobi works
	double** A = allocate_matrix(rows, cols);
	//loop through the A and assign correct value to each entry
	//A has 1204 columns which all are 2880 x 1 vectors that contain the (pixelvalue - mean value)
	//therefore in 1st column corresponds to image 1 and column 1205 corresponds to image 1204

	float val_avg;     //mean x_bar
	float val_face;   //xi
	//loop through the matrix
	for (int i = 0; i < rows; i++) {
		val_avg = mean[i];    //mean is constant for each row of the new matrix
		for (int j = 0; j < cols; j++) { //center the mean of each pixel using every image
			val_face = image[j][i];
			A[i][j] = val_face - val_avg;
		}
	}
	return A;
}
double** transpose(double** matrix, int row, int col)
{
	// dynamically allocate an array
	double** result = allocate_matrix(col, row);

	// transposing
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			result[j][i] = matrix[i][j];
	return result;
}
double** multiply_matrices(double** first, double** second, int r1, int c1, int r2, int c2) {

	if (c1 != r2) {
		cout << "Cannot multiply matrices" << endl;
		exit(1);
	}

	double** result = allocate_matrix(r1, c2);
	// Initializing elements of matrix mult to 0.
	for (int i = 0; i < r1; ++i) {
		for (int j = 0; j < c2; ++j) {
			result[i][j] = 0;
		}
	}

	// Multiplying first and second matrices and storing it in result
	for (int i = 0; i < r1; ++i) {
		for (int j = 0; j < c2; ++j) {
			for (int k = 0; k < c1; ++k) {
				result[i][j] += first[i][k] * second[k][j];
				//result[i][j] /= (float)N_DATA;//normalize (phiT)(phi)
			}
		}
	}
	return result;
}
void printeigenFaces(double** vectors, char* dir_name) { //vectors : (2880X1204)
	ImageType* eigFace;
	eigFace = new ImageType[N_DATA];
	int rows = 60;
	int cols = 48;
	int levels = 1;
	int y;
	for (int i = 1; i <= N_DATA; i++)
	{
		char* fname = new char[28];
		eigFace[i - 1].setImageInfo(rows, cols, levels);
		int h = 1;
		double max = -255;
		double min = 255;
		for (int n = 1; n <= width * length; n++)//set min and max
		{
			if (vectors[n][i] <= min)
				min = vectors[n][i];
			if (vectors[n][i] >= max)
				max = vectors[n][i];
		}
		for (int j = 0; j < rows; j++)
		{
			for (int k = 0; k < cols; k++)
			{
				//cast it here
				y = (int)255 * ((vectors[h][i] - min) / (max - min));
				eigFace[i - 1].setPixelVal(j, k, y);
				h++;
			}
		}
		sprintf(fname, "eigenfaces_%d", i);//eigenfaces_####
		strcat(fname, dir_name);//eigenfaces_####f% _%
		strcat(fname, ".pgm");//eigenfaces_####f% _%.pgm
		writeImage(fname, eigFace[i - 1]);
		delete[] fname;
	}
}
void printValues(double* w)
{
	ofstream ofp;
	ofp.open("eigenvalues.txt");
	for (int i = 1; i < 1204; i++)
	{
		ofp << w[i] << endl;
	}
	ofp.close();
}
void printVectors(double** V, int rows, int cols, char fname[]) {
	ofstream ofp;
	ofp.open(fname);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			ofp << V[i][j] << " ";
		}
		ofp << endl;
	}
	ofp.close();
}
double** readeigenVectors() {
	double** V = allocate_matrix(2880, N_DATA);
	ifstream ifp;
	ifp.open("U.txt");
	for (int i = 0; i < 2880; i++)
		for (int j = 0; j < 1204; j++)
			ifp >> V[i][j];
	ifp.close();
	return V;
}
double* readeigenValues()
{
	double* w = new double[N_DATA + 1];
	ifstream ifp;
	ifp.open("eigenvalues.txt");
	for (int i = 1; i <= 1204; i++)
		ifp >> w[i];
	ifp.close();
	return w;
}
void normalizeMatrix(double** vec, int m, int n) {
	//for every col
	for (int i = 1; i <= n; i++) {
		double sum = 0;
		//for every row
		for (int j = 1; j <= m; j++) {
			//add entry^2 to the sum
			sum += pow(vec[j][i], 2);
		}
		double length = sqrt(sum);
		//for every row
		for (int j = 1; j <= m; j++) {
			//normalize entry
			vec[j][i] /= length;
		}
	}
}
void printmean(float* xbar) {
	ofstream ofp;
	ofp.open("mean.txt");
	for (int i = 0; i < 2880; i++) {
		ofp << xbar[i] << endl;
	}
}