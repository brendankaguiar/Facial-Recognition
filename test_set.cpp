//Version 0.1 Began working on Reconstruction. Tested and confirmed working code up til  - Brendan Aguiar

#include "image.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#define USER 2 //if user == 1, Julia's directory in use, else Brendan's directory in use
#define N_DATA 1204 //number of images used in data set
#define TEST_SET 1196
using namespace std;

void readImageHeader(char fname[], int& N, int& M, int& Q, bool& type);
void readImage(char fname[], ImageType& image);
void writeImage(char fname[], ImageType& image);
double** readOmega_i(int M, int K);
double* readLambda(int K);
void readAlgorithm(ImageType* images, char* dir);
void readTestImage(ImageType& images, char* dir, char im[]);
double** compute_Phi(double* image, double* mean);
double* toVector(ImageType known);
double** transpose(double** vector, int row, int col);
double** multiply_matrices(double** first, double** second, int r1, int c1, int r2, int c2);
void toMatrix(double** image_stack, ImageType* images);
void toVectors(double** image_stack, ImageType* images);
double** allocate_matrix(int N, int M);
void deallocate_matrix(double** matrix, int N);
double* add(double** M, double* x);
double** toDoublePtr(double* x);
double* euclideanDist(double* I, double* I_hat);//I_hat double** because it is formed from doubles but still match I dimensionally.
double min(double* Xm, int K, double ID[], int i);
double* get_reconstruction(double** omega, double** eigenvectors, double* mean);
void printKnownFace(double* mean);
double* getPhiHat(double** PhiTranspose, double** U_Stack);
double* readMean();
double** readU(int N, int K);
double* Mah(double* O, double** O_i, double* L, int K, int M);
void printVectors(double** V, int rows, int cols, char fname[]);
void printErrorReport(double* er, int  M, double* ID);
int width = 48;
int length = 60;
char JuliaDir[] = "C:\\Users\\Julia\\Desktop\\cs479\\Project3\\";
char BrendanDir[] = "C:\\Users\\aguia\\source\\repos\\FacialRecognition2\\FacialRecognition2\\";

int main()
{
	cout << "Setting up directory\n";
	int K = (int)N_DATA * .8; //Using  80% of eigenfaces
	char directory_names[4][5] = { "fa_H", "fa_L", "fb_H", "fb_L" };
	char dir_in_use[5];
	int N = length * width;//<----- Assigning dims to char sized variables
	int M = TEST_SET;
	strcpy(dir_in_use, directory_names[2]);//<---change to work with matching directory

	cout << "Reading Omega_i, x_bar, eigenfaces(U), and lambda...\n";
	double** O_i = readOmega_i(1204, K);//Testing with fb_h
	double* L_i = readLambda(K);// (K x 1)
	double* x_bar = readMean();// (N x 1)
	double** U = readU(N, K);// (N x K) 
	///double** O_iT = transpose(O_i, K, 1204);// U^T (N x K)
	//double** Omega_i = multiply_matrices(O_iT, U, K, N, N, M);//(K x M)
	//double** Omega_iT = transpose(Omega_i, K, M); // (M x K)
	//deallocate_matrix(Omega_i, K);
	//deallocate_matrix(O_iT, K);
	//deallocate_matrix(O_i, N);
	cout << "Creating Gallery Set...\n";// (2880 x 1196)
	ImageType* G = new ImageType[M];
	readAlgorithm(G, dir_in_use);
	double** G_Set = allocate_matrix(M, N);
	toVectors(G_Set, G);// (M x N)

	cout << "Applying PCA Reduction...\n";
	cout << K << endl;

	cout << "Processing Gallery...\n";
	double ID[1204];// ID (M x 1)
	double* S = new double[M];//er (M x 1)
	for (int i = 0; i < M; i++)//For Each query image M
	{
		double* Xk = new double[M];
		double** Phi = compute_Phi(G_Set[i], x_bar);// Phi (N x 1)
		double** PhiTranspose = transpose(Phi, length * width, 1);// PhiTranspose (1 x N)
		double** Omega = multiply_matrices(PhiTranspose, U, 1, N, N, K);// (1 x K)
		Xk = Mah(Omega[0], O_i, L_i, K, M);//Mahalanobis distance produces M distances
		S[i] = min(Xk, M, ID, i);
		delete[] Xk;
		deallocate_matrix(Omega, 1);
		deallocate_matrix(PhiTranspose, 1);
		deallocate_matrix(Phi, N);
	}
	printErrorReport(S, M, ID);
	cout << "Freeing Memory...\n";
	delete[] S;
	deallocate_matrix(U, N);
	deallocate_matrix(G_Set, M);
	delete[] G;
	delete[] x_bar;
	delete[] L_i;
	deallocate_matrix(O_i, M);
	/*
	char image_in_use[3][22] = { "\\01009_960627_fa.pgm", "unknown_image.pgm"
		, "\\average_face.pgm" };//[0]known image, [1] unknown image [2] average face
	//ImageType known, unknown, avg_face;
	//ImageType* U = new ImageType[N_DATA];//eigenfaces with N columns
	//double** U_Stack = allocate_matrix(width * length, N_DATA);
	cout << "Preparing to Reconstruct Known Image...\n"; //matrix reference to right of assignment, N = length* width
	//Reconstruction check
	readTestImage(known, dir_in_use, image_in_use[0]);//getting known image
	double* I = toVector(known);// I (1 x N)
	//readAlgorithm(U, dir_in_use);//<
	//toMatrix(U_Stack, U);// U_stack (N X N_DATA) or (2880 x 1204)  Phi
	double* Phi_h = get_reconstruction(yi, U, x_bar);//also reconstructed image I_hat
	double* I_h = add(Phi_h, x_bar);
	cout << "Reconstructing Known Image using Euclidean Distance...\n";
	printKnownFace(Phi_h);
	cout << "The error for the known image is " << er << endl;
	cout << "The Image is Identified as # " << ID_Known << endl;
	delete[] Xm;
	delete[] I;
	delete[] Phi_h;
	//delete[] U;
	deallocate_matrix(yi, 1);
	deallocate_matrix(Phi, 2880);
	deallocate_matrix(PhiTranspose, 1);
	*/
	return 0;
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
double* readLambda(int K)
{
	double* w = new double[K];
	ifstream ifp;
	ifp.open("eigenvalues.txt");
	for (int i = 0; i < K; i++)
		ifp >> w[i];
	ifp.close();
	return w;
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
void readTestImage(ImageType& images, char* dir_name, char im[]) {
	char text_file[100];
	if (USER == 1)//use Julia's directory
		strcpy(text_file, JuliaDir);
	else//use Brendan's directory
		strcpy(text_file, BrendanDir);
	strcat(text_file, dir_name);
	strcat(text_file, im);//appending test image file name
	int n, m, q;
	bool t;

	readImageHeader(text_file, n, m, q, t);
	images.setImageInfo(n, m, q);
	readImage(text_file, images);
	images.getImageInfo(n, m, q);//line may not be necessary
}
double** compute_Phi(double* image, double* mean) {
	double** Phi = allocate_matrix(length * width, 1);
	for (int i = 0; i < length * width; i++) {
		Phi[i][0] = image[i] - mean[i];
	}
	return Phi;
}
double* toVector(ImageType images)
{
	int q;
	double* X = new double[length * width];
	int h = 0;
	for (int j = 0; j < 60; j++) {
		for (int k = 0; k < 48; k++) {
			images.getPixelVal(j, k, q);
			X[h] = (float)q;
			h++;
		}
	}
	return X;
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
void toMatrix(double** image_stack, ImageType* images)
{
	int q;
	for (int i = 0; i < TEST_SET; i++) {
		int h = 0;
		for (int j = 0; j < 60; j++) {
			for (int k = 0; k < 48; k++) {
				images[i].getPixelVal(j, k, q);
				image_stack[h][i] = (double)q;
				h++;
			}
		}
	}
}
void toVectors(double** image_stack, ImageType* images)
{
	int q;
	for (int i = 0; i < TEST_SET; i++) {
		int h = 0;
		for (int j = 0; j < length; j++) {
			for (int k = 0; k < width; k++) {
				images[i].getPixelVal(j, k, q);
				image_stack[i][h] = (float)q;
				h++;
			}
		}
	}
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
	for (int i = 0; i < N; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}
double** readOmega_i(int K, int M) {
	double** O_i = allocate_matrix(K, M);// Omega_i
	ifstream ifp;
	ifp.open("Omega_fa.txt");
	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < M; j++)
		{
			ifp >> O_i[i][j];
		}
	}
	ifp.close();
	return O_i;
}
double* add(double** M, double* x) {
	double* I_h = new double[2880];
	for (int i = 0; i < length * width; i++)
	{
		I_h[i] = M[0][i] + x[i];
	}
	return I_h;
}
double** toDoublePtr(double* x) {
	double** x_b = allocate_matrix(1, length * width);
	for (int i = 0; i < length * width; i++)
	{
		x_b[0][i] = x[i];
	}
	return x_b;
}
double* euclideanDist(double* I, double* I_hat) {
	double* diff = new double[length * width];
	float sum = 0;
	for (int i = 0; i < length * width; i++)
	{
		diff[i] = I[0] - I_hat[i];
		diff[i] = pow(diff[i], 2);
	}
	return diff;
}
double min(double* Xm, int M, double ID[], int i) {
	double min = Xm[0];
	ID[i] = 1;
	for (int i = 1; i < 1204; i++)
	{
		if (min > Xm[i])
		{
			min = Xm[i];
			ID[i] = i + 1;
		}
	}
	return min;
}
double* get_reconstruction(double** Omega, double** eigenvectors, double* mean) {
	double* sum = new double[width * length];
	for (int i = 0; i < width * length; i++) {
		sum[i] = 0;
	}
	double value;

	//for each row of omega coefficient
	for (int i = 0; i < N_DATA; i++) {
		value = Omega[0][i];
		for (int j = 0; j < width * length; j++) {
			sum[j] += eigenvectors[j][i] * value;
		}
	}

	for (int i = 0; i < width * length; i++) {//add back average face
		sum[i] += mean[i];
	}
	return sum;
}
void printKnownFace(double* mean) {
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
	char fname[] = "known_face.pgm";
	writeImage(fname, avgFace);
}
double* getPhiHat(double** PhiTranspose, double** U_Stack) {

	double** yu = allocate_matrix(N_DATA, length * width);
	double** yi = allocate_matrix(N_DATA, length * width);
	double* Phi_h = new double[2880];

	// Initializing elements of matrix mult to 0.
	for (int i = 0; i < 2880; ++i) {
		Phi_h[i] = 0;
	}
	for (int i = 0; i < 1204; ++i) {
		yi[0][i] = 0;
	}

	for (int j = 0; j < 1204; ++j) {
		for (int k = 0; k < 2880; k++) {
			yi[0][j] += PhiTranspose[0][k] * U_Stack[k][j];//get scaling value
		}
		for (int k = 0; k < 2880; k++)
		{
			yu[j][k] = yi[0][j] * U_Stack[k][j];//scale Ui by yi on yuth vector of images
			Phi_h[k] += yu[j][k];
		}
	}
	return Phi_h;
}
double* readMean() {
	ifstream ifp;
	double* m = new double[2880];
	ifp.open("mean.txt");
	for (int i = 0; i < 2880; i++)
		ifp >> m[i];
	ifp.close();
	return m;
}
double** readU(int N, int K)
{
	ifstream ifp;
	double** U = allocate_matrix(N, K);
	ifp.open("U.txt");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < K; j++)
		{
			ifp >> U[i][j];
		}
	}
	ifp.close();
	return U;
}
double* Mah(double* Om, double** Om_i, double* L, int K, int M) {
	double diff;
	double* sum = new double[1204];
	float min, max;
	for (int j = 0; j < 1204; j++)//for each image of Omega_i
	{
		sum[j] = 0;
		for (int i = 0; i < K; i++)//compare the coefficient K with current image
		{
			diff = Om[i] - Om_i[j][i];//Omega(1 x K) - Omega_i(1204 x [K])
			diff *= diff;
			diff /= L[i];
			sum[j] += diff;//summed to jth image K times 
		}
		sum[j] /= K;
		//Normalize errors
	}	/*
	min = sum[0];
	max = min;
	//float mean = 0;
	//float std_dev = 0;

	for (int i = 0; i < 1204; i++)
	{
		mean += sum[i];
	}
	mean /= 1204;
	for (int i = 0; i < 1204; i++)
	{
		std_dev += pow((sum[i] - mean), 2);
	}
	std_dev /= 1204;
	std_dev = sqrt(std_dev);
	for (int i = 0; i < 1204; i++)
	{
		sum[i] = (sum[i] - min) / (max - min);
	}*/
	return sum; // (M x 1)
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
void printErrorReport(double* er, int  M, double* ID) {
	ofstream ofp;
	ofp.open("ErrorReportfb_h_80percent.txt");
	for (int i = 0; i < M; i++)//for all images
	{
		ofp << "Image # " << i + 1 << " with ID : " << ID[i] << " has error : " << er[i] << endl;
	}
	ofp.close();
}