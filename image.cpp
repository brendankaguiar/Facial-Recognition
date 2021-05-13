#include "image.h"
ImageType::ImageType()
{
	N = 0;
	M = 0;
	Q = 0;
	pixelValue = nullptr;

}

//RGB ** ImageType::allocate_memory(int i, int j)

void ImageType::setPixelVal(int i, int j, int val)
{
	pixelValue[i][j] = val;
}

void ImageType::setImageInfo(int rows, int cols, int levels)
{
	N = rows;
	M = cols;
	Q = levels;
	pixelValue = new int* [N];
	for (int i = 0; i < N; i++)
		pixelValue[i] = new int[M];
}

void ImageType::getPixelVal(int i, int j, int& val) {
	val = pixelValue[i][j];
}

void ImageType::getImageInfo(int& rows, int& cols, int& levels)
{
	rows = N;
	cols = M;
	levels = Q;
}