#ifndef IMAGE_H
#define IMAGE_H
#include <vector>

using namespace std;
class ImageType {
	int N, M, Q; //Rows, Columns, Levels
	int** pixelValue;
public:
	ImageType();
	void getImageInfo(int&, int&, int&);
	void setImageInfo(int, int, int);
	void setPixelVal(int, int, int);
	void getPixelVal(int, int, int&);
};
#endif
