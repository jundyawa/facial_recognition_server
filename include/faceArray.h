
#ifndef FACEARRAY_H
#define FACEARRAY_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dirent.h>

using namespace cv;
using namespace std;

const double golden_ratio = 1.618;

class faceArray{
    
    public:
        // Constructor by Directory Path
        faceArray();

        // Constructor by Directory Path
        faceArray(const string& dirPath, const double minScale, const double maxScale, const int steps);

        // Destructeur
        ~faceArray();

        // Operator Overloading    
    	vector<Mat> operator [](const int& index) const;

        // Access Methods
        int size();
        string getName(const int& index);
        vector<vector<Mat>> getImages();

    private:

        vector<double> scaleFactors_;

        vector<string> names_;
        vector<vector<Mat>> images_;
};

#endif