#include "../include/faceArray.h"

// Constructor by Path
faceArray::faceArray(){}

// Constructor by Path
faceArray::faceArray(const string& dirPath, const double minScale, const double maxScale, const int steps){
    
    // Set scaleFactors attribute
    if(minScale >= maxScale){
        return;
    }

    double delta = (maxScale - minScale)/steps;
    for(int i = 0 ; i <= steps ; i++){
        scaleFactors_.push_back(minScale + delta*i);
    }

    // Fetch images from directory
    DIR *dir;
	struct dirent *ent;

	if ((dir = opendir(dirPath.c_str())) == NULL) {
		return;
	}

	while ((ent = readdir (dir)) != NULL) {

        // Fetch file name
		string fileName = ent->d_name;
        
        // Check if ends with valid format
        if (fileName.length() > 4) {

            string last4chars = fileName.substr(fileName.length() - 4);

            if(last4chars != ".jpg" && last4chars != ".png" && last4chars != ".gif"){
                continue;
            }
        }

		string filePath = dirPath + fileName;

		// Read the file
		Mat img = imread(filePath);
		
		if (img.empty()) {
			continue;
		}

        cvtColor(img, img, COLOR_RGB2GRAY);

        // Soften the image
        //GaussianBlur(img,img,Size(3,3),1.0,0);

        // Show
        //imshow(fileName,img);

        names_.push_back(fileName);

        vector<Mat> image;
        for(double& scaleFactor : scaleFactors_){
            // Resize Face
            Mat imgFaceResized;
            int widthFaceResized = round(img.cols*scaleFactor);
            int heightFaceResized = round(img.rows*scaleFactor);
            int pixelCountFaceResized = widthFaceResized*heightFaceResized;

            Size size(widthFaceResized,heightFaceResized);
            resize(img,imgFaceResized,size);

            // Deep Copy
            image.push_back(imgFaceResized.clone());
        }

        // Add to attribute
        images_.push_back(image);			
	}
	
    closedir (dir);
}

// Destructor
faceArray::~faceArray(){}

// Operator Overloading
vector<Mat> faceArray::operator [](const int& index) const{
    return images_.at(index);
}

// Access Methods
int faceArray::size(){
    return names_.size();
}

string faceArray::getName(const int& index){
    return names_.at(index);
}

vector<vector<Mat>> faceArray::getImages(){
    return images_;
}

