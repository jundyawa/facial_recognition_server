#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <iostream>
#include <sstream>


using namespace cv;
using namespace std;

// RGB Image Attributes
int widthRGB = 1280;
int heightRGB = 960;
Mat imgRGB = Mat::zeros(heightRGB,widthRGB,CV_8UC3);

// HOG Image Attributes
int scanMatrixWidth = 16;
int widthHOG = widthRGB/scanMatrixWidth;
int heightHOG = heightRGB/scanMatrixWidth;
int pixelCountHOG = widthHOG*heightHOG;
Mat imgHOG = Mat::zeros(heightHOG,widthHOG, CV_8UC1);

// Histogram Attributes
int histSize = 8; // Number of Bins
float range[] = { 0, 256 } ; // Set the range of values (upper bound exclusive)
const float* histRange = { range };

// Face Detect Attributes
double faceDetectThreshold = 0.79;
vector<string> names;

// Face Image Attributes
double goldenRatio = 1.618;
int heightFace = 4*round(heightHOG/5.0);
int widthFace = round(heightFace/goldenRatio);
vector<Mat> imgFaces;
/*
	Saves the HOG of the cropped face image in the program folder

		returns true if the write was successful
*/
bool saveFaceHOG(string faceName){

	int startFrameX = round((widthHOG-widthFace)/2.0);
	int startFrameY = round((heightHOG-heightFace)/2.0);

	// Setup a rectangle to define the region of interest
	Rect myROI(startFrameX, startFrameY, widthFace, heightFace);

	// Crop the full image to that image contained by the rectangle myROI
	Mat imgFace = imgHOG(myROI).clone();
	imgFaces.push_back(imgFace);

	// Save image
	string fileName = faceName + ".png";

	return imwrite(fileName,imgFace);
}


/*
	Generates the HOG from the RGB image
*/
void generateHOG(){

	// Initialize Process Matrices
	Mat imgGrayscale = Mat::zeros(heightRGB, widthRGB, CV_8UC1);
	Mat imgGradient = Mat::zeros(heightRGB, widthRGB, CV_8UC1);
	
	// Initialize Scanning Matrix
	int scanMatrix[8];

	// Convert RGB to Grayscale
	cvtColor(imgRGB,imgGrayscale, CV_RGB2GRAY);

	// Check if conversion was successful
	if (imgGrayscale.data == NULL){
		cout << "Couldn't convert to grayscale" << endl;
		return;
	}

	// Scanning Matrix goes CHOO CHOO!
	// We skip the borders [1,N-1]
	for(int i = 1 ; i < heightRGB-1; i++){
		for(int j = 1 ; j < widthRGB-1 ; j++){

			scanMatrix[3] = (int) imgGrayscale.at<uchar>(i-1,j-1);
			scanMatrix[2] = (int) imgGrayscale.at<uchar>(i-1,j);
			scanMatrix[1] = (int) imgGrayscale.at<uchar>(i-1,j+1);

			scanMatrix[4] = (int) imgGrayscale.at<uchar>(i,j-1);
			scanMatrix[0] = (int) imgGrayscale.at<uchar>(i,j+1);

			scanMatrix[5] = (int) imgGrayscale.at<uchar>(i+1,j-1);
			scanMatrix[6] = (int) imgGrayscale.at<uchar>(i+1,j);
			scanMatrix[7] = (int) imgGrayscale.at<uchar>(i+1,j+1);

			// We set the Gradient Image (i,j) to be the index of darkest surrounding pixel
			imgGradient.at<uchar>(i,j) = distance(scanMatrix, max_element(scanMatrix,scanMatrix+8));
		}
	}

	int globalI = 0;
	int globalJ = 0;
	int angleTotal = 0;
	int angleMean = 0;
	for(int k = 0 ; k < pixelCountHOG ; k++){

		// Scan through sub images
		for(int i = 0 ; i < scanMatrixWidth; i++){
			for(int j = 0 ; j < scanMatrixWidth; j++){
				// Sum neighboring angles
				angleTotal += imgGradient.at<uchar>(globalI+i,globalJ+j);
			}
		}

		// We take the mean of angleTotal
		angleMean = round(angleTotal/(1.0*(scanMatrixWidth*scanMatrixWidth)));

		// Set The HOG (i,j) to the local average on the RGB image
		imgHOG.at<uchar>(globalI/scanMatrixWidth, globalJ/scanMatrixWidth) = angleMean*32;

		// Reset angleTotal
		angleTotal = 0;

		// Increment global i
		globalI += scanMatrixWidth;

		// If we reached the max i we go back to the top and increment column (j)
		if(globalI >= imgGradient.rows){
			globalI = 0;
			globalJ += scanMatrixWidth;
		}			
	}	
}


/*
	Show the HOG image
*/
void showHOG(){
	namedWindow("HOG", WINDOW_NORMAL);
	resizeWindow("HOG", widthRGB/5, heightRGB/5);
	imshow("HOG",imgHOG);
}

/*
	Calculate the Histogram
*/
Mat calculateHist(Mat imgToHist){

	bool uniform = true; 
	bool accumulate = false;
	
	/// Compute the histograms
	Mat hist;
	calcHist( &imgToHist, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

	return hist.clone();
}

/*

	Show the histogram

*/
void showHOGHist(Mat histogram){
	
	// Draw Attributes
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	/// Draw for each channel
	for( int i = 1; i < histSize; i++ ){

		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(histogram.at<float>(i-1)) ) ,
						Point( bin_w*(i), hist_h - cvRound(histogram.at<float>(i)) ),
						Scalar( 255, 0, 0), 2, 8, 0  );
	}
	
	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	imshow("calcHist Demo", histImage );
}


/*
	Detect Face
*/
void faceDetect(){
	
	// Resized Face Image
	Mat imgFaceResized;
	double scaleFactor = 1.0;
	int widthFaceResized = 0;
	int heightFaceResized = 0;
	int pixelCountFaceResized = 0;

	// Fit Variables
	double score = 0;
	vector<double> maxScore;
	vector<int> maxI;
	vector<int> maxJ;
	vector<double> maxScaleFactor;

	for(int p = 0 ; p < names.size() ; p++){

		int maxI_temp = 0;
		int maxJ_temp = 0;
		int maxScale_temp = 0;
		double maxScore_temp = 0;

		// Resize Face
		for(int k = 0 ; k < 8 ; k++){
			
			// We set the scale factor [0.7;1.3]
			scaleFactor = 0.7+0.05*k;

			// Resize Face
			widthFaceResized = round(widthFace*scaleFactor);
			heightFaceResized = round(heightFace*scaleFactor);
			pixelCountFaceResized = widthFaceResized*heightFaceResized;
			Size size(widthFaceResized,heightFaceResized);
			resize(imgFaces.at(p),imgFaceResized,size);

			// On divise l'image en 4 fractions
			int halfWidth = round(widthFaceResized/2.0);
			int thirdHeight = round(heightFaceResized/3.0);
			Rect topleftRect(0,0,halfWidth, thirdHeight);
			Rect toprightRect(halfWidth,0,halfWidth-1, thirdHeight);	
			Rect bottomleftRect(0,thirdHeight,halfWidth, 2*thirdHeight-1);
			Rect bottomrightRect(halfWidth,thirdHeight,halfWidth-1, 2*thirdHeight-1);	
			Mat topleftFaceImg = imgFaceResized(topleftRect).clone();
			Mat toprightFaceImg = imgFaceResized(toprightRect).clone();	
			Mat bottomleftFaceImg = imgFaceResized(bottomleftRect).clone();
			Mat bottomrightFaceImg = imgFaceResized(bottomrightRect).clone();

			// Scan through HOG image to see where face fits best
			for(int i = 0 ; i < widthHOG-widthFaceResized; i+=5){
				for(int j = 0 ; j < heightHOG-heightFaceResized; j+=5){
					
					// Move a face sized cropping rectangle over the HOG image
					Rect myROI(i, j, widthFaceResized, heightFaceResized);
					Mat croppedImg = imgHOG(myROI).clone();

					Mat topleftCroppedImg = croppedImg(topleftRect).clone();
					Mat toprightCroppedImg = croppedImg(toprightRect).clone();
					Mat bottomleftCroppedImg = croppedImg(bottomleftRect).clone();
					Mat bottomrightCroppedImg = croppedImg(bottomrightRect).clone();
					
					// Compare the histogram of the face and the cropped rectangle
					Mat histDiffTopLeft;
					Mat histDiffTopRight;
					Mat histDiffBottomLeft;
					Mat histDiffBottomRight;
					absdiff(calculateHist(topleftCroppedImg),calculateHist(topleftFaceImg),histDiffTopLeft);
					absdiff(calculateHist(toprightCroppedImg),calculateHist(toprightFaceImg),histDiffTopRight);
					absdiff(calculateHist(bottomleftCroppedImg),calculateHist(bottomleftFaceImg),histDiffBottomLeft);
					absdiff(calculateHist(bottomrightCroppedImg),calculateHist(bottomrightFaceImg),histDiffBottomRight);
					
					// Calculate Score
					score = 1 - (sum(histDiffTopLeft)[0] + sum(histDiffTopRight)[0] + sum(histDiffBottomLeft)[0] + sum(histDiffBottomRight)[0])/(1.0*pixelCountFaceResized);

					// If we have a better fit
					if(score > maxScore_temp && score > faceDetectThreshold){
						maxI_temp = i;
						maxJ_temp = j;
						maxScale_temp = scaleFactor;
						maxScore_temp = score;
					}			
				}
			}
		}

		maxI.push_back(maxI_temp);
		maxJ.push_back(maxJ_temp);
		maxScaleFactor.push_back(maxScale_temp);
		maxScore.push_back(maxScore_temp);
	}

	for(int i = 0 ; i < maxI.size() ; i++){
		// Add rectangle
		int X1 = round((maxI.at(i)/(1.0*widthHOG))*widthRGB);
		int Y1 = round((maxJ.at(i)/(1.0*heightHOG))*heightRGB);
		int X2 = round(X1 + ((widthFace*maxScaleFactor.at(i))/(1.0*widthHOG))*widthRGB);
		int Y2 = round(Y1 + ((heightFace*maxScaleFactor.at(i))/(1.0*heightHOG))*heightRGB);
		
		rectangle(imgRGB,Point(X1, Y1),Point(X2, Y2),Scalar(255, 255, 0), 20);
		stringstream nametag;
		nametag << names.at(i) << ", score = " << to_string(maxScore.at(i));
		putText(imgRGB, nametag.str(), cvPoint(X1,Y1-25), FONT_HERSHEY_COMPLEX_SMALL, 1.5, cvScalar(255,255,0), 1, CV_AA);
	}
}

int main(int argc, char *argv[]){

	// Webcam
	VideoCapture cam(1);

	// Check if video device is working
	if(!cam.isOpened()){
		cout << "Failed to connect to the camera." << endl;
		return -1;
	}

	// Set cam properties
	cam.set(CV_CAP_PROP_FRAME_WIDTH,widthRGB);
	cam.set(CV_CAP_PROP_FRAME_HEIGHT,heightRGB);

	// Start User interface
	int keyPressed = 0;
	string yesNo = "";

	while(1){
		
		string name;
		cout << "Enter your name : ";
		cin >> name;
		names.push_back(name);

		cout << endl << "Put your face in the frame and press the spacebar" << endl;

		while(1){

			// Capture Image
			cam >> imgRGB;

			// Add Face frame (20px border)
			int heightFaceFrame = 4*round(heightRGB/5.0);
			int widthFaceFrame = round(heightFaceFrame/goldenRatio);
			int X1 = round((widthRGB-widthFaceFrame)/2.0);
			int Y1 = round((heightRGB-heightFaceFrame)/2.0);
			int X2 = X1 + widthFaceFrame;
			int Y2 = Y1 + heightFaceFrame;
			rectangle(imgRGB,Point(X1, Y1),Point(X2, Y2),Scalar(255, 255, 0), 20);	

			// Check if user has asked to stop script
			keyPressed = (waitKey(30) % 256);
			if (keyPressed == 27){		// Esc pressed
				cout << "Exiting program" << endl;
				return -1;
			}else if (keyPressed == 32){		// Space Bar pressed
				generateHOG();
				if(saveFaceHOG(name)){
					cout << name << "'s face saved!" << endl << endl;
					break;
				}else{
					cout << "Couldn't save Face, try again" << endl;
				}
			}

			// Show Image
			namedWindow("Webcam", WINDOW_NORMAL);
			resizeWindow("Webcam", widthRGB/2, heightRGB/2);
			imshow("Webcam",imgRGB);
		}

		cout << "Do you want to save another face (y/n)? ";
		cin >> yesNo;

		if(yesNo.compare("n") == 0){
			break;
		}

		if(yesNo.compare("y") != 0){
			cout << "Wrong input" << endl;
			break;
		}
	}
	

	
	while(1){

		// Capture Image
		cam >> imgRGB;		

		// Check if user has asked to stop script
		keyPressed = (waitKey(30) % 256);
		if (keyPressed == 27){		// Esc pressed
			cout << "Exiting program" << endl;
			break;
		}
		
		generateHOG();
		faceDetect();

		// Show Image
		namedWindow("Webcam", WINDOW_NORMAL);
		resizeWindow("Webcam", widthRGB/2, heightRGB/2);
		imshow("Webcam",imgRGB);

	}
	
	// Close Window
	destroyAllWindows();
	return 0;
}


