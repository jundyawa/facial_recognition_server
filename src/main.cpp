#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>
#include "../include/faceArray.h"

using namespace cv;
using namespace std;

// RGB Attributes
Mat imgRGB;
int widthRGB;
int heightRGB;
int pixelCountRGB;

// HOG Attributes
Mat imgHOG;
int heightHOG;
int widthHOG;
int pixelCountHOG;

// Scan Matrix Attributes
int scanMatrixWidth;
int scanMatrixPixelCount;
Mat imgGrayscale;
Mat imgGradient;

// Faces Attributes
faceArray myFaces;
double faceDetectThreshold = 0.91;

// Histogram Attributes
int histSize = 8; // Number of Bins
float range[] = { 0, 256 } ; // Set the range of values (upper bound exclusive)
const float* histRange = { range };


void show(const Mat& img, const string& title){

    namedWindow(title, WINDOW_NORMAL);
    resizeWindow(title, widthRGB/2, heightRGB/2);
    imshow(title,img);
}

void updateHOG(){

	// Initialize Scanning Matrix
	int scanMatrix[8];

	// Convert img to Grayscale
	cvtColor(imgRGB, imgGrayscale, CV_RGB2GRAY);

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
        angleTotal = 0;
		for(int i = 0 ; i < scanMatrixWidth; i++){
			for(int j = 0 ; j < scanMatrixWidth; j++){
				// Sum neighboring angles
				angleTotal += imgGradient.at<uchar>(globalI+i,globalJ+j);
			}
		}

		// We take the mean of angleTotal
		angleMean = round(angleTotal/(1.0*scanMatrixPixelCount));

		// Set The HOG (i,j) to the local average on the RGB image
		imgHOG.at<uchar>(globalI/scanMatrixWidth, globalJ/scanMatrixWidth) = angleMean*31;

		// Increment global i
		globalI += scanMatrixWidth;

		// If we reached the max i we go back to the top and increment column (j)
		if(globalI >= heightRGB){
			globalI = 0;
			globalJ += scanMatrixWidth;
		}			
	}

    //GaussianBlur(imgHOG,imgHOG,Size(3,3),1.0,0);
}

Mat calculateHist(const Mat& imgToHist){

	bool uniform = true; 
	bool accumulate = false;
	
	/// Compute the histograms
	Mat hist;
	calcHist( &imgToHist, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

	return hist;
}

void showHOGHist(const Mat& histogram){
	
	// Draw Attributes
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	/// Draw for each channel
	for( int i = 0; i < histSize; i++ ){

        line( histImage, Point(bin_w*i, hist_h), 
                         Point(bin_w*i, hist_h - cvRound(histogram.at<float>(i))),
                         Scalar( 255, 0, 0),
                         20,8,0);
	}
	
	imshow("calcHist Demo", histImage );
}


void faceDetect(){

    double textSize = 1.5;

    // We iterate through all images of faces
    for(int k = 0 ; k < myFaces.size() ; k++){

        // Variables that store best fitting face
        int maxI = 0;
		int maxJ = 0;
        int maxWidth = 0;
        int maxHeight = 0;

	    double score = 0;
		double maxScore = 0;
        double maxScale = 0;

		// We iterate through the different format of a face
		for(Mat &image : myFaces[k]){

            int widthFace = image.cols;
            int heightFace = image.rows;
            int pixelCountFace = widthFace*heightFace;

            // On divise l'image en 4 fractions
			int halfWidth = round(widthFace/2.0);
			Rect leftRect(0,0,halfWidth, heightFace);
			Rect rightRect(halfWidth,0,halfWidth-1, heightFace);	
			Mat leftFaceImg = image(leftRect).clone();
			Mat rightFaceImg = image(rightRect).clone();	

			// Scan through HOG image to see where face fits best
			for(int i = 0 ; i < widthHOG-widthFace; i+=6){
				for(int j = 0 ; j < heightHOG-heightFace; j+=6){
					
					// Move a face sized cropping rectangle over the HOG image
					Rect myROI(i, j, widthFace, heightFace);
					Mat croppedHOG = imgHOG(myROI);

                    Mat leftCroppedImg = croppedHOG(leftRect);
					Mat rightCroppedImg = croppedHOG(rightRect);
					
					// Compare the histogram of the face and the cropped rectangle
					//Mat histDiff;
					//absdiff(calculateHist(croppedHOG),calculateHist(image),histDiff);

                    // Compare the histogram of the face and the cropped rectangle
					Mat histDiffLeft;
					Mat histDiffRight;
					absdiff(calculateHist(leftCroppedImg),calculateHist(leftFaceImg),histDiffLeft);
					absdiff(calculateHist(rightCroppedImg),calculateHist(rightFaceImg),histDiffRight);
				
					// Calculate Score
					score = 1 - (sum(histDiffLeft)[0] + sum(histDiffRight)[0])/(1.0*pixelCountFace);

					
					// Calculate Score
					//score = 1 - (sum(histDiff)[0])/(1.0*pixelCountFace);

					// If we have a better fit
					if(score > maxScore && score > faceDetectThreshold && score < 1.0){
						maxI = i;
						maxJ = j;
                        maxWidth = widthFace;
                        maxHeight = heightFace;
						maxScore = score;
					}			
				}
			}
		}

        // Add rectangle
		int X1 = round((maxI/(1.0*widthHOG))*widthRGB);
		int Y1 = round((maxJ/(1.0*heightHOG))*heightRGB);
		int X2 = X1 + round((maxWidth/(1.0*widthHOG))*widthRGB);
		int Y2 = Y1 + round((maxHeight/(1.0*heightHOG))*heightRGB);
		rectangle(imgRGB,Point(X1, Y1),Point(X2, Y2),Scalar(255, 255, 0), 20);

        // Add Name Tag
		stringstream nametag;
		nametag << myFaces.getName(k) << ", score = " << maxScore;
		putText(imgRGB, nametag.str(), cvPoint(X1,Y1-25), FONT_HERSHEY_COMPLEX_SMALL, textSize, cvScalar(255,255,0), 1, CV_AA);
    }
}


void addFaceFrame(){

    int heightFaceFrame = 4*round(heightRGB/5.0);
    int widthFaceFrame = round(heightFaceFrame/1.618);
    int X1 = round((widthRGB-widthFaceFrame)/2.0);
    int Y1 = round((heightRGB-heightFaceFrame)/2.0);
    int X2 = X1 + widthFaceFrame;
    int Y2 = Y1 + heightFaceFrame;
    rectangle(imgRGB,Point(X1, Y1),Point(X2, Y2),Scalar(255, 255, 0), 20);
}

bool saveFaceHOG(){

	int heightFaceFrame = 4*round(heightHOG/5.0);
    int widthFaceFrame = round(heightFaceFrame/1.618);
    int X1 = round((widthHOG-widthFaceFrame)/2.0);
    int Y1 = round((heightHOG-heightFaceFrame)/2.0);
    
	// Setup a rectangle to define the region of interest
	Rect myROI(X1, Y1, widthFaceFrame, heightFaceFrame);

	// Crop the full image to that image contained by the rectangle myROI
	Mat imgFace = imgHOG(myROI);

	// Ask for name
    string faceName;
    string answer;
    cout << "Name : ";
    cin >> faceName;

    // Ask to save
    cout << "Save (y/n)? ";
    cin >> answer;
    if(answer.compare("y") == 0){

        string fileName = faceName + ".png";
        string filePath = "faces/" + fileName;

        return imwrite(filePath,imgFace);
    }

    return false;
}


void modeOfOperation(int mode){

    if(mode == 0){

        show(imgRGB,"RGB");

    }else if(mode == 1){

        // Update HOG
        updateHOG();

        show(imgRGB,"RGB");
        show(imgHOG,"HOG");

    }else if(mode == 2){

        // Update HOG
        updateHOG();

        addFaceFrame();
        show(imgRGB,"RGB");

        if((waitKey(30) % 256) == 32){
            // Space Pressed
            if(saveFaceHOG()){
                cout << "Face saved!" << endl;
                myFaces = faceArray("faces/",0.7,1,6);

            }else{
                cout << "Could'nt save" << endl;
            }
        }
    }else if(mode == 3){

        // Update HOG
        updateHOG();
        faceDetect();

        show(imgRGB,"RGB");

    }else if(mode == 4){

        // Update HOG
        updateHOG();
        faceDetect();

        show(imgRGB,"RGB");
        show(imgHOG,"HOG");
    }

    /* Convert HOG to RGB just to show it
    Mat imgHOG_RGB;
    cvtColor(imgHOG, imgHOG_RGB, COLOR_GRAY2RGB);

    Mat imgMerged;
    hconcat(img,imgHOG_RGB,imgMerged);
    show(imgMerged);
    */

    //Mat hist = calculateHist(imgHOG);
    //showHOGHist(hist);
}

int main(int argc, char *argv[]){

    // Init Webcam Object
	VideoCapture cam(0);

	// Check if webcam is working
	if(!cam.isOpened()){
		cout << "Failed to connect to the camera." << endl;
		return -1;
	}

	// Weird trick to set cam format, you go overboard and it sets it to max
	cam.set(CV_CAP_PROP_FRAME_WIDTH,10000);
	cam.set(CV_CAP_PROP_FRAME_HEIGHT,10000);

    // Define Constants
    widthRGB = cam.get(CV_CAP_PROP_FRAME_WIDTH);
    heightRGB = cam.get(CV_CAP_PROP_FRAME_HEIGHT);
    pixelCountRGB = widthRGB*heightRGB;

    cout << "Scan Matrix Width : ";
    cin >> scanMatrixWidth;
    cout << endl;
    scanMatrixPixelCount = scanMatrixWidth*scanMatrixWidth;

    widthHOG = widthRGB/scanMatrixWidth;
    heightHOG = heightRGB/scanMatrixWidth;
    pixelCountHOG = widthHOG*heightHOG;

    // Initialize Process Variables
    imgRGB = Mat::zeros(heightRGB,widthRGB,CV_8UC3);
    imgGrayscale = Mat::zeros(heightRGB, widthRGB, CV_8UC1);
    imgGradient = Mat::zeros(heightRGB, widthRGB, CV_8UC1);
    imgHOG = Mat::zeros(heightHOG,widthHOG,CV_8UC1);

    // Import all saved faces
    myFaces = faceArray("faces/",0.7,1,6);

    // Start capturing video
    int mode = 0;
    int pressedKey = 0;
    while(1){

        // Capture RGB from webcam
        cam >> imgRGB;

        modeOfOperation(mode);

        // Check for user keyboard input
        pressedKey = (waitKey(30) % 256);
        if(pressedKey == 27){
            // Esc Pressed
            cout << "Exiting program" << endl;
            break;

        }else if(pressedKey >= 48 && pressedKey <= 57){
            // 0-9 Pressed
            mode = pressedKey - 48;
        }
        pressedKey = 0;
    }

    return 0;
}
