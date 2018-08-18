#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>

using namespace cv;
using namespace std;

void contour(Mat image);
int main1(int argc, char** argv)
{
	VideoCapture cap(0); //capture the video from web cam

						 //if (!cap.isOpened())  // if not success, exit program
						 //{
						 //	cout << "Cannot open the web cam" << endl;
						 //	return -1;
						 //}

	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"
	namedWindow("Thresholded Image", CV_WINDOW_AUTOSIZE);
	int iLowH = 0;
	int iHighH = 179;

	int iLowS = 100;
	int iHighS = 250;

	int iLowV = 120;
	int iHighV = 255;

	//Create trackbars in "Control" window
	cvCreateTrackbar("LowH", "Thresholded Image", &iLowH, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Thresholded Image", &iHighH, 179);

	cvCreateTrackbar("LowS", "Thresholded Image", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Thresholded Image", &iHighS, 255);

	cvCreateTrackbar("LowV", "Thresholded Image", &iLowV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Thresholded Image", &iHighV, 255);

	while (true)
	{
		Mat imgOriginal;

		//bool bSuccess = cap.read(imgOriginal); // read a new frame from video

		//if (!bSuccess) //if not success, break loop
		//{
		//	cout << "Cannot read a frame from video stream" << endl;
		//	break;
		//}

		Mat imgHSV;
		imgOriginal = cv::imread("placa17468511.png");


		cv::Mat lambda(2, 4, CV_32FC1);
		cv::Mat input, output;									// The 4 points that select quadilateral on the input , from top-left in clockwise order
		cv::Point2f inputQuad[4]; 										// These four pts are the sides of the rect box used as input 
		cv::Point2f outputQuad[4];
		input = imgOriginal.clone();
		inputQuad[0] = cv::Point2f(1260, 0);
		inputQuad[1] = cv::Point2f(1260, 675);
		inputQuad[2] = cv::Point2f(0, 673);
		inputQuad[3] = cv::Point2f(0, 0);
		// The 4 points where the mapping is to be done , from top-left in clockwise order
		outputQuad[0] = cv::Point2f(1260, 200);
		outputQuad[1] = cv::Point2f(1260, 875);
		outputQuad[2] = cv::Point2f(0, 670);
		outputQuad[3] = cv::Point2f(0, 0);

		// Get the Perspective Transform Matrix i.e. lambda 
		lambda = getPerspectiveTransform(inputQuad, outputQuad);
		// Apply the Perspective Transform just found to the src image
		warpPerspective(input, output, lambda, output.size());



		//cvtColor(output, output, COLOR_BGR2GRAY);
		//cvtColor(output, output, COLOR_GRAY2BGR);
		cvtColor(output, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

		Mat imgThresholded;

		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

																									  //morphological opening (remove small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(1, 1)));
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(1, 1)));

		//morphological closing (fill small holes in the foreground)
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(1, 1)));
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(1, 1)));


		imshow("Thresholded Image", imgThresholded); //show the thresholded image
		//imshow("Original", imgOriginal); //show the original image
		//contour(imgThresholded);
		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	return 0;

}

void contour(Mat gray)
{
	namedWindow("Display window", CV_WINDOW_AUTOSIZE);
	  //imshow( "Display window", image );
	//   Mat gray;
	//  cvtColor(image, gray, CV_BGR2GRAY);
	Canny(gray, gray, 100, 200, 3);
	/// Find contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	findContours(gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));


	////////////////////
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}

	///  Get the mass centers:
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}



	///////////////////
	/// Draw contours

	Mat drawing = Mat::zeros(gray.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(255,255,255);
		drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());

		////////////////////
		vector<Moments> mu(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			mu[i] = moments(contours[i], false);
		}

		///  Get the mass centers:
		vector<Point2f> mc(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
		}



		///////////////////
		cout << "Center = " << mc[i].x << "," << mc[i].y << endl;
	}

	//imshow("Result window", drawing);
	//   waitKey(0);

}