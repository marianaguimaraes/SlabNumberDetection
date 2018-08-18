//by Mariana Guimaraes 14/08/2018

#include <windows.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>
#include<iostream>
#include<sstream>

const int MIN_CONTOUR_AREA = 1000;
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

class ContourWithData {
public:

	std::vector<cv::Point> ptContour;
	cv::Rect boundingRect;
	float fltArea;


	bool checkIfContourIsValid() {
		if (fltArea < MIN_CONTOUR_AREA) return false;

		return true;
	}


	static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      // this function allows us to sort
		return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   // the contours from left to right
	}

};

bool checkRed(cv::Mat image)
{
	std::vector<ContourWithData> allContoursWithData_red;
	cv::Mat imgHSV_red;
	cv::Mat matBlurred_red;
	cv::Mat matThresh_red;
	cv::Mat matThreshCopy_red;
	cv::Mat imgThresholded_red;
	cv::Mat lambda(2, 4, CV_32FC1);
	cv::Mat input, output;
	cv::Point2f inputQuad[4];
	cv::Point2f outputQuad[4];
	int nColorRedStu = 0;

	output = image.clone();

	cv::cvtColor(output, imgHSV_red, CV_BGR2HSV);

	cv::inRange(imgHSV_red, cv::Scalar(0, 104, 120), cv::Scalar(179, 250, 255), imgThresholded_red);

	cv::GaussianBlur(imgThresholded_red,
		matBlurred_red,
		cv::Size(7, 7),
		1);

	cv::adaptiveThreshold(matBlurred_red,
		matThresh_red,
		255,
		cv::ADAPTIVE_THRESH_GAUSSIAN_C,
		cv::THRESH_BINARY_INV,
		11,
		2);

	matThreshCopy_red = matThresh_red.clone();

	std::vector<std::vector<cv::Point> > ptContours_red;        // declare a vector for the contours
	std::vector<cv::Vec4i> v4iHierarchy_red;                    // declare a vector for the hierarchy (we won't use this in this program but this may be helpful for reference)

	cv::findContours(matThreshCopy_red,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
		ptContours_red,                             // output contours
		v4iHierarchy_red,                           // output hierarchy
		cv::RETR_EXTERNAL,							// retrieve the outermost contours only
		cv::CHAIN_APPROX_SIMPLE);					// compress horizontal, vertical, and diagonal segments and leave only their end points

	for (int i = 0; i < ptContours_red.size(); i++) {               // for each contour
		ContourWithData contourWithData_red;                                                    // instantiate a contour with data object
		contourWithData_red.ptContour = ptContours_red[i];                                          // assign contour to contour with data
		contourWithData_red.boundingRect = cv::boundingRect(contourWithData_red.ptContour);         // get the bounding rect
		contourWithData_red.fltArea = cv::contourArea(contourWithData_red.ptContour);               // calculate the contour area
		allContoursWithData_red.push_back(contourWithData_red);                                     // add contour with data object to list of all contours with data
	}

	for (int i = 0; i < allContoursWithData_red.size(); i++) {                      // for all contours
		if (allContoursWithData_red[i].checkIfContourIsValid()) {                   // check if valid
			nColorRedStu = 1;
			return true;
		}
	}
	return false;
}
cv::Mat Perspective(cv::Mat image)
{
	cv::Mat lambda(2, 4, CV_32FC1);
	cv::Mat input, output;
	cv::Point2f inputQuad[4];
	cv::Point2f outputQuad[4];
	input = image.clone();
	inputQuad[0] = cv::Point2f(1260, 0);
	inputQuad[1] = cv::Point2f(1260, 675);
	inputQuad[2] = cv::Point2f(0, 673);
	inputQuad[3] = cv::Point2f(0, 0);

	outputQuad[0] = cv::Point2f(1260, 200);
	outputQuad[1] = cv::Point2f(1260, 875);
	outputQuad[2] = cv::Point2f(0, 670);
	outputQuad[3] = cv::Point2f(0, 0);

	// Get the Perspective Transform Matrix i.e. lambda 
	lambda = getPerspectiveTransform(inputQuad, outputQuad);
	// Apply the Perspective Transform just found to the src image
	warpPerspective(input, output, lambda, output.size());

	cv::Rect RectangleToSelect(250, 470, 800, 130);
	output = output(RectangleToSelect);
	return output;
}

int main() {
	std::cout << "\n\n" << "Welcome to Slab Number Detection - CSP Level 2 Team - by Mariana Guimaraes \n\n";
	for (size_t i = 1; i < 13; i++)
	{
		short nColorRedStu = 0;

		cv::Mat matClassificationInts;      // we will read the classification numbers into this variable as though it is a vector

		cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);        // open the classifications file

		if (fsClassifications.isOpened() == false) {                                                    // if the file was not opened successfully
			std::cout << "error, unable to open training classifications file, exiting program\n\n";    // show error message
			return(0);                                                                                  // and exit program
		}

		fsClassifications["classifications"] >> matClassificationInts;      // read classifications section into Mat classifications variable
		fsClassifications.release();                                        // close the classifications file
																			// read in training images ////////////////////////////////////////////////////////////

		cv::Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector

		cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);          // open the training images file

		if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
			std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
			return(0);                                                                              // and exit program
		}

		fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // read images section into Mat training images variable
		fsTrainingImages.release();                                                 // close the traning images file

																					// train //////////////////////////////////////////////////////////////////////////////

		cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());            // instantiate the KNN object

																					// finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
																					// even though in reality they are multiple images / numbers
		kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);
		cv::Mat slabImage;


		switch (i)
		{
		case 1:
			slabImage = cv::imread("placa17544521.png");            // read in the test numbers image
			break;
		case 2:
			slabImage = cv::imread("placa17467611.png");            // read in the test numbers image
			break;
		case 3:
			slabImage = cv::imread("placa17242071.png");            // read in the test numbers image
			break;
		case 4:
			slabImage = cv::imread("placa17236021.png");            // read in the test numbers image
			break;
		case 5:
			slabImage = cv::imread("placa17003031.png");            // read in the test numbers image
			break;
		case 6:
			slabImage = cv::imread("placa17534051.png");            // read in the test numbers image
			break;
		case 7:
			slabImage = cv::imread("placa17534551.png");            // read in the test numbers image
			break;
		case 8:
			slabImage = cv::imread("placa17534451.png");            // read in the test numbers image
			break;
		case 9:
			slabImage = cv::imread("placa17543541.png");            // read in the test numbers image
			break;
		case 10:
			slabImage = cv::imread("placa17543531.png");            // read in the test numbers image
			break;
		case 11:
			slabImage = cv::imread("placa17543511.png");            // read in the test numbers image
			break;
		case 12:
			slabImage = cv::imread("placa17542541.png");            // read in the test numbers image
			break;
		default:
			break;
		}




		if (slabImage.empty()) {                                // if unable to open image
			std::cout << "error: image not read from file\n\n";         // show error message on command line
			return(0);                                                  // and exit program
		}


		cv::Mat matGrayscale;
		cv::Mat matBlurred;
		cv::Mat matThresh;
		cv::Mat matThreshCopy;
		cv::Mat imgThresholded;
		cv::Mat imgHSV;
		cv::Mat output;

		short HSVSetPoint = 0;

		output = Perspective(slabImage);

		if (checkRed(output))
		{
			HSVSetPoint = 150;
		}
		else
		{
			HSVSetPoint = 95;
		}
		cv::cvtColor(output, output, CV_BGR2GRAY);
		cv::cvtColor(output, output, CV_GRAY2BGR);
		cv::cvtColor(output, imgHSV, CV_BGR2HSV);

		for (size_t b = 0; b < 15; b++)
		{
			std::vector<ContourWithData> allContoursWithData;           // declare empty vectors,
			std::vector<ContourWithData> validContoursWithData;         // we will fill these shortly

			int contouHeight[300] = { 0 };														// read in training classifications ///////////////////////////////////////////////////
			int contouWgt[300] = { 0 };

			HSVSetPoint = HSVSetPoint + 10;
			cv::inRange(imgHSV, cv::Scalar(0, 0, 0), cv::Scalar(179, 255, HSVSetPoint), imgThresholded);


			cv::GaussianBlur(imgThresholded,            // input image
				matBlurred,								// output image
				cv::Size(7, 7),							// smoothing window width and height in pixels
				1);										// sigma value, determines how much the image will be blurred, zero makes function choose the sigma value
														// filter image from grayscale to black and white

			cv::adaptiveThreshold(matBlurred,           // input image
				matThresh,								// output image
				255,									// make pixels that pass the threshold full white
				cv::ADAPTIVE_THRESH_GAUSSIAN_C,			// use gaussian rather than mean, seems to give better results
				cv::THRESH_BINARY_INV,					 // invert so foreground will be white, background will be black
				11,										// size of a pixel neighborhood used to calculate threshold value
				2);										// constant subtracted from the mean or weighted mean

			matThreshCopy = matThresh.clone();              // make a copy of the thresh image, this in necessary b/c findContours modifies the image


			std::vector<std::vector<cv::Point> > ptContours;        // declare a vector for the contours
			std::vector<cv::Vec4i> v4iHierarchy;                    // declare a vector for the hierarchy (we won't use this in this program but this may be helpful for reference)

			cv::findContours(matThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
				ptContours,                             // output contours
				v4iHierarchy,                           // output hierarchy
				cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
				cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points

			for (int i = 0; i < ptContours.size(); i++) {               // for each contour
				if (i < 105)
				{
					ContourWithData contourWithData;                                                    // instantiate a contour with data object
					contourWithData.ptContour = ptContours[i];                                          // assign contour to contour with data
					contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);         // get the bounding rect
					contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);               // calculate the contour area
					contouHeight[i] = contourWithData.boundingRect.height;
					contouWgt[i] = contourWithData.boundingRect.width;
					allContoursWithData.push_back(contourWithData);
				}
				else
				{
					break;
				}
				// add contour with data object to list of all contours with data
			}

			for (int i = 0; i < allContoursWithData.size(); i++) {                      // for all contours
																						// check if valid
				if (contouHeight[i] > 60 && contouWgt[i] < 100)
				{
					validContoursWithData.push_back(allContoursWithData[i]);            // if so, append to valid contour list
				}


			}
			// sort contours from left to right
			std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

			std::string strFinalString;         // declare final string, this will have the final number sequence by the end of the program

			for (int i = 0; i < validContoursWithData.size(); i++) {            // for each contour

																				// draw a green rect around the current char
				cv::rectangle(output,                            // draw rectangle on original image
					validContoursWithData[i].boundingRect,        // rect to draw
					cv::Scalar(0, 255, 0),                        // green
					2);                                           // thickness

				cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);          // get ROI image of bounding rect

				cv::Mat matROIResized;
				cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage

				cv::Mat matROIFloat;
				matROIResized.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest

				cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

				cv::Mat matCurrentChar(0, 0, CV_32F);

				kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     // finally we can call find_nearest !!!

				float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

				strFinalString = strFinalString + char(int(fltCurrentChar));        // append current char to full string
			}

			 
			if (strFinalString != "")
			{
				std::vector<char> cstr(strFinalString.c_str(), strFinalString.c_str() + strFinalString.size() + 1);
				if ((cstr.capacity() > 10) && (cstr[9] != '\0') && (cstr.capacity() < 20))
				{
					if (cstr[3] == '7' && cstr[2] == '1' && cstr[1] == '0')
					{
						std::cout << "\n\n" << "..:: Slab read number = " << strFinalString << " ::.. Cyclic [" << b << "] \n\n";       // show the full string

						//cv::imshow("matThreshCopy", matThresh);
						//cv::imshow("matBlurred", matBlurred);
						cv::imshow("Original", slabImage);
						cv::imshow("Tratamento", imgThresholded);
						cv::imshow("Leitura", output);

						break;

					}
					else
					{
						std::cout << "\n\n" << "Trying again [" << b << "], charge number less then A017xxx = " << strFinalString << "\n\n";       // show the full string
						cv::imshow("Original", slabImage);
						cv::imshow("Tratamento", imgThresholded);
						cv::imshow("Leitura", output);
						char c = cv::waitKey(0);
						if ('a' == c)
						{
							continue;
						}
					}


				}
				else
				{
					std::cout << "\n\n" << "Trying again [" << b <<"] Slab Number Empty Incorrect = " << strFinalString << "\n\n";
					cv::imshow("Original", slabImage);
					cv::imshow("Tratamento", imgThresholded);
					cv::imshow("Leitura", output);
					char c = cv::waitKey(0);
					if ('a' == c)
					{
						continue;
					}
				}

			}
			else
			{
				//std::cout << "\n\n" << "Try again - Slab Number Empty \n\n";       // show the full string
				//cv::imshow("Original", slabImage);
				//cv::imshow("Tratamento", imgThresholded);
				//cv::imshow("Leitura", output);
				//char c = cv::waitKey(0);
				//if ('a' == c)
				//{
				//	continue;
				//}
			}
			

		}
		char c = cv::waitKey(0);
		if ('n' == c)
		{
			continue;
		}


	}



	cv::waitKey(0);



	// wait for user key press

	return(0);
}


