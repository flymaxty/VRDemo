#include <iostream>
#include <string>
#include <vector>
#include <inttypes.h>

#include "opencv2/opencv.hpp"
#include "opencv2/text.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include "baseapi.h"
#include "strngs.h"


//ERStat extraction is done in parallel for different channels
class Parallel_extractCSER : public cv::ParallelLoopBody
{
private:
	std::vector<cv::Mat> &channels;
	std::vector<std::vector<cv::text::ERStat>> &regions;
	std::vector<cv::Ptr<cv::text::ERFilter>> er_filter1;
	std::vector<cv::Ptr<cv::text::ERFilter>> er_filter2;

public:
	Parallel_extractCSER(std::vector<cv::Mat> &_channels, std::vector<std::vector<cv::text::ERStat> > &_regions,
		std::vector<cv::Ptr<cv::text::ERFilter> >_er_filter1, std::vector<cv::Ptr<cv::text::ERFilter> >_er_filter2)
		: channels(_channels), regions(_regions), er_filter1(_er_filter1), er_filter2(_er_filter2){}

	virtual void operator()(const cv::Range &r) const
	{
		for (int c = r.start; c < r.end; c++)
		{
			er_filter1[c]->run(channels[c], regions[c]);
			er_filter2[c]->run(channels[c], regions[c]);
		}
	}
	Parallel_extractCSER & operator=(const Parallel_extractCSER &a);
};


struct RectData{
	std::string name;
	std::vector<cv::Point2f> vertexes;
};

struct ImageData{
	cv::Mat image;
	std::string name;
	std::vector<cv::Point2f> vertexes;
};

struct TextBox
{
	bool hasRect;
	std::string word;
	cv::Rect rect;
};

cv::VideoCapture cap;										//Camera
cv::Mat sceneImage, step1Image, outputImage;				//Input and output image

//Rectangle detection
double area;
std::vector<cv::Point> poly;
std::vector<std::vector<cv::Point>> contours;
cv::Mat grayImage, thresholdImage, cannyImage;

cv::Mat transH;
RectData tmpImageRect;
std::vector <RectData> showRect;							//Filterd contours
std::map<std::string, ImageData> imageData;					//Stored Image

//Text detection
std::vector<cv::Mat> channels;								//Split channels
std::vector<cv::Ptr<cv::text::ERFilter>> er_filters1;		//ERFilters1
std::vector<cv::Ptr<cv::text::ERFilter>> er_filters2;		//ERFilters2
std::vector<std::vector<cv::text::ERStat>> regions;			//Recognized letters

//Word group
bool isRepeative;
cv::Mat tmpWordImage;
std::string tmpWordString;
std::vector<TextBox> textBoxes;								//Recognized word struct list
std::vector<cv::Rect> groupBoxes;							//Recognized word list
std::vector<std::vector<cv::Vec2i>> regionGroups;			//Recognized letters grouped by words index

//Tesseract api
tesseract::TessBaseAPI tess;

void initOCR()
{
	std::cout << "Init OCR......";
	for (uint8_t i = 0; i < 6; i++)
	{
		cv::Ptr<cv::text::ERFilter::Callback> NM1Callback = cv::text::loadClassifierNM1("..\\..\\model\\trained_classifierNM1.xml");
		cv::Ptr<cv::text::ERFilter::Callback> NM2Callback = cv::text::loadClassifierNM2("..\\..\\model\\trained_classifierNM2.xml");
		cv::Ptr<cv::text::ERFilter> er_filter1 = cv::text::createERFilterNM1(NM1Callback, 16, 0.00015f, 0.13f, 0.2f, true, 0.1f);
		cv::Ptr<cv::text::ERFilter> er_filter2 = cv::text::createERFilterNM2(NM2Callback, 0.5);
		er_filters1.push_back(er_filter1);
		er_filters2.push_back(er_filter2);
	}

	tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
	tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
	tess.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
	std::cout << "Done!" << std::endl;
}

void loadImage()
{
	std::cout << "Load Image......";
	imageData["GUNDAM"].image = cv::imread("..\\..\\image\\GUNDAM.jpg");
	imageData["GUNDAM"].vertexes.resize(4);
	imageData["GUNDAM"].vertexes[0] = cv::Point2f(0, 0);
	imageData["GUNDAM"].vertexes[1] = cv::Point2f(0, imageData["GUNDAM"].image.rows);
	imageData["GUNDAM"].vertexes[2] = cv::Point2f(imageData["GUNDAM"].image.cols, imageData["GUNDAM"].image.rows);
	imageData["GUNDAM"].vertexes[3] = cv::Point2f(imageData["GUNDAM"].image.cols, 0);

	imageData["MIKU"].image = cv::imread("..\\..\\image\\MIKU.jpg");
	imageData["MIKU"].vertexes.resize(4);
	imageData["MIKU"].vertexes[0] = cv::Point2f(0, 0);
	imageData["MIKU"].vertexes[1] = cv::Point2f(0, imageData["MIKU"].image.rows);
	imageData["MIKU"].vertexes[2] = cv::Point2f(imageData["MIKU"].image.cols, imageData["MIKU"].image.rows);
	imageData["MIKU"].vertexes[3] = cv::Point2f(imageData["MIKU"].image.cols, 0);

	imageData["YUI"].image = cv::imread("..\\..\\image\\YUI.jpg");
	imageData["YUI"].vertexes.resize(4);
	imageData["YUI"].vertexes[0] = cv::Point2f(0, 0);
	imageData["YUI"].vertexes[1] = cv::Point2f(0, imageData["YUI"].image.rows);
	imageData["YUI"].vertexes[2] = cv::Point2f(imageData["YUI"].image.cols, imageData["YUI"].image.rows);
	imageData["YUI"].vertexes[3] = cv::Point2f(imageData["YUI"].image.cols, 0);

	imageData["UNKNOWN"].image = cv::imread("..\\..\\image\\UNKNOWN.jpg");
	imageData["UNKNOWN"].vertexes.resize(4);
	imageData["UNKNOWN"].vertexes[0] = cv::Point2f(0, 0);
	imageData["UNKNOWN"].vertexes[1] = cv::Point2f(0, imageData["UNKNOWN"].image.rows);
	imageData["UNKNOWN"].vertexes[2] = cv::Point2f(imageData["UNKNOWN"].image.cols, imageData["UNKNOWN"].image.rows);
	imageData["UNKNOWN"].vertexes[3] = cv::Point2f(imageData["UNKNOWN"].image.cols, 0);

	/*std::map<std::string, ImageData>::iterator it;
	for (it = imageData.begin(); it != imageData.end(); it++)
	{
		cv::imshow(it->first, it->second.image);;
	}

	cv::waitKey(0);*/
	std::cout << "Done!" << std::endl;
}

void findOutputRect()
{
	tmpImageRect.vertexes.resize(4);
	showRect.clear();

	cv::cvtColor(sceneImage, grayImage, cv::COLOR_BGR2GRAY);
	grayImage.convertTo(grayImage, CV_8UC1);
	cv::Canny(grayImage, cannyImage, 150, 200);

	cv::findContours(cannyImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat testImage;
	outputImage.copyTo(testImage);
	testImage.setTo(0);
	for (uint16_t i = 0; i<contours.size(); i++)
	{
		area = fabs(cv::contourArea(contours[i]));
		if (area > 1000 && !cv::isContourConvex(contours[i]))
		{
			cv::approxPolyDP(contours[i], poly, 5, 1);
			if (poly.size() == 4)
			{
				if (poly[3].y > poly[1].y)
				{
					cv::Point2f tmpPoint = poly[0];
					poly.erase(poly.begin());
					poly.push_back(tmpPoint);
				}

				for (uint8_t jj = 0; jj < 4; jj++)
				{
					cv::circle(outputImage, poly[jj], 5, cv::Scalar(150, 0, 0), cv::FILLED, cv::LineTypes::LINE_AA);
					tmpImageRect.vertexes[jj].x = poly[jj].x;
					tmpImageRect.vertexes[jj].y = poly[jj].y;
				}

				tmpImageRect.name = "UNKNOWN";
				showRect.push_back(tmpImageRect);
				//cv::drawContours(outputImage, contours, i, cv::Scalar(0, 255, 255), cv::LineTypes::FILLED);
				//tmpImageRect.transH = cv::getPerspectiveTransform(imageData["GUNDAM"].vertexes, tmpImageRect.vertexes);
				//cv::warpPerspective(imageData["GUNDAM"].image, tmpImage, transH, outputImage.size());
				//tmpImage.copyTo(maskImage);
				//cv::cvtColor(maskImage, maskImage, CV_BGR2GRAY);
				//tmpImage.copyTo(outputImage, maskImage);
			}
		}
		cv::drawContours(testImage, contours, i, cv::Scalar(0, 255, 255));
	}
	cv::imshow("testImage", testImage);
}

void matching()
{
	double shortestDistance = 99999, tmpDistance;
	uint16_t shortestIndex = 0;
	for (uint16_t i = 0; i < textBoxes.size(); i++)
	{
		if (imageData.count(textBoxes[i].word))
		{
			shortestDistance = 99999;
			for (uint16_t j = 0; j < showRect.size(); j++)
			{
				for (uint8_t x = 0; x < 4; x++)
				{
					tmpDistance = cv::norm(cv::Point2f(textBoxes[i].rect.tl()) - showRect[j].vertexes[x]);
					if (tmpDistance < shortestDistance)
					{
						shortestDistance = tmpDistance;
						shortestIndex = j;
					}
				}
			}

			std::vector<RectData>::iterator it;
			if (showRect[shortestIndex].name == "UNKNOWN")
			{
				std::cout << "set name: " << textBoxes[i].word << std::endl;
				showRect[shortestIndex].name = textBoxes[i].word;
			}
		}
	}
}

void showing()
{
	cv::Mat transH;
	cv::Mat tmpImage, maskImage;

	std::vector<RectData>::iterator it;
	for (it = showRect.begin(); it != showRect.end(); it++)
	{
		transH = cv::getPerspectiveTransform(imageData[it->name].vertexes, it->vertexes);
		cv::warpPerspective(imageData[it->name].image, tmpImage, transH, outputImage.size());
		tmpImage.copyTo(maskImage);
		cv::cvtColor(maskImage, maskImage, CV_BGR2GRAY);
		tmpImage.copyTo(outputImage, maskImage);
	}
}

void doOCR()
{
	cv::text::computeNMChannels(sceneImage, channels);
	regions.clear();
	regions.resize(channels.size());

	parallel_for_(cv::Range(0, (int)channels.size()), Parallel_extractCSER(channels, regions, er_filters1, er_filters2));

	regionGroups.clear();
	groupBoxes.clear();
	erGrouping(sceneImage, channels, regions, regionGroups, groupBoxes, cv::text::ERGROUPING_ORIENTATION_HORIZ);

	textBoxes.clear();
	for (uint16_t x = 0; x < groupBoxes.size(); x++)
	{
		TextBox tempBox;
		tempBox.rect = groupBoxes[x];

		if (tempBox.rect.x < 0)
			tempBox.rect.x = 0;
		if (tempBox.rect.y < 0)
			tempBox.rect.y = 0;
		if (tempBox.rect.br().x > sceneImage.cols)
			tempBox.rect.width = sceneImage.cols - tempBox.rect.x;
		if (tempBox.rect.br().y> sceneImage.rows)
			tempBox.rect.height = sceneImage.rows - tempBox.rect.y;

		tmpWordImage = sceneImage(tempBox.rect);
		tess.SetImage((unsigned char*)tmpWordImage.data, tmpWordImage.cols, tmpWordImage.rows, tmpWordImage.channels(), tmpWordImage.step);
		tmpWordString = tess.GetUTF8Text();
		tmpWordString.erase(remove(tmpWordString.begin(), tmpWordString.end(), '\n'), tmpWordString.end());
		tempBox.word = tmpWordString;
		tempBox.hasRect = false;

		isRepeative = false;
		for (uint16_t i = 0; i < textBoxes.size(); i++)
		{
			if (textBoxes[i].word == tmpWordString && textBoxes[i].rect == tempBox.rect)
			{
				isRepeative = true;
				break;
			}
		}
		if (!isRepeative)
		{
			textBoxes.push_back(tempBox);
		}
	}

	std::cout << "===================   Words   ===================" << std::endl;
	for (uint16_t xx = 0; xx < textBoxes.size(); xx++)
	{
		std::cout << xx << " : ";
		std::cout << textBoxes[xx].rect.tl() << ", " << textBoxes[xx].rect.br();
		std::cout << ", " << textBoxes[xx].word << std::endl;

		cv::Point tmpPoint(textBoxes[xx].rect.tl().x, textBoxes[xx].rect.br().y+20);
		cv::putText(outputImage, textBoxes[xx].word, tmpPoint,
			cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2, cv::LineTypes::LINE_AA);
	}
}

int main(int argc, const char * argv[])
{
	cap.open(1);
	initOCR();
	loadImage();
	while (1)
	{
		cap >> sceneImage;
		//Get mirror image
		cv::flip(sceneImage, sceneImage, 0);
		cv::flip(sceneImage, sceneImage, 1);
		sceneImage.copyTo(outputImage);

		//Detect vaild Rectangle
		findOutputRect();
		//OCR
		doOCR();
		//Get match between words an rectangle
		matching();
		//Show the result
		showing();

		imshow("Output Image", outputImage);
		cv::waitKey(1);
	}

	return 0;
}
