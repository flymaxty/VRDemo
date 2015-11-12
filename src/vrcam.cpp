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

int main(int argc, const char * argv[])
{
	cv::Ptr<cv::text::ERFilter::Callback> NM1Callback = cv::text::loadClassifierNM1("..\\..\\model\\trained_classifierNM1.xml");
	cv::Ptr<cv::text::ERFilter::Callback> NM2Callback = cv::text::loadClassifierNM2("..\\..\\model\\trained_classifierNM2.xml");
	cv::Ptr<cv::text::ERFilter> er_filter1 = cv::text::createERFilterNM1(NM1Callback, 16, 0.00015f, 0.13f, 0.2f, true, 0.1f);
	cv::Ptr<cv::text::ERFilter> er_filter2 = cv::text::createERFilterNM2(NM2Callback, 0.5);

	tesseract::TessBaseAPI tess;
	tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
	tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
	tess.SetVariable("tessedit_char_whitelist", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");

	cv::VideoCapture cap(0);
	cv::Mat sceneImage, outputImage;
	std::vector<cv::Mat> channels;
	std::vector<std::vector<cv::text::ERStat>> regions;
	std::vector<std::vector<cv::Vec2i>> regionGroups;
	std::vector<cv::Rect> groupBoxes;

	struct TextBox
	{
		std::string word;
		cv::Rect rect;
	};

	std::vector<TextBox> textBoxes;
	TextBox tempBox;

	cv::Mat wordImage;
	std::string wordString;
	bool isRepeative;
	while (1)
	{
		cap >> sceneImage;
		cv::text::computeNMChannels(sceneImage, channels);

		regions.clear();
		regions.resize(channels.size());

		for (uint8_t i = 0; i < regions.size(); i++)
		{
			er_filter1->run(channels[i], regions[i]);
			er_filter2->run(channels[i], regions[i]);
		}

		regionGroups.clear();
		groupBoxes.clear();
		erGrouping(sceneImage, channels, regions, regionGroups, groupBoxes, cv::text::ERGROUPING_ORIENTATION_HORIZ);

		//std::cout << "=================== Raw Words ===================" << std::endl;
		textBoxes.clear();
		for (uint16_t x = 0; x < groupBoxes.size(); x++)
		{
			wordImage = sceneImage(groupBoxes[x]);
			tess.SetImage((unsigned char*)wordImage.data, wordImage.cols, wordImage.rows, wordImage.channels(), wordImage.step);
			wordString = tess.GetUTF8Text();
			wordString.erase(remove(wordString.begin(), wordString.end(), '\n'), wordString.end());

			//std::cout << x << " : ";
			//std::cout << groupBoxes[x].tl() << ", " << groupBoxes[x].br();
			//std::cout << ", " << wordString << std::endl;

			isRepeative = false;
			for (uint16_t i = 0; i < textBoxes.size(); i++)
			{
				if (textBoxes[i].word == wordString && textBoxes[i].rect == groupBoxes[i])
				{
					isRepeative = true;
					break;
				}
			}
			if (!isRepeative)
			{
				tempBox.word = wordString;
				tempBox.rect = groupBoxes[x];
				textBoxes.push_back(tempBox);
			}
		}

		sceneImage.copyTo(outputImage);
		//std::cout << "===================   Words   ===================" << std::endl;
		for (uint16_t xx = 0; xx < textBoxes.size(); xx++)
		{
			//std::cout << xx << " : ";
			//std::cout << textBoxes[xx].rect.tl() << ", " << textBoxes[xx].rect.br();
			//std::cout << ", " << textBoxes[xx].word << std::endl;

			cv::putText(outputImage, textBoxes[xx].word, textBoxes[xx].rect.tl(), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0));
		}

		imshow("Output Image", outputImage);
		cv::waitKey(1);
	}

	return 0;
}
