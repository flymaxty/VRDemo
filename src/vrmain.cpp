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
	cv::Mat sceneImage = cv::imread(argv[1]);

	cv::Ptr<cv::text::ERFilter::Callback> NM1Callback = cv::text::loadClassifierNM1("..\\..\\model\\trained_classifierNM1.xml");
	cv::Ptr<cv::text::ERFilter::Callback> NM2Callback = cv::text::loadClassifierNM2("..\\..\\model\\trained_classifierNM2.xml");
	std::vector<cv::Ptr<cv::text::ERFilter>> er_filter1;
	std::vector<cv::Ptr<cv::text::ERFilter>> er_filter2;

	for (int i = 0; i < 5; i++)
	{
		er_filter1.push_back(cv::text::createERFilterNM1(NM1Callback, 16, 0.00015f, 0.13f, 0.2f, true, 0.1f));
		er_filter2.push_back(cv::text::createERFilterNM2(NM2Callback, 0.5));
	}

	tesseract::TessBaseAPI tess;
	tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
	tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
	tess.SetVariable("tessedit_char_whitelist", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");

	std::vector<cv::Mat> channels;
	cv::text::computeNMChannels(sceneImage, channels);
	std::cout << channels.size() << std::endl;

	std::vector<std::vector<cv::text::ERStat>> regions(channels.size());
	for (uint8_t i = 0; i < regions.size(); i++)
	{
		er_filter1[i]->run(channels[i], regions[i]);
		er_filter2[i]->run(channels[i], regions[i]);
	}

	std::vector<std::vector<cv::Vec2i>> regionGroups;
	std::vector<cv::Rect> groupBoxes;
	erGrouping(sceneImage, channels, regions, regionGroups, groupBoxes, cv::text::ERGROUPING_ORIENTATION_HORIZ);

	struct TextBox
	{
		std::string word;
		cv::Rect rect;
	};

	std::vector<TextBox> textBoxes;
	TextBox tempBox;

	cv::Mat outputImage;
	sceneImage.copyTo(outputImage);
	for (uint16_t x = 0; x < groupBoxes.size(); x++)
	{
		cv::Mat group_img = sceneImage(groupBoxes[x]);
		tess.SetImage((unsigned char*)group_img.data, group_img.cols, group_img.rows, group_img.channels(), group_img.step);
		std::string ss(tess.GetUTF8Text());
		ss.erase(remove(ss.begin(), ss.end(), '\n'), ss.end());

		std::cout << x << " : ";
		std::cout << groupBoxes[x].tl() << ", " << groupBoxes[x].br();
		//std::cout << groupBoxes[0]
		std::cout << ", " << ss << std::endl;

		bool isRepeative = false;
		for (uint16_t i = 0; i < textBoxes.size(); i++)
		{
			if (textBoxes[i].word == ss && textBoxes[i].rect == groupBoxes[i])
			{
				isRepeative = true;
				break;
			}
		}
		if (!isRepeative)
		{
			tempBox.word = ss;
			tempBox.rect = groupBoxes[x];
			textBoxes.push_back(tempBox);
		}

		cv::rectangle(outputImage, groupBoxes[x].tl(), groupBoxes[x].br(), cv::Scalar(0, 255, 255), 2, cv::LineTypes::LINE_AA, 0);
	}

	std::cout << "==================================" << std::endl;
	for (uint16_t xx = 0; xx < textBoxes.size(); xx++)
	{
		std::cout << xx << " : ";
		std::cout << textBoxes[xx].rect.tl() << ", " << textBoxes[xx].rect.br();
		//std::cout << groupBoxes[0]
		std::cout << ", " << textBoxes[xx].word << std::endl;
	}

	imshow("Output Image", outputImage);
	cv::waitKey(0);

	return 0;
}
