#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;

// 用拉普拉斯算子的方差来计算指定路径图像的模糊程度
double getImageBlurriness(const std::string& img_path) {
	Mat img = cv::imread(img_path);
	Mat img2gray, laplacian;
	Scalar mean, stddev;
  
  // imread函数默认读取图像的格式是BGR，需要先把BGR转为灰度图
	cvtColor(img, img2gray, COLOR_BGR2GRAY);
  // 将拉普拉斯算子的计算结果保存到laplacian变量
	Laplacian(img2gray, laplacian, CV_64F);
  // laplacian的均值和标准差
	meanStdDev(laplacian, mean, stddev);
  // 拉普拉斯算子的方差等于标准差的平方
	return stddev[0] * stddev[0];
}
