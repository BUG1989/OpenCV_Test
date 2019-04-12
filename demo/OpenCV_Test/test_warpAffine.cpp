#include "fbc_cv_funset.hpp"
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <core/mat.hpp>
#include <warpAffine.hpp>

// Blog: http://blog.csdn.net/fengbingchun/article/details/51923392

int test_getAffineTransform()
{
#ifdef _MSC_VER
	cv::Mat matSrc = cv::imread("E:/GitCode/OpenCV_Test/test_images/lena.png", 1);
#else	
	cv::Mat matSrc = cv::imread("test_images/lena.png", 1);
#endif
	if (!matSrc.data) {
		std::cout << "read image fail" << std::endl;
		return -1;
	}

	fbc::Point2f srcTri[3];
	fbc::Point2f dstTri[3];

	// Set your 3 points to calculate the  Affine Transform
	srcTri[0] = fbc::Point2f(0, 0);
	srcTri[1] = fbc::Point2f(matSrc.cols - 1, 0);
	srcTri[2] = fbc::Point2f(0, matSrc.rows - 1);

	dstTri[0] = fbc::Point2f(matSrc.cols*0.0, matSrc.rows*0.33);
	dstTri[1] = fbc::Point2f(matSrc.cols*0.85, matSrc.rows*0.25);
	dstTri[2] = fbc::Point2f(matSrc.cols*0.15, matSrc.rows*0.7);

	// Get the Affine Transform
	fbc::Mat_<double, 1> warp_mat(2, 3);
	int ret = fbc::getAffineTransform(srcTri, dstTri, warp_mat);
	assert(ret == 0);

	cv::Point2f srcTri_[3];
	cv::Point2f dstTri_[3];

	// Set your 3 points to calculate the  Affine Transform
	srcTri_[0] = cv::Point2f(0, 0);
	srcTri_[1] = cv::Point2f(matSrc.cols - 1, 0);
	srcTri_[2] = cv::Point2f(0, matSrc.rows - 1);

	dstTri_[0] = cv::Point2f(matSrc.cols*0.0, matSrc.rows*0.33);
	dstTri_[1] = cv::Point2f(matSrc.cols*0.85, matSrc.rows*0.25);
	dstTri_[2] = cv::Point2f(matSrc.cols*0.15, matSrc.rows*0.7);

	// Get the Affine Transform
	cv::Mat warp_mat_(2, 3, CV_64FC1);
	warp_mat_ = cv::getAffineTransform(srcTri_, dstTri_);

	assert(warp_mat.cols == warp_mat_.cols && warp_mat.rows == warp_mat_.rows);
	assert(warp_mat.step == warp_mat_.step);
	for (int y = 0; y < warp_mat.rows; y++) {
		const fbc::uchar* p = warp_mat.ptr(y);
		const uchar* p_ = warp_mat_.ptr(y);

		for (int x = 0; x < warp_mat.step; x++) {
			assert(p[x] == p_[x]);
		}
	}

	return 0;
}

int test_warpAffine_uchar()
{
#ifdef _MSC_VER
	cv::Mat matSrc = cv::imread("E:/GitCode/OpenCV_Test/test_images/lena.png", 1);
#else	
	cv::Mat matSrc = cv::imread("test_images/lena.png", 1);
#endif
	if (!matSrc.data) {
		std::cout << "read image fail" << std::endl;
		return -1;
	}

	for (int interpolation = 0; interpolation < 5; interpolation++) {
		fbc::Point2f srcTri[3];
		fbc::Point2f dstTri[3];

		// Set your 3 points to calculate the  Affine Transform
		srcTri[0] = fbc::Point2f(0, 0);
		srcTri[1] = fbc::Point2f(matSrc.cols - 1, 0);
		srcTri[2] = fbc::Point2f(0, matSrc.rows - 1);

		dstTri[0] = fbc::Point2f(matSrc.cols*0.0, matSrc.rows*0.33);
		dstTri[1] = fbc::Point2f(matSrc.cols*0.85, matSrc.rows*0.25);
		dstTri[2] = fbc::Point2f(matSrc.cols*0.15, matSrc.rows*0.7);

		// Get the Affine Transform
		fbc::Mat_<double, 1> warp_mat(2, 3);
		int ret = fbc::getAffineTransform(srcTri, dstTri, warp_mat);
		assert(ret == 0);

		fbc::Mat_<uchar, 3> mat(matSrc.rows, matSrc.cols, matSrc.data);
		fbc::Mat_<uchar, 3> warp_dst;
		warp_dst.zeros(mat.rows, mat.cols);
		fbc::warpAffine(mat, warp_dst, warp_mat, interpolation);

		cv::Point2f srcTri_[3];
		cv::Point2f dstTri_[3];

		// Set your 3 points to calculate the  Affine Transform
		srcTri_[0] = cv::Point2f(0, 0);
		srcTri_[1] = cv::Point2f(matSrc.cols - 1, 0);
		srcTri_[2] = cv::Point2f(0, matSrc.rows - 1);

		dstTri_[0] = cv::Point2f(matSrc.cols*0.0, matSrc.rows*0.33);
		dstTri_[1] = cv::Point2f(matSrc.cols*0.85, matSrc.rows*0.25);
		dstTri_[2] = cv::Point2f(matSrc.cols*0.15, matSrc.rows*0.7);

		// Get the Affine Transform
		cv::Mat warp_mat_(2, 3, CV_64FC1);
		warp_mat_ = cv::getAffineTransform(srcTri_, dstTri_);

		// Set the dst image the same type and size as src
		cv::Mat warp_dst_ = cv::Mat::zeros(matSrc.rows, matSrc.cols, matSrc.type());
		cv::Mat mat_;
		matSrc.copyTo(mat_);

		// Apply the Affine Transform just found to the src image
		cv::warpAffine(mat_, warp_dst_, warp_mat_, warp_dst_.size(), interpolation);

		assert(warp_mat.cols == warp_mat_.cols && warp_mat.rows == warp_mat_.rows);
		assert(warp_mat.step == warp_mat_.step);
		for (int y = 0; y < warp_mat.rows; y++) {
			const fbc::uchar* p = warp_dst.ptr(y);
			const uchar* p_ = warp_dst_.ptr(y);

			for (int x = 0; x < warp_mat.step; x++) {
				assert(p[x] == p_[x]);
			}
		}
	}

	return 0;
}

int test_warpAffine_float()
{
#ifdef _MSC_VER
	cv::Mat matSrc = cv::imread("E:/GitCode/OpenCV_Test/test_images/lena.png", 1);
#else	
	cv::Mat matSrc = cv::imread("test_images/lena.png", 1);
#endif
	if (!matSrc.data) {
		std::cout << "read image fail" << std::endl;
		return -1;
	}
	cv::cvtColor(matSrc, matSrc, CV_BGR2GRAY);
	matSrc.convertTo(matSrc, CV_32FC1);

	for (int interpolation = 0; interpolation < 5; interpolation++) {
		fbc::Point2f srcTri[3];
		fbc::Point2f dstTri[3];

		// Set your 3 points to calculate the  Affine Transform
		srcTri[0] = fbc::Point2f(0, 0);
		srcTri[1] = fbc::Point2f(matSrc.cols - 1, 0);
		srcTri[2] = fbc::Point2f(0, matSrc.rows - 1);

		dstTri[0] = fbc::Point2f(matSrc.cols*0.0, matSrc.rows*0.33);
		dstTri[1] = fbc::Point2f(matSrc.cols*0.85, matSrc.rows*0.25);
		dstTri[2] = fbc::Point2f(matSrc.cols*0.15, matSrc.rows*0.7);

		// Get the Affine Transform
		fbc::Mat_<double, 1> warp_mat(2, 3);
		int ret = fbc::getAffineTransform(srcTri, dstTri, warp_mat);
		assert(ret == 0);

		fbc::Mat_<float, 1> mat(matSrc.rows, matSrc.cols, matSrc.data);
		fbc::Mat_<float, 1> warp_dst;
		warp_dst.zeros(mat.rows, mat.cols);

		fbc::warpAffine(mat, warp_dst, warp_mat, interpolation);

		cv::Point2f srcTri_[3];
		cv::Point2f dstTri_[3];

		// Set your 3 points to calculate the  Affine Transform
		srcTri_[0] = cv::Point2f(0, 0);
		srcTri_[1] = cv::Point2f(matSrc.cols - 1, 0);
		srcTri_[2] = cv::Point2f(0, matSrc.rows - 1);

		dstTri_[0] = cv::Point2f(matSrc.cols*0.0, matSrc.rows*0.33);
		dstTri_[1] = cv::Point2f(matSrc.cols*0.85, matSrc.rows*0.25);
		dstTri_[2] = cv::Point2f(matSrc.cols*0.15, matSrc.rows*0.7);

		// Get the Affine Transform
		cv::Mat warp_mat_(2, 3, CV_64FC1);
		warp_mat_ = cv::getAffineTransform(srcTri_, dstTri_);

		// Set the dst image the same type and size as src
		cv::Mat warp_dst_ = cv::Mat::zeros(matSrc.rows, matSrc.cols, matSrc.type());
		cv::Mat mat_;
		matSrc.copyTo(mat_);

		// Apply the Affine Transform just found to the src image
		cv::warpAffine(mat_, warp_dst_, warp_mat_, warp_dst_.size(), interpolation);

		assert(warp_mat.cols == warp_mat_.cols && warp_mat.rows == warp_mat_.rows);
		assert(warp_mat.step == warp_mat_.step);
		for (int y = 0; y < warp_mat.rows; y++) {
			const fbc::uchar* p = warp_mat.ptr(y);
			const uchar* p_ = warp_mat_.ptr(y);

			for (int x = 0; x < warp_mat.step; x++) {
				assert(p[x] == p_[x]);
			}
		}
	}

	return 0;
}

#include <sys/time.h>
#include <stdio.h>
#include <string>

static double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void save_mat_u8c3(const char *path, const cv::Mat& src_mat)
{
	cv::Mat dst = src_mat;

	printf("save the mat to: %s\n", path);

	FILE *pf = fopen(path, "w");
	for (int y = 0; y < dst.rows; y++) 
	{
		uchar *pline = dst.ptr(y);
		for (int x = 0; x < dst.cols*3; x++) 
		{
			fprintf(pf, "%02X ", pline[x]);
		}
		fprintf(pf, "\n");
	}
	fclose(pf);
}

static void save_mat_u8c3(const char *path, const fbc::Mat_<uchar, 3>& src_mat)
{
	printf("save the mat to: %s\n", path);

	FILE *pf = fopen(path, "w");
	for (int y = 0; y < src_mat.rows; y++) 
	{
		const uchar *pline = src_mat.ptr(y);
		for (int x = 0; x < src_mat.cols*3; x++) 
		{
			fprintf(pf, "%02X ", pline[x]);
		}
		fprintf(pf, "\n");
	}
	fclose(pf);
}

int test_warpAffine_tq()
{
	cv::Mat matSrc = cv::imread("sao.jpg");
	if (!matSrc.data) {
		std::cout << "read image fail" << std::endl;
		return -1;
	}

	// Test with dg_cv API
	cv::Point center = cv::Point(matSrc.cols/2, matSrc.rows/2);
	double angle = 45.0;
	double scale = 0.7;

	cv::Mat warp_mat_; // (2, 3, CV_32FC1);
	warp_mat_ = cv::getRotationMatrix2D(center, angle, scale); //	

	fbc::Mat_<double, 1> warp_mat(warp_mat_.rows, warp_mat_.cols, warp_mat_.data);
	fbc::Mat_<uchar, 3> mat(matSrc.rows, matSrc.cols, matSrc.data);
	fbc::Mat_<uchar, 3> warp_dst;
	warp_dst.zeros(mat.rows, mat.cols);

	double start = get_current_time();
	fbc::warpAffine(mat, warp_dst, warp_mat);
	double end = get_current_time();
	printf("dg_cv cost time     : %8.3f ms\n", end - start);


	save_mat_u8c3("result_tq.txt", warp_dst);
	

	cv::Mat warp_dst_cv(warp_dst.rows, warp_dst.cols, CV_8UC3, warp_dst.data);
	cv::imwrite("result_sao_tq.jpg", warp_dst_cv);
	// cv::imshow("test", warp_dst_cv);
	// cv::waitKey(0);
#if 1
	// Test with OpenCV API
	// Get the Affine Transform
	// Set the dst image the same type and size as src
	cv::Mat warp_dst_ = cv::Mat::zeros(matSrc.rows, matSrc.cols, matSrc.type());
	cv::Mat mat_;
	matSrc.copyTo(mat_);

	// Apply the Affine Transform just found to the src image
	start = get_current_time();
	cv::warpAffine(mat_, warp_dst_, warp_mat_, warp_dst_.size());	
	end = get_current_time();
	printf("opencv_cv cost time : %8.3f ms\n", end - start);	

	// cv::imshow("test", warp_dst_);
	// cv::waitKey(0);

	save_mat_u8c3("result_cv.txt", warp_dst_);
	cv::imwrite("result_sao_cv.jpg", warp_dst_);
#endif
	return 0;
}