#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <book.h>
#include <omp.h>
#include <cmath>
#include <iostream>
#include <string>

#define START_CPU \
	double start = omp_get_wtime();
#define END_CPU \
	double stop = omp_get_wtime(); \
	printf("CPU Time used: %3.1f ms\n", (stop - start) * 1000);

#define START_GPU {\
	cudaEvent_t start, stop; \
	float elapsedTime; \
	HANDLE_ERROR(cudaEventCreate(&start)); \
	HANDLE_ERROR(cudaEventCreate(&stop)); \
	HANDLE_ERROR(cudaEventRecord(start, 0));

#define END_GPU \
	HANDLE_ERROR(cudaEventRecord(stop, 0)); \
	HANDLE_ERROR(cudaEventSynchronize(stop)); \
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop)); \
	printf("GPU Time used: %3.1f ms\n", elapsedTime); \
	HANDLE_ERROR(cudaEventDestroy(start)); \
	HANDLE_ERROR(cudaEventDestroy(stop)); }

//#define GPU
//using namespace cv;
//using namespace std;
typedef unsigned char uchar;
const double EPS = 0.01;
const int WINDOW = 15;
const int MARGIN = WINDOW * 3;
const int SIGMA = 12;
const int FEATURES_NUM = 100;
float bilinear_interpolation(cv::Mat m, float u0, float v0) {
	int u = (int)u0, v = (int)v0;
	float a = u0 - u, b = v0 - v;
	float f1 = m.at<float>(v0, u0), f2 = m.at<float>(v0, u0 + 1), f3 = m.at<float>(v0 + 1, u0), f4 = m.at <float>(v0 + 1, u0 + 1);
	float f5 = f1 + a * (f2 - f1), f6 = f3 + a * (f4 - f3);
	float f7 = f5 + b * (f6 - f5);
	return f7;
}
struct matrix {
	int rows;
	int cols;
	float *data;
	matrix(int rows = 0, int cols = 0) : rows(rows), cols(cols) {
		data = (float*)malloc(rows * cols * sizeof(float));
		memset(data, 0, sizeof(float)* rows * cols);
	}
};
struct k_point {
	int x, y;
	float min_value;
	float dx, dy;
	bool flag;
	bool operator < (const k_point &other) const {
		return min_value > other.min_value;
	}
	void print() {
		printf("(%d %d), %lf, (%lf, %lf)", x, y, min_value, dx, dy);
		puts(flag ? "" : "***");
	}
};

bool is_valid(const k_point &p, const std::vector<k_point> &ps) {
	for (int i = 0; i < ps.size(); ++i) {
		if (sqrt((p.x - ps[i].x) * (p.x - ps[i].x) + (p.y - ps[i].y) * (p.y - ps[i].y)) < WINDOW) {
			return false;
		}
	}
	return true;
}
__global__ void kernel(float *dev_i, float *dev_j, float *dev_gx, float *dev_gy, int height, int width, float *dev_dx, float *dev_dy, float *dev_z) {
	//printf("ok\n");
	int x = threadIdx.x + blockIdx.x * 15;
	int y = threadIdx.y + blockIdx.y * 15;
	if (x < MARGIN || y < MARGIN || width - 15 - x < MARGIN || height - 15 - y < MARGIN) {
		//dev_z[y*width + x] = 0;
		return;
	}
	__shared__ float temp_i[29][29];
	__shared__ float temp_j[29][29];
	__shared__ float temp_gx[29][29];
	__shared__ float temp_gy[29][29];
	temp_i[threadIdx.y][threadIdx.x] = dev_i[y * width + x];
	temp_j[threadIdx.y][threadIdx.x] = dev_j[y * width + x];
	temp_gx[threadIdx.y][threadIdx.x] = dev_gx[y * width + x];
	temp_gy[threadIdx.y][threadIdx.x] = dev_gy[y * width + x];
	if (threadIdx.x >= 15 || threadIdx.y >= 15) {
		return;
	}
	__syncthreads();
	float z00 = 0, z01 = 0, z10 = 0, z11 = 0;
	float e00 = 0, e10 = 0;
	for (int k = 0; k < WINDOW; ++k) {
		for (int l = 0; l < WINDOW; ++l) {
			float g00 = temp_gx[threadIdx.y + k][threadIdx.x + l];
			float g10 = temp_gy[threadIdx.y + k][threadIdx.x + l];
			z00 += g00 * g00;
			z10 += g00 * g10;
			z01 += g00 * g10;
			z11 += g10 * g10;
			e00 += (temp_i[threadIdx.y + k][threadIdx.x + l] - temp_j[threadIdx.y + k][threadIdx.x + l]) * g00;
			e10 += (temp_i[threadIdx.y + k][threadIdx.x + l] - temp_j[threadIdx.y + k][threadIdx.x + l]) * g10;
		}
	}
	float z_values00 = (z00 + z11 + sqrt((z00 - z11)*(z00 - z11) + 4 * z10*z01)) / 2;
	float z_values10 = (z00 + z11 - sqrt((z00 - z11)*(z00 - z11) + 4 * z10*z01)) / 2;
	float z_inv00 = z11 / (z00*z11 - z10*z01);
	float z_inv01 = -z01 / (z00*z11 - z10*z01);
	float z_inv10 = -z10 / (z00*z11 - z10*z01);
	float z_inv11 = z00 / (z00*z11 - z10*z01);
	float d00 = z_inv00*e00 + z_inv01*e10;
	float d10 = z_inv10*e00 + z_inv11*e10;
	dev_dx[y*width + x] = d00;
	dev_dy[y*width + x] = d10;
	dev_z[y*width + x] = z_values00 < z_values10 ? z_values00 : z_values10;
	__syncthreads();
}
int main() {
	std::cout << "SIGMA = " << SIGMA << std::endl;
	cv::Mat img0 = cv::imread("0.jpg", 0);
	cv::Mat img0_show = cv::imread("0.jpg", 1);
	cv::Mat img1 = cv::imread("1.jpg", 0);
	cv::Mat img1_show = cv::imread("1.jpg", 1);
	int width = img0.cols, height = img0.rows;
	cv::Mat ave = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);
	cv::Mat img_i = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);
	cv::Mat img_j = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);
	GaussianBlur(img0, img0, cv::Size(), SIGMA, SIGMA);
	GaussianBlur(img1, img1, cv::Size(), SIGMA, SIGMA);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			img_i.at<float>(i, j) = img0.at<uchar>(i, j);
			img_j.at<float>(i, j) = img1.at<uchar>(i, j);
			ave.at<float>(i, j) = ((float)img0.at<uchar>(i, j) + img1.at<uchar>(i, j)) / 2;
		}
	}
	cv::Mat gx = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);
	cv::Mat gy = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);
	for (int i = 0; i < height - 1; ++i) {
		for (int j = 0; j < width - 1; ++j) {
			gx.at<float>(i, j) = ave.at<float>(i, j + 1) - ave.at<float>(i, j);
			gy.at<float>(i, j) = ave.at<float>(i + 1, j) - ave.at<float>(i, j);
		}
	}
	std::vector<k_point> k_points;
	matrix mi(height, width), mj(height, width), mgx(height, width), mgy(height, width);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			mi.data[i * width + j] = img0.at<uchar>(i, j);
			mj.data[i * width + j] = img1.at<uchar>(i, j);
			mgx.data[i * width + j] = gx.at<float>(i, j);
			mgy.data[i * width + j] = gy.at<float>(i, j);

		}
	}
#ifndef GPU
	START_CPU;

	for (int i = MARGIN; i < height - MARGIN - WINDOW; ++i) {
		for (int j = MARGIN; j < width - MARGIN - WINDOW; ++j) {
			float z00 = 0, z01 = 0, z10 = 0, z11 = 0;
			float e00 = 0, e10 = 0;
			for (int k = 0; k < WINDOW; ++k) {
				for (int l = 0; l < WINDOW; ++l) {
					float g00 = mgx.data[(i + k) * width + j + l];
					float g10 = mgy.data[(i + k) * width + j + l];
					z00 += g00 * g00;
					z10 += g00 * g10;
					z01 += g00 * g10;
					z11 += g10 * g10;
					e00 += (mi.data[(i + k) * width + j + l] - mj.data[(i + k) * width + j + l]) * g00;
					e10 += (mi.data[(i + k) * width + j + l] - mj.data[(i + k) * width + j + l]) * g10;
				}
			}
			float z_values00 = (z00 + z11 + sqrt((z00 - z11)*(z00 - z11) + 4 * z10*z01)) / 2;
			float z_values10 = (z00 + z11 - sqrt((z00 - z11)*(z00 - z11) + 4 * z10*z01)) / 2;
			float z_inv00 = z11 / (z00*z11 - z10*z01);
			float z_inv01 = -z01 / (z00*z11 - z10*z01);
			float z_inv10 = -z10 / (z00*z11 - z10*z01);
			float z_inv11 = z00 / (z00*z11 - z10*z01);
			float d00 = z_inv00*e00 + z_inv01*e10;
			float d10 = z_inv10*e00 + z_inv11*e10;
			k_points.push_back(
				k_point{ i, j,
				z_values00 < z_values10 ? z_values00 : z_values10,
				d00, d10, false });
		}
	}

	END_CPU;
#else
	START_GPU;
	float *dev_dx, *dev_dy, *dev_z;
	cudaMalloc((void**)&dev_dx, height * width * sizeof(float));
	cudaMalloc((void**)&dev_dy, height * width * sizeof(float));
	cudaMalloc((void**)&dev_z, height * width * sizeof(float));
	cudaMemset(dev_z, 0, height * width * sizeof(float));
	float *dev_i, *dev_j, *dev_gx, *dev_gy;
	cudaMalloc((void**)&dev_i, height * width * sizeof(float));
	cudaMalloc((void**)&dev_j, height * width * sizeof(float));
	cudaMemcpy(dev_i, mi.data, height * width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_j, mj.data, height * width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_gx, height * width * sizeof(float));
	cudaMalloc((void**)&dev_gy, height * width * sizeof(float));
	cudaMemcpy(dev_gx, mgx.data, height * width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_gy, mgy.data, height * width * sizeof(float), cudaMemcpyHostToDevice);
	kernel << <dim3(36, 48), dim3(29, 29) >> >(dev_i, dev_j, dev_gx, dev_gy, height, width, dev_dx, dev_dy, dev_z);
	//kernel2 << <dim3(720, 540), dim3(15, 15) >> >(dev_i, dev_j, dev_gx, dev_gy, height, width, dev_dx, dev_dy, dev_z);
	float *host_dx, *host_dy, *host_z;
	host_dx = (float*)malloc(height * width * sizeof(float));
	host_dy = (float*)malloc(height * width * sizeof(float));
	host_z = (float*)malloc(height * width * sizeof(float));
	cudaMemcpy(host_dx, dev_dx, height*width*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_dy, dev_dy, height*width*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_z, dev_z, height*width*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < height * width; ++i) {
		if (host_z[i] != 0) {
			k_points.push_back(
				k_point{ i / width, i % width,
				host_z[i],
				host_dx[i], host_dy[i], false });
		}
	}
	END_GPU;
	//cudaDeviceReset();
#endif
	std::sort(k_points.begin(), k_points.end());
	std::vector<k_point> selected_points;
	for (int i = 0; i < k_points.size(); ++i) {
		if (is_valid(k_points[i], selected_points)) {
			selected_points.push_back(k_points[i]);
		}
		if (selected_points.size() == FEATURES_NUM) {
			break;
		}
	}
	for (int i = 0; i < FEATURES_NUM; ++i) {
		float dx = selected_points[i].dx, dy = selected_points[i].dy;
		int cnt = 0;
		while (++cnt < 10) {
			cv::Mat w1 = cv::Mat::zeros(cv::Size(WINDOW + 1, WINDOW + 1), CV_32FC1);
			cv::Mat w2 = cv::Mat::zeros(cv::Size(WINDOW + 1, WINDOW + 1), CV_32FC1);
			cv::Mat w3 = cv::Mat::zeros(cv::Size(WINDOW + 1, WINDOW + 1), CV_32FC1);
			cv::Mat w_ave = cv::Mat::zeros(cv::Size(WINDOW + 1, WINDOW + 1), CV_32FC1);
			for (int j = 0; j < WINDOW + 1; ++j) {
				for (int k = 0; k < WINDOW + 1; ++k) {
					w1.at<float>(j, k) = bilinear_interpolation(img_i,
						selected_points[i].y + k - selected_points[i].dx,
						selected_points[i].x + j - selected_points[i].dy);
					w2.at<float>(j, k) = img_j.at<float>(selected_points[i].x + j, selected_points[i].y + k);
				}
			}
			w_ave = (w1 + w2) / 2;
			cv::Mat z = cv::Mat::zeros(cv::Size(2, 2), CV_32FC1);
			cv::Mat e = cv::Mat::zeros(cv::Size(1, 2), CV_32FC1);
			//Mat d = Mat::zeros(Size(1, 2), CV_32FC1);
			for (int j = 0; j < WINDOW; ++j) {
				for (int k = 0; k < WINDOW; ++k) {
					cv::Mat g = cv::Mat::zeros(cv::Size(1, 2), CV_32FC1);
					g.at<float>(0, 0) = w_ave.at<float>(j, k + 1) - w_ave.at<float>(j, k);
					g.at<float>(1, 0) = w_ave.at<float>(j + 1, k) - w_ave.at<float>(j, k);
					z += g * g.t();
					e += (w1.at<float>(j, k) - w2.at<float>(j, k)) * g;
				}
			}
			cv::Mat d = z.inv() * e;
			dx = d.at<float>(0, 0);
			dy = d.at<float>(1, 0);
			if (dx * dx + dy * dy < EPS) {
				selected_points[i].flag = true;
				break;
			}
			selected_points[i].dx += dx;
			selected_points[i].dy += dy;
			if (fabs(selected_points[i].dx) >= MARGIN || fabs(selected_points[i].dy) >= MARGIN) {
				break;
			}
		}
	}
	int cnt = 0;
	for (int i = 0; i < FEATURES_NUM; ++i) {
		//selected_points[i].print();
		img0_show.at<cv::Vec3b>(selected_points[i].x + WINDOW / 2, selected_points[i].y + WINDOW / 2) = cv::Vec3b(0, 0, 255);
		img0_show.at<cv::Vec3b>(selected_points[i].x + WINDOW / 2 - 1, selected_points[i].y + WINDOW / 2) = cv::Vec3b(0, 0, 255);
		img0_show.at<cv::Vec3b>(selected_points[i].x + WINDOW / 2, selected_points[i].y + WINDOW / 2 - 1) = cv::Vec3b(0, 0, 255);
		img0_show.at<cv::Vec3b>(selected_points[i].x + WINDOW / 2 + 1, selected_points[i].y + WINDOW / 2) = cv::Vec3b(0, 0, 255);
		img0_show.at<cv::Vec3b>(selected_points[i].x + WINDOW / 2, selected_points[i].y + WINDOW / 2 + 1) = cv::Vec3b(0, 0, 255);
		if (selected_points[i].flag) {
			++cnt;
			img1_show.at<cv::Vec3b>(
				cvRound(selected_points[i].x + WINDOW / 2 + selected_points[i].dy),
				cvRound(selected_points[i].y + WINDOW / 2 + selected_points[i].dx)) = cv::Vec3b(0, 0, 255);
			img1_show.at<cv::Vec3b>(
				cvRound(selected_points[i].x + WINDOW / 2 - 1 + selected_points[i].dy),
				cvRound(selected_points[i].y + WINDOW / 2 + selected_points[i].dx)) = cv::Vec3b(0, 0, 255);
			img1_show.at<cv::Vec3b>(
				cvRound(selected_points[i].x + WINDOW / 2 + selected_points[i].dy),
				cvRound(selected_points[i].y + WINDOW / 2 - 1 + selected_points[i].dx)) = cv::Vec3b(0, 0, 255);
			img1_show.at<cv::Vec3b>(
				cvRound(selected_points[i].x + WINDOW / 2 + 1 + selected_points[i].dy),
				cvRound(selected_points[i].y + WINDOW / 2 + selected_points[i].dx)) = cv::Vec3b(0, 0, 255);
			img1_show.at<cv::Vec3b>(
				cvRound(selected_points[i].x + WINDOW / 2 + selected_points[i].dy),
				cvRound(selected_points[i].y + WINDOW / 2 + 1 + selected_points[i].dx)) = cv::Vec3b(0, 0, 255);
		}
	}
	cv::imwrite("show0.png", img0_show);
	cv::imwrite("show1.png", img1_show);
	cv::imshow("show0", img0_show);
	cv::imshow("show1", img1_show);
	std::cout << "ratio: " << cnt * 1.0 / FEATURES_NUM << std::endl;
	cvWaitKey();
	
	return 0;
}