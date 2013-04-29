#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#include <QImage>

#include <pvkernel/core/picviz_bench.h>

constexpr uint32_t IMAGE_WIDTH = 1024;
constexpr uint32_t IMAGE_HEIGHT = IMAGE_WIDTH;
constexpr uint32_t PIXEL_COUNT = IMAGE_WIDTH*IMAGE_HEIGHT;

#define RANDOM 0

struct random_point_t
{
	random_point_t(uint32_t r, uint32_t c) : row(r), col(c) {}
	uint32_t row;
	uint32_t col;
};

int main()
{
	srand (time(NULL));

	// Points generation
	std::vector<random_point_t> vec;
	vec.reserve(PIXEL_COUNT);
	uint32_t row = 0;
	uint32_t col = 0;
#if RANDOM
	for (uint32_t i = 0; i < PIXEL_COUNT ; i++) {
		row = rand() % IMAGE_HEIGHT;
		col = rand() % IMAGE_WIDTH;
		vec[i] = random_point_t(row, col);
	}
#else // SEQUENTIAL
	for (uint32_t row = 0; row < IMAGE_WIDTH ; row++) {
		for (uint32_t col = 0; col < IMAGE_HEIGHT ; col++) {
			vec.emplace_back(row, col);
		}
	}
#endif

	// 32 bits image generation
	QImage img(QSize(IMAGE_WIDTH, IMAGE_HEIGHT), QImage::Format_RGB32);
	BENCH_START(random_points);
	QRgb* first_pixel = (QRgb*) &img.scanLine(0)[0];
	for (uint32_t i = 0; i < PIXEL_COUNT ; i++) {
		first_pixel[vec[i].row*IMAGE_WIDTH + vec[i].col] = 0xFFFFFF;
	}
	BENCH_END(random_points, "random_points", sizeof(random_point_t), PIXEL_COUNT, sizeof(uint32_t), PIXEL_COUNT);
	QString image_path = "/tmp/random_points.bmp";
	bool res = img.save(image_path, "bmp");
	std::cout << "image saved to '"<< qPrintable(image_path) << "': "  << std::boolalpha << res << std::endl;

	// 8 bits image generation
	std::vector<char> vec_char;
	vec_char.reserve(PIXEL_COUNT);
	BENCH_START(random_points_8bits);
	char* first_8bits_pixel = (char*) &vec_char[0];
	for (uint32_t i = 0; i < PIXEL_COUNT ; i++) {
		first_8bits_pixel[vec[i].row*IMAGE_WIDTH + vec[i].col] = 0xFF;
	}
	BENCH_END(random_points_8bits, "random_points_8bits", sizeof(random_point_t), PIXEL_COUNT, sizeof(char), PIXEL_COUNT);
}
