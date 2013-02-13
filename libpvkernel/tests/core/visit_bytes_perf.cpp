/**
 * \file visit_bits.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVByteVisitor.h>

#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv)
{
	if (argc <= 1) {
		std::cerr << "Usage: " << argv[0] << " number_slices" << std::endl;
		return 1;
	}

	size_t n = atoll(argv[1]);
	std::vector<std::string> slices;
	slices.reserve(n);
	srand(0);
	for (size_t i = 0; i < n; i++) {
		size_t str_size = rand() % 2048;

		std::string rand_str;
		rand_str.reserve(str_size);
		for (size_t c = 0; c < str_size; c++) {
			rand_str.push_back((rand() % 26) + 'a');
		}
		slices.push_back(std::move(rand_str));
	}
	size_t buf_size = 0;
	for (std::string const& s: slices) {
		buf_size += s.size() + 1;
	}
	char* buf = (char*) malloc(buf_size);
	char* cur_buf = buf;
	for (std::string const& s: slices) {
		memcpy(cur_buf, s.c_str(), s.size()+1);
		cur_buf += s.size()+1;
	}

	BENCH_START(slices_serial);
	for (size_t i = 0; i < n; i++) {
		size_t ret;
		if (PVCore::PVByteVisitor::__impl::get_nth_slice_serial((const uint8_t*) buf, buf_size, i, ret) == nullptr) {
			std::cerr << "Fatal error: unable to find the " << i << "-th slice." << std::endl;
			return 1;
		}
	}
	BENCH_END(slices_serial, "serial", buf_size, 1, 1, 1);

	BENCH_START(slices_sse);
	for (size_t i = 0; i < n; i++) {
		size_t ret;
		if (PVCore::PVByteVisitor::__impl::get_nth_slice_sse((const uint8_t*) buf, buf_size, i, ret) == nullptr) {
			std::cerr << "Fatal error: unable to find the " << i << "-th slice." << std::endl;
			return 1;
		}
	}
	BENCH_END(slices_sse, "sse", buf_size, 1, 1, 1);

	free(buf);

	return 0;
}
