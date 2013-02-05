#include <iostream>
#include <pvkernel/core/PVSelBitField.h>

#define N 100000000

inline const PVRow gather_func(const PVRow r) { return 4*r; }

void show_diff(std::vector<PVRow> const& packed, std::vector<PVRow> const& ref)
{
	size_t size_ref = ref.size();
	size_t size_packed = packed.size();
	if (size_ref != size_packed) {
		std::cerr << "Size differs: " << size_packed << " != " << size_ref << std::endl;
	}	
	for (size_t i = 0; i < std::min(size_packed, size_ref); i++) {
		if (packed[i] != ref[i]) {
			std::cerr << i << ": " << packed[i] << " != " << ref[i] << std::endl;
		}
	}
}

int main()
{
	PVCore::PVSelBitField bits;
	bits.select_random();

	std::vector<PVRow> ref;
	ref.reserve(N);
	bits.visit_selected_lines([&ref](const PVRow r) { ref.push_back(gather_func(r)); }, N);

	std::vector<PVRow> packed;
	packed.reserve(N);
	bits.visit_selected_lines_packed<8>(
		[&packed](const PVRow rows[8])
		{
			for (int i = 0; i < 8; i++) {
				packed.push_back(gather_func(rows[i]));
			}
		},
		[&packed](const PVRow r) { packed.push_back(gather_func(r)); },
		N);

	if (packed != ref) {
		std::cerr << "Packed on 8 bytes is != ref" << std::endl;
		show_diff(packed, ref);
		return 1;
	}

	std::vector<PVRow> sse;
	sse.reserve(N);
	bits.visit_selected_lines_gather_sse(
		[&sse](__m128i const v)
		{
			sse.push_back(_mm_extract_epi32(v, 0));
			sse.push_back(_mm_extract_epi32(v, 1));
			sse.push_back(_mm_extract_epi32(v, 2));
			sse.push_back(_mm_extract_epi32(v, 3));
		},
		[&sse](PVRow const res) { sse.push_back(res); },
		gather_func,
		N);

	if (sse != ref) {
		std::cerr << "gather_sse != ref" << std::endl;
		show_diff(sse, ref);
		return 1;
	}
}
