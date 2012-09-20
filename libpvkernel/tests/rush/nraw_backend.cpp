#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/rush/PVNrawDiskBackend.h>

#include <iostream>
#include <sstream>

#include <tbb/tick_count.h>

#define MIN_SIZE 1
#define MAX_SIZE 256
#define N 2100000
#define LATENCY_N 100000

size_t get_buf_size(size_t i)
{
	return (i%(MAX_SIZE-MIN_SIZE+1))+MIN_SIZE;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " path_nraw" << std::endl;
		return 1;
	}

	const char* nraw_path = argv[1];
	PVRush::PVNRawDiskBackend<> backend(nraw_path, 1);

	std::vector<std::string> vec;
	vec.reserve(N);
	for (int i = 0 ; i < N; i++) {
		std::stringstream st;
		st << i << " ";
		vec.push_back(st.str());
		backend.add(0, st.str().c_str(), st.str().length());
	}
	backend.flush();

	size_t ret;

	std::vector<unsigned int> shuffled_fields_sequence;
	shuffled_fields_sequence.reserve(LATENCY_N);
	for (unsigned int i = 0; i < LATENCY_N; i++) {
		shuffled_fields_sequence.push_back(i);
	}
	std::random_shuffle(shuffled_fields_sequence.begin(), shuffled_fields_sequence.end());

	tbb::tick_count t1 = tbb::tick_count::now();
	bool test = true;
	for (unsigned int i : shuffled_fields_sequence) {
		test &= backend.at(i, 0, ret) != nullptr;
	}
	tbb::tick_count t2 = tbb::tick_count::now();
	std::cout << "latency (random)=" << ((t2-t1).seconds()*1000)/LATENCY_N << " milli sec" << std::endl;
	std::cout << test << std::endl;

	bool test_passed = true;
	for (int i = 0 ; i < N && test_passed; i++) {
		const char* field = backend.at(i, 0, ret);
		test_passed &= (strcmp(field, vec[i].c_str()) == 0);
		if (i % 1000 == 0) {
			std::cout << ((double)i/N)*100 << "%" << std::endl;
		}
	}

	std::cout << "test passed: " << std::boolalpha << test_passed << std::endl;
	backend.print_stats();

	/*
		const char* nraw_path = argv[1];
		PVRush::PVNRawDiskBackend<> backend(nraw_path, 5);

		char buf[MAX_SIZE];
		for (size_t i = 0; i < N; i++) {
				//const size_t sbuf = (rand()%(MAX_SIZE-MIN_SIZE+1))+MIN_SIZE;
				const size_t sbuf = get_buf_size(i);
				memset(buf, 'a', sbuf);
				backend.add(0, buf, sbuf);
		}
		backend.flush();
		backend.print_indexes();

		for (size_t i = 0; i < N; i++) {
				size_t sret;
				if (i == 8143) {
						printf("hi\n");
				}
				const char* bread = backend.at(i, 0, sret);
				printf("sret/theorical: %lu/%lu\n", sret, get_buf_size(i));
				ASSERT_VALID(sret == get_buf_size(i));
				printf("%lu: ", i);
				fwrite(bread, 1, sret, stdout);
				printf("\n");
		}
	*/

	return !test_passed;
}
