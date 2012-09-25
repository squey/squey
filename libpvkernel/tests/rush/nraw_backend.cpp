#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/rush/PVNrawDiskBackend.h>
#include <pvkernel/core/picviz_bench.h>

#include <iostream>
#include <sstream>

#include <tbb/tick_count.h>

#define MIN_SIZE 1
#define MAX_SIZE 256
#define N 4096000
#define LATENCY_N 100000

size_t get_buf_size(size_t i)
{
	//return (i%(MAX_SIZE-MIN_SIZE+1))+MIN_SIZE;
	return 10;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " path_nraw" << std::endl;
		return 1;
	}

	const char* nraw_path = argv[1];
	PVRush::PVNrawDiskBackend backend;
	backend.init(nraw_path, 2);

#if 0
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

	backend.clear();
#endif

	PVLOG_INFO("Writing NRAW...\n");
	char buf[MAX_SIZE];
	size_t stotal = 0;
	for (size_t i = 0; i < N; i++) {
		//const size_t sbuf = (rand()%(MAX_SIZE-MIN_SIZE+1))+MIN_SIZE;
		const size_t sbuf = get_buf_size(i);
		stotal += sbuf;
		memset(buf, 'a' + i%26, sbuf);
		if (i == 3038487) {
			printf("hello\n");
		}
		backend.add(0, buf, sbuf);
		//backend.add(1, buf, sbuf);
	}
	backend.flush();

#if 0
	size_t sret;
	PVLOG_INFO("Checking values with at()...\n");
	BENCH_START(at);
	for (size_t i = 0; i < N; i++) {
		backend.at(i, 0, sret);
		/*printf("sret/theorical: %lu/%lu\n", sret, get_buf_size(i));
		printf("%lu: ", i);
		fwrite(bread, 1, sret, stdout);
		printf("\n");*/
		ASSERT_VALID(sret == get_buf_size(i));
	}
	BENCH_END(at, "at", sizeof(char), stotal, 1, 1);
#endif

	PVLOG_INFO("Checking values with visit_column()...\n");
	BENCH_START(visit);
	ASSERT_VALID(backend.visit_column2(0, [=](size_t r, const char*, size_t n)
			{
				ASSERT_VALID(n == get_buf_size(r));
			}));
	BENCH_END(visit, "visit", sizeof(char), stotal, 1, 1);

	PVLOG_INFO("Checking values with visit_column_tbb()...\n");
	BENCH_START(visit2);
	ASSERT_VALID(backend.visit_column_tbb(0, [=](size_t r, const char*, size_t n)
			{
				ASSERT_VALID(n == get_buf_size(r));
			}));
	BENCH_END(visit2, "visit", sizeof(char), stotal, 1, 1);

	return 0;
}
