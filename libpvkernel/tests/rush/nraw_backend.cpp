#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/rush/PVNrawDiskBackend.h>
#include <pvkernel/core/picviz_bench.h>

#include <iostream>
#include <sstream>

#include <tbb/tick_count.h>

#define MIN_SIZE 1
#define MAX_SIZE 256
#define N (4096000*5)
#define LATENCY_N 100000

size_t get_buf_size(size_t /*i*/)
{
	//return (i%(MAX_SIZE-MIN_SIZE+1))+MIN_SIZE;
	return 10;
}

void check_value(const char* buf, size_t s, size_t r)
{
	size_t read_r;
	char rbuf[MAX_SIZE+1];
	sscanf(buf, "%lu %s", &read_r, rbuf);
	bool valid = (read_r == r) && (strlen(rbuf) == get_buf_size(r));
	if (!valid) {
		printf("line %lu: ", r);
		fwrite(buf, 1, s, stdout);
		printf("  INVALID!!");
		printf("\n");
	}
	PV_ASSERT_VALID(valid);
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
	char buf[MAX_SIZE+20];
	size_t stotal = 0;
	for (size_t i = 0; i < N; i++) {
		//const size_t sbuf = (rand()%(MAX_SIZE-MIN_SIZE+1))+MIN_SIZE;
		const size_t sbuf = get_buf_size(i);
		stotal += sbuf;
		int swrite = snprintf(buf, 20, "%lu ", i);
		memset(&buf[swrite], 'a' + i%26, sbuf);
		backend.add(0, buf, sbuf+swrite);
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
		PV_ASSERT_VALID(sret == get_buf_size(i));
	}
	BENCH_END(at, "at", sizeof(char), stotal, 1, 1);
#endif

	PVLOG_INFO("Checking values with visit_column()...\n");
	BENCH_START(visit);
	PV_ASSERT_VALID(backend.visit_column2(0, [=](size_t r, const char* buf, size_t n)
			{
				check_value(buf, n, r);
			}));
	BENCH_END(visit, "visit", sizeof(char), stotal, 1, 1);

	PVLOG_INFO("Checking values with visit_column_tbb()...\n");
	BENCH_START(visit2);
	PV_ASSERT_VALID(backend.visit_column_tbb(0, [=](size_t r, const char* buf, size_t n)
			{
				check_value(buf, n, r);
			}));
	BENCH_END(visit2, "visit", sizeof(char), stotal, 1, 1);

	PVCore::PVSelBitField sel;
	sel.select_all();
	//sel.set_bit_fast((N/2)-7);
	backend.visit_column_tbb_sel(0, [&](size_t r, const char* bread, size_t n)
		{
			check_value(bread, n, r);
		},
		sel);

	sel.select_none();
	sel.set_bit_fast((N/2)-7);
	backend.visit_column_tbb_sel(0, [&](size_t r, const char* bread, size_t n)
		{
			check_value(bread, n, r);
		},
		sel);

	sel.select_none();
	sel.set_bit_fast(0);
	sel.set_bit_fast(2);
	sel.set_bit_fast(4);
	sel.set_bit_fast(6);
	sel.set_bit_fast(7);
	sel.set_bit_fast(10000);
	sel.set_bit_fast(16384);
	sel.set_bit_fast(65534);
	sel.set_bit_fast(N-1);
	backend.visit_column2_sel(0, [&](size_t r, const char* bread, size_t n)
		{
			check_value(bread, n, r);
			printf("line: %lu, ", r);
			fwrite(bread, 1, n, stdout);
			printf("\n");
		},
		sel);

	backend.visit_column_tbb_sel(0, [&](size_t r, const char* bread, size_t n)
		{
			check_value(bread, n, r);
			printf("line: %lu, ", r);
			fwrite(bread, 1, n, stdout);
			printf("\n");
		},
		sel);

	sel.select_odd();

	BENCH_START(visit_sel);
	backend.visit_column2_sel(0, [&](size_t r, const char* bread, size_t n)
		{
			/*printf("line: %llu, ", r);
			fwrite(bread, 1, n, stdout);
			printf("\n");*/
			//PV_ASSERT_VALID(n == get_buf_size(r));
		},
		sel);
	BENCH_END(visit_sel, "visit_sel", sizeof(char), stotal, 1, 1);

	BENCH_START(visit_sel2);
	backend.visit_column2_sel(0, [&](size_t r, const char* bread, size_t n)
		{
			/*printf("line: %llu, ", r);
			fwrite(bread, 1, n, stdout);
			printf("\n");*/
			//PV_ASSERT_VALID(n == get_buf_size(r));
		},
		sel);
	BENCH_END(visit_sel2, "visit_sel (cached)", sizeof(char), stotal, 1, 1);

	BENCH_START(visit_sel_tbb);
	backend.visit_column_tbb_sel(0, [&](size_t r, const char* bread, size_t n)
		{
			//PV_ASSERT_VALID(n == get_buf_size(r));
		},
		sel);
	BENCH_END(visit_sel_tbb, "visit_sel_tbb (cached)", sizeof(char), stotal, 1, 1);

	sel.select_none();
	sel.set_bit_fast(0);
	sel.set_bit_fast(N-1);
	BENCH_START(visit_sel4);
	backend.visit_column2_sel(0, [&](size_t r, const char* bread, size_t n)
		{
			/*printf("line: %llu, ", r);
			fwrite(bread, 1, n, stdout);
			printf("\n");*/
			//PV_ASSERT_VALID(n == get_buf_size(r));
		},
		sel);
	BENCH_END(visit_sel4, "visit_sel (cached) (two rows)", sizeof(char), stotal, 1, 1);

	BENCH_START(visit_sel_tbb2);
	backend.visit_column_tbb_sel(0, [&](size_t r, const char* bread, size_t n)
		{
			//PV_ASSERT_VALID(n == get_buf_size(r));
		},
		sel);
	BENCH_END(visit_sel_tbb2, "visit_sel_tbb (cached) (two rows)", sizeof(char), stotal, 1, 1);

	return 0;
}
