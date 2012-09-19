#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/rush/PVNrawDiskBackend.h>

#include <iostream>

#define MIN_SIZE 1
#define MAX_SIZE 256
#define N 8155

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
		//ASSERT_VALID(sret == get_buf_size(i));
		printf("%lu: ", i);
		fwrite(bread, 1, sret, stdout);
		printf("\n");
	}


	return 0;
}
