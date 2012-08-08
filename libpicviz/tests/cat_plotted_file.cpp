#include <iostream>
#include <stdlib.h>
#include <string.h>

#pragma pack(push)
#pragma pack(1)
struct PlottedFileHeader
{
	uint32_t ncols;
	bool is_transposed;
};
#pragma pack(pop)

#define SIZE_TMP (1024*10)

static_assert(sizeof(PlottedFileHeader) == 5, "Packing of PlottedFile structure isn't good !");

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " plotted1 plotted2 ... outfile" << std::endl;
		return 1;
	}

	size_t n = argc-2;
	FILE* files[n];
	size_t nrows[n];
	PlottedFileHeader headers[n];

	// Header of plotted files is:
	// [4 bytes: number of cols][1 byte: is_transposed]
	
	// Open file
	int transposed = -1;
	uint32_t ncols;
	for (size_t i = 0; i < n; i++) {
		FILE* f = fopen(argv[i+1], "r");
		if (!f) {
			std::cerr << "Error opening " << argv[i+1] << ": " << strerror(errno) << std::endl;
			return 1;
		}
		PlottedFileHeader& header(headers[i]);
		if (fread(&header, sizeof(PlottedFileHeader), 1, f) != 1) {
			std::cerr << "Error reading " << argv[i+1] << ": " << strerror(errno) << std::endl;
			return 1;
		}
		if (transposed == -1) {
			transposed = header.is_transposed;
			ncols = header.ncols;
		}
		else
		{
			if (transposed != header.is_transposed) {
				std::cerr << "All files must be transposed or not, not a mix of both !" << std::endl;
				return 1;
			}
			if (ncols != header.ncols) {
				std::cerr << "All files must have the same number of columns !" << std::endl;
				return 1;
			}
		}

		// Compute number of rows
		static_assert(sizeof(off_t) == sizeof(uint64_t), "off_t isn't a 64 bit integer !");
		fseek(f, 0, SEEK_END);
		nrows[i] = ((ftello(f)-sizeof(PlottedFileHeader))/sizeof(float))/header.ncols;
		fseek(f, 0, SEEK_SET);

		files[i] = f;

	}

	FILE* fout = fopen(argv[argc-1], "w");
	if (!fout) {
		std::cerr << "Unable to open " << argv[argc-1] << " for writing: " << strerror(errno) << std::endl;
		return 1;
	}

	float* tmp_buf = (float*) malloc(sizeof(float)*SIZE_TMP);

	if (transposed == 1) {
		fwrite(&ncols, sizeof(uint32_t), 1, fout);
		fwrite(&transposed, 1, 1, fout);

		for (uint32_t c = 0; c < ncols; c++) {
			std::cerr << "Processing column " << c << "..." << std::endl;
			for (size_t i = 0; i < n; i++) {
				std::cerr << "Processing file " << argv[i+1] << "..." << std::endl;
				FILE* cur_f = files[i];
				// Write this file's column
				size_t cur_r = 0;
				for (; cur_r < (nrows[i]/SIZE_TMP)*SIZE_TMP; cur_r += SIZE_TMP) {
					if (fread(tmp_buf, sizeof(float), SIZE_TMP, cur_f) != SIZE_TMP) {
						std::cerr << "Error reading " << argv[i+1] << ": " << strerror(errno) << std::endl;
						return 1;
					}
					fwrite(tmp_buf, sizeof(float), SIZE_TMP, fout);
				}
				size_t r = fread(tmp_buf, sizeof(float), nrows[i]-cur_r, cur_f);
				if (r != (int)nrows[i]-cur_r) {
					std::cerr << "Error reading " << argv[i+1] << ": " << strerror(errno) << std::endl;
					return 1;
				}
				fwrite(tmp_buf, sizeof(float), r, fout);
			}
		}
	}

	for (size_t i = 0; i < n; i++) {
		fclose(files[i]);
	}
	fclose(fout);

	return 0;
}
