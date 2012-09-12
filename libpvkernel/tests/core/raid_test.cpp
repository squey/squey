#include <iostream>
#include <sstream>
#include <string>

#include <fcntl.h>

#include <boost/tokenizer.hpp>

#include <pvkernel/core/picviz_bench.h>

constexpr uint64_t DEFAULT_CONTENT_SIZE = 1024*1024*1024/2;
constexpr uint64_t DEFAULT_WRITE_CHUNK_SIZE = 8*1024*1024;
constexpr uint64_t DEFAULT_READ_CHUNK_SIZE = 32*1024*1024;
constexpr uint64_t BUF_SIZE = 256*1024*1024;
constexpr uint64_t BUF_ALIGN = 512;


// Wiping system disk cache:
// sync ; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

// Changing opened files limit per process (valid only for the current shell)
// $sudo su
// #ulimit -n <max_opened_files_limit>

// Generate 4.2G dictionary of words
// cat /usr/share/hunspell/en_US.dic |cut -d'/' -f1 > ./words
// for i in `seq 13`; do cat words >> words_big && cat words >> words_big && mv words_big words; done

typedef std::basic_string<char, std::char_traits<char>, PVCore::PVAlignedAllocator<char, BUF_ALIGN>> aligned_string_t;


/*
 *
 * Policy classes
 *
 *
 */

struct BufferedPolicy
{
	typedef FILE* file_t;

	file_t Open(std::string const& filename)
	{
		return fopen(filename.c_str(), "rw");
	}

	inline bool Write(const char* content, uint64_t buf_size, file_t file)
	{
		return fwrite(content, buf_size, 1, file) > 0;
	}

	inline int64_t Read(file_t file, void* content, uint64_t buf_size)
	{
		return fread(content, 1, buf_size, file);
	}

	inline int64_t Seek(file_t file, int64_t offset)
	{
		return fseek(file, offset, SEEK_CUR);
	}

	void Flush(file_t file)
	{
		fflush(file);
	}

	void Close(file_t file)
	{
		fclose(file);
	}
};

struct UnbufferedPolicy
{
	typedef int file_t;

	file_t Open(std::string const& filename)
	{
		file_t f = open(filename.c_str(), O_RDWR | O_CREAT);
		if (f == -1) {
			std::cout << strerror(errno) << std::endl;
		}
		return f;
	}

	inline bool Write(const char* content, uint64_t buf_size, file_t file)
	{
		int64_t r = write(file, content, buf_size);
		if (r == -1) {
			std::cout << "errno=" << errno << std::endl;
			std::cout << strerror(errno) << std::endl;
		}
		return r;
	}

	inline int64_t Read(file_t file, void* buffer,  uint64_t buf_size)
	{
		int64_t r = read(file, buffer, buf_size);
		if (r == -1) {
			std::cout << strerror(errno) << std::endl;
		}
		return r;
	}

	inline int64_t Seek(file_t file, int64_t offset)
	{
		int64_t r = lseek(file, offset, SEEK_CUR);
		if (r == -1) {
			std::cout << strerror(errno) << std::endl;
		}
		return r;
	}

	void Flush(file_t)
	{
	}

	void Close(file_t file)
	{
		close(file);
	}
};

struct RawPolicy : public UnbufferedPolicy
{

	file_t Open(std::string const& filename)
	{
		file_t f = open(filename.c_str(), O_RDWR | O_CREAT | O_DIRECT);
		if (f == -1) {
			std::cout << strerror(errno) << std::endl;
		}
		return f;
	}
};

struct RawBufferedPolicy : public BufferedPolicy
{
	file_t Open(std::string const& filename)
	{
		int fd = open(filename.c_str(), O_RDWR | O_CREAT | O_DIRECT);
		return fdopen(fd, "rw");
	}
};


class CustomBufferedPolicy : public BufferedPolicy
{
public:
	file_t Open(std::string const& filename)
	{
		file_t f = fopen(filename.c_str(), "rw");

		char* buffer = new char[8192];
		setbuf(f, buffer);
		_buffers.push_back(buffer);

		return f;
	}

	~CustomBufferedPolicy()
	{
		for (auto buffer : _buffers) {
			delete [] buffer;
		}
	}
private:
	std::vector<char*> _buffers;
};

/*
 *
 * Writer class
 *
 *
 */

template <typename BufferPolicy>
class Writer : public BufferPolicy
{
public:
	Writer(std::string const& folder, uint64_t num_cols) : _num_cols(num_cols)
	{
		_files = new typename BufferPolicy::file_t[num_cols];
		_filenames = new std::string[num_cols];

		this->CreateFolder(folder, _num_cols);

		for (int i = 0 ; i < _num_cols ; i++) {
			_files[i] = this->Open(this->_filenames[i]);
		}
	}

	void CreateFolder(std::string const& folder, uint64_t num_cols)
	{
		_folder = folder;

		DeleteFolder();

		system((std::string("mkdir ") + _folder + " 2> /dev/null").c_str());

		for (uint64_t i = 0 ; i < num_cols ; i++) {
			std::stringstream st;
			st << _folder << "/file_" << i;
			_filenames[i] = st.str().c_str();
		}
	}

	void DeleteFolder()
	{
		system((std::string("rm -rf ") + _folder).c_str());
	}

	inline void write_cols(const char* content, uint64_t buf_size)
	{
		bool res = true;
		for (int i = 0 ; i < _num_cols ; i++) {
			res &= this->Write(content, buf_size, _files[i]);
		}
		if (!res) {
			std::cout << "write failed" << std::endl;
		}
		flush_all();
	}

	inline void write_cols(aligned_string_t const& content)
	{
		write_cols(content.c_str(), content.length());
	}

	inline void write(const char* buffer, uint64_t chunk_size, uint64_t num_chunks)
	{
		std::stringstream st;
		st << "sequential writes (" << typeid(this).name() << ") [chunk_size=" << chunk_size << " num_chunks=" <<  num_chunks << " num_cols=" << _num_cols << "]";

		BENCH_START(w);
		for (uint64_t j = 0 ; j < num_chunks; j++) {
			write_cols(buffer, chunk_size);
		}
		flush_all();
		BENCH_END(w, st.str().c_str(), 1, 1, chunk_size, (uint64_t) _num_cols*num_chunks); //!\\ Ensure result fits on uint64_t
	}

	void flush_all()
	{
		for (int i = 0 ; i < _num_cols ; i++) {
			this->Flush(_files[i]);
		}
	}

	~Writer()
	{
		for (int i = 0 ; i < _num_cols ; i++) {
			this->Close(_files[i]);
		}

		this->DeleteFolder();

		delete [] _filenames;
		delete [] _files;
	}
private:
	std::string* _filenames = nullptr;
	std::string _folder;
	uint64_t _num_cols;
	typename BufferPolicy::file_t* _files = nullptr;
};

/*
 *
 * Writer test
 *
 *
 */

void write_test(std::string const& folder)
{
	char* buffer = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(BUF_SIZE);
	memset(buffer, '$', sizeof(char)*BUF_SIZE);

	for (uint64_t num_cols : {1, 2, 32, 128, 256, 512, 4096, 8192, 16384}) {
		for (uint64_t chunk_size : {4*1024, 16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 1*1024*1024, 2*1024*1024, 8*1024*1024, 16*1024*1024, 32*1024*1024, 64*1024*1024, 128*1024*1024, 256*1024*1024}) {
			uint64_t num_chunks = std::max(DEFAULT_CONTENT_SIZE/chunk_size/num_cols, (uint64_t)2);

			/*{
			Writer<BufferedPolicy> writer_buffered(folder, num_cols);
			writer_buffered.write(buffer, chunk_size, num_chunks);
			}

			{
			Writer<UnbufferedPolicy> writer_unbuffered(folder, num_cols);
			writer_unbuffered.write(buffer, chunk_size, num_chunks);
			}*/

			{
			Writer<RawPolicy> writer_raw(folder, num_cols);
			writer_raw.write(buffer, chunk_size, num_chunks);
			}
		}
	}

	PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(buffer, BUF_SIZE);
}

/*
 *
 * Reader class
 *
 *
 */

template <typename BufferPolicy>
class Reader : public BufferPolicy
{
public:
	typedef typename BufferPolicy::file_t file_t;

public:
	uint64_t Search(const std::string& filename, uint64_t num_cols, uint64_t chunk_size, std::string const& content_to_find)
	{
		char* const buffer = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(chunk_size*2);

		file_t file = this->Open(filename);

		std::stringstream st;
		st << "sequential read (" << typeid(this).name() << ") [num_cols=" << num_cols << " chunk_size=" << chunk_size << "]";
		BENCH_START(r);

		uint64_t total_read_size = 0;
		uint64_t nb_occur = 0;
		uint64_t read_size = 0;
		uint64_t end_of_file_pos = 0;

		char* buffer_ptr = buffer+BUF_ALIGN;
		bool last_chunk = false;
		do
		{

			read_size = this->Read(file, buffer+BUF_ALIGN, chunk_size-BUF_ALIGN);
			last_chunk = read_size < (chunk_size-BUF_ALIGN);

			while (true) {

				char* endl = nullptr;
				if (last_chunk) {
					endl = (char*) memchr(buffer_ptr, '\n', chunk_size);

					if (endl == nullptr || buffer_ptr >= buffer+BUF_ALIGN+read_size) {
						buffer_ptr = buffer;

						break;
					}
				}
				else {
					endl = (char*) memchr(buffer_ptr, '\n', chunk_size);

					if (endl == nullptr) {
						uint64_t partial_line_length = buffer+chunk_size-buffer_ptr;

						char* dst = buffer+BUF_ALIGN-partial_line_length;
						memcpy(dst, buffer_ptr, partial_line_length);
						buffer_ptr = dst;

						break;
					}
				}

				int64_t line_length = (&endl[0] - &buffer_ptr[0]);
				buffer_ptr[line_length] = '\0';

				nb_occur += (memcmp(buffer_ptr, content_to_find.c_str(), content_to_find.length()) == 0);

				buffer_ptr += (line_length+1);
			}

			total_read_size += read_size;

			// Skip other columns if any:
			if (num_cols > 1) {
				this->Seek(file, (num_cols-1)*chunk_size);
			}

		} while(!last_chunk);

		std::cout << "total_read_size=" << total_read_size << std::endl;
		std::cout << "nb_occur=" << nb_occur << std::endl;

		BENCH_END(r, st.str().c_str(), sizeof(char), total_read_size, 1, 1);

		PVCore::PVAlignedAllocator<char, BUF_ALIGN>().deallocate(buffer, chunk_size*2);

		return nb_occur;
	}
};

/*
 *
 * Reader test
 *
 *
 */

void read_test(std::string const& path)
{
	Reader<RawPolicy> reader;
	for (uint64_t num_cols : {1, 2, 4, 8, 16 , 32, 128, 256, 512, 1024, 4096, 8192, 16384}) {
		for (uint64_t chunk_size : {16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 1*1024*1024, 2*1024*1024, 8*1024*1024, 16*1024*1024, 32*1024*1024, 64*1024*1024, 128*1024*1024, 256*1024*1024}) {
			uint64_t nb_occur = reader.Search(path, num_cols, chunk_size, std::string("motherfucker"));
		}
	}
}

void usage(const char* app_name)
{
	std::cerr << "Usage: " << app_name << " [folder_path]" << std::endl;
}

int main(int argc, const char* argv[])
{
	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	const std::string folder(argv[1]);

	write_test(folder);

	read_test(folder);
}
