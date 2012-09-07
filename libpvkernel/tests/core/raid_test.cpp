#include <iostream>
#include <sstream>
#include <string>

#include <fcntl.h>

#include <pvkernel/core/picviz_bench.h>

constexpr uint64_t DEFAULT_CONTENT_SIZE = 1024*1024*1024/2;
constexpr uint64_t DEFAULT_CHUNK_SIZE = 8*1024*1024;
constexpr uint64_t BUF_SIZE = 32*1024*1024;
constexpr uint64_t BUF_ALIGN = 512;


// Wiping system disk cache:
// sync ; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

// Changing opened files limit per process (valid only for the current shell)
// $sudo su
// #ulimit -n <max_opened_files_limit>

// Generate dictinary of words
// cat /usr/share/hunspell/en_US.dic |cut -d'/' -f1 > ./words

typedef std::basic_string<char, std::char_traits<char>, PVCore::PVAlignedAllocator<char, BUF_ALIGN>> aligned_string_t;

class BaseBufferPolicy
{
public:
	BaseBufferPolicy(uint64_t num_cols)
	{
		_filenames = new std::string[num_cols];
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

	~BaseBufferPolicy()
	{
		delete [] _filenames;
	}
protected:
	std::string* _filenames = nullptr;
private:
	std::string _folder;
};

struct BufferedPolicy : public BaseBufferPolicy
{
	typedef FILE* file_t;

	BufferedPolicy(uint64_t num_cols) : BaseBufferPolicy(num_cols) {}

	file_t Open(std::string const& filename)
	{
		return fopen(filename.c_str(), "w");
	}

	inline bool Write(const char* content, uint64_t buf_size, file_t file)
	{
		return fwrite(content, buf_size, 1, file) > 0;
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

struct UnbufferedPolicy : public BaseBufferPolicy
{
	typedef int file_t;

	UnbufferedPolicy(uint64_t num_cols) : BaseBufferPolicy(num_cols) {}

	file_t Open(std::string const& filename)
	{
		return open(filename.c_str(), O_WRONLY | O_CREAT);
	}

	inline bool Write(const char* content, uint64_t buf_size, file_t file)
	{
		return write(file, content, buf_size) != -1;
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
	RawPolicy(uint64_t num_cols) : UnbufferedPolicy(num_cols) {}

	file_t Open(std::string const& filename)
	{
		return open(filename.c_str(), O_WRONLY | O_CREAT | O_DIRECT);
	}
};

struct RawBufferedPolicy : public BufferedPolicy
{
	RawBufferedPolicy(uint64_t num_cols) : BufferedPolicy(num_cols){}

	file_t Open(std::string const& filename)
	{
		int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_DIRECT);
		return fdopen(fd, "w");
	}
};


class CustomBufferedPolicy : public BufferedPolicy
{
public:
	CustomBufferedPolicy(uint64_t num_cols) : BufferedPolicy(num_cols)
	{
	}

	file_t Open(std::string const& filename)
	{
		file_t f = fopen(filename.c_str(), "w");

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

template <typename BufferPolicy>
class Writer : public BufferPolicy
{
public:
	Writer(std::string const& folder, uint64_t num_cols) : BufferPolicy(num_cols), _num_cols(num_cols)
	{
		_files = new typename BufferPolicy::file_t[num_cols];

		this->CreateFolder(folder, _num_cols);

		for (int i = 0 ; i < _num_cols ; i++) {
			_files[i] = this->Open(this->_filenames[i]);
		}
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
		BENCH_END(w, st.str().c_str(), 1, 1, chunk_size, (uint64_t) _num_cols*num_chunks); //!\\ Ensure result fit on uint64_t
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

		delete [] _files;
	}
private:
	uint64_t _num_cols;
	typename BufferPolicy::file_t* _files = nullptr;
};

void write_test(std::string const& folder)
{
	char* buffer = PVCore::PVAlignedAllocator<char, BUF_ALIGN>().allocate(BUF_SIZE);
	memset(buffer, '$', sizeof(char)*BUF_SIZE);

	for (uint64_t num_cols : {/*1, 2, 32,*/ 512/*, 1024, 4096, 8192*/}) {
		for (uint64_t chunk_size : {/*4*1024, 16*1024, */32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 1*1024*1024, 2*1024*1024, 4*1024*1024, 16*1024*1024, 32*1024*1024,64*1024*1024, 128*1024*1024}) {
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

void read_test(std::string const& folder)
{

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
