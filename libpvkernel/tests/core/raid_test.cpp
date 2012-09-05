#include <iostream>
#include <sstream>
#include <string>

#include <fcntl.h>

#include <pvkernel/core/picviz_bench.h>

#define NUM_ROWS 500000000

// sync ; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

class BaseBufferPolicy
{
public:
	BaseBufferPolicy(uint32_t num_cols)
	{
		_filenames = new std::string[num_cols];
	}

	void CreateFolder(std::string const& folder, uint32_t num_cols)
	{
		_folder = folder;

		DeleteFolder();

		system((std::string("mkdir ") + _folder + " 2> /dev/null").c_str());

		for (uint32_t i = 0 ; i < num_cols ; i++) {
			std::stringstream st;
			st << _folder << "file_" << i;
			_filenames[i] = st.str();
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

	BufferedPolicy(uint32_t num_cols) : BaseBufferPolicy(num_cols) {}

	file_t Open(std::string const& filename)
	{
		return fopen(filename.c_str(), "w");
	}

	void Write(std::string const& content, file_t file)
	{
		fwrite(content.c_str(), content.length() , 1, file);
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

	UnbufferedPolicy(uint32_t num_cols) : BaseBufferPolicy(num_cols) {}

	file_t Open(std::string const& filename)
	{
		return open(filename.c_str(), O_WRONLY | O_CREAT);
	}

	void Write(std::string const& content, file_t file)
	{
		write(file, content.c_str(), content.length());
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
	RawPolicy(uint32_t num_cols) : UnbufferedPolicy(num_cols) {}

	file_t Open(std::string const& filename)
	{
		return open(filename.c_str(), O_WRONLY | O_CREAT | O_DIRECT);
	}
};

struct RawBufferedPolicy : public BufferedPolicy
{
	RawBufferedPolicy(uint32_t num_cols) : BufferedPolicy(num_cols) {}

	file_t Open(std::string const& filename)
	{
		int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_DIRECT);
		return fdopen(fd, "w");
	}
};

template <typename BufferPolicy>
class Writer : public BufferPolicy
{
public:
	Writer(std::string const& folder, uint32_t num_cols) : BufferPolicy(num_cols), _num_cols(num_cols)
	{
		_files = new typename BufferPolicy::file_t[num_cols];

		this->CreateFolder(folder, _num_cols);

		for (int i = 0 ; i < _num_cols ; i++) {
			_files[i] = this->Open(this->_filenames[i]);
		}
	}

	void write_cols(std::string const& content)
	{
		for (int i = 0 ; i < _num_cols ; i++) {
			this->Write(content, _files[i]);
		}
	}

	/*void write(std::string const& content, int file_num)
	{
		this->Write(content, _files[file_num]);
	}*/

	void flush_all()
	{
		for (int i = 0 ; i < _num_cols ; i++) {
			this->Flush(_files[i]);
		}
	}

	inline uint32_t get_num_cols() { return _num_cols; }

	~Writer()
	{
		for (int i = 0 ; i < _num_cols ; i++) {
			this->Close(_files[i]);
		}

		this->DeleteFolder();

		delete [] _files;
	}
private:
	uint32_t _num_cols;
	typename BufferPolicy::file_t* _files = nullptr;
};


const std::string folder("/mnt/raid0_ext2/raid_test/");
static const std::string buffer = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9";

template <typename Writer>
void do_write(Writer& writer)
{
	BENCH_START(w);

	uint32_t num_cols = writer.get_num_cols();
	uint32_t num_rows = NUM_ROWS / num_cols;
	for (uint32_t j = 0 ; j < num_rows; j++) {
		writer.write_cols(buffer);
	}
	writer.flush_all();

	std::stringstream st;
	st << "sequential writes (" << typeid(writer).name() << ") [num_cols=" << num_cols << "]";
	BENCH_END(w, st.str().c_str(), 1, 1, buffer.length(), num_cols*num_rows);
}

int main()
{
	for (uint32_t num_cols : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048})
	{
		{
		Writer<BufferedPolicy> writer_buffered(folder, num_cols);
		do_write(writer_buffered);
		}

		{
		Writer<UnbufferedPolicy> writer_unbuffered(folder, num_cols);
		do_write(writer_unbuffered);
		}

		{
		Writer<RawPolicy> writer_raw(folder, num_cols);
		do_write(writer_raw);
		}
		std::cout << "---" << std::endl;
	}
}


/*int main()
{

#ifdef BUFFER
	FILE* files[NUM_COLS];
#else
	int files[NUM_COLS];
#endif

	std::string folder("/mnt/raid0_ext2/raid_test/");
	//std::string folder("/tmp/noraid_test/");

	system((std::string("mkdir ") + folder + " 2> /dev/null").c_str());

	for (int i = 0 ; i < NUM_COLS ; i++) {
		std::stringstream st;
		st << folder << "file_" << i;

#ifdef BUFFER
		files[i] = fopen(st.str().c_str(), "w");
#else
		files[i] = open(st.str().c_str(), O_WRONLY | O_CREAT);
#endif
	}

	std::string buffer = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9";

	BENCH_START(w);

	for (int j = 0 ; j < NUM_ROWS ; j++) {
		for (int i = 0 ; i < NUM_COLS ; i++) {
#ifdef BUFFER
			fwrite(buffer.c_str(), buffer.length() , 1, files[i]);
#else
			write(files[i], buffer.c_str(), buffer.length());
#endif
		}
	}

	for (int i = 0 ; i < NUM_COLS ; i++) {
#ifdef BUFFER
		fclose(files[i]);
#else
		close(files[i]);
#endif

	}

	BENCH_END(w, "sequential writes", 1, 1, buffer.length(), NUM_COLS*NUM_ROWS);

	system((std::string("rm -rf ") + folder).c_str());
}*/
