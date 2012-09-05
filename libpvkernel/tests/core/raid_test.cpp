#include <iostream>
#include <sstream>
#include <string>

#include <fcntl.h>

#include <pvkernel/core/picviz_bench.h>

#define NUM_COLS 10 // Files
#define NUM_ROWS 50000000

// sync ; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

class BaseBufferPolicy
{
public:
	void CreateFolder(std::string const& folder)
	{
		_folder = folder;

		DeleteFolder();

		system((std::string("mkdir ") + _folder + " 2> /dev/null").c_str());

		for (int i = 0 ; i < NUM_COLS ; i++) {
			std::stringstream st;
			st << _folder << "file_" << i;
			_filenames[i] = st.str();
		}
	}

	void DeleteFolder()
	{
		system((std::string("rm -rf ") + _folder).c_str());
	}
protected:
	std::string _filenames[NUM_COLS];
private:
	std::string _folder;
};

struct BufferedPolicy : public BaseBufferPolicy
{
	typedef FILE* file_t;

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

	file_t Open(std::string const& filename)
	{
		return open(filename.c_str(), O_WRONLY | O_CREAT);
	}

	void Write(std::string const& content, file_t file)
	{
		write(file, content.c_str(), content.length());
	}

	void Flush(file_t file)
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
		return open(filename.c_str(), O_WRONLY | O_CREAT | O_DIRECT);
	}
};

struct RawBufferedPolicy : public BufferedPolicy
{
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
	Writer(std::string const& folder)
	{
		this->CreateFolder(folder);

		for (int i = 0 ; i < NUM_COLS ; i++) {
			_files[i] = this->Open(this->_filenames[i]);
		}
	}

	void write(std::string const& content, int file_num)
	{
		this->Write(content, _files[file_num]);
	}

	void flush_all()
	{
		for (int i = 0 ; i < NUM_COLS ; i++) {
			this->Flush(_files[i]);
		}
	}

	~Writer()
	{
		for (int i = 0 ; i < NUM_COLS ; i++) {
			this->Close(_files[i]);
		}

		this->DeleteFolder();
	}
private:
	typename BufferPolicy::file_t _files[NUM_COLS];
};


int main()
{
	const std::string folder("/mnt/raid0_ext2/raid_test/");
	const std::string buffer = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9";

	//Writer<BufferedPolicy> writer(folder);
	//Writer<UnbufferedPolicy> writer(folder);
	//Writer<RawPolicy> writer(folder);
	Writer<RawBufferedPolicy> writer(folder);

	BENCH_START(w);

	for (int j = 0 ; j < NUM_ROWS ; j++) {
		for (int i = 0 ; i < NUM_COLS ; i++) {
			writer.write(buffer, i);
		}
	}
	writer.flush_all();

	BENCH_END(w, "sequential writes", 1, 1, buffer.length(), NUM_COLS*NUM_ROWS);
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
