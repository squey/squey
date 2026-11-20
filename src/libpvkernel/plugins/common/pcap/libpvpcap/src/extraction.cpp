//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <libpvpcap/extraction.h>

#include <libpvpcap/capture.h>
#include <libpvpcap/flow.h>
#include <libpvpcap/process.h>
#include <libpvpcap.h>

#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVUtils.h>

#include <QDir>
#include <QFile>
#include <QString>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace pvpcap
{

namespace
{

/* What a tshark process keeps of what it reads, measured with the profiles Squey ships: it
 * grows by about 4.3 times the bytes of the packets it read, on top of what it takes from
 * the start. Rounded up, for the protocols that keep more. */
constexpr uint64_t TSHARK_BYTES_PER_CAPTURE_BYTE = 5;
constexpr uint64_t TSHARK_BASE_MEMORY = 200 * 1024 * 1024;

//! What a tshark that reads its part twice keeps on top: measured from 5 to over 40 %.
constexpr uint64_t TWO_PASS_EXTRA_MEMORY_PERCENT = 50;

//! What a compressed capture is assumed to grow to, as long as it was not read.
constexpr uint64_t COMPRESSION_RATIO = 4;

//! Past this, parts get larger rather than more numerous.
constexpr uint64_t MOST_PARTS = 4096;

constexpr size_t QUEUE_BLOCK = 1024 * 1024;
constexpr size_t QUEUE_CAPACITY = 16 * QUEUE_BLOCK;
constexpr size_t FILE_BUFFER = 256 * 1024;
constexpr size_t MOST_OPEN_FILES = 128;

//! Records read before choosing how to split, so as to know what the capture holds.
constexpr size_t PROBED_RECORDS = 256;

//! Fragments of datagrams whose first fragment was seen, forgotten past that many.
constexpr size_t MOST_FRAGMENTED_DATAGRAMS = 1 << 20;

//! What tshark writes before the fields asked for: which frame the line is about.
const std::vector<std::string> FRAME_FIELDS = {"-e", "frame.number", "-e", "frame.len"};

// Packed: millions of frames make every byte count.
#pragma pack(push, 4)
struct frame_info {
	uint16_t part = 0;
	uint16_t wire_length = 0; //!< lower 16 bits, to tell the line of the frame from another
	uint32_t stream = NO_STREAM;
	uint32_t bytes = 0;    //!< of the file, read since the previous frame
	int64_t timestamp = 0; //!< nanoseconds since 1970
};
#pragma pack(pop)
static_assert(sizeof(frame_info) == 20);

/**
 * Frames appended by one thread, and read by another up to what was published.
 *
 * Segments of fixed size are never moved, so that reading needs no lock.
 */
class frame_table
{
	static constexpr size_t SEGMENT = 1 << 16;
	static constexpr size_t MOST_SEGMENTS = 1 << 16;

  public:
	frame_table() : _segments(new std::atomic<frame_info*>[MOST_SEGMENTS]) {}
	~frame_table()
	{
		for (size_t i = 0; i * SEGMENT < _count; i++) {
			delete[] _segments[i].load();
		}
	}

	//! @return false if there are more frames than a NRAW can hold
	bool append(const frame_info& frame)
	{
		const size_t segment = _count / SEGMENT;
		if (segment >= MOST_SEGMENTS) {
			return false;
		}
		if (_count % SEGMENT == 0) {
			_segments[segment].store(new frame_info[SEGMENT], std::memory_order_release);
		}
		_segments[segment].load(std::memory_order_relaxed)[_count % SEGMENT] = frame;
		_count++;
		return true;
	}
	uint64_t count() const { return _count; }
	void publish() { _published.store(_count, std::memory_order_release); }
	uint64_t published() const { return _published.load(std::memory_order_acquire); }
	const frame_info& operator[](uint64_t i) const
	{
		return _segments[i / SEGMENT].load(std::memory_order_acquire)[i % SEGMENT];
	}

  private:
	std::unique_ptr<std::atomic<frame_info*>[]> _segments;
	uint64_t _count = 0;
	std::atomic<uint64_t> _published = 0;
};

/**
 * Bytes on their way from the thread reading the capture to the one feeding a tshark.
 */
class byte_queue
{
  public:
	//! Blocks while the queue is full. @return false if it was aborted
	bool push(std::vector<uint8_t>&& block)
	{
		std::unique_lock<std::mutex> lock(_mutex);
		_cv.wait(lock, [&]() { return _aborted or _queued + block.size() <= QUEUE_CAPACITY; });
		if (_aborted) {
			return false;
		}
		_queued += block.size();
		_blocks.push_back(std::move(block));
		_cv.notify_all();
		return true;
	}

	//! Blocks while the queue is empty. @return false once closed and empty, or aborted
	bool pop(std::vector<uint8_t>& block)
	{
		std::unique_lock<std::mutex> lock(_mutex);
		_cv.wait(lock, [&]() { return _aborted or _closed or not _blocks.empty(); });
		if (_aborted or _blocks.empty()) {
			return false;
		}
		block = std::move(_blocks.front());
		_blocks.pop_front();
		_queued -= block.size();
		_cv.notify_all();
		return true;
	}

	void close()
	{
		const std::lock_guard<std::mutex> lock(_mutex);
		_closed = true;
		_cv.notify_all();
	}

	void abort()
	{
		const std::lock_guard<std::mutex> lock(_mutex);
		_aborted = true;
		_blocks.clear();
		_cv.notify_all();
	}

  private:
	std::mutex _mutex;
	std::condition_variable _cv;
	std::deque<std::vector<uint8_t>> _blocks;
	size_t _queued = 0;
	bool _closed = false;
	bool _aborted = false;
};

/**
 * Reads the lines a tshark writes to a file, as they come.
 */
class line_reader
{
  public:
	~line_reader()
	{
		if (_file != nullptr) {
			std::fclose(_file);
		}
	}

	/**
	 * The next whole line, without its end, if there is one yet.
	 *
	 * @param ended whether the writer is done: whatever follows the last line end is then a
	 *        line of its own
	 */
	bool peek(const std::string& path, bool ended, std::string_view& line)
	{
		if (_file == nullptr) {
			_file = open_file(path, "rb");
			if (_file == nullptr) {
				return false;
			}
			_buffer.resize(QUEUE_BLOCK);
		}
		while (true) {
			const char* begin = _buffer.data() + _begin;
			const char* eol = static_cast<const char*>(std::memchr(begin, '\n', _end - _begin));
			if (eol != nullptr) {
				line = std::string_view(begin, eol - begin);
				_next = eol - _buffer.data() + 1;
				return true;
			}
			if (not fill()) {
				if (ended and _end > _begin) {
					line = std::string_view(begin, _end - _begin);
					_next = _end;
					return true;
				}
				return false;
			}
		}
	}

	//! Moves past the line peek() returned.
	void consume() { _begin = _next; }

  private:
	bool fill()
	{
		if (_begin > 0) {
			std::memmove(_buffer.data(), _buffer.data() + _begin, _end - _begin);
			_end -= _begin;
			_next -= std::min(_next, _begin);
			_begin = 0;
		}
		if (_end == _buffer.size()) {
			_buffer.resize(_buffer.size() * 2);
		}
		// What the writer added since the end of file was last met.
		std::clearerr(_file);
		const size_t read = std::fread(_buffer.data() + _end, 1, _buffer.size() - _end, _file);
		_end += read;
		return read > 0;
	}

	std::FILE* _file = nullptr;
	std::vector<char> _buffer;
	size_t _begin = 0;
	size_t _end = 0;
	size_t _next = 0;
};

void remove_file(const std::string& path)
{
	QFile::remove(QString::fromStdString(path));
}

/**
 * The part of a capture written to a file. Only the most recently written to stay open, so
 * that thousands of parts don't run out of file descriptors.
 */
class part_file
{
  public:
	using open_files = std::list<part_file*>;

	part_file(std::string path, open_files& open) : _path(std::move(path)), _open(open) {}
	~part_file() { close(); }

	bool write(const uint8_t* data, size_t size)
	{
		_buffer.insert(_buffer.end(), data, data + size);
		return _buffer.size() < FILE_BUFFER or flush();
	}

	bool close()
	{
		const bool flushed = flush();
		return close_file() and flushed;
	}

	const std::string& path() const { return _path; }

  private:
	bool flush()
	{
		if (_buffer.empty()) {
			return true;
		}
		if (_file == nullptr) {
			_file = open_file(_path, _created ? "ab" : "wb");
			if (_file == nullptr) {
				return false;
			}
			_created = true;
			_open.push_back(this);
			_where = std::prev(_open.end());
			if (_open.size() > MOST_OPEN_FILES) {
				_open.front()->close_file();
			}
		} else {
			_open.splice(_open.end(), _open, _where);
		}
		const bool written =
		    std::fwrite(_buffer.data(), 1, _buffer.size(), _file) == _buffer.size();
		_buffer.clear();
		return written;
	}

	bool close_file()
	{
		if (_file == nullptr) {
			return true;
		}
		const bool closed = std::fclose(_file) == 0;
		_file = nullptr;
		_open.erase(_where);
		return closed;
	}

	std::string _path;
	open_files& _open;
	open_files::iterator _where;
	std::vector<uint8_t> _buffer;
	std::FILE* _file = nullptr;
	bool _created = false;
};

enum class part_state { waiting, running, done, failed };

struct part {
	size_t index = 0;
	std::string input;  //!< file of the records, for a part read once written
	std::string output; //!< of tshark
	std::string errors; //!< of tshark

	// Thread reading the capture
	uint64_t assigned = 0; //!< bytes of the records given to the part
	uint64_t frames = 0;
	std::vector<uint8_t> block;
	std::unique_ptr<part_file> file;

	// Thread feeding the tshark of a part fed as the capture is read
	byte_queue queue;
	//! Bytes its tshark got: as they are fed, or all of them once it read them from a file.
	std::atomic<uint64_t> fed = 0;

	std::unique_ptr<child_process> process; //!< under implementation::processes
	std::atomic<part_state> state = part_state::waiting;

	// Thread calling next()
	line_reader lines;
	uint64_t walked = 0; //!< frames of the part the lines were looked for

	/* Number in the capture of each frame of the part, from 0: tshark numbers the frames it
	 * read, and the fields naming other frames have to be told which they are. Appended to
	 * as the capture is read, before tshark gets the frame, and read as its lines come. */
	std::mutex numbers_mutex;
	std::vector<uint32_t> numbers;
};

struct owned_record {
	capture_record record;
	std::vector<uint8_t> bytes; //!< what the spans of a record kept for later point to
	uint64_t file_position = 0;
};

struct fragment_key {
	flow_key datagram;
	uint32_t id = 0;
	bool operator==(const fragment_key&) const = default;
};

struct fragment_key_hash {
	size_t operator()(const fragment_key& key) const noexcept
	{
		return flow_key_hash()(key.datagram) ^ (size_t(key.id) * 0x9e3779b97f4a7c15ULL);
	}
};

struct conversation {
	uint32_t stream;
	uint16_t part;
};

//! Days since 1970-01-01 of a date of the proleptic Gregorian calendar.
int64_t days_from_civil(int64_t y, unsigned m, unsigned d)
{
	y -= m <= 2;
	const int64_t era = (y >= 0 ? y : y - 399) / 400;
	const unsigned yoe = static_cast<unsigned>(y - era * 400);
	const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
	const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
	return era * 146097 + static_cast<int64_t>(doe) - 719468;
}

void civil_from_days(int64_t z, int64_t& y, unsigned& m, unsigned& d)
{
	z += 719468;
	const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
	const unsigned doe = static_cast<unsigned>(z - era * 146097);
	const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
	const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
	const unsigned mp = (5 * doy + 2) / 153;
	d = doy - (153 * mp + 2) / 5 + 1;
	m = mp < 10 ? mp + 3 : mp - 9;
	y = static_cast<int64_t>(yoe) + era * 400 + (m <= 2);
}

constexpr int64_t NANOSECONDS = 1000000000;

int64_t floor_div(int64_t a, int64_t b)
{
	return a / b - ((a % b != 0) and ((a < 0) != (b < 0)));
}

void append_utc(std::string& out, int64_t ns)
{
	const int64_t seconds = floor_div(ns, NANOSECONDS);
	const int64_t fraction = ns - seconds * NANOSECONDS;
	const int64_t days = floor_div(seconds, 86400);
	const int64_t time = seconds - days * 86400;
	int64_t year;
	unsigned month, day;
	civil_from_days(days, year, month, day);
	char text[64];
	const int length =
	    std::snprintf(text, sizeof(text), "%04lld-%02u-%02uT%02lld:%02lld:%02lld.%09lld+0000",
	                  static_cast<long long>(year), month, day, static_cast<long long>(time / 3600),
	                  static_cast<long long>(time / 60 % 60), static_cast<long long>(time % 60),
	                  static_cast<long long>(fraction));
	out.append(text, length);
}

void append_duration(std::string& out, int64_t ns)
{
	if (ns < 0) {
		out += '-';
		ns = -ns;
	}
	char text[48];
	const int length =
	    std::snprintf(text, sizeof(text), "%lld.%09lld", static_cast<long long>(ns / NANOSECONDS),
	                  static_cast<long long>(ns % NANOSECONDS));
	out.append(text, length);
}

bool read_digits(std::string_view& text, size_t count, int64_t& value)
{
	if (text.size() < count) {
		return false;
	}
	value = 0;
	for (size_t i = 0; i < count; i++) {
		if (text[i] < '0' or text[i] > '9') {
			return false;
		}
		value = value * 10 + (text[i] - '0');
	}
	text.remove_prefix(count);
	return true;
}

bool skip(std::string_view& text, char c)
{
	if (text.empty() or text.front() != c) {
		return false;
	}
	text.remove_prefix(1);
	return true;
}

//! Nanoseconds of what follows a decimal point, as many digits as there are.
int64_t read_fraction(std::string_view& text)
{
	int64_t fraction = 0;
	int digits = 0;
	while (not text.empty() and text.front() >= '0' and text.front() <= '9') {
		if (digits < 9) {
			fraction = fraction * 10 + (text.front() - '0');
			digits++;
		}
		text.remove_prefix(1);
	}
	for (; digits < 9; digits++) {
		fraction *= 10;
	}
	return fraction;
}

/**
 * Reads an absolute time as tshark writes it: in ISO 8601, with an offset or as UTC, or as
 * seconds since 1970.
 */
bool parse_absolute_time(std::string_view text, int64_t& ns)
{
	if (text.find('-') == std::string_view::npos) {
		// 1304466181.545400000
		const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), ns);
		if (error != std::errc()) {
			return false;
		}
		text.remove_prefix(end - text.data());
		ns *= NANOSECONDS;
		if (skip(text, '.')) {
			ns += read_fraction(text);
		}
		return text.empty();
	}

	// 2011-05-03T23:43:01.545400000+0000, or ...Z, or ...+02:00
	int64_t year, month, day, hour, minute, second;
	if (not read_digits(text, 4, year) or not skip(text, '-') or not read_digits(text, 2, month) or
	    not skip(text, '-') or not read_digits(text, 2, day) or
	    not(skip(text, 'T') or skip(text, ' ')) or not read_digits(text, 2, hour) or
	    not skip(text, ':') or not read_digits(text, 2, minute) or not skip(text, ':') or
	    not read_digits(text, 2, second) or month < 1 or month > 12 or day < 1 or day > 31) {
		return false;
	}
	int64_t fraction = 0;
	if (skip(text, '.')) {
		fraction = read_fraction(text);
	}
	int64_t offset = 0;
	if (not text.empty() and (text.front() == '+' or text.front() == '-')) {
		const int64_t sign = text.front() == '-' ? -1 : 1;
		text.remove_prefix(1);
		int64_t hours, minutes;
		if (not read_digits(text, 2, hours)) {
			return false;
		}
		skip(text, ':');
		if (not read_digits(text, 2, minutes)) {
			return false;
		}
		offset = sign * (hours * 3600 + minutes * 60);
	} else {
		skip(text, 'Z');
	}
	if (not text.empty()) {
		return false;
	}
	const int64_t days = days_from_civil(year, unsigned(month), unsigned(day));
	ns = ((days * 86400 + hour * 3600 + minute * 60 + second) - offset) * NANOSECONDS + fraction;
	return true;
}

/**
 * Calls `f` on each value of a field, which tshark separates by the aggregator when it
 * writes them all, and appends what it returns in place of the value.
 */
template <typename F>
void rewrite_values(std::string& out, std::string_view field, const std::string& aggregator, F&& f)
{
	while (true) {
		const size_t end = aggregator.empty() ? std::string_view::npos : field.find(aggregator);
		f(out, field.substr(0, end));
		if (end == std::string_view::npos) {
			break;
		}
		out += aggregator;
		field.remove_prefix(end + aggregator.size());
	}
}

//! Reads a number, and the separator after it unless it ends the line.
template <typename T>
bool parse_number(std::string_view& text, T& value)
{
	const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
	const size_t read = end - text.data();
	if (error != std::errc() or (read < text.size() and *end != SEPARATOR[0])) {
		return false;
	}
	text.remove_prefix(std::min(read + 1, text.size()));
	return true;
}

} // namespace

struct extraction::implementation {
	enum class mode {
		fed,     //!< tshark processes fed through pipes while the capture is read
		written, //!< parts written to files, read by tshark processes once written
		direct,  //!< a single tshark reads the capture, which this can't read
	};

	implementation(const std::string& capture, const options& opts);
	~implementation();

	void choose_mode();
	void start_tshark(part& p, const std::string& input);
	void read_capture();
	bool dispatch(const owned_record& r);
	void broadcast(std::span<const uint8_t> bytes);
	bool give(part& p, std::span<const uint8_t> bytes);
	uint16_t choose_part(const flow_key& key);
	void feed(part& p);
	void run_written_parts();
	void finished(part& p, const child_process::status& status);
	void fail(const std::string& why);
	void stop();
	bool next_line(part& p, std::string_view& line);
	bool next_direct(line& l);

	std::string capture;
	options opts;
	std::string directory;
	mode how = mode::fed;
	std::vector<std::unique_ptr<part>> parts;

	std::unique_ptr<capture_reader> reader;
	std::vector<owned_record> probed;
	part_file::open_files open_parts;
	frame_table table;
	std::unordered_map<flow_key, conversation, flow_key_hash> conversations;
	std::unordered_map<fragment_key, flow_key, fragment_key_hash> fragments;
	uint64_t previous_file_position = 0;
	//! For read_bytes(): where the records given to the parts end in the file, and their bytes.
	std::atomic<uint64_t> given_position = 0;
	std::atomic<uint64_t> given_bytes = 0;
	mutable std::atomic<uint64_t> told_read_bytes = 0;

	std::thread reading;
	std::vector<std::thread> workers;
	std::atomic<size_t> next_written_part = 0;

	std::mutex processes; //!< guards part::process, and the decision to start one
	std::atomic<bool> stopping = false;
	mutable std::mutex failure_mutex;
	std::string failure;

	std::atomic<bool> read = false; //!< the capture was read: the members below are set
	uint64_t frame_count = 0;
	uint32_t stream_count = 0;
	std::string damage;

	/**
	 * The fields asked for, from what tshark wrote of the frame: `p` is the part it read,
	 * null for a tshark that read the whole capture.
	 */
	std::string_view rewrite(std::string_view fields, part* p, uint64_t frame, uint32_t length);

	enum class own_field : uint8_t { no, number, length };
	std::vector<own_field> own; //!< of each field asked for

	bool numbering = false; //!< whether fields name frames, for which part::numbers is kept
	bool rewriting = false; //!< whether some field is rewritten

	void rewrite_summary(std::string& out, std::string_view summary, part& p);
	bool global_number(part& p, std::string_view local, uint64_t& global);

	// Thread calling next()
	std::string rewritten;
	bool shown = false;          //!< whether next() returned a line yet
	int64_t shown_timestamp = 0; //!< of the frame of the line next() returned last
	uint64_t walked = 0;
	uint64_t pending_bytes = 0;
	part* consumed = nullptr; //!< whose line next() returned last, to move past
};

extraction::implementation::implementation(const std::string& capture_path, const options& o)
    : capture(capture_path), opts(o)
{
	if (opts.workers == 0) {
		opts.workers = std::max(1u, std::thread::hardware_concurrency());
	}
	for (const field& f : opts.fields) {
		numbering |= f.type == field::kind::frames or f.type == field::kind::summary;
		rewriting |= f.type != field::kind::text;
		own.push_back(f.name == "frame.number" ? own_field::number
		              : f.name == "frame.len"  ? own_field::length
		                                       : own_field::no);
		rewriting |= own.back() != own_field::no;
	}
	QDir().mkpath(QString::fromStdString(opts.work_directory));
	// Named as a NRAW directory, so that what a crash left behind is removed as one is.
	directory =
	    PVCore::mkdtemp(QString::fromStdString(opts.work_directory) + "/nraw-XXXXXX").toStdString();
	if (directory.empty()) {
		throw extraction_error("Can't create a temporary directory in '" + opts.work_directory +
		                       "'");
	}

	try {
		reader = std::make_unique<capture_reader>(capture);
		owned_record r;
		while (probed.size() < PROBED_RECORDS and reader->next(r.record)) {
			const size_t frame_offset =
			    r.record.frame.empty() ? 0 : r.record.frame.data() - r.record.bytes.data();
			r.bytes.assign(r.record.bytes.begin(), r.record.bytes.end());
			r.record.frame = {r.bytes.data() + frame_offset, r.record.frame.size()};
			r.record.bytes = {r.bytes.data(), r.bytes.size()};
			r.file_position = reader->file_position();
			probed.push_back(std::move(r));
			r = owned_record();
		}
	} catch (const capture_error&) {
		// Read by tshark alone, which knows more formats than capture_reader does.
		reader.reset();
	}

	choose_mode();

	try {
		if (how == mode::direct) {
			part& p = *parts.front();
			start_tshark(p, capture);
			workers.emplace_back([this, &p]() {
				const child_process::status status = p.process->wait();
				finished(p, status);
			});
			return;
		}
		for (const auto& p : parts) {
			if (how == mode::fed) {
				start_tshark(*p, "");
				if (reader->file_format() == capture_reader::format::pcap) {
					p->block.assign(reader->header().begin(), reader->header().end());
				}
				workers.emplace_back([this, &p = *p]() { feed(p); });
			} else {
				p->input = directory + "/" + std::to_string(p->index) + ".pcap";
				p->file = std::make_unique<part_file>(p->input, open_parts);
				p->file->write(reader->header().data(), reader->header().size());
			}
		}
		reading = std::thread([this]() { read_capture(); });
	} catch (...) {
		stop();
		for (std::thread& worker : workers) {
			worker.join();
		}
		parts.clear();
		PVCore::PVDirectory::remove_rec(QString::fromStdString(directory));
		throw;
	}
}

void extraction::implementation::choose_mode()
{
	if (reader == nullptr) {
		how = mode::direct;
		parts.push_back(std::make_unique<part>());
		parts.back()->output = directory + "/0.csv";
		parts.back()->errors = directory + "/0.err";
		return;
	}

	uint64_t size = reader->size();
	if (reader->compressed()) {
		size *= COMPRESSION_RATIO;
	}
	uint64_t memory = opts.memory;
	if (memory == 0) {
		memory = std::max<uint64_t>(PVCore::available_memory() / 2, 1024ULL * 1024 * 1024);
	}

	// Of each byte it reads, in percent.
	uint64_t kept = TSHARK_BYTES_PER_CAPTURE_BYTE * 100;
	if (opts.two_pass) {
		kept += kept * TWO_PASS_EXTRA_MEMORY_PERCENT / 100;
	}

	size_t count = 1;
	// A pipe can't be read twice.
	how = opts.two_pass ? mode::written : mode::fed;
	// A few frames are not worth more than one tshark.
	if (reader->splittable() and size >= opts.smallest_split and opts.workers > 1 and
	    opts.most_frames == 0) {
		const uint64_t fed_memory = size * kept / 100 + opts.workers * TSHARK_BASE_MEMORY;
		if (fed_memory <= memory) {
			count = std::clamp<uint64_t>(size / std::max<uint64_t>(opts.smallest_part, 1), 1,
			                             opts.workers);
		} else {
			// Parts small enough for as many tshark as workers to hold them at once.
			const uint64_t per_worker = memory / opts.workers;
			const uint64_t part_size = std::max<uint64_t>(
			    (per_worker > TSHARK_BASE_MEMORY ? per_worker - TSHARK_BASE_MEMORY : 0) * 100 /
			        kept,
			    std::max<uint64_t>(opts.smallest_part, 1));
			count = std::clamp<uint64_t>((size + part_size - 1) / part_size, opts.workers,
			                             std::max<uint64_t>(MOST_PARTS, opts.workers));
			how = mode::written;
		}
	}
	for (size_t i = 0; i < count; i++) {
		parts.push_back(std::make_unique<part>());
		part& p = *parts.back();
		p.index = i;
		p.output = directory + "/" + std::to_string(i) + ".csv";
		p.errors = directory + "/" + std::to_string(i) + ".err";
	}
}

void extraction::implementation::start_tshark(part& p, const std::string& input)
{
	child_process::options process;
	process.argv.push_back(opts.program.empty() ? tshark_path() : opts.program);
	if (opts.two_pass) {
		// tshark refuses to read its standard input twice, even from a file.
		process.argv.insert(process.argv.end(), {"-2", "-r", input});
	} else {
		process.argv.insert(process.argv.end(), {"-r", "-"});
	}
	process.argv.insert(process.argv.end(),
	                    {"-T", "fields", "-E", std::string("separator=") + SEPARATOR, "-E",
	                     "quote=n", "-E", "header=n"});
	process.argv.insert(process.argv.end(), FRAME_FIELDS.begin(), FRAME_FIELDS.end());
	process.argv.insert(process.argv.end(), opts.tshark_options.begin(), opts.tshark_options.end());
	if (how == mode::direct and opts.most_frames != 0) {
		process.argv.insert(process.argv.end(), {"-c", std::to_string(opts.most_frames)});
	}
	for (size_t i = 0; i < opts.fields.size(); i++) {
		if (own[i] == own_field::no) {
			process.argv.push_back("-e");
			process.argv.push_back(opts.fields[i].name);
		}
	}
	process.input = input;
	process.output = p.output;
	process.errors = p.errors;
	process.environment = opts.environment;

	const std::lock_guard<std::mutex> lock(processes);
	if (stopping) {
		p.state = part_state::failed;
		return;
	}
	try {
		p.process = std::make_unique<child_process>(process);
	} catch (const process_error& e) {
		p.state = part_state::failed;
		throw extraction_error(e.what());
	}
	p.state = part_state::running;
}

extraction::implementation::~implementation()
{
	stop();
	if (reading.joinable()) {
		reading.join();
	}
	for (std::thread& worker : workers) {
		worker.join();
	}
	parts.clear();
	PVCore::PVDirectory::remove_rec(QString::fromStdString(directory));
}

void extraction::implementation::stop()
{
	stopping = true;
	for (const auto& p : parts) {
		p->queue.abort();
	}
	const std::lock_guard<std::mutex> lock(processes);
	for (const auto& p : parts) {
		if (p->process) {
			p->process->kill();
		}
	}
}

void extraction::implementation::fail(const std::string& why)
{
	{
		const std::lock_guard<std::mutex> lock(failure_mutex);
		if (failure.empty()) {
			failure = why;
		}
	}
	stop();
}

void extraction::implementation::finished(part& p, const child_process::status& status)
{
	if (stopping) {
		p.state = part_state::failed;
		return;
	}
	const bool succeeded = status.exited and status.code == 0;
	if (succeeded) {
		p.state = part_state::done;
		return;
	}
	std::string said = tshark_errors(p.errors);
	if (how == mode::direct and
	    said.find("cut short in the middle of a packet") != std::string::npos) {
		// Reading the capture itself, tshark says when it ends in the middle of a packet,
		// and exits with an error once it wrote all the packets before.
		{
			const std::lock_guard<std::mutex> lock(failure_mutex);
			damage = said;
		}
		p.state = part_state::done;
		return;
	}
	if (said.empty()) {
		said = status.exited ? "tshark exited with code " + std::to_string(status.code)
		                     : "tshark was killed by signal " + std::to_string(status.code);
	}
	// Failed before the part says so, so that whoever waits for its lines sees why it stopped.
	fail(said);
	p.state = part_state::failed;
}

void extraction::implementation::feed(part& p)
{
	if (p.process == nullptr) {
		return;
	}
	std::vector<uint8_t> block;
	while (p.queue.pop(block)) {
		if (not p.process->write(block.data(), block.size())) {
			// It stopped reading: what it says as it ends tells why.
			p.queue.abort();
			break;
		}
		p.fed += block.size();
	}
	p.process->close_input();
	const child_process::status status = p.process->wait();
	finished(p, status);
}

void extraction::implementation::run_written_parts()
{
	const size_t count = std::min(opts.workers, parts.size());
	for (size_t w = 0; w < count; w++) {
		workers.emplace_back([this]() {
			while (not stopping) {
				const size_t i = next_written_part++;
				if (i >= parts.size()) {
					break;
				}
				part& p = *parts[i];
				if (p.frames == 0) {
					// Nothing for tshark to read, and nothing to say.
					remove_file(p.input);
					p.fed = p.assigned;
					p.state = part_state::done;
					continue;
				}
				try {
					start_tshark(p, p.input);
				} catch (const extraction_error& e) {
					fail(e.what());
					break;
				}
				if (p.process == nullptr) {
					break;
				}
				const child_process::status status = p.process->wait();
				remove_file(p.input);
				p.fed = p.assigned;
				finished(p, status);
			}
		});
	}
}

uint16_t extraction::implementation::choose_part(const flow_key& key)
{
	if (parts.size() == 1) {
		return 0;
	}
	if (how == mode::fed) {
		// The tshark with the least left to read: a slow one gets fewer new conversations.
		size_t best = 0;
		uint64_t best_backlog = UINT64_MAX;
		for (size_t i = 0; i < parts.size(); i++) {
			const uint64_t backlog = parts[i]->assigned - parts[i]->fed;
			if (backlog < best_backlog) {
				best = i;
				best_backlog = backlog;
			}
		}
		return static_cast<uint16_t>(best);
	}
	// Of two parts picked by the conversation, the smaller: as even as looking at all of
	// them, which thousands of parts would make slow.
	const size_t hash = flow_key_hash()(key);
	const size_t first = hash % parts.size();
	const size_t second = (hash / parts.size()) % parts.size();
	return static_cast<uint16_t>(parts[first]->assigned <= parts[second]->assigned ? first
	                                                                               : second);
}

bool extraction::implementation::give(part& p, std::span<const uint8_t> bytes)
{
	p.assigned += bytes.size();
	given_bytes += bytes.size();
	if (how == mode::written) {
		if (not p.file->write(bytes.data(), bytes.size())) {
			fail("Can't write to '" + p.input + "': " + std::strerror(errno));
			return false;
		}
		return true;
	}
	p.block.insert(p.block.end(), bytes.begin(), bytes.end());
	if (p.block.size() >= QUEUE_BLOCK) {
		if (not p.queue.push(std::move(p.block))) {
			return false;
		}
		p.block = std::vector<uint8_t>();
		p.block.reserve(QUEUE_BLOCK + 64 * 1024);
	}
	return true;
}

void extraction::implementation::broadcast(std::span<const uint8_t> bytes)
{
	for (const auto& p : parts) {
		const uint64_t assigned = p->assigned;
		give(*p, bytes);
		// What every part gets says nothing of how busy one is.
		p->assigned = assigned;
	}
}

bool extraction::implementation::dispatch(const owned_record& r)
{
	given_position = r.file_position;
	const capture_record& record = r.record;
	if (record.type == capture_record::kind::metadata) {
		broadcast(record.bytes);
		return not stopping;
	}

	frame_info frame;
	frame.bytes = static_cast<uint32_t>(
	    std::min<uint64_t>(r.file_position - previous_file_position, UINT32_MAX));
	previous_file_position = r.file_position;

	flow_key key;
	if (record.type == capture_record::kind::packet) {
		const packet_flow flow = flow_of(record.link_type, record.frame);
		key = flow.key;
		if (flow.fragment) {
			const fragment_key datagram{flow.datagram, flow.datagram_id};
			if (flow.first_fragment) {
				if (fragments.size() >= MOST_FRAGMENTED_DATAGRAMS) {
					fragments.clear();
				}
				fragments[datagram] = flow.key;
			} else if (auto it = fragments.find(datagram); it != fragments.end()) {
				key = it->second;
			}
		}
		frame.wire_length = static_cast<uint16_t>(record.wire_length);
		frame.timestamp = record.timestamp;
	} else {
		// A frame that holds no packet goes with the other frames of its kind.
		key.family = 0xfe;
	}

	auto [it, inserted] = conversations.try_emplace(key, conversation{0, 0});
	if (inserted) {
		it->second.stream = stream_count++;
		it->second.part = choose_part(key);
	}
	frame.part = it->second.part;
	frame.stream = record.type == capture_record::kind::packet ? it->second.stream : NO_STREAM;
	parts[frame.part]->frames++;

	if (not table.append(frame)) {
		fail("The capture holds more frames than can be imported");
		return false;
	}
	if (numbering) {
		part& p = *parts[frame.part];
		const std::lock_guard<std::mutex> lock(p.numbers_mutex);
		p.numbers.push_back(static_cast<uint32_t>(table.count() - 1));
	}
	if (table.count() % 1024 == 0) {
		table.publish();
	}
	return give(*parts[frame.part], record.bytes);
}

void extraction::implementation::read_capture()
{
	try {
		const auto wanted = [this]() {
			return opts.most_frames == 0 or table.count() < opts.most_frames;
		};
		bool going = true;
		for (const owned_record& r : probed) {
			if (not wanted() or not(going = dispatch(r))) {
				break;
			}
		}
		probed.clear();
		owned_record r;
		while (going and not stopping and wanted() and reader->next(r.record)) {
			r.file_position = reader->file_position();
			going = dispatch(r);
		}
		const std::lock_guard<std::mutex> lock(failure_mutex);
		damage = reader->damage();
	} catch (const std::exception& e) {
		fail(std::string("Can't read the capture: ") + e.what());
	}
	table.publish();
	frame_count = table.count();

	if (how == mode::fed) {
		for (const auto& p : parts) {
			if (not p->block.empty()) {
				p->queue.push(std::move(p->block));
			}
			p->queue.close();
		}
	} else {
		for (const auto& p : parts) {
			if (not p->file->close()) {
				fail("Can't write to '" + p->input + "': " + std::strerror(errno));
			}
		}
		if (not stopping) {
			run_written_parts();
		}
	}
	read = true;
}

bool extraction::implementation::next_line(part& p, std::string_view& line)
{
	using namespace std::chrono_literals;
	auto pause = 100us;
	while (true) {
		if (stopping) {
			return false;
		}
		const part_state state = p.state;
		const bool ended = state == part_state::done or state == part_state::failed;
		if (state != part_state::waiting and p.lines.peek(p.output, ended, line)) {
			return true;
		}
		if (ended) {
			return false;
		}
		std::this_thread::sleep_for(pause);
		pause = std::min<std::chrono::microseconds>(pause * 2, 5ms);
	}
}

bool extraction::implementation::next_direct(line& l)
{
	part& p = *parts.front();
	std::string_view text;
	if (not next_line(p, text)) {
		return false;
	}
	p.lines.consume();
	if (text.ends_with('\r')) {
		text.remove_suffix(1);
	}
	uint64_t number = 0;
	uint32_t length = 0;
	if (not parse_number(text, number) or not parse_number(text, length)) {
		fail("tshark wrote a line that is not about a frame: " + std::string(text.substr(0, 200)));
		return false;
	}
	l.frame = number;
	l.stream = NO_STREAM;
	l.bytes = 0;
	l.fields = rewriting ? rewrite(text, nullptr, number - 1, length) : text;
	frame_count = number;
	return true;
}

bool extraction::implementation::global_number(part& p, std::string_view local, uint64_t& global)
{
	uint64_t number = 0;
	const auto [end, error] = std::from_chars(local.data(), local.data() + local.size(), number);
	if (error != std::errc() or end != local.data() + local.size() or number == 0) {
		return false;
	}
	const std::lock_guard<std::mutex> lock(p.numbers_mutex);
	if (number > p.numbers.size()) {
		return false;
	}
	global = uint64_t(p.numbers[number - 1]) + 1;
	return true;
}

/**
 * Of the frames the Info column names, those where a PDU is reassembled: "[TCP PDU
 * reassembled in 927]", "[Reassembled in #12]". Other numbers in it are left as they are.
 */
void extraction::implementation::rewrite_summary(std::string& out,
                                                 std::string_view summary,
                                                 part& p)
{
	static constexpr std::string_view mention = "eassembled in ";
	while (true) {
		const size_t at = summary.find(mention);
		if (at == std::string_view::npos) {
			out += summary;
			return;
		}
		size_t begin = at + mention.size();
		if (begin < summary.size() and summary[begin] == '#') {
			begin++;
		}
		size_t end = begin;
		while (end < summary.size() and summary[end] >= '0' and summary[end] <= '9') {
			end++;
		}
		uint64_t global = 0;
		out += summary.substr(0, begin);
		if (global_number(p, summary.substr(begin, end - begin), global)) {
			out += std::to_string(global);
		} else {
			out += summary.substr(begin, end - begin);
		}
		summary.remove_prefix(end);
	}
}

std::string_view extraction::implementation::rewrite(std::string_view fields,
                                                     part* p,
                                                     uint64_t frame,
                                                     uint32_t length)
{
	// The frame of a line of the tshark that read the whole capture, which is all the
	// fields name already.
	const bool whole = p == nullptr;
	const frame_info* info = whole ? nullptr : &table[frame];
	rewritten.clear();
	size_t at = 0;
	const auto next_value = [&]() {
		const size_t begin = at;
		while (at < fields.size() and fields[at] != SEPARATOR[0]) {
			// What tshark escapes is part of the value.
			at += fields[at] == '\\' ? 2 : 1;
		}
		const std::string_view value = fields.substr(begin, std::min(at, fields.size()) - begin);
		at++;
		return value;
	};

	for (size_t index = 0; index < opts.fields.size(); index++) {
		if (index > 0) {
			rewritten += SEPARATOR[0];
		}
		// What tshark writes first, which it is not asked for twice: it would only write
		// the last of them.
		if (own[index] == own_field::number) {
			rewritten += std::to_string(frame + 1);
			continue;
		}
		if (own[index] == own_field::length) {
			rewritten += std::to_string(length);
			continue;
		}
		const std::string_view value = next_value();
		switch (opts.fields[index].type) {
		case field::kind::text:
			rewritten += value;
			break;
		case field::kind::frames:
			if (whole) {
				rewritten += value;
				break;
			}
			rewrite_values(rewritten, value, opts.aggregator,
			               [&](std::string& out, std::string_view v) {
				               uint64_t global = 0;
				               if (global_number(*p, v, global)) {
					               out += std::to_string(global);
				               } else {
					               out += v;
				               }
			               });
			break;
		case field::kind::absolute_time:
			rewrite_values(rewritten, value, opts.aggregator,
			               [](std::string& out, std::string_view v) {
				               int64_t ns = 0;
				               if (parse_absolute_time(v, ns)) {
					               append_utc(out, ns);
				               } else {
					               out += v;
				               }
			               });
			break;
		case field::kind::time_since_first:
			if (whole) {
				rewritten += value;
			} else {
				append_duration(rewritten, info->timestamp - table[0].timestamp);
			}
			break;
		case field::kind::time_since_previous:
			if (whole) {
				rewritten += value;
			} else if (frame > 0) {
				// tshark leaves it empty for the first frame it read, which only the first
				// frame of the capture is.
				append_duration(rewritten, info->timestamp - table[frame - 1].timestamp);
			}
			break;
		case field::kind::time_since_shown:
			if (whole) {
				rewritten += value;
			} else if (shown) {
				append_duration(rewritten, info->timestamp - shown_timestamp);
			}
			break;
		case field::kind::summary:
			if (whole) {
				rewritten += value;
			} else {
				rewrite_summary(rewritten, value, *p);
			}
			break;
		}
	}
	if (info != nullptr) {
		shown_timestamp = info->timestamp;
		shown = true;
	}
	return rewritten;
}

extraction::extraction(const std::string& capture, const options& opts)
    : _impl(std::make_unique<implementation>(capture, opts))
{
}

extraction::~extraction() = default;

bool extraction::next(line& l)
{
	implementation& x = *_impl;
	const auto failed = [&x]() {
		const std::lock_guard<std::mutex> lock(x.failure_mutex);
		if (not x.failure.empty()) {
			throw extraction_error(x.failure);
		}
	};

	if (x.consumed != nullptr) {
		x.consumed->lines.consume();
		x.consumed = nullptr;
	}

	using namespace std::chrono_literals;
	if (x.how == implementation::mode::direct) {
		if (x.next_direct(l)) {
			return true;
		}
		while (not x.stopping and x.parts.front()->state == part_state::running) {
			std::this_thread::sleep_for(1ms);
		}
		failed();
		return false;
	}

	auto pause = 100us;
	while (not x.stopping) {
		uint64_t published = x.table.published();
		if (x.walked >= published) {
			if (x.read) {
				published = x.table.published();
				if (x.walked >= published) {
					break;
				}
				continue;
			}
			std::this_thread::sleep_for(pause);
			pause = std::min<std::chrono::microseconds>(pause * 2, 5ms);
			continue;
		}
		pause = 100us;

		const frame_info& frame = x.table[x.walked];
		part& p = *x.parts[frame.part];
		const uint64_t local = p.walked + 1;
		std::string_view text;
		if (not x.next_line(p, text)) {
			if (x.stopping) {
				break;
			}
			// The display filter dropped this frame, and those after it in its part.
			p.walked = local;
			x.walked++;
			x.pending_bytes += frame.bytes;
			continue;
		}

		if (text.ends_with('\r')) {
			text.remove_suffix(1);
		}
		uint64_t number = 0;
		uint32_t length = 0;
		if (not parse_number(text, number) or not parse_number(text, length)) {
			x.fail("tshark wrote a line that is not about a frame: " +
			       std::string(text.substr(0, 200)));
			break;
		}
		if (number > local) {
			p.walked = local;
			x.walked++;
			x.pending_bytes += frame.bytes;
			continue;
		}
		if (number < local or
		    (frame.stream != NO_STREAM and frame.wire_length != static_cast<uint16_t>(length))) {
			x.fail("tshark did not read the frames of the capture as they were given to it: "
			       "frame " +
			       std::to_string(x.walked + 1) + " of " + std::to_string(frame.wire_length) +
			       " bytes, written as frame " + std::to_string(number) + " of part " +
			       std::to_string(frame.part) + ", of " + std::to_string(length) + " bytes");
			break;
		}

		p.walked = local;
		x.walked++;
		l.frame = x.walked;
		l.stream = frame.stream;
		l.bytes = x.pending_bytes + frame.bytes;
		l.fields = x.rewriting ? x.rewrite(text, &p, x.walked - 1, length) : text;
		x.pending_bytes = 0;
		x.consumed = &p;
		return true;
	}

	// What a tshark says as it exits, once it wrote all its lines, is said too.
	for (const auto& p : x.parts) {
		while (not x.stopping and
		       (p->state == part_state::waiting or p->state == part_state::running)) {
			std::this_thread::sleep_for(1ms);
		}
	}
	failed();
	return false;
}

void extraction::cancel()
{
	_impl->stop();
}

uint64_t extraction::frames() const
{
	return _impl->frame_count;
}

uint32_t extraction::streams() const
{
	return _impl->stream_count;
}

std::string extraction::damage() const
{
	const std::lock_guard<std::mutex> lock(_impl->failure_mutex);
	return _impl->damage;
}

size_t extraction::parts() const
{
	return _impl->parts.size();
}

uint64_t extraction::read_bytes() const
{
	const implementation& x = *_impl;
	const uint64_t given = x.given_bytes;
	if (given == 0) {
		return 0;
	}
	uint64_t fed = 0;
	for (const auto& p : x.parts) {
		fed += p->fed;
	}
	// As much of the file read so far as tshark got of what it was given: what it was given is
	// decompressed, if the file is not.
	const uint64_t position = x.given_position;
	const uint64_t read =
	    std::min<uint64_t>(position, uint64_t(double(position) * double(fed) / double(given)));
	// Never less than told before, though the file may compress some parts better than others.
	uint64_t told = x.told_read_bytes;
	while (told < read and not x.told_read_bytes.compare_exchange_weak(told, read)) {
	}
	return std::max(told, read);
}

} // namespace pvpcap
