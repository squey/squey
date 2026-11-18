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

#include "../include/libpvpcap/sample.h"

#include "../include/libpvpcap.h"
#include "../include/libpvpcap/capture.h"
#include "../include/libpvpcap/extraction.h"
#include "../include/libpvpcap/process.h"
#include "../include/libpvpcap/ws.h"

#include <QFileInfo>
#include <QString>
#include <QTemporaryDir>

#include <rapidjson/filereadstream.h>
#include <rapidjson/reader.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace pvpcap
{

namespace
{

//! Of an example of a value, in characters.
constexpr size_t EXAMPLE_LENGTH = 120;

std::string example_of(std::string_view value)
{
	std::string example;
	for (const char c : value.substr(0, EXAMPLE_LENGTH)) {
		example.push_back(static_cast<unsigned char>(c) < 0x20 ? ' ' : c);
	}
	if (value.size() > EXAMPLE_LENGTH) {
		// Not to end in the middle of a character.
		while (not example.empty() and
		       (static_cast<unsigned char>(example.back()) & 0xc0) == 0x80) {
			example.pop_back();
		}
		if (not example.empty() and static_cast<unsigned char>(example.back()) >= 0xc0) {
			example.pop_back();
		}
		example += "…";
	}
	return example;
}

/**
 * Counts the frames holding each field and protocol, out of what `tshark -T json` writes: an
 * array of frames, each an object holding its protocols, each holding fields named after
 * their filter names, whose values are strings.
 */
class sample_reader : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, sample_reader>
{
  public:
	sample_reader(capture_sample& sample, const std::atomic<bool>& canceled)
	    : _sample(sample), _canceled(canceled)
	{
	}

	bool StartObject()
	{
		_keys.push_back(std::move(_key));
		_key.clear();
		if (_keys.size() == FRAME_DEPTH) {
			_frame_fields.clear();
			_frame_protocols.clear();
		}
		return not _canceled;
	}

	bool EndObject(rapidjson::SizeType)
	{
		if (_keys.size() == FRAME_DEPTH) {
			_sample.frames++;
		}
		_key = std::move(_keys.back());
		_keys.pop_back();
		return true;
	}

	bool StartArray()
	{
		// Its values are those of the key before it, which it keeps.
		_keys.push_back(_key);
		return true;
	}

	bool EndArray(rapidjson::SizeType)
	{
		_key = std::move(_keys.back());
		_keys.pop_back();
		return true;
	}

	bool Key(const char* text, rapidjson::SizeType length, bool)
	{
		_key.assign(text, length);
		return true;
	}

	bool String(const char* text, rapidjson::SizeType length, bool)
	{
		if (_keys.size() <= FRAME_DEPTH or _key.empty()) {
			return true;
		}
		const std::string_view value(text, length);
		if (_frame_fields.insert(_key).second) {
			capture_sample::field& field = _sample.fields[_key];
			field.frames++;
			if (field.example.empty()) {
				field.example = example_of(value);
			}
		}
		if (_key == "frame.protocols") {
			count_protocols(value);
		}
		return true;
	}

  private:
	void count_protocols(std::string_view chain)
	{
		uint32_t depth = 0;
		while (not chain.empty()) {
			const size_t end = std::min(chain.find(':'), chain.size());
			const std::string name(chain.substr(0, end));
			chain.remove_prefix(std::min(end + 1, chain.size()));
			if (not _frame_protocols.insert(name).second) {
				continue;
			}
			auto [it, added] = _protocol_indices.try_emplace(name, _sample.protocols.size());
			if (added) {
				_sample.protocols.push_back({name, 0, depth});
			}
			capture_sample::protocol& protocol = _sample.protocols[it->second];
			protocol.frames++;
			protocol.depth = std::min(protocol.depth, depth);
			depth++;
		}
	}

	//! Of the containers a frame is in: the array of frames.
	static constexpr size_t FRAME_DEPTH = 2;

	capture_sample& _sample;
	const std::atomic<bool>& _canceled;
	std::vector<std::string> _keys; //!< of the containers the reader is in
	std::string _key;               //!< of the value to come
	std::unordered_set<std::string> _frame_fields;
	std::unordered_set<std::string> _frame_protocols;
	std::unordered_map<std::string, size_t> _protocol_indices;
};

} // namespace

capture_overview overview_capture(const std::string& capture, uint64_t most_frames)
{
	capture_overview overview;
	overview.size = uint64_t(QFileInfo(QString::fromStdString(capture)).size());
	try {
		capture_reader reader(capture);
		overview.format = reader.file_format() == capture_reader::format::pcap ? "pcap" : "pcapng";
		overview.compressed = reader.compressed();
		capture_record record;
		bool more = true;
		while (overview.frames < most_frames and (more = reader.next(record))) {
			if (record.type == capture_record::kind::metadata) {
				continue;
			}
			overview.frames++;
			if (record.type == capture_record::kind::packet) {
				if (overview.first_timestamp == 0) {
					overview.first_timestamp = record.timestamp;
				}
				overview.last_timestamp = record.timestamp;
			}
		}
		overview.whole = not more;
		overview.damage = reader.damage();
		// The bytes read of a compressed file only come in blocks.
		const uint64_t read = reader.compressed() ? reader.file_position() : reader.position();
		overview.estimated_frames =
		    overview.whole or read == 0
		        ? overview.frames
		        : uint64_t(double(overview.frames) * double(overview.size) / double(read));
	} catch (const capture_error&) {
		// A format only tshark reads.
	}
	return overview;
}

capture_sample sample_capture(const std::string& capture,
                              uint64_t frames,
                              const std::vector<std::string>& tshark_options,
                              const std::atomic<bool>& canceled)
{
	QTemporaryDir directory;
	if (not directory.isValid()) {
		throw extraction_error("Can't create a temporary directory: " +
		                       directory.errorString().toStdString());
	}
	child_process::options process;
	process.argv = {tshark_path(), "-r", capture, "-c", std::to_string(frames), "-T", "json"};
	process.argv.insert(process.argv.end(), tshark_options.begin(), tshark_options.end());
	process.output = directory.filePath("sample.json").toStdString();
	process.errors = directory.filePath("errors").toStdString();
	process.environment = tshark_environment();

	child_process::status status;
	try {
		child_process tshark(process);
		tshark.close_input();
		std::mutex mutex;
		std::condition_variable waking;
		bool ended = false;
		std::thread watcher([&]() {
			std::unique_lock<std::mutex> lock(mutex);
			while (not waking.wait_for(lock, std::chrono::milliseconds(20),
			                           [&]() { return ended or canceled.load(); })) {
			}
			if (not ended) {
				tshark.kill();
			}
		});
		status = tshark.wait();
		{
			const std::lock_guard<std::mutex> lock(mutex);
			ended = true;
		}
		waking.notify_one();
		watcher.join();
	} catch (const process_error& e) {
		throw extraction_error(e.what());
	}

	capture_sample sample;
	if (canceled) {
		return sample;
	}
	if (not status.exited or status.code != 0) {
		const std::string said = tshark_errors(process.errors);
		// tshark writes the frames before the end of a capture cut short, then says so.
		if (said.find("cut short in the middle of a packet") == std::string::npos) {
			throw extraction_error(said.empty() ? "tshark could not read the capture." : said);
		}
	}

	const std::unique_ptr<std::FILE, int (*)(std::FILE*)> output(open_file(process.output, "rb"),
	                                                             std::fclose);
	if (output == nullptr) {
		throw extraction_error("tshark wrote nothing of the capture.");
	}
	char buffer[64 * 1024];
	rapidjson::FileReadStream stream(output.get(), buffer, sizeof(buffer));
	sample_reader reader(sample, canceled);
	rapidjson::Reader().Parse(stream, reader);
	return sample;
}

} // namespace pvpcap
