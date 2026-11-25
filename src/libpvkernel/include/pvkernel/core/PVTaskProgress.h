/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVCORE_PVTASKPROGRESS_H__
#define __PVCORE_PVTASKPROGRESS_H__

#include <QString>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>

namespace PVCore
{

/**
 * The progress of a task that runs in a thread of its own while another thread shows it: the task
 * tells how far it got, and reads whether to stop. Every member can be called from any thread.
 */
class PVTaskProgress
{
  public:
	/** 0 while the end of the task is unknown. */
	void set_maximum(uint64_t maximum) { _maximum = maximum; }
	uint64_t maximum() const { return _maximum; }

	void set_value(uint64_t value) { _value = value; }
	uint64_t value() const { return _value; }

	void set_status(QString status)
	{
		std::lock_guard<std::mutex> lock(_status_mutex);
		_status = std::move(status);
	}
	QString status() const
	{
		std::lock_guard<std::mutex> lock(_status_mutex);
		return _status;
	}

	/** Asks the task to stop, which it does once it reads canceled(). */
	void cancel() { _canceled = true; }
	bool canceled() const { return _canceled; }

  private:
	std::atomic<uint64_t> _maximum = 0;
	std::atomic<uint64_t> _value = 0;
	std::atomic<bool> _canceled = false;
	mutable std::mutex _status_mutex;
	QString _status;
};

} // namespace PVCore

#endif // __PVCORE_PVTASKPROGRESS_H__
