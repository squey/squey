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

#ifndef PCAP_GUI_BACKGROUND_H
#define PCAP_GUI_BACKGROUND_H

#include <QObject>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

namespace PVPcapsicum
{

/**
 * Runs work away from the thread of the widgets, one at a time: running work cancels the
 * work run before, whose result is dropped.
 *
 * Thread-safety: for the thread it belongs to, which gets the results.
 */
class BackgroundJob : public QObject
{
  public:
	explicit BackgroundJob(QObject* parent = nullptr) : QObject(parent) {}

	//! Cancels what runs, and waits for it to end.
	~BackgroundJob() override
	{
		cancel();
		for (worker& w : _workers) {
			w.thread.join();
		}
	}

	BackgroundJob(const BackgroundJob&) = delete;
	BackgroundJob& operator=(const BackgroundJob&) = delete;

	/**
	 * Runs `work` in a thread of its own, then `done` with what it returned, in the thread
	 * of this object, unless work was run or canceled meanwhile.
	 *
	 * @param work called with a flag telling it to stop; must not throw
	 */
	template <typename Work, typename Done>
	void run(Work work, Done done)
	{
		using Result = std::invoke_result_t<Work&, const std::atomic<bool>&>;
		cancel();
		reap();
		const uint64_t generation = _generation;
		auto canceled = std::make_shared<std::atomic<bool>>(false);
		auto finished = std::make_shared<std::atomic<bool>>(false);
		_canceled = canceled;
		_running = true;
		std::thread thread([this, work = std::move(work), done = std::move(done), canceled,
		                    finished, generation]() mutable {
			auto result = std::make_shared<Result>(work(*canceled));
			QMetaObject::invokeMethod(
			    this,
			    [this, done, result, generation]() {
				    if (generation == _generation) {
					    _running = false;
					    done(std::move(*result));
				    }
			    },
			    Qt::QueuedConnection);
			*finished = true;
		});
		_workers.push_back({std::move(thread), finished});
	}

	//! The work running, if any, is to stop: its result is dropped.
	void cancel()
	{
		if (_canceled) {
			*_canceled = true;
		}
		_running = false;
		_generation++;
	}

	//! Whether work runs, whose result is to come.
	bool running() const { return _running; }

  private:
	//! Joins the threads done with their work, which they are about to end.
	void reap()
	{
		for (auto it = _workers.begin(); it != _workers.end();) {
			if (*it->finished) {
				it->thread.join();
				it = _workers.erase(it);
			} else {
				++it;
			}
		}
	}

	struct worker {
		std::thread thread;
		std::shared_ptr<std::atomic<bool>> finished;
	};
	std::vector<worker> _workers;
	std::shared_ptr<std::atomic<bool>> _canceled; //!< of the work run last
	uint64_t _generation = 0;                     //!< of the work whose result is wanted
	bool _running = false;
};

} // namespace PVPcapsicum

#endif // PCAP_GUI_BACKGROUND_H
