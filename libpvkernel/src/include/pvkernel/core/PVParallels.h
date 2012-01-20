#ifndef PVCORE_PVPARALLELS_H
#define PVCORE_PVPARALLELS_H

#include <boost/thread.hpp>

namespace PVCore {

namespace __impl {

template <class F>
static void worker_thread(F f)
{
	try {
		f();
	}
	catch (boost::thread_interrupted) {
	}
}

template <class Tret, class F>
static void worker_thread(F f, Tret& ret)
{
	try {
		ret = f();
	}
	catch (boost::thread_interrupted) {
	}
}

}

template <typename F1, typename F2, typename TimeDuration>
void launch_adaptive(F1 const& f1, F2 const& f2, TimeDuration const& dur)
{
	boost::thread thread(boost::bind(&__impl::worker_thread<F1>, boost::ref(f1)));
	if (!thread.timed_join(dur)) {
		// f1 takes too much time. We tell thread to stop, and launch f2. Then, waits for thread to finish.
		thread.interrupt();
		PVLOG_INFO("launch_adaptive: wait for end of f1.\n");
		thread.join();
		PVLOG_INFO("f1 finished.\n");
		f2();
	}
}

template <typename F1, typename F2, typename TimeDuration, typename Tret>
void launch_adaptive(F1 const& f1, F2 const& f2, TimeDuration const& dur, Tret& ret)
{
	boost::thread thread(boost::bind(&__impl::worker_thread<F1, Tret>, boost::ref(f1), boost::ref(ret)));
	if (!thread.timed_join(dur)) {
		// f1 takes too much time. We tell thread to stop, and launch f2. Then, waits for thread to finish.
		thread.interrupt();
		PVLOG_INFO("launch_adaptive: wait for end of f1.\n");
		thread.join();
		PVLOG_INFO("f1 finished.\n");
		ret = f2();
	}
}

}

#endif
