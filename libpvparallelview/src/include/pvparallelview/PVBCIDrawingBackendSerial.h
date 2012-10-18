#ifndef PVPARALLELVIEW_PVBCIDRAWINGBACKENDSERIAL_H
#define PVPARALLELVIEW_PVBCIDRAWINGBACKENDSERIAL_H

#include <pvkernel/core/PVFunctionTraits.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIDrawingBackend.h>

#include <tbb/concurrent_queue.h>

namespace PVParallelView {

template <class Engine>
class PVBCIDrawingBackendSerial: protected Engine
{
	typedef PVCore::PVTypeTraits::function_traits<decltype(&PVBCIDrawingBackend<10>::operator())> backend10_functor_traits_t;
	typedef PVCore::PVTypeTraits::function_traits<decltype(&PVBCIDrawingBackend<11>::operator())> backend11_functor_traits_t;

	typedef typename PVBCIDrawingBackend<>::render_group_t render_group_t;

	struct drawing_job
	{
		size_t _bbits;
		struct {
			backend10_functor_traits_t::arguments_deep_copy_type args_10;
			backend11_functor_traits_t::arguments_deep_copy_type args_11;
		} _args;

		drawing_job()
		{ }

		drawing_job(drawing_job const& o):
			_bbits(o._bbits)
		{
			assert((_bbits == 10) || (_bbits == 11));
			if (_bbits == 10) {
				_args.args_10 = o._args.args_10;
			}
			else {
				_args.args_11 = o._args.args_11;
			}
		}

		void set_args(backend10_functor_traits_t::arguments_type const& args)
		{
			_args.args_10 = args;
		}

		void set_args(backend11_functor_traits_t::arguments_type const& args)
		{
			_args.args_11 = args;
		}

		inline void run(Engine& e)
		{
			if (_bbits == 10) {
				PVCore::PVTypeTraits::function_traits<decltype(&(Engine::template run<10>))>::template call<&Engine::template run<10>>(e, _args.args_10);
			}
			else
			if (_bbits == 11) {
				PVCore::PVTypeTraits::function_traits<decltype(&(Engine::template run<11>))>::template call<&Engine::template run<11>>(e, _args.args_11);
			}
			else {
				assert(false);
			}
		}

		std::function<void()> get_cleaning_func()
		{
			if (_bbits == 10) {
				return std::get<7>(_args.args_10);
			}
			else {
				return std::get<7>(_args.args_11);
			}
			assert(false);
			return std::function<void()>();
		}
	};

public:
	PVBCIDrawingBackendSerial():
		Engine()
	{
		_cur_grp = -1;
	}

public:
	template <size_t Bbits>
	inline void add_job(PVBCIBackendImage<Bbits>& dst_img, size_t x_start, size_t width, PVBCICode<Bbits>* codes, size_t n, const float zoom_y, bool reverse, typename PVBCIDrawingBackend<Bbits>::func_cleaning_t const& cleaning_func, typename PVBCIDrawingBackend<Bbits>::func_drawing_done_t const& drawing_done, render_group_t const rgrp)
	{
		typename PVCore::PVTypeTraits::function_traits<decltype(&PVBCIDrawingBackend<Bbits>::operator())>::arguments_type args;
		args.set_args(dst_img, x_start, width, codes, n, zoom_y, reverse, cleaning_func, drawing_done, rgrp);

		static_assert((Bbits == 10) || (Bbits == 11), "Unsupported Bbits size.");

		drawing_job job;
		job._bbits = Bbits;
		job.set_args(args);
		if (rgrp != -1) {
			_grp_order.push(rgrp);
			_jobs[rgrp].push(job);
		}
		else {
			job.run(*this);
		}
	}

	void cancel_group(render_group_t const g)
	{
		// Warning: this function is *not* thread-safe !
		typename decltype(_jobs)::iterator it = _jobs.find(g);
		if (it != _jobs.end()) {
			auto& jobs = it->second;
			PVLOG_INFO("Cancel group %d with %d jobs..\n", g, jobs.size());
			drawing_job job;
			while (!jobs.empty()) {
				if (jobs.try_pop(job)) {
					// Call cleaning func
					std::function<void()> cfunc = job.get_cleaning_func();
					if (cfunc) {
						cfunc();
					}
				}
			}
		}
	}

	render_group_t new_render_group()
	{
		// Warning: this function is *not* thread-safe !
		if (_jobs.size() == 0) {
			_jobs[0].set_capacity(50);
			return 0;
		}

		// Get last group id
		render_group_t new_g = _jobs.rbegin()->first + 1;
		_jobs[new_g].set_capacity(50);
		return new_g;
	}

	void remove_render_group(render_group_t const g)
	{
		assert(_jobs.find(g) != _jobs.end());
		// Warning: this function is *not* thread-safe !
		_jobs.erase(g);
	}

public:
	void run()
	{
		Engine::init_engine_thread();
		try {
			render_group_t cur_grp;
			drawing_job job;
			while (true) {
				_grp_order.pop(cur_grp);
				if (_jobs[cur_grp].try_pop(job)) {
					PVLOG_INFO("Running job for group %d (%d remaining)..\n", cur_grp, _jobs[cur_grp].size());
					tbb::atomic<render_group_t> cur_grp_tmp;
					cur_grp_tmp = cur_grp;
					_cur_grp = cur_grp_tmp;
					job.run(*this);
				}
			}
		}
		catch (...) {
		}
		Engine::free_engine_thread();
	}

private:
	std::map<render_group_t, tbb::concurrent_bounded_queue<drawing_job> > _jobs;
	tbb::concurrent_bounded_queue<render_group_t> _grp_order;
	tbb::atomic<render_group_t> _cur_grp;
};

}

#endif
