
#include <iostream>
#include <atomic>

#include <libgen.h>

#include <pvkernel/core/picviz_bench.h>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/task.h>
#include <tbb/task_scheduler_init.h>

/*****************************************************************************
 * about program's parameters
 */
enum {
	P_PROG = 0,
	P_QT_HEIGHT,
	P_QT_LIMIT,
	P_THREADS,
	P_MAX_VALUE
};

void usage(char *program)
{
	std::cerr << "usage: " << basename(program) << " height limit thread_num" << std::endl;
	std::cerr << "\theight    : height of quadtree" << std::endl;
	std::cerr << "\tlimit     : depth limit for tasks creation" << std::endl;
	std::cerr << "\tthread_num: number of used thread" << std::endl;
}

/*****************************************************************************
 * code \o/
 */

enum child_pos_t
{
	SW = 0,
	SE,
	NW,
	NE
};

class context_t
{
public:
	context_t()
	{
	}

	void accum_value(size_t v)
	{
		_value += v;
	}

private:
	size_t _value;
};

typedef tbb::enumerable_thread_specific<context_t> tls_set_t;

#if 0
template <uint64_t y1min, uint64_t y1max, uint64_t y2min, uint64_t y2max>
class QuadTreeBase
{
	constexpr static uint64_t y1_min = y1min;
	constexpr static uint64_t y1_max = y1max;
	constexpr static uint64_t y2_min = y2min;
	constexpr static uint64_t y2_max = y2max;

public:
	QuadTreeBase()
	{}

	void do_job(tls_set_t &tls) const
	{
		size_t s = 0;
		for(uint64_t y1 = y1_min; y1 < y1_max; ++y1) {
			for(uint64_t y2 = y2_min; y2 < y2_max; ++y2) {
				s += y1 + y2;
			}
		}

		tls.local().accum_value(s);
	}
};

template <uint64_t y1min, uint64_t y1max, uint64_t y2min, uint64_t y2max, int depth = 4>
class QuadTreeTmpl : public QuadTreeBase<y1min, y1max, y2min, y2max>
{
public:
	QuadTreeTmpl()
	{
	}

	void do_job(tls_set_t &tls) const
	{
		childSW.do_job(tls);
		childSE.do_job(tls);
		childNW.do_job(tls);
		childNE.do_job(tls);
	}

private:
	QuadTreeTmpl<y1min, (y1min + y1max) / 2, y2min, (y2min + y2max) / 2, depth-1>  childSW;
	QuadTreeTmpl<(y1min + y1max) / 2, y1max, y2min, (y2min + y2max) / 2, depth-1> childSE;
	QuadTreeTmpl<y1min, (y1min + y1max) / 2, (y2min + y2max) / 2, y2max, depth-1> childNW;
	QuadTreeTmpl<(y1min + y1max) / 2, y1max, (y2min + y2max) / 2, y2max, depth-1> childNE;
};

template <uint64_t y1min, uint64_t y1max, uint64_t y2min, uint64_t y2max>
class QuadTreeTmpl<y1min, y1max, y2min, y2max, 0> : public QuadTreeBase<y1min, y1max, y2min, y2max>
{
	typedef QuadTreeBase<y1min, y1max, y2min, y2max> parent_class;
public:
	QuadTreeTmpl()
	{}

	void do_job(tls_set_t &tls) const
	{
		parent_class::do_job(tls);
	}
};
#endif // 0

class QuadTree
{
public:
	QuadTree(int height, int depth, uint64_t y1_min, uint64_t y1_max, uint64_t y2_min, uint64_t y2_max) :
		_y1_min(y1_min), _y1_max(y1_max),
		_y2_min(y2_min), _y2_max(y2_max),
		_heigth(height), _depth(depth)
	{
		if (height == 0) {
			_nodes[SW] = nullptr;
			_nodes[SE] = nullptr;
			_nodes[NW] = nullptr;
			_nodes[NE] = nullptr;
		} else {
			_nodes[SW] = new QuadTree(height - 1, depth + 1,
			                          y1_min,                        (y1_min + y1_max) / 2,
			                          y2_min,                        (y2_min + y2_max) / 2);
			_nodes[SE] = new QuadTree(height - 1, depth + 1,
			                          (y1_min + y1_max) / 2,                        y1_max,
			                          y2_min,                        (y2_min + y2_max) / 2);
			_nodes[NW] = new QuadTree(height - 1, depth + 1,
			                          y1_min,                        (y1_min + y1_max) / 2,
			                          (y2_min + y2_max) / 2,                        y2_max);
			_nodes[NE] = new QuadTree(height - 1, depth + 1,
			                          (y1_min + y1_max) / 2,                        y1_max,
			                          (y2_min + y2_max) / 2,                        y2_max);
		}
	}

	QuadTree(int height, uint64_t y1_min, uint64_t y1_max, uint64_t y2_min, uint64_t y2_max) :
		QuadTree(height, 0, y1_min, y1_max, y2_min, y2_max)
	{}

	void do_job(tls_set_t &tls) const
	{
		if (is_splitted()) {
			for (int i = 0; i < 4; ++i) {
				_nodes[i]->do_job(tls);
			}
		} else {
			size_t s = 0;
			for(uint64_t y1 = _y1_min; y1 < _y1_max; ++y1) {
				for(uint64_t y2 = _y2_min; y2 < _y2_max; ++y2) {
					s += y1 + y2;
				}
			}
			context_t &ctx = tls.local();
			ctx.accum_value(s);
		}
	}

	bool is_splitted() const
	{
		return (_nodes[SW] != nullptr);
	}

	bool depth_reached(int depth) const
	{
		return (_depth == depth);
	}

	QuadTree *get_child(child_pos_t child) const
	{
		return _nodes[child];
	}

private:
	QuadTree* _nodes[4];
	uint64_t _y1_min;
	uint64_t _y1_max;
	uint64_t _y2_min;
	uint64_t _y2_max;
	int      _heigth;
	int      _depth;
};

typedef std::atomic<size_t> atomic_size_t;

class JobTask : public tbb::task
{
public:
	JobTask (QuadTree &qt, tls_set_t &tls) : _qt(qt), _tls(tls)
	{
		++_task_count;
	}

	virtual ~JobTask()
	{}

	virtual tbb::task* execute()
	{
		if (_qt.is_splitted() && !_qt.depth_reached(_limit)) {
			tbb::empty_task &c = *new(allocate_continuation()) tbb::empty_task;

			JobTask &jNE = *new(c.allocate_child()) JobTask(*_qt.get_child(NE), _tls);
			JobTask &jNW = *new(c.allocate_child()) JobTask(*_qt.get_child(NW), _tls);
			JobTask &jSE = *new(c.allocate_child()) JobTask(*_qt.get_child(SE), _tls);
			JobTask &jSW = *new(c.allocate_child()) JobTask(*_qt.get_child(SW), _tls);

			c.set_ref_count(4);

			c.spawn(jNE);
			c.spawn(jNW);
			c.spawn(jSE);
			c.spawn(jSW);

		} else {
			_qt.do_job(_tls);
		}
		return nullptr;
	}

	static void clear_task_count()
	{
		_task_count = 0;
	}

	static size_t get_task_count()
	{
		return _task_count.load();
	}

	static void set_limit(size_t l)
	{
		_limit = l;
	}

private:
	QuadTree  &_qt;
	tls_set_t &_tls;

private:
	static atomic_size_t _task_count;
	static size_t        _limit;
};

atomic_size_t JobTask::_task_count(0);
size_t JobTask::_limit;

/*****************************************************************************
 * main
 */
#define MAX_VALUE (1ULL << 16)

int main(int argc, char **argv)
{
	if (argc != P_MAX_VALUE) {
		usage(argv[P_PROG]);
		exit(1);
	}

#if 0
	typedef QuadTreeTmpl<0, MAX_VALUE, 0, MAX_VALUE, 5> quadtree_t;

	quadtree_t qtt;
	tls_set_t tls_tmpl;

	std::cout << "sizeof(quadtree_t) = " << sizeof(quadtree_t) << std::endl;
	BENCH_START(run_tmpl);
	qtt.do_job(tls_tmpl);
	BENCH_STOP(run_tmpl);

	BENCH_SHOW(run_tmpl, "run_tmpl", 1, 1, 1, 1);
#endif // 0

	int height = atoi(argv[P_QT_HEIGHT]);
	int limit = atoi(argv[P_QT_LIMIT]);
	int thread_num = atoi(argv[P_THREADS]);

	tbb::task_scheduler_init init(thread_num);

	JobTask::set_limit(limit);

	QuadTree qt(height, 0, MAX_VALUE, 0, MAX_VALUE);

	tls_set_t tls_seq;

	BENCH_START(run_seq);
	qt.do_job(tls_seq);
	BENCH_STOP(run_seq);

	double time_seq = BENCH_END_TIME(run_seq);

	tls_set_t tls_tbb;

	JobTask &t = *new(tbb::task::allocate_root()) JobTask(qt, tls_tbb);

	JobTask::clear_task_count();
	BENCH_START(run_tbb);
	tbb::task::spawn_root_and_wait(t);
	BENCH_STOP(run_tbb);

	double time_tbb = BENCH_END_TIME(run_tbb);

	std::cout << time_seq / time_tbb << std::endl;

	return 0;
}
