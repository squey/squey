
#include <iostream>

#include <libgen.h>

#include <pvkernel/core/picviz_bench.h>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/task.h>

/*****************************************************************************
 * about program's parameters
 */
enum {
	P_PROG = 0,
	P_NUM,
	P_MAX_VALUE
};

void usage(char *program)
{
	std::cerr << "usage: " << basename(program) << " num\n" << std::endl;
	std::cerr << "\tnum  : number of values along each coordinate" << std::endl;
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

class QuadTree
{
public:
	QuadTree(int depth, uint64_t y1_min, uint64_t y1_max, uint64_t y2_min, uint64_t y2_max) :
		_y1_min(y1_min), _y1_max(y1_max),
		_y2_min(y2_min), _y2_max(y2_max)
	{
		if (depth == 0) {
			_nodes[SW] = nullptr;
			_nodes[SE] = nullptr;
			_nodes[NW] = nullptr;
			_nodes[NE] = nullptr;
		} else {
			_nodes[SW] = new QuadTree(depth - 1,
			                          y1_min,                        (y1_min + y1_max) / 2,
			                          y2_min,                        (y2_min + y2_max) / 2);
			_nodes[SE] = new QuadTree(depth - 1,
			                          (y1_min + y1_max) / 2,                        y1_max,
			                          y2_min,                        (y2_min + y2_max) / 2);
			_nodes[NW] = new QuadTree(depth - 1,
			                          y1_min,                        (y1_min + y1_max) / 2,
			                          (y2_min + y2_max) / 2,                        y2_max);
			_nodes[NE] = new QuadTree(depth - 1,
			                          (y1_min + y1_max) / 2,                        y1_max,
			                          (y2_min + y2_max) / 2,                        y2_max);
		}
	}

	void do_job(tls_set_t &tls) const
	{
		if (_nodes[SW] == nullptr) {
			size_t s = 0;
			for(uint64_t y1 = _y1_min; y1 < _y1_max; ++y1) {
				for(uint64_t y2 = _y2_min; y2 < _y2_max; ++y2) {
					s += y1 + y2;
				}
			}
			tls.local().accum_value(s);
		} else {
			for (int i = 0; i < 4; ++i) {
				_nodes[i]->do_job(tls);
			}
		}
	}

private:
	QuadTree* _nodes[4];
	uint64_t _y1_min;
	uint64_t _y1_max;
	uint64_t _y2_min;
	uint64_t _y2_max;
};

template <class C>
class JobTask
{
public:
	JobTask () {}

	tbb::task* execute()
	{
		return nullptr;
	}
};


/*****************************************************************************
 * main
 */
#define MAX_VALUE (1ULL << 17)

typedef QuadTreeTmpl<0, MAX_VALUE, 0, MAX_VALUE, 5> quadtree_t;

int main(int argc, char **argv)
{
	if (argc != P_MAX_VALUE) {
		usage(argv[P_PROG]);
		exit(1);
	}

	quadtree_t qtt;
	tls_set_t tls;

	std::cout << "sizeof(quadtree_t) = " << sizeof(quadtree_t) << std::endl;
	BENCH_START(run_tmpl);
	qtt.do_job(tls);
	BENCH_STOP(run_tmpl);

	BENCH_SHOW(run_tmpl, "run_tmpl", 1, 1, 1, 1);

	QuadTree qt(5, 0, MAX_VALUE, 0, MAX_VALUE);

	BENCH_START(run);
	qt.do_job(tls);
	BENCH_STOP(run);

	BENCH_SHOW(run, "run", 1, 1, 1, 1);

	return 0;
}
