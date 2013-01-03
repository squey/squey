#include <pvkernel/core/general.h>
#include <pvkernel/core/picviz_bench.h>

#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVVector.h>
#include <pvkernel/core/PVSharedPointer.h>

#include <picviz/PVSelection.h>

#include <QVector>

#include <tbb/concurrent_vector.h>

#include <omp.h>

#define TBB_PREVIEW_DETERMINISTIC_REDUCE 1
#include <tbb/parallel_reduce.h>
#include <tbb/task_scheduler_init.h>

// #define TEST_OMP_TBC

/*****************************************************************************
 */
template <typename T>
class Buffer
{
public:
	typedef PVCore::PVSharedPtr<T>             data_ptr_t;
	typedef PVCore::PVAlignedAllocator<T, 16>  allocator_type;

	typedef typename allocator_type::pointer         pointer;
	typedef typename allocator_type::const_pointer   const_pointer;
	typedef typename allocator_type::reference       reference;
	typedef typename allocator_type::const_reference const_reference;
	typedef typename allocator_type::size_type       size_type;
	typedef pointer                                  iterator;
	typedef const_pointer                            const_iterator;

public:
	Buffer()
	{
		clear();
	}

	~Buffer()
	{
		clear();
	}

	inline void clear()
	{
		_data = data_ptr_t();
		_index = 0;
	}

	inline void reserve(size_t n)
	{
		_data = data_ptr_t(allocator_type().allocate(n));
		_index = 0;
	}

	inline pointer get()
	{
		return _data.get();
	}

	inline const_pointer get() const
	{
		return _data.get();
	}

	inline size_type size() const
	{
		return _index;
	}

	inline void set_size(size_type s)
	{
		_index = s;
	}

	inline const T &at(size_t i) const
	{
		return _data.get()[i];
	}

	inline T &at(size_t i)
	{
		return _data.get()[i];
	}

	inline Buffer<T> &operator=(const Buffer<T> &buffer)
	{
		_data = buffer._data;
		_index = buffer._index;
		return *this;
	}

	inline void push_back(const T &v)
	{
		_data.get()[_index++] = v;
	}

	iterator begin() { return get(); }
	const_iterator begin() const { return get(); }

	iterator end() { return get() + size(); }
	const_iterator end() const { return get() + size(); }

	bool operator==(const QVector<T> &v)
	{
		if (size() != (size_t)v.size()) {
			return false;
		}

		for(size_t i = 0; i < _index; ++i) {
			if (at(i) != v.at(i)) {
				return false;
			}
		}

		return true;
	}

private:
	data_ptr_t _data;
	size_t     _index;
};

template <typename T>
class FilterIndexes
{
public:
	typedef Buffer<T> buffer_t;
	typedef tbb::blocked_range<size_t> blocked_range;

public:
	FilterIndexes() : _count(0)
	{}

	FilterIndexes(buffer_t const& src_idxes_in, buffer_t& src_idxes_out, Picviz::PVSelection const* sel) :
		_offset(0), _count(0),
		_in(src_idxes_in), _out(src_idxes_out), _sel(sel)
	{}

	FilterIndexes(FilterIndexes& s, tbb::split) :
		_offset(0), _count(0),
		_in(s._in), _out(s._out), _sel(s._sel)
	{}

	size_t get_count()
	{
		return _count;
	}

	void operator()(const blocked_range& r)
	{
		size_t offset = _offset = r.begin();
		for(size_t i = r.begin(); i != r.end(); ++i) {
			const PVRow line = _in.at(i);
			if (_sel->get_line(line)) {
				_out.at(offset) = line;
				++_count;
				++offset;
			}
		}
	}

	void join(FilterIndexes& rhs)
	{
		memmove(_out.get() + _offset + _count,
		        _out.get() + rhs._offset, sizeof(T) * rhs._count);
		_count += rhs._count;
	}

private:
	size_t                     _offset;
	size_t                     _count;
	const buffer_t            &_in;
	buffer_t                  &_out;
	const Picviz::PVSelection *_sel;

};

/*****************************************************************************
 */
void filter_indexes_ref(QVector<PVRow> const& src_idxes_in, QVector<PVRow>& src_idxes_out, Picviz::PVSelection const* sel, size_t n)
{
	src_idxes_out.clear();
	const PVRow nvisible_lines = sel->get_number_of_selected_lines_in_range(0, n);
	if (nvisible_lines == 0) {
		return;
	} else if (nvisible_lines == n) {
		src_idxes_out = src_idxes_in;
		return;
	}

	src_idxes_out.reserve(nvisible_lines);
	QVector<PVRow>::const_iterator it;
	for (it = src_idxes_in.begin(); it != src_idxes_in.end(); it++) {
		const PVRow line = *it;
		if (sel->get_line(line)) {
			src_idxes_out.push_back(line);
		}
	}
}

typedef Buffer<PVRow> vector_t;

void filter_indexes_simple(vector_t const& src_idxes_in, vector_t& src_idxes_out, Picviz::PVSelection const* sel, size_t n)
{
	src_idxes_out.clear();
	const PVRow nvisible_lines = sel->get_number_of_selected_lines_in_range(0, n);
	if (nvisible_lines == 0) {
		return;
	} else if (nvisible_lines == n) {
		src_idxes_out = src_idxes_in;
		return;
	}

	src_idxes_out.reserve(nvisible_lines);
	vector_t::const_iterator it;
	for (it = src_idxes_in.begin(); it != src_idxes_in.end(); it++) {
		const PVRow line = *it;
		if (sel->get_line(line)) {
			src_idxes_out.push_back(line);
		}
	}
}

typedef tbb::concurrent_vector<PVRow> concurrent_vector_t;

void filter_indexes_omp_tcv(QVector<PVRow> const& src_idxes_in, concurrent_vector_t& src_idxes_out, Picviz::PVSelection const* sel, size_t n)
{
	src_idxes_out.clear();
	const PVRow nvisible_lines = sel->get_number_of_selected_lines_in_range(0, n);
	if (nvisible_lines == 0) {
		return;
	}

	src_idxes_out.reserve(nvisible_lines);

	size_t i;
#pragma omp parallel for num_threads(4)
	for (i = 0; i < (size_t)src_idxes_in.size(); ++i) {
		const PVRow line = src_idxes_in.at(i);
		if (sel->get_line(line)) {
			src_idxes_out.push_back(line);
		}
	}
}

static PVRow *local_buffer[4];
static size_t local_index[4];

int get_local_index_sum()
{
	return local_index[0] + local_index[1] + local_index[2] + local_index[3];

}

void filter_indexes_omp_4buf(vector_t const& src_idxes_in, vector_t &src_idxes_out, Picviz::PVSelection const* sel, size_t n)
{
	const PVRow nvisible_lines = sel->get_number_of_selected_lines_in_range(0, n);
	if (nvisible_lines == 0) {
		local_index[0] = 0;
		local_index[1] = 0;
		local_index[2] = 0;
		local_index[3] = 0;
		return;
	} else if (nvisible_lines == n) {
		src_idxes_out = src_idxes_in;
		return;
	}

	size_t i;
#pragma omp parallel num_threads(4)
	{
		int tid = omp_get_thread_num();
		size_t idx = 0;
		PVRow *buffer = local_buffer[tid];

#pragma omp for
		for (i = 0; i < (size_t)src_idxes_in.size(); ++i) {
			const PVRow line = src_idxes_in.at(i);
			if (sel->get_line(line)) {
				buffer[idx] = line;
				++idx;
			}
		}

		local_index[tid] = idx;
	}

	/* merge
	 */

	BENCH_START(b);
	src_idxes_out.reserve(get_local_index_sum());
	PVRow *buffer = src_idxes_out.get();
	size_t offset = 0;
	for(int i = 0; i < 4; ++i) {
		size_t size = local_index[i];
		memcpy(buffer + offset, local_buffer[i], size * sizeof(PVRow));
		offset += size;
	}
	BENCH_END(b, "merge", 1, 1, offset, sizeof(PVRow));
	src_idxes_out.set_size(offset);
}

void filter_indexes_par_red(Buffer<PVRow> const& src_idxes_in, Buffer<PVRow>& src_idxes_out, Picviz::PVSelection const* sel,
                            size_t n, int thread_num, size_t block_num = 0)
{
	const PVRow nvisible_lines = sel->get_number_of_selected_lines_in_range(0, n);
	if (nvisible_lines == 0) {
		return;
	} else if (nvisible_lines == n) {
		src_idxes_out = src_idxes_in;
		return;
	}

	src_idxes_out.reserve(src_idxes_in.size());

	typedef FilterIndexes<PVRow> filter_indexes_t;

	tbb::task_scheduler_init init(thread_num);

	if (block_num == 0) {
		block_num = thread_num;
	}

	filter_indexes_t fi(src_idxes_in, src_idxes_out, sel);
	tbb::parallel_deterministic_reduce(filter_indexes_t::blocked_range(0, n, n / block_num), fi);
	src_idxes_out.set_size(fi.get_count());
}

int main(int argc, char** argv)
{
	// Simulate PVGuiQt::PVListingSortFilterProxyModel::filter_source_indexes in order
	// to benchmark its performance and experiment other methods

	if ((argc != 3) && (argc != 4)) {
		std::cerr << "Usage: " << argv[0] << " row_count num_thread [block_num]" << std::endl;
		return 1;
	}

	srand(0);
	size_t n = atoll(argv[1]);
	size_t thread_num = atoll(argv[2]);

	size_t block_num = 0;

	if (argc == 4) {
		block_num = atoll(argv[3]);
	}

	for(int i = 0; i < 4; ++i) {
		local_buffer[i] = new PVRow [n];
	}

	// First, get a continuous vector of indexes
	QVector<PVRow> src_idxes_in;
	src_idxes_in.reserve(n);

	if (block_num == 0) {
		std::cout << "##############################################################################" << std::endl;
		std::cout << "#" << std::endl;
	}

	BENCH_START(b0);
	for (size_t i = 0; i < n; i++) {
		src_idxes_in.push_back(i);
	}
	BENCH_STOP(b0);

	if (block_num == 0) {
		BENCH_SHOW(b0, "vector filling w/ continous indexes", 1, 1, n, sizeof(PVRow));
	}

	vector_t src_idxes_in_2;
	src_idxes_in_2.reserve(n);

	BENCH_START(b1);
	for (size_t i = 0; i < n; i++) {
		src_idxes_in_2.push_back(i);
	}
	BENCH_STOP(b1);
	if (block_num == 0) {
		BENCH_SHOW(b1, "vector_simple filling w/ continous indexes", 1, 1, n, sizeof(PVRow));
	}

	QVector<PVRow> src_idxes_out;
	vector_t src_idxes_out_2;
	concurrent_vector_t src_idxes_out_3;

	Picviz::PVSelection sel;
	// Create a full selection
	if (block_num == 0) {
		std::cout << "##############################################################################" << std::endl;
		std::cout << "#" << std::endl;
		sel.select_all();
		{
			BENCH_START(b);
			filter_indexes_ref(src_idxes_in, src_idxes_out, &sel, n);
			BENCH_END(b, "full-selection_ref", n, sizeof(PVRow), src_idxes_out.size(), sizeof(PVRow));
		}
		{
			BENCH_START(b);
			filter_indexes_simple(src_idxes_in_2, src_idxes_out_2, &sel, n);
			BENCH_END(b, "full-selection_simple", n, sizeof(PVRow), src_idxes_out_2.size(), sizeof(PVRow));
		}
		if (src_idxes_out_2 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
#ifdef TEST_OMP_TBC
		{
			BENCH_START(b);
			filter_indexes_omp_tcv(src_idxes_in, src_idxes_out_3, &sel, n);
			BENCH_END(b, "full-selection_omp_tcv", n, sizeof(PVRow), src_idxes_out_3.size(), sizeof(PVRow));
		}
		if (src_idxes_out_3 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
#endif
		{
			BENCH_START(b);
			filter_indexes_omp_4buf(src_idxes_in_2, src_idxes_out_2, &sel, n);
			BENCH_END(b, "full-selection_omp_4buf", n, sizeof(PVRow), src_idxes_out_2.size(), sizeof(PVRow));
		}
		if (src_idxes_out_2 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
		{
			BENCH_START(b);
			filter_indexes_par_red(src_idxes_in_2, src_idxes_out_2, &sel, n, thread_num);
			BENCH_END(b, "full-selection_par_red", n, sizeof(PVRow), src_idxes_out_2.size(), sizeof(PVRow));
		}
		if (src_idxes_out_2 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
	}

	// Create an "even" selection
	sel.select_even();
	if (block_num == 0) {
		std::cout << "##############################################################################" << std::endl;
		std::cout << "#" << std::endl;
	}
	if (thread_num == 1) {
		BENCH_START(b);
		filter_indexes_ref(src_idxes_in, src_idxes_out, &sel, n);
		BENCH_END(b, "even-selection_ref", n, sizeof(PVRow), src_idxes_out.size(), sizeof(PVRow));
	}
	if (block_num == 0) {
		{
			BENCH_START(b);
			filter_indexes_simple(src_idxes_in_2, src_idxes_out_2, &sel, n);
			BENCH_END(b, "even-selection_simple", n, sizeof(PVRow), src_idxes_out_2.size(), sizeof(PVRow));
		}
		if (src_idxes_out_2 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
#ifdef TEST_OMP_TBC
		{
			BENCH_START(b);
			filter_indexes_omp_tcv(src_idxes_in, src_idxes_out_3, &sel, n);
			BENCH_END(b, "even-selection_omp_tcv", n, sizeof(PVRow), src_idxes_out_3.size(), sizeof(PVRow));
		}
		if (src_idxes_out_3 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
#endif
		{
			BENCH_START(b);
			filter_indexes_omp_4buf(src_idxes_in_2, src_idxes_out_2, &sel, n);
			BENCH_END(b, "even-selection_omp_4buf", n, sizeof(PVRow), src_idxes_out_2.size(), sizeof(PVRow));
		}
		if (src_idxes_out_2 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
	}
	if (thread_num != 1) {
		BENCH_START(b);
		if (block_num) {
			filter_indexes_par_red(src_idxes_in_2, src_idxes_out_2, &sel, n, thread_num, block_num);
		} else {
			filter_indexes_par_red(src_idxes_in_2, src_idxes_out_2, &sel, n, thread_num);
		}
		BENCH_END(b, "even-selection_par_red", n, sizeof(PVRow), src_idxes_out_2.size(), sizeof(PVRow));
	}
	if (block_num == 0) {
		if (src_idxes_out_2 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
	}

	// Create a random selection
	if (block_num == 0) {
		std::cout << "##############################################################################" << std::endl;
		std::cout << "#" << std::endl;
		sel.select_random();
		{
			BENCH_START(b);
			filter_indexes_ref(src_idxes_in, src_idxes_out, &sel, n);
			BENCH_END(b, "rand-selection_ref", n, sizeof(PVRow), src_idxes_out.size(), sizeof(PVRow));
		}
		{
			BENCH_START(b);
			filter_indexes_simple(src_idxes_in_2, src_idxes_out_2, &sel, n);
			BENCH_END(b, "rand-selection_simple", n, sizeof(PVRow), src_idxes_out_2.size(), sizeof(PVRow));
		}
		if (src_idxes_out_2 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
#ifdef TEST_OMP_TBC
		{
			BENCH_START(b);
			filter_indexes_omp_tcv(src_idxes_in, src_idxes_out_3, &sel, n);
			BENCH_END(b, "rand-selection_omp_tcv", sizeof(PVRow), n, sizeof(PVRow), src_idxes_out_3.size());
		}
		if (src_idxes_out_3 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
#endif
		{
			BENCH_START(b);
			filter_indexes_omp_4buf(src_idxes_in_2, src_idxes_out_2, &sel, n);
			BENCH_END(b, "rand-selection_omp_4buf", n, sizeof(PVRow), src_idxes_out_2.size(), sizeof(PVRow));
		}
		if (src_idxes_out_2 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
		{
			BENCH_START(b);
			filter_indexes_par_red(src_idxes_in_2, src_idxes_out_2, &sel, n, thread_num);
			BENCH_END(b, "rand-selection_par_red", n, sizeof(PVRow), src_idxes_out_2.size(), sizeof(PVRow));
		}
		if (src_idxes_out_2 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
	}

	// Create an empty selection
	if (block_num == 0) {
		std::cout << "##############################################################################" << std::endl;
		std::cout << "#" << std::endl;
		sel.select_none();
		{
			BENCH_START(b);
			filter_indexes_ref(src_idxes_in, src_idxes_out, &sel, n);
			BENCH_END(b, "empty-selection_ref", n, sizeof(PVRow), src_idxes_out.size(), sizeof(PVRow));
		}
		{
			BENCH_START(b);
			filter_indexes_simple(src_idxes_in_2, src_idxes_out_2, &sel, n);
			BENCH_END(b, "empty-selection_simple", n, sizeof(PVRow), src_idxes_out_2.size(), sizeof(PVRow));
		}
		if (src_idxes_out_2 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
#ifdef TEST_OMP_TBC
		{
			BENCH_START(b);
			filter_indexes_omp_tcv(src_idxes_in, src_idxes_out_3, &sel, n);
			BENCH_END(b, "empty-selection_omp_tcv", n, sizeof(PVRow), src_idxes_out_3.size(), sizeof(PVRow));
		}
		if (src_idxes_out_3 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
#endif
		{
			BENCH_START(b);
			filter_indexes_omp_4buf(src_idxes_in_2, src_idxes_out_2, &sel, n);
			BENCH_END(b, "empty-selection_omp_4buf", n, sizeof(PVRow), src_idxes_out_2.size(), sizeof(PVRow));
		}
		if (src_idxes_out_2 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
		{
			BENCH_START(b);
			filter_indexes_par_red(src_idxes_in_2, src_idxes_out_2, &sel, n, thread_num);
			BENCH_END(b, "empty-selection_par_red", n, sizeof(PVRow), src_idxes_out_2.size(), sizeof(PVRow));
		}
		if (src_idxes_out_2 == src_idxes_out) {
			std::cout << "outputs are equal" << std::endl;
		} else {
			std::cout << "outputs differs" << std::endl;
		}
	}

	return 0;
}
