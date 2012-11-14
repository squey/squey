#include <pvkernel/core/general.h>
#include <pvkernel/core/picviz_bench.h>

#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVVector.h>
#include <pvkernel/core/PVSharedPointer.h>

#include <picviz/PVSelection.h>

#include <QVector>

#include <tbb/concurrent_vector.h>

#include <omp.h>

// #define TEST_OMP_TBC

template <typename T>
class Buffer
{
	typedef PVCore::PVSharedPtr<T>             data_ptr_t;
	typedef T*                                 pointer_t;
	typedef PVCore::PVReallocableCAllocator<T> allocator_t;

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
		_data = data_ptr_t(allocator_t().allocate(n));
		_index = 0;
	}

	inline pointer_t get()
	{
		return _data.get();
	}

	inline size_t size() const
	{
		return _index;
	}

	inline void set_size(size_t s)
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
	size_t i;
	for (i = 0; i < src_idxes_in.size(); ++i) {
		const PVRow line = src_idxes_in.at(i);
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

int main(int argc, char** argv)
{
	// Simulate PVGuiQt::PVListingSortFilterProxyModel::filter_source_indexes in order
	// to benchmark its performance.
	
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " row_count" << std::endl;
		return 1;
	}

	size_t n = atoll(argv[1]);

	for(int i = 0; i < 4; ++i) {
		local_buffer[i] = new PVRow [n];
	}

	// First, get a continuous vector of indexes
	QVector<PVRow> src_idxes_in;
	src_idxes_in.reserve(n);

	std::cout << "##############################################################################" << std::endl;
	std::cout << "#" << std::endl;

	BENCH_START(b0);
	for (size_t i = 0; i < n; i++) {
		src_idxes_in.push_back(i);
	}
	BENCH_END(b0, "vector filling w/ continous indexes", 1, 1, n, sizeof(PVRow));

	vector_t src_idxes_in_2;
	src_idxes_in_2.reserve(n);

	BENCH_START(b1);
	for (size_t i = 0; i < n; i++) {
		src_idxes_in_2.push_back(i);
	}
	BENCH_END(b1, "vector_simple filling w/ continous indexes", 1, 1, n, sizeof(PVRow));

	QVector<PVRow> src_idxes_out;
	vector_t src_idxes_out_2;
	concurrent_vector_t src_idxes_out_3;

	Picviz::PVSelection sel;
	// Create a full selection
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

	// Create an "even" selection
	std::cout << "##############################################################################" << std::endl;
	std::cout << "#" << std::endl;
	sel.select_even();
	{
		BENCH_START(b);
		filter_indexes_ref(src_idxes_in, src_idxes_out, &sel, n);
		BENCH_END(b, "even-selection_ref", n, sizeof(PVRow), src_idxes_out.size(), sizeof(PVRow));
	}
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

	// Create a random selection
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

	// Create an empty selection
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

	return 0;
}
