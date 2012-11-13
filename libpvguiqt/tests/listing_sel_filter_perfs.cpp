#include <pvkernel/core/general.h>
#include <pvkernel/core/picviz_bench.h>

#include <picviz/PVSelection.h>

#include <QVector>

void filter_indexes(QVector<PVRow> const& src_idxes_in, QVector<PVRow>& src_idxes_out, Picviz::PVSelection const* sel, size_t n)
{
	src_idxes_out.clear();
	const PVRow nvisible_lines = sel->get_number_of_selected_lines_in_range(0, n);
	if (nvisible_lines == 0) {
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

int main(int argc, char** argv)
{
	// Simulate PVGuiQt::PVListingSortFilterProxyModel::filter_source_indexes in order
	// to benchmark its performance.
	
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " row_count" << std::endl;
		return 1;
	}

	size_t n = atoll(argv[1]);

	// First, get a continuous vector of indexes
	QVector<PVRow> src_idxes_in;
	src_idxes_in.reserve(n);
	BENCH_START(b0);
	for (size_t i = 0; i < n; i++) {
		src_idxes_in.push_back(i);
	}
	BENCH_END(b0, "vector filling w/ continous indexes", 1, 1, sizeof(PVRow), n);

	QVector<PVRow> src_idxes_out;

	Picviz::PVSelection sel;
	// Create a full selection
	sel.select_all();
	{
		BENCH_START(b);
		filter_indexes(src_idxes_in, src_idxes_out, &sel, n);
		BENCH_END(b, "full-selection", sizeof(PVRow), n, sizeof(PVRow), src_idxes_out.size());
	}

	// Create an "even" selection
	sel.select_even();
	{
		BENCH_START(b);
		filter_indexes(src_idxes_in, src_idxes_out, &sel, n);
		BENCH_END(b, "even-selection", sizeof(PVRow), n, sizeof(PVRow), src_idxes_out.size());
	}

	// Create a random selection
	sel.select_random();
	{
		BENCH_START(b);
		filter_indexes(src_idxes_in, src_idxes_out, &sel, n);
		BENCH_END(b, "rand-selection", sizeof(PVRow), n, sizeof(PVRow), src_idxes_out.size());
	}

	// Create an empty selection
	sel.select_none();
	{
		BENCH_START(b);
		filter_indexes(src_idxes_in, src_idxes_out, &sel, n);
		BENCH_END(b, "empty-selection", sizeof(PVRow), n, sizeof(PVRow), src_idxes_out.size());
	}

	return 0;
}
