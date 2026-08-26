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

#ifndef __SQUEY_PVDUCKDBQUERY__
#define __SQUEY_PVDUCKDBQUERY__

#include <squey/export.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace PVRush
{
class PVNraw;
} // namespace PVRush

namespace PVCore
{
class PVSelBitField;
} // namespace PVCore

namespace Squey
{

class PVSource;
class PVView;

/**
 * Runs SQL queries against the columns of a PVNraw and turns their result into
 * a selection.
 *
 * The nraw columns are not copied into DuckDB: they are exposed as a virtual
 * table through a table function that hands DuckDB a pointer into the mmapped
 * column, so a query holds a single copy of the data and always sees the
 * current schema (including columns added at runtime through Python).
 *
 * A query names the rows it reads the way the application names them:
 *   - "selection": what is selected right now;
 *   - "layers": what the layer stack lets through, which is what the listing
 *     shows and what a layer filter takes as input;
 *   - "layer('name')": one layer, by the name it carries in the layer stack.
 *     The base layer is locked, so "layer('All events')" is how a query reads
 *     every row of the source.
 *
 * A business type is exposed as the integer it is stored as, whose order is its
 * own order -- an IPv4 as a UINTEGER, a datetime as an epoch. Named conversions
 * bridge the two: "WHERE ip = ipv4('192.168.1.1')" and "SELECT ipv4_text(ip)".
 * The literal-side one is the one to reach for, since it leaves the comparison
 * on the stored integer rather than building a string per row.
 *
 * Each of them also takes a "text" argument -- "selection(text := true)",
 * "layer('All events', text := true)" -- which renders every column the way the
 * listing does, as the text the cell was written with. That is the form to
 * reach for to read a cell the format could not parse: the typed form exposes
 * such a cell as NULL, since what the storage holds for it is an encoding
 * rather than a value.
 *
 * Queries come in two shapes:
 *   - a bare predicate -- "port = 80 AND host LIKE '%.fr'" -- which is wrapped
 *     into "SELECT rowid FROM <table> WHERE (...)". Only the condition carries
 *     intent, so this is the form to reach for;
 *   - a full statement, for anything the predicate form cannot express
 *     (aggregates, subqueries, CTEs). It must project a single integer "rowid"
 *     column, the physical row index and the only stable identity of a row --
 *     SQL guarantees no ordering, so rowid is what maps a result onto a
 *     selection.
 */
class PVSQUEY_EXPORT PVDuckDBQuery
{
  public:
	/**
	 * What the row scopes stand for. Each is called when a query runs rather
	 * than read once, so a selection made after this object was built is the one
	 * a query sees, and a layer created between two queries is nameable by the
	 * second.
	 *
	 * A null return means "every row"; for layer(), it means no layer carries
	 * that name, which is an error rather than a silently wider result.
	 *
	 * Empty functions are the honest answer for a caller that has no view: the
	 * scopes then cover every row and no layer can be named.
	 */
	struct Scopes {
		std::function<const PVCore::PVSelBitField*()> selection;
		std::function<const PVCore::PVSelBitField*()> layers;
		std::function<const PVCore::PVSelBitField*(const std::string& name)> layer;
		//! The layer names that do exist, to tell a typo what it could have meant.
		std::function<std::vector<std::string>()> layer_names;
	};

	/**
	 * One source a query can read, with everything needed to name and scope it.
	 *
	 * A query object exposes several so that a join can reach across sources,
	 * the way a Python script reaching for squey.source() already can. The
	 * first one is what a bare "selection" resolves to; the others are named.
	 *
	 * @a name is the source name as Squey shows it, and @a position tells it
	 * apart from its namesakes -- source names are not unique, which is why
	 * the Python API indexes them by that same pair.
	 */
	struct Source {
		const PVRush::PVNraw* nraw = nullptr;
		std::function<std::string(size_t)> name_of;
		Scopes scopes;
		std::string name;
		size_t position = 0;
	};

	/**
	 * Expose the source a view reads, with the scopes that view defines.
	 *
	 * This is the constructor to reach for wherever a view is at hand: a
	 * selection and a layer stack belong to a view, and a source may carry
	 * several.
	 *
	 * @param view must outlive this object.
	 */
	explicit PVDuckDBQuery(const Squey::PVView& view);

	/**
	 * Expose a source: its nraw provides the data, its format the column names.
	 *
	 * Nothing is copied. Names are read from the format every time a query is
	 * bound, so renaming an axis or adding a column at runtime is picked up
	 * without rebuilding this object.
	 *
	 * The scopes resolve through the source's current view, so they follow
	 * whichever view is active. Until one exists, they cover every row.
	 *
	 * @param source must outlive this object, and must not be structurally
	 *               modified while a query runs.
	 */
	explicit PVDuckDBQuery(const Squey::PVSource& source);

	/**
	 * Expose a bare nraw, for callers that have no source (tests, or data built
	 * in memory). Columns are then named "col_<index>": neither pvcop nor the
	 * nraw carries a column name of its own -- pvcop addresses everything by
	 * index, and axis names live in the Squey format.
	 *
	 * There is no view either, so "selection" and "layers" cover every row and
	 * no layer can be named.
	 */
	explicit PVDuckDBQuery(const PVRush::PVNraw& nraw);
	~PVDuckDBQuery();

	PVDuckDBQuery(const PVDuckDBQuery&) = delete;
	PVDuckDBQuery& operator=(const PVDuckDBQuery&) = delete;

  public:
	/**
	 * Run a query and store the rows it returns into @a out.
	 *
	 * @param sql a query projecting a single "rowid" column
	 * @param out receives the matching rows; it is cleared first and must be
	 *            sized for the source row count
	 *
	 * @throws std::runtime_error if the query fails or does not project a
	 *         single integer column
	 */
	void select(const std::string& sql, PVCore::PVSelBitField& out) const;

	/**
	 * Same as select(), with "selection" standing for @a in rather than for what
	 * the view has selected -- for a caller holding the selection it means.
	 *
	 * A bare predicate is then wrapped against "selection" rather than against
	 * "layers", which is what "filter within what I already have" reads as.
	 */
	void select(const std::string& sql,
	            const PVCore::PVSelBitField& in,
	            PVCore::PVSelBitField& out) const;

	/**
	 * Run a query and hand each returned row id to @a fn, in result order.
	 *
	 * A selection is a bit field and therefore unordered, so this is the way to
	 * observe what ORDER BY produced -- which also makes it what a result view
	 * needs in order to display rows the way the query asked for.
	 *
	 * @param in what "selection" stands for; null leaves it to the view
	 *
	 * @throws std::runtime_error under the same conditions as select()
	 */
	void for_each_row(const std::string& sql,
	                  const std::function<void(size_t)>& fn,
	                  const PVCore::PVSelBitField* in = nullptr) const;

	/**
	 * Quote @a name if a query could not carry it as-is.
	 *
	 * Axis names are exposed verbatim, so a name holding a dash, a space or any
	 * other non-identifier character has to be double quoted to be written in a
	 * query -- "font-infos-family_name" rather than font-infos-family_name,
	 * which SQL would read as a subtraction. Names that need nothing are
	 * returned unchanged, to keep queries readable.
	 */
	static std::string quote_identifier(const std::string& name);

	/**
	 * A query result that is not a selection: values already rendered as text.
	 *
	 * Aggregates, group-bys and the like return rows that are not rows of the
	 * source, so they cannot become a selection -- but they are exactly what one
	 * writes to understand a dataset, so they are worth showing rather than
	 * rejecting.
	 */
	struct Table {
		std::vector<std::string> column_names;
		//! SQL type of each column, as reported by the engine.
		std::vector<std::string> column_types;
		std::vector<std::vector<std::string>> rows;
		//! True when the query returned more rows than the requested cap.
		bool truncated = false;

		/**
		 * True when the result is a value/count pair, i.e. what a GROUP BY
		 * produces: two columns whose second one is integral.
		 *
		 * Read from the declared types rather than guessed from column names,
		 * which the query is free to choose.
		 */
		bool is_value_count() const;
	};

	/**
	 * Run any query and return its result as text.
	 *
	 * @param max_rows cap on the rows materialized, since the whole result is
	 *                 held in memory. A result worth reading is small; a large
	 *                 one is better expressed as a selection.
	 */
	Table run_tabular(const std::string& sql,
	                  const PVCore::PVSelBitField* in = nullptr,
	                  size_t max_rows = 10000) const;

	/**
	 * Whether a query's result can become a selection, i.e. projects a single
	 * integer column named "rowid". Lets a caller choose between applying a
	 * selection and displaying a table without running the query twice.
	 */
	bool yields_selection(const std::string& sql) const;

	/**
	 * SQL type of each column, aligned with column_names(). Useful to tell a
	 * user, or a completer, which columns take a number and which take a
	 * string -- the mapping is not obvious from an axis name.
	 */
	std::vector<std::string> column_types() const;

	/**
	 * Storage type of each column as Squey names it -- "ipv4", "mac_address",
	 * "datetime", "number_uint32", "string" -- aligned with column_names(), the
	 * first entry empty since rowid is not an axis.
	 *
	 * The SQL type alone does not say what a column holds: an address and a
	 * counter are both UINTEGER, and only one of them has an ipv4() conversion
	 * worth reaching for. This is what a completer needs to say which.
	 */
	std::vector<std::string> column_axis_types() const;

	/**
	 * How many filters the last query pushed into the scan and were let go.
	 *
	 * A pushed filter pvcop cannot answer is evaluated on the emitted chunk --
	 * except one with no static form, whose value moves as the query runs, which
	 * DuckDB renders as the constant true. That is not a predicate the scan owes
	 * an answer to; it is offered so the scan may read less, while an operator
	 * above computes the result regardless. This counts those.
	 *
	 * Exposed so a test can say that path is still exercised. It is the one place
	 * the scan takes DuckDB at its word, and a rule nothing exercises is a rule
	 * nobody would notice going stale.
	 */
	size_t dropped_optional_filters() const;

	/**
	 * Column names as exposed to SQL, in nraw order: the axis names, verbatim.
	 * Pass them through quote_identifier() to write them into a query.
	 */
	std::vector<std::string> column_names() const;

  private:
	/**
	 * Shared constructor. @a name_of returns the SQL name of a column, and is
	 * called at bind time rather than stored, so names always reflect their
	 * current source. An empty function names columns "col_<index>".
	 *
	 * It exists because the translation unit holding the DuckDB code is built
	 * as C++17 (see the note in that file), which rules out including the Squey
	 * headers that carry the format -- they require C++20 or later. The
	 * PVSource and PVView constructors above are therefore defined in a separate
	 * C++23 file, and @a scopes is how the row scopes reach the DuckDB code
	 * without it ever seeing a view.
	 */
	PVDuckDBQuery(const PVRush::PVNraw& nraw,
	              std::function<std::string(size_t)> name_of,
	              Scopes scopes);

	/**
	 * Same, for the several sources a query may name at once.
	 *
	 * @a sources must not be empty: its first entry is the one this object
	 * speaks for -- the source a bare scope reads, the row count a selection
	 * is sized against, and the columns a completer is offered.
	 */
	explicit PVDuckDBQuery(std::vector<Source> sources);

	void run(const std::string& sql,
	         const PVCore::PVSelBitField* in,
	         PVCore::PVSelBitField& out) const;

	//! Shared body: fills @a out when non-null, calls @a fn per row when set.
	void run(const std::string& sql,
	         const PVCore::PVSelBitField* in,
	         PVCore::PVSelBitField* out,
	         const std::function<void(size_t)>& fn) const;

  private:
	struct impl;
	std::unique_ptr<impl> _d;
};

} // namespace Squey

#endif // __SQUEY_PVDUCKDBQUERY__
