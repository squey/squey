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

#include <squey/PVDuckDBQuery.h>

#include <pvkernel/core/PVSelBitField.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvcop/db/algo.h>
#include <pvcop/db/array.h>
#include <pvcop/db/read_dict.h>
#include <pvcop/db/string_index_types.h>

#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/execution/expression_executor.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/filter/conjunction_filter.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/filter/in_filter.hpp>
#include <duckdb/planner/filter/optional_filter.hpp>
#include <duckdb/planner/table_filter.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{

// Rows are handed out to threads in blocks, so that a thread walking a sparse
// selection skips whole empty regions without contending on shared state. The
// block is a multiple of PVSelBitField's 64-bit chunk so that
// visit_selected_lines() never has to start mid-word.
constexpr size_t BLOCK_ROWS = 256 * 1024;

/**
 * How one nraw column is exposed to SQL.
 *
 * `data` is a pointer into the mmapped column when the pvcop storage and the
 * DuckDB physical type share a binary layout, which lets the scan reference the
 * column instead of copying it. It is null for the types that need a
 * conversion, and those fall back to the textual representation.
 */
struct column_binding {
	duckdb::LogicalType type;
	const uint8_t* data = nullptr;
	size_t elem_size = 0;
	bool zero_copy = false;

	// A string column does not store its text per row: it stores one dictionary
	// index per row, and the distinct strings once. Handing DuckDB that same
	// shape -- a dictionary vector over the distinct values -- costs one string
	// per distinct value and per query, where building the text row by row costs
	// one per row and per chunk. Null when the column is not a string, or when
	// the scan chose the row-by-row path (see scan_init_global).
	const pvcop::db::read_dict* dict = nullptr;
	const pvcop::string_index_t* indices = nullptr;

	// A cell the format could not read is still given a slot in the storage, and
	// what sits there is an internal encoding rather than a value: an empty cell
	// of a number column reads back as 0. Handing that to SQL would let it be
	// compared and summed as if it were data, so those rows are emitted as NULL
	// -- which is also how the rest of the application treats them, its search
	// filter never matching one against a valid literal.
	//
	// Only for the columns read straight from storage. A textual column has no
	// such gap: pvcop hands back the cell as it was written, empty string
	// included, and the listing shows exactly that.
	bool null_on_invalid = false;

	// Whether pvcop may answer a filter on this column instead of DuckDB.
	//
	// False for every column of a text-mode scan: what SQL sees there is the
	// cell as it was written, while pvcop compares against what the storage
	// holds -- and for a cell the format could not read, the two are different
	// strings. Refusing the shortcut leaves DuckDB to filter the emitted text,
	// which is the text the query is written about.
	bool pushdown = true;
};

/**
 * pvcop storage types whose binary layout is identical to a DuckDB physical
 * type. Every one of these is order-preserving: comparing the stored integer
 * yields the semantic order, which is what makes ORDER BY correct without any
 * custom comparator.
 *
 * The business types are deliberately mapped onto unsigned integers rather than
 * onto text: pvcop already stores an IPv4 as a host-order uint32 (see
 * ntohl() in pvcop's src/types/ipv4.cpp), so the numeric order is the address
 * order. Rendering them as VARCHAR would sort them lexicographically, which is
 * wrong.
 */
struct storage_type {
	duckdb::LogicalTypeId id;
	size_t size;
};

const std::unordered_map<std::string, storage_type>& zero_copy_types()
{
	static const std::unordered_map<std::string, storage_type> types = {
	    {"number_int8", {duckdb::LogicalTypeId::TINYINT, 1}},
	    {"number_uint8", {duckdb::LogicalTypeId::UTINYINT, 1}},
	    {"number_int16", {duckdb::LogicalTypeId::SMALLINT, 2}},
	    {"number_uint16", {duckdb::LogicalTypeId::USMALLINT, 2}},
	    {"number_int32", {duckdb::LogicalTypeId::INTEGER, 4}},
	    {"number_uint32", {duckdb::LogicalTypeId::UINTEGER, 4}},
	    {"number_int64", {duckdb::LogicalTypeId::BIGINT, 8}},
	    {"number_uint64", {duckdb::LogicalTypeId::UBIGINT, 8}},
	    {"number_float", {duckdb::LogicalTypeId::FLOAT, 4}},
	    {"number_double", {duckdb::LogicalTypeId::DOUBLE, 8}},
	    // Business types: the stored integer is already the sort key. pvcop
	    // converts an IPv4 with ntohl() precisely so that the numeric order is
	    // the address order (see pvcop's src/types/ipv4.cpp).
	    {"ipv4", {duckdb::LogicalTypeId::UINTEGER, 4}},
	    {"mac_address", {duckdb::LogicalTypeId::UBIGINT, 8}},
	    // datetime and datetime_ms are stored as uint64 epochs, so the numeric
	    // order is the chronological one. They are exposed as integers rather
	    // than as TIMESTAMP because that would need a scale conversion per row,
	    // which would give up the zero-copy path; comparing against a literal
	    // date is the ergonomic gain that would justify revisiting it.
	    {"datetime", {duckdb::LogicalTypeId::UBIGINT, 8}},
	    {"datetime_ms", {duckdb::LogicalTypeId::UBIGINT, 8}},
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	    // pvcop stores an IPv6 as a __uint128_t in host order (ntoh128() in its
	    // src/types/ipv6.cpp), and DuckDB's uhugeint_t is {lower, upper}, which
	    // is the little-endian layout of __uint128_t. The guard is deliberate:
	    // on a big-endian host the two disagree and every comparison would be
	    // silently wrong, so the type falls back to text there instead.
	    {"ipv6", {duckdb::LogicalTypeId::UHUGEINT, 16}},
#endif
	    // Deliberately absent: datetime_us, stored as a boost::posix_time::ptime
	    // whose layout is not guaranteed, and duration, a
	    // boost::posix_time::time_duration. Both keep the textual fallback.
	};
	return types;
}

/**
 * Whether an identifier can appear in a query without double quotes.
 *
 * Axis names are exposed verbatim rather than folded into something
 * SQL-friendly: a name the user never saw in Squey would be one more thing to
 * remember, and SQL already covers the case -- any character is allowed inside
 * "double quotes". Callers wanting a name they can paste into a query should
 * use quote_identifier().
 */
bool needs_quoting(const std::string& name)
{
	if (name.empty() || (std::isdigit(static_cast<unsigned char>(name.front())) != 0)) {
		return true;
	}
	return std::any_of(name.begin(), name.end(), [](char c) {
		return std::isalnum(static_cast<unsigned char>(c)) == 0 && c != '_';
	});
}

/**
 * Which rows a table function stands for. One registration per scope, so the
 * name a query writes is the name DuckDB resolves.
 */
enum class scan_scope {
	//! What is selected right now.
	selection,
	//! Every row the layer stack lets through.
	layers,
	//! One layer, named by the function's argument.
	layer
};

/**
 * Shared state the table function reads. It is owned by PVDuckDBQuery::impl and
 * outlives every query, but `input` is rebound per query: only one query runs at
 * a time (select() serializes them), so the scan always observes the selection
 * of the query that is running.
 */
struct scan_context : public duckdb::TableFunctionInfo {
	scan_scope scope = scan_scope::selection;
	// Every source a query may name, owned by the impl. Entry 0 is the one this
	// query object speaks for: what a bare scope reads when no source argument
	// says otherwise. The others are reachable by name, which is what lets a
	// join cross sources.
	//
	// Each carries its own column naming and its own scopes, queried when a
	// query runs rather than read once: a console outlives any particular
	// selection, and each source follows its own current view.
	//
	// Neither pvcop nor the nraw carries a column name: pvcop::db::collection
	// addresses everything by index, and axis names live in the Squey format.
	// That indirection is what lets this code stay free of the Squey headers
	// that would pull in a newer C++ standard than it can be built with.
	const std::vector<Squey::PVDuckDBQuery::Source>* sources = nullptr;
	// Overrides the "selection" scope for the duration of one query, for a
	// caller holding the selection it means rather than reading the view's.
	// It speaks for the query object's own source, so it applies to entry 0
	// alone: a joined source has its own selection and no caller named it.
	const PVCore::PVSelBitField* input = nullptr;
	// Where to count the optional filters this scan let go. Owned by the impl
	// and shared by its three contexts, since one query may read several.
	std::atomic<size_t>* dropped_optional = nullptr;

	//! The source a bare scope reads. Never null once the impl is built.
	const Squey::PVDuckDBQuery::Source* primary() const
	{
		return sources != nullptr && not sources->empty() ? &(*sources)[0] : nullptr;
	}
};

/**
 * @a text as an SQL string literal: single quotes, doubled inside.
 *
 * A source is named by whoever imported it, so its name can hold a quote;
 * writing one into a generated query unescaped would end the literal early and
 * turn the rest of the name into syntax.
 */
std::string quote_literal(const std::string& text)
{
	std::string quoted = "'";
	for (char c : text) {
		if (c == '\'') {
			quoted += '\'';
		}
		quoted += c;
	}
	return quoted + "'";
}

/**
 * The source names a query could have meant, to turn a typo into an error that
 * says what exists. Namesakes are counted rather than repeated, since what
 * tells them apart is a position and not another name.
 */
std::string known_sources(const scan_context& ctx)
{
	if (ctx.sources == nullptr || ctx.sources->size() <= 1) {
		return ". This console exposes a single source.";
	}
	std::vector<std::pair<std::string, size_t>> counted;
	for (const Squey::PVDuckDBQuery::Source& src : *ctx.sources) {
		auto it = std::find_if(counted.begin(), counted.end(),
		                       [&src](const std::pair<std::string, size_t>& seen) {
			                       return seen.first == src.name;
		                       });
		if (it == counted.end()) {
			counted.emplace_back(src.name, 1);
		} else {
			++it->second;
		}
	}
	std::string known = ". Known sources: ";
	for (size_t i = 0; i < counted.size(); ++i) {
		known += (i == 0 ? "'" : ", '") + counted[i].first + "'";
		if (counted[i].second > 1) {
			known += " (" + std::to_string(counted[i].second) + ")";
		}
	}
	return known + ".";
}

/**
 * Which source a scan reads, from the "source" and "source_position" arguments.
 *
 * No argument means the source this query object speaks for, so every query
 * written before sources could be named keeps its meaning.
 *
 * Source names are not unique -- a project may hold two sources called the
 * same -- so "source_position" tells namesakes apart, the way the Python
 * API's source(name, position) already does. Naming a source that does not exist is
 * an error listing the ones that do: a query that silently read the wrong rows
 * would answer with a plausible selection, which is worse than not answering.
 */
const Squey::PVDuckDBQuery::Source*
resolve_source(const scan_context& ctx,
               const duckdb::named_parameter_map_t& named_parameters)
{
	const auto name_param = named_parameters.find("source");
	if (name_param == named_parameters.end() || name_param->second.IsNull()) {
		const auto position_param = named_parameters.find("source_position");
		if (position_param != named_parameters.end() && not position_param->second.IsNull()) {
			throw duckdb::BinderException(
			    "source_position := takes a source := to disambiguate");
		}
		return ctx.primary();
	}

	const std::string wanted = name_param->second.ToString();
	size_t position = 0;
	const auto position_param = named_parameters.find("source_position");
	if (position_param != named_parameters.end() && not position_param->second.IsNull()) {
		const int64_t asked = position_param->second.GetValue<int64_t>();
		if (asked < 0) {
			throw duckdb::BinderException("source_position := must not be negative");
		}
		position = size_t(asked);
	}

	size_t seen = 0;
	for (const Squey::PVDuckDBQuery::Source& candidate : *ctx.sources) {
		if (candidate.name == wanted) {
			if (seen == position) {
				return &candidate;
			}
			++seen;
		}
	}

	if (seen == 0) {
		throw duckdb::BinderException("no source named '" + wanted + "'" +
		                              known_sources(ctx));
	}
	throw duckdb::BinderException("source '" + wanted + "' has no source_position " +
	                              std::to_string(position) + ": there " +
	                              (seen == 1 ? "is 1 source" : "are " + std::to_string(seen) +
	                                                               " sources") +
	                              " by that name");
}

/**
 * Add a hint to a query error when its cause is a common mistake.
 *
 * SQL quoting is the reverse of most languages: "x" names a column, 'x' is a
 * string. Writing WHERE name == "value" therefore asks to compare against a
 * column called "value", and the resulting "column not found" says nothing
 * about the quotes -- which is what makes it worth catching here rather than
 * leaving every caller, human or not, to rediscover it.
 */
std::string explain_error(const std::string& error, const std::string& sql)
{
	const bool column_not_found = error.find("not found in FROM clause") != std::string::npos ||
	                              error.find("Referenced column") != std::string::npos;
	if (column_not_found && sql.find('"') != std::string::npos) {
		return error +
		       "\n\nHint: in SQL, \"double quotes\" name a column and 'single quotes' hold a "
		       "string. If you meant a text value, write 'value' rather than \"value\".";
	}
	return error;
}

/**
 * Turn a bare predicate into a full query over @a scope.
 *
 * Only the WHERE clause carries intent; "SELECT rowid FROM layers WHERE" is a
 * ritual repeated on every query. So an input that does not start a SQL
 * statement is read as a predicate and wrapped, which makes `port = 80` a valid
 * query on its own. A real statement is passed through untouched, so nothing
 * that already worked stops working -- and a query naming a scope of its own
 * says so where it can be read.
 */
std::string wrap_predicate(const std::string& input, const std::string& scope)
{
	const size_t begin = input.find_first_not_of(" \t\r\n");
	if (begin == std::string::npos) {
		return input;
	}

	size_t end = begin;
	while (end < input.size() && (std::isalpha(static_cast<unsigned char>(input[end])) != 0)) {
		++end;
	}
	std::string first_word = input.substr(begin, end - begin);
	std::transform(first_word.begin(), first_word.end(), first_word.begin(),
	               [](unsigned char c) { return std::toupper(c); });

	// Every keyword that can open a statement, FROM included: DuckDB accepts
	// FROM-first queries.
	static const std::unordered_set<std::string> STATEMENT_HEADS = {
	    "SELECT", "WITH",   "FROM",     "VALUES", "TABLE",
	    "SHOW",   "DESCRIBE", "EXPLAIN", "PRAGMA", "SUMMARIZE"};
	if (STATEMENT_HEADS.count(first_word) != 0) {
		return input;
	}

	return "SELECT rowid FROM " + scope + " WHERE (" + input + ")";
}

std::string column_name(const Squey::PVDuckDBQuery::Source& src, PVCol col)
{
	if (src.name_of) {
		const std::string name = src.name_of(size_t(col));
		if (not name.empty()) {
			return name;
		}
	}
	return std::string("col_") + std::to_string(col);
}

/**
 * The tail of the error a misnamed layer produces: what it could have meant.
 *
 * A layer name is typed by hand and is whatever the user called it, so the list
 * is the answer to the question the error raises.
 */
std::string known_layers(const Squey::PVDuckDBQuery::Source& src)
{
	if (not src.scopes.layer_names) {
		return ". This source exposes no layers.";
	}
	const std::vector<std::string> names = src.scopes.layer_names();
	if (names.empty()) {
		return ". This source exposes no layers.";
	}
	std::string known = ". Known layers: ";
	for (size_t i = 0; i < names.size(); ++i) {
		known += (i == 0 ? "'" : ", '") + names[i] + "'";
	}
	return known + ".";
}

struct scan_bind_data : public duckdb::TableFunctionData {
	scan_context* ctx = nullptr;
	// Which source this scan reads, resolved once when the query was bound.
	// Points into the impl's registry, which outlives every query.
	const Squey::PVDuckDBQuery::Source* src = nullptr;
	size_t row_count = 0;
	// Indexed by nraw column, offset by one against the emitted schema because
	// column 0 is rowid.
	std::vector<column_binding> columns;
	// The name each of those was emitted under.
	std::vector<std::string> column_names;
	// The layer this scan reads, for scan_scope::layer and empty otherwise.
	std::string layer;
	// Every column rendered the way the listing renders it, unreadable cells
	// included. Asked for with "text := true".
	bool text = false;
};

struct scan_global_state : public duckdb::GlobalTableFunctionState {
	// Blocks are claimed with a single atomic increment, so a thread that lands
	// on an empty region moves on without blocking the others.
	std::atomic<size_t> next_block{0};
	size_t block_count = 0;
	size_t max_threads = 1;
	// Which table columns the query actually reads, in output order.
	duckdb::vector<duckdb::column_t> column_ids;
	// Rows kept by the filters pvcop answered, already composed with the input
	// selection. Null when no filter was taken that way.
	std::unique_ptr<PVCore::PVSelBitField> filtered;
	// What the scan walks: the input selection narrowed by the above, or null
	// for "every row".
	const PVCore::PVSelBitField* selection = nullptr;
	// The filters pvcop did not take, as one expression over the emitted chunk.
	// Accepting a filter means the operator that would have applied it is gone,
	// so what is not answered here has to be answered per chunk.
	duckdb::unique_ptr<duckdb::Expression> residual;
	// The distinct values of each dictionary-backed column, built once for the
	// whole query and indexed by nraw column. Null where the scan builds the
	// text row by row instead. Written before any thread starts and only read
	// afterwards, so the chunks share it without locking.
	std::vector<duckdb::unique_ptr<duckdb::Vector>> dictionaries;

	idx_t MaxThreads() const override { return max_threads; }
};

struct scan_local_state : public duckdb::LocalTableFunctionState {
	// Row range of the block currently being walked, and how far into it we are.
	size_t block_end = 0;
	size_t cursor = 0;
	bool has_block = false;
	// Reused across chunks to avoid reallocating per call.
	std::vector<uint32_t> rows;
	// Evaluates the filters pvcop did not take. One per thread: an executor
	// holds the intermediate state of the expression it runs.
	duckdb::unique_ptr<duckdb::ExpressionExecutor> executor;
};

/**
 * Read @a filter as a set of values the column must be one of, appending them to
 * @a values.
 *
 * Returns false as soon as the filter says anything else, in which case @a
 * values is meaningless. Membership is the shape pvcop settles by looking the
 * values up in the column's dictionary once and then comparing indices -- a
 * different algorithm, not the same one moved elsewhere. A range or a pattern
 * would still be a comparison per row, which DuckDB does at least as well.
 *
 * An OR of equalities on one column is membership spelled out, and DuckDB
 * spells it that way for a short IN list, so it is read here as one.
 */
bool collect_membership(const duckdb::TableFilter& filter, std::vector<std::string>& values)
{
	switch (filter.filter_type) {
	case duckdb::TableFilterType::CONSTANT_COMPARISON: {
		const auto& constant = filter.Cast<duckdb::ConstantFilter>();
		if (constant.comparison_type != duckdb::ExpressionType::COMPARE_EQUAL ||
		    constant.constant.IsNull()) {
			return false;
		}
		values.emplace_back(constant.constant.ToString());
		return true;
	}
	case duckdb::TableFilterType::IN_FILTER: {
		const auto& in_filter = filter.Cast<duckdb::InFilter>();
		if (in_filter.values.empty()) {
			return false;
		}
		for (const duckdb::Value& value : in_filter.values) {
			if (value.IsNull()) {
				return false;
			}
			values.emplace_back(value.ToString());
		}
		return true;
	}
	case duckdb::TableFilterType::CONJUNCTION_OR: {
		const auto& disjunction = filter.Cast<duckdb::ConjunctionOrFilter>();
		if (disjunction.child_filters.empty()) {
			return false;
		}
		for (const auto& child : disjunction.child_filters) {
			if (not collect_membership(*child, values)) {
				return false;
			}
		}
		return true;
	}
	case duckdb::TableFilterType::OPTIONAL_FILTER: {
		// DuckDB pushes an IN list this way: the scan may use it to read less,
		// but an operator above still applies the predicate. Reading through the
		// wrapper is what turns it into a membership pvcop can answer.
		const auto& optional = filter.Cast<duckdb::OptionalFilter>();
		return optional.child_filter != nullptr &&
		       collect_membership(*optional.child_filter, values);
	}
	default:
		return false;
	}
}

/**
 * Whether @a expression is the constant true, i.e. says nothing.
 *
 * This is how DuckDB renders a filter that has no static form -- one whose value
 * changes as the query runs. It answers with a tautology rather than refusing,
 * so the tautology is the signal.
 */
bool expresses_nothing(const duckdb::Expression& expression)
{
	if (expression.GetExpressionClass() != duckdb::ExpressionClass::BOUND_CONSTANT) {
		return false;
	}
	const duckdb::Value& value = expression.Cast<duckdb::BoundConstantExpression>().value;
	return not value.IsNull() && value.type().id() == duckdb::LogicalTypeId::BOOLEAN &&
	       duckdb::BooleanValue::Get(value);
}

/**
 * Answer @a filter with pvcop, narrowing @a in into @a out.
 *
 * Returns false when pvcop brings nothing over DuckDB for this filter, leaving
 * it to be evaluated on the emitted chunk instead.
 */
bool select_with_pvcop(const pvcop::db::array& array,
                       const duckdb::TableFilter& filter,
                       const PVCore::PVSelBitField& in,
                       PVCore::PVSelBitField& out)
{
	std::vector<std::string> values;
	if (not collect_membership(filter, values)) {
		return false;
	}

	// A literal that does not convert into the column's type matches nothing as
	// far as pvcop is concerned, but saying so is DuckDB's call: it may have a
	// cast in mind that pvcop does not know about.
	std::vector<std::string> unconverted;
	const pvcop::db::array wanted = pvcop::db::algo::to_array(array, values, &unconverted);
	if (not unconverted.empty()) {
		return false;
	}

	// subselect() intersects with its input selection, so filters compose by
	// being applied one after the other.
	out.select_none();
	pvcop::db::algo::subselect(array, wanted, in, out);
	return true;
}

// The signature must use duckdb::vector and duckdb::string: they are distinct
// types from their std:: counterparts, and the function pointer would not
// convert to table_function_bind_t otherwise.
duckdb::unique_ptr<duckdb::FunctionData> scan_bind(duckdb::ClientContext&,
                                                   duckdb::TableFunctionBindInput& input,
                                                   duckdb::vector<duckdb::LogicalType>& return_types,
                                                   duckdb::vector<duckdb::string>& names)
{
	auto* ctx = dynamic_cast<scan_context*>(input.info.get());
	if (ctx == nullptr || ctx->primary() == nullptr || ctx->primary()->nraw == nullptr) {
		throw duckdb::BinderException("pvcop scan is not bound to a source");
	}

	auto bind_data = duckdb::make_uniq<scan_bind_data>();
	bind_data->ctx = ctx;
	bind_data->src = resolve_source(*ctx, input.named_parameters);
	bind_data->row_count = bind_data->src->nraw->row_count();

	const auto text_param = input.named_parameters.find("text");
	if (text_param != input.named_parameters.end() && not text_param->second.IsNull()) {
		bind_data->text = text_param->second.GetValue<bool>();
	}

	if (ctx->scope == scan_scope::layer) {
		if (input.inputs.empty() || input.inputs[0].IsNull()) {
			throw duckdb::BinderException("layer() takes the name of a layer");
		}
		bind_data->layer = input.inputs[0].ToString();

		// Resolved again when the scan starts, since a layer can be removed
		// between planning and execution. Checked here so that the ordinary
		// case, a typo, is an error naming what does exist rather than an empty
		// result that reads like an answer.
		if (not bind_data->src->scopes.layer ||
		    bind_data->src->scopes.layer(bind_data->layer) == nullptr) {
			throw duckdb::BinderException("no layer named '" + bind_data->layer + "'" +
			                              known_layers(*bind_data->src));
		}
	}

	// Column 0 is the physical row index. It is the only stable identity of a
	// row, and the column a query must project for its result to be turned
	// back into a selection.
	// BIGINT rather than UBIGINT: Vector::Sequence(), used to emit contiguous
	// row ranges without materializing them, is only implemented for the signed
	// types, and BIGINT is what DuckDB uses for row ids anyway.
	return_types.emplace_back(duckdb::LogicalType::BIGINT);
	names.emplace_back("rowid");

	const PVCol column_count = bind_data->src->nraw->column_count();
	std::unordered_map<std::string, int> used_names;
	for (PVCol col(0); col < column_count; ++col) {
		const pvcop::db::array& array = bind_data->src->nraw->column(col);

		column_binding binding;
		// A text scan renders every column through pvcop's own formatter, which
		// is what a cell of the listing draws -- one the format could not read
		// included, since pvcop kept the text it was written with. That is the
		// whole point of the mode, so the storage shortcuts are skipped rather
		// than chosen against.
		//
		// Except where the storage already is that text: a string column with
		// nothing unreadable in it reads the same either way, so pvcop can still
		// answer a filter on it -- which is the column a text query is usually
		// written about.
		binding.pushdown = not bind_data->text ||
		                   (array.is_string() && array.has_invalid() == pvcop::db::NONE);
		const auto it = bind_data->text ? zero_copy_types().end()
		                                : zero_copy_types().find(array.type());
		if (it != zero_copy_types().end()) {
			binding.type = duckdb::LogicalType(it->second.id);
			binding.elem_size = it->second.size;
			binding.data = static_cast<const uint8_t*>(array.data());
			binding.zero_copy = binding.data != nullptr;
			binding.null_on_invalid =
			    binding.zero_copy && array.has_invalid() != pvcop::db::NONE;
		}
		if (not binding.zero_copy) {
			// Everything the mapping does not cover is exposed through its
			// textual form. That keeps any source queryable, but such a column
			// sorts lexicographically: extend zero_copy_types() rather than
			// relying on this for a type whose order matters.
			binding.type = duckdb::LogicalType::VARCHAR;

			// A string column carries its dictionary, so it can be handed over
			// without building the text of every row. Whether that is worth it
			// depends on how many rows the query reads, which is only known once
			// the scan starts: this records what the option needs, and
			// scan_init_global decides.
			//
			// Not when the column holds cells the format could not read: pvcop
			// keeps their text in a dictionary of its own and reads the stored
			// value as an index into it, so those rows are the one place where
			// the column's dictionary and what a cell draws part company.
			if (array.is_string() && array.has_invalid() == pvcop::db::NONE) {
				const pvcop::db::read_dict* dict = array.dict();
				if (dict != nullptr && dict->size() > 0) {
					binding.dict = dict;
					binding.indices =
					    array.to_core_array<pvcop::string_index_t>().data();
				}
			}
		}

		std::string name = column_name(*bind_data->src, col);
		const int seen = used_names[name]++;
		if (seen > 0) {
			name += "_" + std::to_string(col);
		}

		return_types.emplace_back(binding.type);
		bind_data->column_names.emplace_back(name);
		names.emplace_back(std::move(name));
		bind_data->columns.emplace_back(std::move(binding));
	}

	return bind_data;
}

/**
 * The rows this scan's scope stands for, or null for every row.
 *
 * Resolved when the scan starts rather than when it was planned: a selection
 * changes under a console, and the scope is what the query asked to read now.
 */
const PVCore::PVSelBitField* resolve_scope(const scan_context& ctx,
                                           const scan_bind_data& bind_data)
{
	const Squey::PVDuckDBQuery::Scopes& scopes = bind_data.src->scopes;
	const PVCore::PVSelBitField* rows = nullptr;

	switch (ctx.scope) {
	case scan_scope::selection:
		// A caller holding the selection it means says so; otherwise the scope
		// reads whatever is selected now. The override speaks for the query
		// object's own source, so a joined one keeps reading its own view --
		// otherwise "filter within what I have" would silently narrow the
		// other side of the join by a selection taken from this one.
		rows = (ctx.input != nullptr && bind_data.src == ctx.primary())
		           ? ctx.input
		           : (scopes.selection ? scopes.selection() : nullptr);
		break;
	case scan_scope::layers:
		rows = scopes.layers ? scopes.layers() : nullptr;
		break;
	case scan_scope::layer:
		rows = scopes.layer ? scopes.layer(bind_data.layer) : nullptr;
		// The name was checked when the query was bound, so getting here means
		// the layer was removed in between. Reading every row instead would
		// answer a question nobody asked.
		if (rows == nullptr) {
			throw duckdb::InvalidInputException("the layer '" + bind_data.layer +
			                                    "' no longer exists");
		}
		break;
	}

	// A selection holding every row is no restriction at all, and saying so lets
	// the scan emit contiguous ranges rather than walk a bit field. Worth the
	// count: nothing hidden is the ordinary case, and counting bits is a pass
	// over a bit field rather than over the data.
	if (rows != nullptr && rows->bit_count() == bind_data.row_count) {
		return nullptr;
	}
	return rows;
}

duckdb::unique_ptr<duckdb::GlobalTableFunctionState>
scan_init_global(duckdb::ClientContext& context, duckdb::TableFunctionInitInput& input)
{
	const auto& bind_data = input.bind_data->Cast<scan_bind_data>();

	auto state = duckdb::make_uniq<scan_global_state>();
	state->block_count = (bind_data.row_count + BLOCK_ROWS - 1) / BLOCK_ROWS;
	state->column_ids = input.column_ids;

	// One thread per block at most, so a small source does not pay for a wide
	// fan-out it cannot fill.
	const size_t threads = duckdb::NumericCast<size_t>(context.db->NumberOfThreads());
	state->max_threads = std::max<size_t>(1, std::min(threads, state->block_count));

	const scan_context& ctx = *bind_data.ctx;
	const PVCore::PVSelBitField* scoped = resolve_scope(ctx, bind_data);

	// --- Filters DuckDB pushed into the scan ---------------------------------
	// Taking them is a commitment rather than a hint: the operator that would
	// have applied them above the scan is removed, so a filter accepted and not
	// applied widens the result. Whatever pvcop does not answer is therefore
	// kept and run per chunk through DuckDB's own expression machinery -- the
	// very code that removed operator would have run.
	// A filter is keyed by its column's position in the scan's projection, and
	// that is also its position in the emitted chunk -- but only while the two
	// lists agree. They would part company if DuckDB dropped the filter columns
	// before the chunk is handed over, which is what filter_prune asks for and
	// this scan does not. Checked rather than assumed: a mismatch would move
	// every residual filter onto the wrong column, and answer with a plausible
	// selection rather than an error.
	// NotImplementedException rather than InternalException: DuckDB reads the
	// latter as its own state being unsound and invalidates the database, so
	// every later query of the session fails too. This is a condition the scan
	// detects and declines, not evidence that the engine is broken -- it should
	// cost one query.
	if (input.CanRemoveFilterColumns()) {
		throw duckdb::NotImplementedException(
		    "pvcop scan: filter columns pruned from the chunk, which its filter mapping "
		    "does not account for");
	}

	if (input.filters != nullptr && not input.filters->filters.empty()) {
		PVCore::PVSelBitField kept(bind_data.row_count);
		if (scoped != nullptr) {
			kept = *scoped;
		} else {
			kept.select_all();
		}
		bool taken_any = false;

		for (const auto& entry : input.filters->filters) {
			const duckdb::idx_t key = entry.first;
			const duckdb::TableFilter& filter = *entry.second;

			// Filters are keyed by position in the scan's projection, which is
			// also the column's position in the emitted chunk: nothing is
			// pruned before the filter runs.
			const duckdb::column_t id =
			    key < state->column_ids.size() ? state->column_ids[key] : 0;

			bool taken = false;
			if (id != 0 && id <= bind_data.columns.size() && bind_data.columns[id - 1].pushdown) {
				const PVCol col(static_cast<PVCol::value_type>(id - 1));
				PVCore::PVSelBitField narrowed(bind_data.row_count);
				if (select_with_pvcop(bind_data.src->nraw->column(col), filter, kept,
				                      narrowed)) {
					kept = std::move(narrowed);
					taken = taken_any = true;
				}
			}

			if (not taken) {
				const duckdb::LogicalType& type = (id == 0 || id > bind_data.columns.size())
				                                      ? duckdb::LogicalType::BIGINT
				                                      : bind_data.columns[id - 1].type;
				const duckdb::BoundReferenceExpression column(type,
				                                              duckdb::NumericCast<idx_t>(key));

				// Whatever pvcop declined is evaluated on the emitted chunk --
				// the very work the operator DuckDB removed would have done.
				duckdb::unique_ptr<duckdb::Expression> expression;
				try {
					expression = filter.ToExpression(column);
				} catch (const std::exception&) {
					expression = nullptr;
				}

				// Unless there is nothing to evaluate. A filter whose value moves
				// as the query runs -- what "ORDER BY col LIMIT n" pushes, or a
				// join's bloom filter -- has no static form, and DuckDB says so by
				// rendering it as the constant true. Such a thing was never a
				// predicate the scan owed an answer to: it is offered so the scan
				// may read less, while an operator above computes the result
				// regardless. Letting it go can only pass more rows than intended,
				// which that operator then cuts.
				//
				// This is what stands in for a list of filter kinds: the question
				// is not which kind it is but whether it says anything, and the
				// answer comes from DuckDB rather than from a list to keep up to
				// date.
				if (expression == nullptr || expresses_nothing(*expression)) {
					if (filter.filter_type != duckdb::TableFilterType::OPTIONAL_FILTER) {
						// Not optional, so nothing above will apply it, and the
						// scan cannot either. NotImplementedException rather than
						// InternalException: DuckDB reads the latter as its own
						// state being unsound and invalidates the database, which
						// would cost the session rather than the query.
						throw duckdb::NotImplementedException(
						    "pvcop scan: DuckDB pushed a filter this scan can neither answer nor "
						    "express, and which nothing above it will apply");
					}
					if (ctx.dropped_optional != nullptr) {
						ctx.dropped_optional->fetch_add(1, std::memory_order_relaxed);
					}
					continue;
				}

				if (state->residual == nullptr) {
					state->residual = std::move(expression);
				} else {
					state->residual = duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
					    duckdb::ExpressionType::CONJUNCTION_AND, std::move(state->residual),
					    std::move(expression));
				}
			}
		}

		if (taken_any) {
			state->filtered = std::make_unique<PVCore::PVSelBitField>(std::move(kept));
		}
	}

	state->selection = state->filtered != nullptr ? state->filtered.get() : scoped;

	// A dictionary is built once and holds one string per distinct value; the
	// row-by-row path builds one per row read. Which is cheaper is therefore a
	// question of how many rows the query reads, and a query restricted to a
	// handful of them would not repay a dictionary of the whole column.
	const size_t rows_read =
	    state->selection != nullptr ? state->selection->bit_count() : bind_data.row_count;

	state->dictionaries.resize(bind_data.columns.size());
	for (const duckdb::column_t id : input.column_ids) {
		// Column 0 is rowid, and DuckDB has sentinel ids of its own.
		if (id == 0 || id > bind_data.columns.size()) {
			continue;
		}
		const column_binding& binding = bind_data.columns[id - 1];
		if (binding.dict == nullptr || binding.dict->size() >= rows_read) {
			continue;
		}

		auto values =
		    duckdb::make_uniq<duckdb::Vector>(duckdb::LogicalType::VARCHAR, binding.dict->size());
		auto* data = duckdb::FlatVector::GetData<duckdb::string_t>(*values);
		for (size_t i = 0; i < binding.dict->size(); ++i) {
			data[i] = duckdb::StringVector::AddString(*values, binding.dict->key(i));
		}
		state->dictionaries[id - 1] = std::move(values);
	}

	return state;
}

duckdb::unique_ptr<duckdb::LocalTableFunctionState>
scan_init_local(duckdb::ExecutionContext& context, duckdb::TableFunctionInitInput&,
                duckdb::GlobalTableFunctionState* global)
{
	auto state = duckdb::make_uniq<scan_local_state>();
	state->rows.reserve(STANDARD_VECTOR_SIZE);

	// One executor per thread over the expression the global state holds: the
	// expression is shared and read-only, the state of its evaluation is not.
	const auto& gstate = global->Cast<scan_global_state>();
	if (gstate.residual != nullptr) {
		state->executor =
		    duckdb::make_uniq<duckdb::ExpressionExecutor>(context.client, *gstate.residual);
	}

	return state;
}

/**
 * Claim the next block of rows. Returns false once the source is exhausted.
 */
bool next_block(const scan_bind_data& bind_data, scan_global_state& gstate,
                scan_local_state& lstate)
{
	const size_t block = gstate.next_block.fetch_add(1, std::memory_order_relaxed);
	if (block >= gstate.block_count) {
		return false;
	}
	lstate.cursor = block * BLOCK_ROWS;
	lstate.block_end = std::min(lstate.cursor + BLOCK_ROWS, bind_data.row_count);
	lstate.has_block = true;
	return true;
}

/**
 * Collect up to STANDARD_VECTOR_SIZE selected rows from the current block.
 *
 * Only used when a selection restricts the scan; the unrestricted scan emits
 * contiguous ranges and never needs the row list.
 */
void gather_selected_rows(const PVCore::PVSelBitField& selection, scan_local_state& lstate)
{
	lstate.rows.clear();
	const PVRow begin = static_cast<PVRow>(lstate.cursor);
	const PVRow end = static_cast<PVRow>(lstate.block_end);

	// visit_selected_lines() walks 64-bit words and skips the empty ones, so a
	// sparse block costs a scan of the bitfield rather than of the data.
	selection.visit_selected_lines(
	    [&](const PVRow row) {
		    if (lstate.rows.size() < STANDARD_VECTOR_SIZE) {
			    lstate.rows.push_back(static_cast<uint32_t>(row));
		    }
	    },
	    end, begin);

	// Resume after the last row taken; the remainder of the block is picked up
	// by the next call.
	if (lstate.rows.size() >= STANDARD_VECTOR_SIZE) {
		lstate.cursor = static_cast<size_t>(lstate.rows.back()) + 1;
	} else {
		lstate.cursor = lstate.block_end;
	}
}

/**
 * Fill one output column with a contiguous run of rows.
 */
void emit_dense_column(const column_binding& binding, const pvcop::db::array& array,
                       const duckdb::Vector* dictionary, size_t offset, size_t count,
                       duckdb::Vector& out)
{
	if (dictionary != nullptr) {
		// The rows' dictionary indices are what the column already stores, so
		// the chunk is that index array turned into a selection vector: no
		// string is built here at all.
		duckdb::SelectionVector sel(count);
		for (size_t i = 0; i < count; ++i) {
			sel.set_index(i, binding.indices[offset + i]);
		}
		out.Slice(*dictionary, sel, count);
		return;
	}

	if (binding.zero_copy) {
		// The vector references the mmapped column: nothing is copied, and the
		// data outlives the chunk because the nraw owns it. The vector type is
		// set explicitly because the previous chunk may have left it as a
		// dictionary, and SetData() is only meaningful on a flat vector.
		out.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
		duckdb::FlatVector::SetData(
		    out, const_cast<duckdb::data_ptr_t>(
		             static_cast<duckdb::const_data_ptr_t>(binding.data + offset * binding.elem_size)));
		if (binding.null_on_invalid) {
			// Reset first: the mask is the one thing SetData() does not replace,
			// so it still carries what the previous chunk wrote.
			auto& validity = duckdb::FlatVector::Validity(out);
			validity.SetAllValid(count);
			for (size_t i = 0; i < count; ++i) {
				if (not array.is_valid(offset + i)) {
					validity.SetInvalid(i);
				}
			}
		}
		return;
	}

	// Textual fallback: the string is built per row, so this path costs a copy
	// the zero-copy one avoids.
	out.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
	for (size_t i = 0; i < count; ++i) {
		const std::string value = array.at(offset + i);
		duckdb::FlatVector::GetData<duckdb::string_t>(out)[i] =
		    duckdb::StringVector::AddString(out, value);
	}
}

/**
 * Fill one output column with an arbitrary set of rows.
 */
void emit_sparse_column(const column_binding& binding, const pvcop::db::array& array,
                        const duckdb::Vector* dictionary, const std::vector<uint32_t>& rows,
                        duckdb::Vector& out)
{
	const size_t count = rows.size();

	if (dictionary != nullptr) {
		// Same indirection as the dense case, composed with the row list: the
		// selection vector goes straight from output position to dictionary
		// entry, so the two indirections cost one.
		duckdb::SelectionVector sel(count);
		for (size_t i = 0; i < count; ++i) {
			sel.set_index(i, binding.indices[rows[i]]);
		}
		out.Slice(*dictionary, sel, count);
		return;
	}

	if (binding.zero_copy) {
		// A dictionary vector keeps the data zero-copy: only the index array is
		// written, and DuckDB resolves the indirection downstream. This is the
		// representation it produces itself after a filter.
		duckdb::Vector base(binding.type,
		                    const_cast<duckdb::data_ptr_t>(
		                        static_cast<duckdb::const_data_ptr_t>(binding.data)));
		duckdb::SelectionVector sel(count);
		for (size_t i = 0; i < count; ++i) {
			sel.set_index(i, rows[i]);
		}
		out.Slice(base, sel, count);
		if (binding.null_on_invalid) {
			// A dictionary vector takes its mask from the vector it references,
			// which here covers the whole column; flattening the chunk first
			// gives it a mask of its own to write into. It copies at most one
			// vector's worth of values.
			out.Flatten(count);
			auto& validity = duckdb::FlatVector::Validity(out);
			validity.SetAllValid(count);
			for (size_t i = 0; i < count; ++i) {
				if (not array.is_valid(rows[i])) {
					validity.SetInvalid(i);
				}
			}
		}
		return;
	}

	out.SetVectorType(duckdb::VectorType::FLAT_VECTOR);
	for (size_t i = 0; i < count; ++i) {
		const std::string value = array.at(rows[i]);
		duckdb::FlatVector::GetData<duckdb::string_t>(out)[i] =
		    duckdb::StringVector::AddString(out, value);
	}
}

void scan_function(duckdb::ClientContext&, duckdb::TableFunctionInput& input,
                   duckdb::DataChunk& output)
{
	const auto& bind_data = input.bind_data->Cast<scan_bind_data>();
	auto& gstate = input.global_state->Cast<scan_global_state>();
	auto& lstate = input.local_state->Cast<scan_local_state>();

	// Set once per query: the input selection, narrowed by whatever the pushed
	// filters resolved.
	const PVCore::PVSelBitField* selection = gstate.selection;

	// An empty block must not end the scan: returning a zero-sized chunk is how
	// DuckDB is told this thread is done, so keep claiming blocks until one
	// yields rows or the source is exhausted.
	while (true) {
		if (not lstate.has_block || lstate.cursor >= lstate.block_end) {
			if (not next_block(bind_data, gstate, lstate)) {
				output.SetCardinality(0);
				return;
			}
		}

		size_t count = 0;
		if (selection == nullptr) {
			count = std::min<size_t>(STANDARD_VECTOR_SIZE, lstate.block_end - lstate.cursor);
			const size_t offset = lstate.cursor;
			lstate.cursor += count;

			for (size_t i = 0; i < gstate.column_ids.size(); ++i) {
				const duckdb::column_t id = gstate.column_ids[i];
				if (id == 0) {
					output.data[i].Sequence(static_cast<int64_t>(offset), 1, count);
					continue;
				}
				const PVCol col(static_cast<PVCol::value_type>(id - 1));
				emit_dense_column(bind_data.columns[id - 1], bind_data.src->nraw->column(col),
				                  gstate.dictionaries[id - 1].get(), offset, count,
				                  output.data[i]);
			}
		} else {
			gather_selected_rows(*selection, lstate);
			if (lstate.rows.empty()) {
				continue; // block held no selected row, try the next one
			}

			count = lstate.rows.size();
			for (size_t i = 0; i < gstate.column_ids.size(); ++i) {
				const duckdb::column_t id = gstate.column_ids[i];
				if (id == 0) {
					output.data[i].SetVectorType(duckdb::VectorType::FLAT_VECTOR);
					auto* rowids = duckdb::FlatVector::GetData<int64_t>(output.data[i]);
					for (size_t k = 0; k < count; ++k) {
						rowids[k] = static_cast<int64_t>(lstate.rows[k]);
					}
					continue;
				}
				const PVCol col(static_cast<PVCol::value_type>(id - 1));
				emit_sparse_column(bind_data.columns[id - 1], bind_data.src->nraw->column(col),
				                   gstate.dictionaries[id - 1].get(), lstate.rows,
				                   output.data[i]);
			}
		}

		output.SetCardinality(count);

		if (lstate.executor != nullptr) {
			duckdb::SelectionVector sel(STANDARD_VECTOR_SIZE);
			const duckdb::idx_t kept = lstate.executor->SelectExpression(output, sel);
			if (kept == 0) {
				// The whole chunk was filtered out. Handing it back would tell
				// DuckDB this thread is done, so claim more work instead.
				output.Reset();
				continue;
			}
			if (kept != count) {
				output.Slice(sel, kept);
			}
		}
		return;
	}
}

} // namespace

struct Squey::PVDuckDBQuery::impl {
	explicit impl(std::vector<Squey::PVDuckDBQuery::Source> sources_p)
	    : db(nullptr), con(db), sources(std::move(sources_p))
	{
		for (scan_context* ctx : {&sel_ctx, &layers_ctx, &layer_ctx}) {
			ctx->sources = &sources;
			ctx->dropped_optional = &dropped_optional;
		}
		sel_ctx.scope = scan_scope::selection;
		layers_ctx.scope = scan_scope::layers;
		layer_ctx.scope = scan_scope::layer;

		register_table("selection", {}, sel_ctx);
		register_table("layers", {}, layers_ctx);
		register_table("layer", {duckdb::LogicalType::VARCHAR}, layer_ctx);

		// A view of the same name over the zero-argument call, so the everyday
		// query reads "FROM selection" rather than "FROM selection()": a table
		// function needs its parentheses, and they are noise on the form that is
		// written most. A bare name resolves the view and a call resolves the
		// function, which is what leaves "selection(text := true)" free to mean
		// something else.
		//
		// layer() has no view: it takes the name of a layer, so there is no
		// zero-argument form to spell without parentheses.
		expect_success(con.Query("CREATE VIEW selection AS SELECT * FROM selection()"));
		expect_success(con.Query("CREATE VIEW layers AS SELECT * FROM layers()"));

		create_source_schemas();
		create_source_listing();
		create_conversions();
		confine();
	}

	/**
	 * Shut the doors a query has no business opening.
	 *
	 * A query is written about the source the console was opened on, and needs
	 * nothing else: not the filesystem, not the network, not another database.
	 * DuckDB opens all of those by default -- read_csv() on any path, ATTACH,
	 * INSTALL -- so a query typed into the console could read whatever the
	 * process can.
	 *
	 * Set here rather than left to a whitelist of statement keywords: that
	 * whitelist exists to tell a predicate from a statement, and a boundary that
	 * happens to fall out of a parsing convenience is one a refactoring silently
	 * removes.
	 *
	 * Last, so that lock_configuration cannot catch the setup itself.
	 */
	void confine()
	{
		// Files and attached databases.
		expect_success(con.Query("SET enable_external_access = false"));
		// An INSTALL reaches the network and a LOAD runs code that was fetched.
		expect_success(con.Query("SET autoinstall_known_extensions = false"));
		expect_success(con.Query("SET autoload_known_extensions = false"));
		expect_success(con.Query("SET allow_unsigned_extensions = false"));
		// And none of the above can be undone from a query.
		expect_success(con.Query("SET lock_configuration = true"));
	}

	/**
	 * Refuse anything that is not a question.
	 *
	 * Reading the first keyword is not enough: DuckDB runs every statement a
	 * string holds, so "SELECT 1; DROP VIEW layers" passes a check that looks at
	 * "SELECT" and then destroys the view (which is how this was found). The
	 * statements are therefore parsed and counted before anything runs.
	 *
	 * Everything the console offers -- SELECT, WITH, FROM-first, VALUES, TABLE,
	 * SHOW, DESCRIBE, PRAGMA, SUMMARIZE -- parses as a SELECT; only EXPLAIN
	 * carries a type of its own. Anything else changes something, and this object
	 * exposes a source to be read.
	 */
	void require_read_only(const std::string& statement)
	{
		duckdb::vector<duckdb::unique_ptr<duckdb::SQLStatement>> statements;
		try {
			statements = con.ExtractStatements(statement);
		} catch (const std::exception&) {
			// Not parseable: running it reports that better than this could.
			return;
		}

		if (statements.size() > 1) {
			throw std::runtime_error(
			    "a query runs one statement at a time, and this text holds " +
			    std::to_string(statements.size()) +
			    ". Everything after the first semicolon would run too, so none of it does.");
		}
		if (statements.empty()) {
			return;
		}

		const duckdb::StatementType type = statements[0]->type;
		if (type != duckdb::StatementType::SELECT_STATEMENT &&
		    type != duckdb::StatementType::EXPLAIN_STATEMENT) {
			throw std::runtime_error(
			    "only a query that reads is allowed here, and this one is a " +
			    std::string(duckdb::StatementTypeToString(type)) +
			    " statement. The tables a query reads are views over the source, which is "
			    "opened read-only.");
		}
	}

	/**
	 * Name the conversions between a business type and the integer it is stored
	 * as, one pair per type: one to write a literal, one to read a column.
	 *
	 * An address is stored as the integer whose order is the address order, which
	 * is what makes ORDER BY right and the scan able to hand DuckDB a pointer into
	 * the column. It also leaves a query reading 3232235777 where the listing
	 * shows 192.168.1.1, which is what these are for.
	 *
	 * The literal-side one is the important half. Converting the constant leaves
	 * the comparison on the stored integer, so the scan reads the column without
	 * building anything and the filter can still be answered by pvcop; converting
	 * the column instead builds a string per row for the same answer. The two
	 * spell the same question and do not cost the same, which is not something a
	 * user can be expected to guess -- hence a name for the fast one.
	 *
	 * Deliberately absent: the datetime types. Their stored epoch depends on the
	 * time format the axis was given -- an axis parsed from "71-07-27 04:32:28"
	 * and one parsed from the epoch of that same instant differ by the local UTC
	 * offset -- so a single conversion would be right for one axis and an hour off
	 * for the other, silently. Reading one as an instant is DuckDB's own
	 * to_timestamp(); writing a literal against one waits for that to be settled.
	 *
	 * Also absent: ipv6. Its text form allows a "::" run whose expansion is not
	 * something to write in SQL.
	 */
	void create_conversions()
	{
		// pvcop stores an IPv4 in host order (ntohl in its src/types/ipv4.cpp), so
		// the four written bytes read most significant first.
		expect_success(con.Query(R"(CREATE MACRO ipv4(address) AS (
		    (CAST(str_split(address, '.')[1] AS BIGINT) * 16777216 +
		     CAST(str_split(address, '.')[2] AS BIGINT) * 65536 +
		     CAST(str_split(address, '.')[3] AS BIGINT) * 256 +
		     CAST(str_split(address, '.')[4] AS BIGINT))::UINTEGER))"));
		expect_success(con.Query(R"(CREATE MACRO ipv4_text(address) AS (
		    (address >> 24)::VARCHAR || '.' || ((address >> 16) & 255)::VARCHAR || '.' ||
		    ((address >> 8) & 255)::VARCHAR || '.' || (address & 255)::VARCHAR))"));

		// A MAC is packed most significant byte first as well (see the union in
		// pvcop's src/types/mac_address.cpp), so the six hex bytes read as one
		// number. The three separators pvcop's parser accepts are all dropped.
		expect_success(con.Query(R"(CREATE MACRO mac_address(address) AS (
		    ('0x' || replace(replace(replace(address, ':', ''), '-', ''), '.', ''))::UBIGINT))"));
		expect_success(con.Query(R"(CREATE MACRO mac_address_text(address) AS (
		    printf('%02X:%02X:%02X:%02X:%02X:%02X',
		           (address >> 40) & 255, (address >> 32) & 255, (address >> 24) & 255,
		           (address >> 16) & 255, (address >> 8) & 255, address & 255)))"));
	}

	void register_table(const std::string& name,
	                    duckdb::vector<duckdb::LogicalType> arguments,
	                    scan_context& ctx)
	{
		duckdb::TableFunction fn(name, std::move(arguments), scan_function, scan_bind,
		                         scan_init_global, scan_init_local);
		// Every scope reads the same rows either as values or as the text they
		// were written with, so the argument belongs on all of them rather than
		// on a fourth name.
		fn.named_parameters["text"] = duckdb::LogicalType::BOOLEAN;
		// Which source the scope reads. Absent means the one this query object
		// speaks for, so a query written when a console showed a single source
		// keeps meaning what it did. "source_position" tells namesakes apart: source
		// names are not unique, and the Python API indexes them by that same
		// pair rather than by name alone.
		fn.named_parameters["source"] = duckdb::LogicalType::VARCHAR;
		// Not "position": POSITION(x IN y) is SQL, so the parser reads a named
		// parameter of that name as the start of that call and stops at the ":=".
		fn.named_parameters["source_position"] = duckdb::LogicalType::BIGINT;
		fn.projection_pushdown = true;
		// Taking filters is a commitment: DuckDB drops the operator that would
		// have applied them (verified -- with the flag set and the filters
		// ignored, a WHERE clause selects every row). scan_init_global answers
		// what pvcop can and keeps the rest for evaluation per chunk.
		fn.filter_pushdown = true;
		// The context is shared, not owned: it lives as long as this object.
		fn.function_info = duckdb::shared_ptr<duckdb::TableFunctionInfo>(
		    &ctx, [](duckdb::TableFunctionInfo*) {});

		duckdb::CreateTableFunctionInfo info(fn);
		con.BeginTransaction();
		auto& catalog = duckdb::Catalog::GetSystemCatalog(*con.context);
		catalog.CreateTableFunction(*con.context, info);
		con.Commit();
	}

	/**
	 * A schema per source, so that a join can read "ventes.selection" rather
	 * than "selection(source := 'ventes')".
	 *
	 * Sugar over the argument form, not a replacement: only a source whose name
	 * is unique gets one, since a schema cannot tell namesakes apart -- and
	 * inventing a suffix for them would put a name in queries that nothing in
	 * the interface ever shows. Names colliding with a schema DuckDB already
	 * has are left out for the same reason: what "main.selection" would mean is
	 * then ambiguous, and the argument form still reaches the source.
	 *
	 * The views are what carry the sugar; layer() stays a function, reached as
	 * "ventes.layer('...')".
	 */
	void create_source_schemas()
	{
		for (const Squey::PVDuckDBQuery::Source& src : sources) {
			if (src.name.empty() || not unique_source_name(src.name) ||
			    reserved_schema(src.name)) {
				continue;
			}
			const std::string schema = Squey::PVDuckDBQuery::quote_identifier(src.name);
			const std::string source_arg = quote_literal(src.name);
			expect_success(con.Query("CREATE SCHEMA " + schema));
			expect_success(con.Query("CREATE VIEW " + schema +
			                         ".selection AS SELECT * FROM selection(source := " +
			                         source_arg + ")"));
			expect_success(con.Query("CREATE VIEW " + schema +
			                         ".layers AS SELECT * FROM layers(source := " +
			                         source_arg + ")"));
		}
	}

	//! True while @a name belongs to exactly one source.
	bool unique_source_name(const std::string& name) const
	{
		size_t seen = 0;
		for (const Squey::PVDuckDBQuery::Source& src : sources) {
			seen += size_t(src.name == name);
		}
		return seen == 1;
	}

	//! Schemas DuckDB defines itself, which a source must not shadow.
	static bool reserved_schema(const std::string& name)
	{
		static const std::set<std::string> RESERVED = {"main", "temp", "system", "pg_catalog",
		                                               "information_schema"};
		return RESERVED.count(name) != 0;
	}

	/**
	 * A "sources" table naming what a query can read.
	 *
	 * Without it the names have to be guessed: a Python script can loop over
	 * the sources it reaches, whereas a query can only name one it already
	 * knows. Built as a literal VALUES list rather than a scan: it describes
	 * the registry, which does not change while this object lives.
	 */
	void create_source_listing()
	{
		std::string rows;
		size_t position = 0;
		std::string previous;
		for (size_t i = 0; i < sources.size(); ++i) {
			const Squey::PVDuckDBQuery::Source& src = sources[i];
			// Position counts namesakes, in registration order, which is the
			// order resolve_source() walks.
			position = 0;
			for (size_t j = 0; j < i; ++j) {
				position += size_t(sources[j].name == src.name);
			}
			rows += (i == 0 ? "" : ", ");
			rows += "(" + quote_literal(src.name) + ", " + std::to_string(position) + ", " +
			        std::to_string(src.nraw != nullptr ? src.nraw->row_count() : 0) + ", " +
			        std::to_string(src.nraw != nullptr ? size_t(src.nraw->column_count()) : 0) +
			        ", " + (i == 0 ? "true" : "false") + ")";
		}
		expect_success(con.Query("CREATE VIEW sources AS SELECT * FROM (VALUES " + rows +
		                         ") AS t(name, position, rows, columns, current)"));
	}

	/**
	 * Hold every source still for the length of a query.
	 *
	 * A join reads more than one, and any of them may have a column appended
	 * or removed under it. Ordered by address rather than by registration, so
	 * two queries taking the same pair take it in the same order and cannot
	 * deadlock against each other; deduplicated because locking the same mutex
	 * twice would.
	 */
	[[nodiscard]] std::vector<std::shared_lock<std::shared_mutex>> hold_sources() const
	{
		std::vector<const PVRush::PVNraw*> ordered;
		for (const Squey::PVDuckDBQuery::Source& src : sources) {
			if (src.nraw != nullptr) {
				ordered.push_back(src.nraw);
			}
		}
		std::sort(ordered.begin(), ordered.end());
		ordered.erase(std::unique(ordered.begin(), ordered.end()), ordered.end());

		std::vector<std::shared_lock<std::shared_mutex>> held;
		held.reserve(ordered.size());
		for (const PVRush::PVNraw* nraw : ordered) {
			held.emplace_back(nraw->lock_structure());
		}
		return held;
	}

	//! The source this query object speaks for. Never null.
	const Squey::PVDuckDBQuery::Source& primary() const { return sources.front(); }

	/**
	 * Run a statement with the interrupt flag under control.
	 *
	 * The flag lives on the connection rather than on a query, so it is cleared
	 * on the way in: one raised after a query ended -- a cancel that lost the
	 * race against the last chunk -- would otherwise kill the query that comes
	 * next. And "running" is what tells interrupt() there is something to stop.
	 */
	duckdb::unique_ptr<duckdb::MaterializedQueryResult> run_interruptible(const std::string& sql)
	{
		con.context->ClearInterrupt();
		running.store(true, std::memory_order_release);
		struct clear_on_exit {
			std::atomic<bool>& flag;
			~clear_on_exit() { flag.store(false, std::memory_order_release); }
		} guard{running};
		return con.Query(sql);
	}

	static void expect_success(duckdb::unique_ptr<duckdb::MaterializedQueryResult> result)
	{
		if (result->HasError()) {
			throw std::runtime_error(result->GetError());
		}
	}

	duckdb::DuckDB db;
	duckdb::Connection con;
	// Owns what every scan reads. Entry 0 is the source this query object
	// speaks for; the rest are the ones a join can name. Never empty, and never
	// resized after construction: the scan holds pointers into it.
	std::vector<Squey::PVDuckDBQuery::Source> sources;
	scan_context sel_ctx;
	scan_context layers_ctx;
	scan_context layer_ctx;
	// Queries are serialized: the scan reads sel_ctx.input without
	// synchronization, which is only sound while a single query runs.
	mutable std::mutex query_lock;
	//! Reset per query. See PVDuckDBQuery::dropped_optional_filters().
	std::atomic<size_t> dropped_optional{0};
	//! Whether a statement is in DuckDB's hands. See PVDuckDBQuery::interrupt().
	std::atomic<bool> running{false};
};

Squey::PVDuckDBQuery::PVDuckDBQuery(const PVRush::PVNraw& nraw,
                                    std::function<std::string(size_t)> name_of,
                                    Scopes scopes)
    : PVDuckDBQuery(std::vector<Source>{
          Source{&nraw, std::move(name_of), std::move(scopes), std::string(), 0}})
{
}

Squey::PVDuckDBQuery::PVDuckDBQuery(std::vector<Source> sources)
    : _d(std::make_unique<impl>(std::move(sources)))
{
}

Squey::PVDuckDBQuery::PVDuckDBQuery(const PVRush::PVNraw& nraw) : PVDuckDBQuery(nraw, {}, {}) {}

Squey::PVDuckDBQuery::~PVDuckDBQuery() = default;

void Squey::PVDuckDBQuery::select(const std::string& sql, PVCore::PVSelBitField& out) const
{
	// No input selection: "selection" then stands for what the view has
	// selected, and a bare predicate reads the layer stack.
	run(sql, nullptr, out);
}

void Squey::PVDuckDBQuery::select(const std::string& sql,
                                  const PVCore::PVSelBitField& in,
                                  PVCore::PVSelBitField& out) const
{
	run(sql, &in, out);
}

void Squey::PVDuckDBQuery::for_each_row(const std::string& sql,
                                        const std::function<void(size_t)>& fn,
                                        const PVCore::PVSelBitField* in) const
{
	run(sql, in, nullptr, fn);
}

void Squey::PVDuckDBQuery::run(const std::string& sql,
                               const PVCore::PVSelBitField* in,
                               PVCore::PVSelBitField& out) const
{
	run(sql, in, &out, {});
}

void Squey::PVDuckDBQuery::run(const std::string& sql,
                               const PVCore::PVSelBitField* in,
                               PVCore::PVSelBitField* out,
                               const std::function<void(size_t)>& fn) const
{
	std::lock_guard<std::mutex> lock(_d->query_lock);

	// A bare predicate reads the selection when a caller handed one over, which
	// is what "filter within what I already selected" means, and the layer stack
	// otherwise -- the rows the application itself searches.
	const std::string statement = wrap_predicate(sql, in != nullptr ? "selection" : "layers");
	const bool was_wrapped = statement != sql;
	_d->require_read_only(statement);
	_d->dropped_optional.store(0, std::memory_order_relaxed);

	// Held for the whole query: the scan hands DuckDB pointers into the columns
	// and walks them from several threads, so the set of columns has to stay
	// where it is until the last chunk has been read.
	const auto held = _d->hold_sources();

	_d->sel_ctx.input = in;
	auto result = _d->run_interruptible(statement);
	_d->sel_ctx.input = nullptr;

	if (result->HasError()) {
		std::string error = explain_error(result->GetError(), sql);
		if (was_wrapped) {
			// The user did not write the wrapper, so show what actually ran --
			// otherwise the error points at line and column numbers that do not
			// match anything they typed.
			error += "\n\nThe input was read as a predicate and run as:\n  " + statement;
		}
		throw std::runtime_error(error);
	}
	if (result->ColumnCount() != 1) {
		throw std::runtime_error(
		    "the query must project exactly one column, the row id (got " +
		    std::to_string(result->ColumnCount()) + "). Either project rowid only, "
		    "e.g. SELECT rowid FROM layers WHERE ..., or write just the condition, "
		    "e.g. port = 80.");
	}

	const size_t row_count = _d->primary().nraw->row_count();
	if (out != nullptr) {
		out->select_none();
	}

	// Chunks are consumed in result order, so a caller reading rows through the
	// callback observes whatever ORDER BY produced. Filling a selection loses
	// that order, which is inherent to a bit field.
	for (auto& chunk : result->Collection().Chunks()) {
		duckdb::UnifiedVectorFormat format;
		chunk.data[0].ToUnifiedFormat(chunk.size(), format);
		const auto* rows = duckdb::UnifiedVectorFormat::GetData<int64_t>(format);

		for (duckdb::idx_t i = 0; i < chunk.size(); ++i) {
			const auto idx = format.sel->get_index(i);
			if (not format.validity.RowIsValid(idx)) {
				continue;
			}
			const int64_t row = rows[idx];
			if (row < 0 || static_cast<size_t>(row) >= row_count) {
				throw std::runtime_error("the query returned a row id outside the source: " +
				                         std::to_string(row));
			}
			if (out != nullptr) {
				out->set_line(static_cast<PVRow>(row), true);
			}
			if (fn) {
				fn(static_cast<size_t>(row));
			}
		}
	}
}

std::string Squey::PVDuckDBQuery::quote_identifier(const std::string& name)
{
	if (not needs_quoting(name)) {
		return name;
	}
	// A double quote inside a quoted identifier is escaped by doubling it.
	std::string quoted = "\"";
	for (char c : name) {
		if (c == '"') {
			quoted += '"';
		}
		quoted += c;
	}
	quoted += '"';
	return quoted;
}

bool Squey::PVDuckDBQuery::Table::is_value_count() const
{
	if (column_types.size() != 2) {
		return false;
	}
	static const std::unordered_set<std::string> INTEGRAL = {
	    "TINYINT",  "SMALLINT",  "INTEGER",  "BIGINT",  "HUGEINT",
	    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT"};
	return INTEGRAL.count(column_types[1]) != 0;
}

Squey::PVDuckDBQuery::Table Squey::PVDuckDBQuery::run_tabular(const std::string& sql,
                                                              const PVCore::PVSelBitField* in,
                                                              size_t max_rows) const
{
	std::lock_guard<std::mutex> lock(_d->query_lock);

	const std::string statement = wrap_predicate(sql, in != nullptr ? "selection" : "layers");
	_d->require_read_only(statement);
	_d->dropped_optional.store(0, std::memory_order_relaxed);

	const auto held = _d->hold_sources();

	_d->sel_ctx.input = in;
	auto result = _d->run_interruptible(statement);
	_d->sel_ctx.input = nullptr;

	if (result->HasError()) {
		std::string error = explain_error(result->GetError(), sql);
		if (statement != sql) {
			error += "\n\nThe input was read as a predicate and run as:\n  " + statement;
		}
		throw std::runtime_error(error);
	}

	Table table;
	for (const auto& name : result->names) {
		table.column_names.emplace_back(name);
	}
	for (const auto& type : result->types) {
		table.column_types.emplace_back(type.ToString());
	}

	for (auto& chunk : result->Collection().Chunks()) {
		for (duckdb::idx_t i = 0; i < chunk.size(); ++i) {
			if (table.rows.size() >= max_rows) {
				table.truncated = true;
				return table;
			}
			std::vector<std::string> row;
			row.reserve(chunk.ColumnCount());
			for (duckdb::idx_t col = 0; col < chunk.ColumnCount(); ++col) {
				const duckdb::Value value = chunk.GetValue(col, i);
				// A NULL renders as an empty cell rather than the string
				// "NULL", which would be indistinguishable from the value.
				row.emplace_back(value.IsNull() ? std::string() : value.ToString());
			}
			table.rows.emplace_back(std::move(row));
		}
	}
	return table;
}

bool Squey::PVDuckDBQuery::yields_selection(const std::string& sql) const
{
	std::lock_guard<std::mutex> lock(_d->query_lock);

	const std::string statement = wrap_predicate(sql, "layers");
	try {
		_d->require_read_only(statement);
	} catch (const std::runtime_error&) {
		// A statement that will be refused yields no selection; saying so here
		// rather than throwing lets the caller reach the error where it belongs,
		// which is the run.
		return false;
	}

	// Prepare rather than run: the shape of the result is known from the
	// statement alone, and a query meant for display should not be executed
	// twice just to be classified.
	auto prepared = _d->con.Prepare(statement);
	if (prepared->HasError()) {
		return false;
	}
	const auto& names = prepared->GetNames();
	const auto& types = prepared->GetTypes();
	if (names.size() != 1 || names[0] != "rowid") {
		return false;
	}
	return types[0].IsIntegral();
}

std::vector<std::string> Squey::PVDuckDBQuery::column_types() const
{
	const auto held = _d->hold_sources();
	auto result = _d->con.Query("SELECT * FROM layers LIMIT 0");
	if (result->HasError()) {
		throw std::runtime_error(result->GetError());
	}
	std::vector<std::string> types;
	types.reserve(result->types.size());
	for (const auto& type : result->types) {
		types.emplace_back(type.ToString());
	}
	return types;
}

std::vector<std::string> Squey::PVDuckDBQuery::layer_names() const
{
	const Source& primary = _d->primary();
	return primary.scopes.layer_names ? primary.scopes.layer_names() : std::vector<std::string>();
}

std::vector<Squey::PVDuckDBQuery::SourceInfo> Squey::PVDuckDBQuery::sources() const
{
	const auto held = _d->hold_sources();

	std::vector<SourceInfo> described;
	described.reserve(_d->sources.size());
	for (size_t i = 0; i < _d->sources.size(); ++i) {
		const Source& src = _d->sources[i];

		SourceInfo info;
		info.name = src.name;
		info.position = src.position;
		info.current = (i == 0);
		// The same test create_source_schemas() applied, so that a completer
		// offers the short form exactly where it resolves.
		info.has_schema = not src.name.empty() && _d->unique_source_name(src.name) &&
		                  not impl::reserved_schema(src.name);

		// Bound rather than read off the nraw: what a query sees is the schema
		// the scan emits, rowid included, and its types are DuckDB's names for
		// them rather than pvcop's.
		const std::string from =
		    i == 0 ? std::string("layers")
		           : "layers(source := " + quote_literal(src.name) +
		                 ", source_position := " + std::to_string(src.position) + ")";
		auto result = _d->con.Query("SELECT * FROM " + from + " LIMIT 0");
		if (result->HasError()) {
			throw std::runtime_error(result->GetError());
		}
		for (const auto& name : result->names) {
			info.column_names.emplace_back(name);
		}
		for (const auto& type : result->types) {
			info.column_types.emplace_back(type.ToString());
		}
		described.emplace_back(std::move(info));
	}
	return described;
}

void Squey::PVDuckDBQuery::interrupt() const
{
	if (_d->running.load(std::memory_order_acquire)) {
		_d->con.Interrupt();
	}
}

size_t Squey::PVDuckDBQuery::dropped_optional_filters() const
{
	return _d->dropped_optional.load(std::memory_order_relaxed);
}

std::vector<std::string> Squey::PVDuckDBQuery::column_axis_types() const
{
	const PVRush::PVNraw& nraw = *_d->primary().nraw;
	const auto held = nraw.lock_structure();

	std::vector<std::string> types;
	types.reserve(size_t(nraw.column_count()) + 1);
	types.emplace_back(); // rowid is not an axis of the source
	for (PVCol col(0); col < nraw.column_count(); ++col) {
		types.emplace_back(nraw.column(col).type());
	}
	return types;
}

std::vector<std::string> Squey::PVDuckDBQuery::column_names() const
{
	const auto held = _d->hold_sources();
	auto result = _d->con.Query("SELECT * FROM layers LIMIT 0");
	if (result->HasError()) {
		throw std::runtime_error(result->GetError());
	}
	return result->names;
}
