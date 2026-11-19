//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/rush/PVRowFilter.h>

#include <duckdb.hpp>
#include <duckdb/common/error_data.hpp>
#include <duckdb/function/table/arrow.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

/**
 * Hands DuckDB the stream @p stream points to, which it releases once it has scanned it.
 */
duckdb::unique_ptr<duckdb::ArrowArrayStreamWrapper> produce(uintptr_t stream,
                                                            duckdb::ArrowStreamParameters&)
{
	auto& rows = *reinterpret_cast<ArrowArrayStream*>(stream);
	auto wrapper = duckdb::make_uniq<duckdb::ArrowArrayStreamWrapper>();
	wrapper->arrow_array_stream = rows;
	rows.release = nullptr;
	return wrapper;
}

void get_schema(ArrowArrayStream* stream, ArrowSchema& schema)
{
	if (stream->get_schema(stream, &schema) != 0) {
		const char* error = stream->get_last_error(stream);
		throw std::runtime_error(error != nullptr ? error
		                                          : "Unable to read the columns of the rows");
	}
}

void release(ArrowArrayStream& rows)
{
	if (rows.release != nullptr) {
		rows.release(&rows);
	}
}

std::string quoted(const std::string& name)
{
	std::string quoted = "\"";
	for (char c : name) {
		quoted += c == '"' ? "\"\"" : std::string(1, c);
	}
	return quoted + "\"";
}

void check(const duckdb::unique_ptr<duckdb::MaterializedQueryResult>& result)
{
	if (result->HasError()) {
		throw std::runtime_error(result->GetError());
	}
}

class DuckDBRowFilter : public PVRush::PVRowFilter
{
  public:
	explicit DuckDBRowFilter(std::string condition)
	    : _condition(std::move(condition)), _database(nullptr), _connection(_database)
	{
		// The rows come by blocks, one after another: a thread is enough for each of them.
		check(_connection.Query("SET threads = 1"));
		// Every instant in UTC, as in the SQL console, which explains why.
		check(_connection.Query("SET TimeZone = 'UTC'"));
		// The condition reads the rows, and nothing else: no file, no network, no extension.
		check(_connection.Query("SET enable_external_access = false"));
		check(_connection.Query("SET autoinstall_known_extensions = false"));
		check(_connection.Query("SET autoload_known_extensions = false"));
		check(_connection.Query("SET allow_unsigned_extensions = false"));
		check(_connection.Query("SET lock_configuration = true"));
	}

	std::vector<int64_t> matching_rows(ArrowArrayStream& rows,
	                                   const std::string& index_column) override
	{
		std::vector<int64_t> matching;
		try {
			const duckdb::vector<duckdb::Value> parameters{
			    duckdb::Value::POINTER(uintptr_t(&rows)),
			    duckdb::Value::POINTER(uintptr_t(&produce)),
			    duckdb::Value::POINTER(uintptr_t(&get_schema))};
			// The scan that pushes nothing down to the stream, so that DuckDB applies the
			// condition itself. Filter() parses a single expression, which keeps the condition
			// from closing the clause it is put in.
			const duckdb::unique_ptr<duckdb::QueryResult> result =
			    _connection.TableFunction("arrow_scan_dumb", parameters)
			        ->Filter(_condition)
			        ->Project(quoted(index_column))
			        ->Execute();
			if (result->HasError()) {
				throw std::runtime_error(result->GetError());
			}
			while (const duckdb::unique_ptr<duckdb::DataChunk> chunk = result->Fetch()) {
				for (duckdb::idx_t row = 0; row < chunk->size(); row++) {
					matching.push_back(chunk->GetValue(0, row).GetValue<int64_t>());
				}
			}
		} catch (const std::exception& e) {
			release(rows);
			throw std::runtime_error(duckdb::ErrorData(e).RawMessage());
		}
		release(rows);
		return matching;
	}

  private:
	std::string _condition;
	duckdb::DuckDB _database;
	duckdb::Connection _connection;
};

/**
 * Registers the engine as the library is loaded, before anything reads a file.
 */
const struct Registration {
	Registration()
	{
		PVRush::PVRowFilter::set_factory([](const std::string& condition) {
			return std::make_unique<DuckDBRowFilter>(condition);
		});
	}
} registration;

} // namespace
