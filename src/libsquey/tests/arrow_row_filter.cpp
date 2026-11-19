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

// The rows of Arrow tables that a SQL condition holds for, as the imports of files filter them.
// The condition reads the columns by their names, quoted when they hold a dot, with their types:
// times, nulls, the dictionaries of the names of the files, and half floats, which the engine
// reads as floats. A column of the name the index of the rows would take keeps its own values.
// What the engine refuses comes back as its message.

#include "common.h"

#include "../../libpvkernel/plugins/common/arrow/PVArrowRowFilter.h"

#include <pvkernel/core/squey_assert.h>

#include <arrow/api.h>
#include <arrow/json/from_string.h>
#include <arrow/util/float16.h>

#include <memory>
#include <string>
#include <vector>

namespace
{

std::shared_ptr<arrow::Array> array_of(const std::shared_ptr<arrow::DataType>& type,
                                       const std::string& values)
{
	return arrow::json::ArrayFromJSONString(type, values).ValueOrDie();
}

std::shared_ptr<arrow::Table> events()
{
	const std::shared_ptr<arrow::DataType> file = arrow::dictionary(arrow::int32(), arrow::utf8());
	const arrow::FieldVector fields{arrow::field("id", arrow::int64()),
	                                arrow::field("event_type", arrow::utf8()),
	                                arrow::field("dns.rrname", arrow::utf8()),
	                                arrow::field("when", arrow::timestamp(arrow::TimeUnit::MICRO)),
	                                arrow::field("ratio", arrow::float16()),
	                                arrow::field("filename", file),
	                                arrow::field("squey_row", arrow::int64())};
	arrow::HalfFloatBuilder ratios;
	for (float ratio : {0.25F, 0.75F, 0.5F, 1.F, 0.F, 0.875F, 0.125F}) {
		PV_ASSERT_VALID(ratios.Append(arrow::util::Float16(ratio).bits()).ok());
	}
	const std::shared_ptr<arrow::Array> files =
	    arrow::DictionaryArray::FromArrays(file, array_of(arrow::int32(), "[0,1,0,1,0,1,0]"),
	                                       array_of(arrow::utf8(), R"(["a.json","b.json"])"))
	        .ValueOrDie();
	return arrow::Table::Make(
	    arrow::schema(fields),
	    {array_of(arrow::int64(), "[0,10,20,30,40,50,null]"),
	     array_of(arrow::utf8(), R"(["flow","alert","flow","alert","flow","alert","dns"])"),
	     array_of(arrow::utf8(), R"(["x.example.com",null,"y.org","x.example.com",null,null,"z"])"),
	     array_of(arrow::timestamp(arrow::TimeUnit::MICRO),
	              R"(["2024-01-02 03:04:05","2024-01-02 03:04:06","2024-01-02 03:04:07",)"
	              R"("2024-01-02 03:04:08","2024-01-02 03:04:09","2024-01-02 03:04:10",null])"),
	     ratios.Finish().ValueOrDie(), files, array_of(arrow::int64(), "[6,5,4,3,2,1,0]")});
}

/** The ids of the rows of @p table that @p condition holds for, in their order. */
std::string ids_of(const std::shared_ptr<arrow::Table>& table, const std::string& condition)
{
	const std::shared_ptr<arrow::Table> rows = PVRush::PVArrowRowFilter(condition).filter(table);
	PV_VALID(rows->num_columns(), table->num_columns(), "condition", condition);
	std::string ids;
	const std::shared_ptr<arrow::Table> combined = rows->CombineChunks().ValueOrDie();
	for (int64_t row = 0; row < combined->num_rows(); row++) {
		const auto& column = static_cast<const arrow::Int64Array&>(*combined->column(0)->chunk(0));
		ids += (ids.empty() ? "" : ",") +
		       (column.IsNull(row) ? std::string("null") : std::to_string(column.Value(row)));
	}
	return ids;
}

} // namespace

int main()
{
	pvtest::TestEnv env;

	const std::shared_ptr<arrow::Table> table = events();
	PV_VALID(ids_of(table, "event_type = 'alert'"), std::string("10,30,50"));
	PV_VALID(ids_of(table, R"(id >= 20 AND "dns.rrname" LIKE '%.example.com')"), std::string("30"));
	PV_VALID(ids_of(table, R"("when" > TIMESTAMP '2024-01-02 03:04:07')"), std::string("30,40,50"));
	PV_VALID(ids_of(table, "filename = 'b.json'"), std::string("10,30,50"));
	PV_VALID(ids_of(table, "id IS NULL"), std::string("null"));
	PV_VALID(ids_of(table, "ratio > 0.5"), std::string("10,30,50"));
	PV_VALID(ids_of(table, "squey_row = 4"), std::string("20"));
	PV_VALID(ids_of(table, "false"), std::string());

	const std::shared_ptr<arrow::Schema> schema = table->schema();
	PV_VALID(PVRush::PVArrowRowFilter::error_of("event_type = 'alert'", schema), std::string());
	const std::string unknown = PVRush::PVArrowRowFilter::error_of("nope = 1", schema);
	PV_ASSERT_VALID(unknown.find("nope") != std::string::npos, "error", unknown);
	PV_ASSERT_VALID(not PVRush::PVArrowRowFilter::error_of("id = ", schema).empty());
	// A single expression, which cannot close the clause it is put in.
	PV_ASSERT_VALID(not PVRush::PVArrowRowFilter::error_of("1 = 1) OR (1 = 1", schema).empty());
	// A name with a dot is one column, between double quotes, rather than a field of another.
	PV_ASSERT_VALID(not PVRush::PVArrowRowFilter::error_of("dns.rrname = 'z'", schema).empty());

	return 0;
}
