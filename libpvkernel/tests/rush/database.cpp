//
// MIT License
//
// © ESI Group, 2015
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

#include "../../plugins/common/database/PVDBInfos.h"
#include "../../plugins/common/database/PVDBQuery.h"

#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/rush/PVCSVExporter.h>
#include "common.h"
#include "helpers.h"

#include <fstream>
#include <memory>

#ifndef INSPECTOR_BENCH
static constexpr const char* ref_file = TEST_FOLDER "/pvkernel/rush/sources/database.out";
constexpr static size_t nb_dup = 1;
#else
constexpr static size_t nb_dup = 200;
#endif

int main()
{
	pvtest::init_ctxt();

	// Setup database infos
	PVRush::PVDBInfos infos;
	infos.set_type("QMYSQL");
	infos.set_host("connectors.srv.picviz");
	infos.set_port(3306);
	infos.set_dbname("logs");
	infos.set_username("inendi");
	infos.set_password("changeme");
	std::string query_str = R"###(
        SELECT * from proxy_sample WHERE http_method = 'POST';
	)###";

	// setup import structures
	auto* qr =
	    new PVRush::PVDBQuery(std::make_shared<PVRush::PVDBServ>(infos), query_str.c_str());
	PV_ASSERT_VALID(qr->connect_serv());
	PVRush::PVSourceCreator_p sc =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("database");
	QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
	for (size_t dup = 0; dup < nb_dup; dup++) {
		list_inputs << PVRush::PVInputDescription_p(qr);
	}
	PVRush::PVNraw nraw;
	PVRush::PVNrawOutput output(nraw);
	PVRush::PVFormat format(qr->get_format_from_db_schema().documentElement());
	PVRush::PVExtractor extractor(format, output, sc, list_inputs);

	// Import data
	auto start = std::chrono::system_clock::now();
	PVRush::PVControllerJob_p job =
	    extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT);
	job->wait_end();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

	// Export selected lines
	PVCore::PVSelBitField sel(nraw.row_count());
	sel.select_all();
	std::string output_file = pvtest::get_tmp_filename();
	PVRush::PVCSVExporter::export_func_f export_func =
	    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
	        const std::string& quote) { return nraw.export_line(row, cols, sep, quote); };
	PVRush::PVCSVExporter exp(format.get_axes_comb(), nraw.row_count(), export_func);
	exp.export_rows(output_file, sel);

#ifndef INSPECTOR_BENCH
	// Check output is the same as the reference
	std::cout << std::endl << output_file << " - " << ref_file << std::endl;
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file, ref_file));
#endif

	std::remove(output_file.c_str());

	return 0;
}
