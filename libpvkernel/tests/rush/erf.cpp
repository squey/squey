/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#include <pvkernel/core/inendi_assert.h>

#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVCSVExporter.h>

#include "common.h"

#include "../../plugins/common/erf/PVERFAPI.h"
#include "../../plugins/common/erf/PVERFDescription.h"

int main(int argc, char** argv)
{
	if (argc <= 3) {
		std::cerr
		    << "Usage: " << argv[0]
		    << " <erf_file> <json_selected_nodes> <output_ref1>  <output_ref2>  <output_ref...>"
		    << std::endl;
		return 1;
	}

	const std::string& erf_file = argv[1];
	const std::string& json_selected_nodes = argv[2];
	std::vector<std::string> output_refs;
	for (int i = 3; i < argc; i++) {
		output_refs.emplace_back(argv[i]);
	}

	pvtest::init_ctxt();

	/*
	 *  Set Up the API
	 */
	PVRush::PVERFAPI erf(erf_file);

	rapidjson::Document selected_nodes;
	selected_nodes.Parse<0>(json_selected_nodes.c_str());

	/**************************************************************************
	 * Import data
	 *************************************************************************/
	PVRush::PVSourceCreator_p sc =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("erf");

	size_t ref_index = 0;
	for (auto& [selected_nodes, source_name, format] : erf.get_sources_info(selected_nodes)) {

		QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
		PVRush::PVERFDescription* erf_desc = new PVRush::PVERFDescription(
		    QString::fromStdString(erf_file), source_name.c_str(), std::move(selected_nodes));
		list_inputs << PVRush::PVInputDescription_p(erf_desc);

		PVRush::PVNraw nraw;
		PVRush::PVNrawOutput output(nraw);
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
		const std::string& output_file = pvtest::get_tmp_filename();
		PVRush::PVCSVExporter::export_func_f export_func =
		    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
		        const std::string& quote) { return nraw.export_line(row, cols, sep, quote); };
		PVRush::PVCSVExporter exp(format.get_axes_comb(), nraw.row_count(), export_func);

		exp.export_rows(output_file, sel);

#ifndef INSPECTOR_BENCH
		// Check output is the same as the reference
		std::cout << std::endl << output_file << " - " << output_refs[ref_index] << std::endl;
		PV_ASSERT_VALID(
		    PVRush::PVUtils::files_have_same_content(output_file, output_refs[ref_index]));
#endif

		std::remove(output_file.c_str());

		ref_index++;
	}

	return 0;
}
