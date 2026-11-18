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

// The settings of the plugins: those the input plugins declare, and those
// generated from the arguments of the layer filters. What the dialog of a layer
// filter starts with in a new view is the defaults of the plugin, and in place
// of some, the defaults the user chose in the settings. The defaults of the
// plugin stay what a search is described against.

#include "common.h"

#include <squey/PVAnalysisHistory.h>
#include <squey/PVLayerFilter.h>
#include <squey/PVSqueySettings.h>
#include <squey/PVView.h>

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVSettings.h>
#include <pvkernel/core/squey_assert.h>

#include <QDir>
#include <QFile>

#include <string>

constexpr char FILENAME[] = TEST_FOLDER "/picviz/errors_search.csv";
constexpr char FORMAT[] = TEST_FOLDER "/picviz/errors_search.csv.format";

static std::string selected(PVCore::PVArgumentList const& args, const char* key)
{
	return args.at(key).value<PVCore::PVEnumType>().get_sel().toStdString();
}

int main(int argc, char** argv)
{
	// SQUEY_CONFIG_DIR gives the test a configuration of its own, which needs
	// the config.ini that loading the plugins reads: a copy of pvconfig.ini, the
	// argument, as the packaged testsuite may not ship one where PVConfig looks.
	PV_ASSERT_VALID(argc == 2);
	PV_ASSERT_VALID(QDir().mkpath(PVCore::PVConfig::user_dir()));
	if (not QFile::exists(PVCore::PVConfig::user_path())) {
		PV_ASSERT_VALID(QFile::copy(argv[1], PVCore::PVConfig::user_path()));
	}
	const QString user_file = PVCore::PVConfig::user_dir() + PVCore::PVSettings::USER_FILENAME;
	QFile::remove(user_file);

	// Loading the plugins declares their settings.
	pvtest::TestEnv env;
	PVCore::PVSettings& settings = PVCore::PVSettings::get();
	PV_ASSERT_VALID(
	    settings.descriptor("plugins.input.elasticsearch.verify_certificates").has_value());

	const auto match_case = settings.descriptor("plugins.layer_filters.search-multiple.case");
	PV_ASSERT_VALID(match_case.has_value());
	PV_ASSERT_VALID(match_case->type == PVCore::PVSettingType::Choice);
	PV_VALID(match_case->default_value.toString().toStdString(),
	         std::string("Does not match case"));
	PV_VALID(settings.section_title("plugins.layer_filters.search-multiple").toStdString(),
	         std::string("Text Search › Multiple values"));
	PV_ASSERT_VALID(settings.descriptor("plugins.layer_filters.errors-search.type").has_value());
	PV_ASSERT_VALID(
	    settings.descriptor("plugins.layer_filters.frequency-gradient.scale").has_value());
	// An axis, or the text of a search, belongs to one use of the filter.
	PV_ASSERT_VALID(
	    not settings.descriptor("plugins.layer_filters.search-multiple.axis").has_value());
	PV_ASSERT_VALID(
	    not settings.descriptor("plugins.layer_filters.search-multiple.exps").has_value());
	PV_ASSERT_VALID(
	    not settings.descriptor("plugins.layer_filters.errors-search.axis").has_value());

	PV_ASSERT_VALID(settings.set_value("plugins.layer_filters.search-multiple.case", "Match case"));
	PV_ASSERT_VALID(settings.set_value("plugins.layer_filters.errors-search.type", "Empty values"));

	env.add_source(FILENAME, FORMAT);
	env.compute_mappings();
	env.compute_scalings();
	env.compute_views();
	Squey::PVView* view = env.root.current_view();

	PV_VALID(selected(view->get_last_args_filter("search-multiple"), "case"),
	         std::string("Match case"));
	PV_VALID(selected(view->get_last_args_filter("search-multiple"), "interpret"),
	         std::string("Plain text"));
	PV_VALID(selected(view->get_last_args_filter("errors-search"), "type"),
	         std::string("Empty values"));

	Squey::PVLayerFilter::p_type search_multiple =
	    LIB_CLASS(Squey::PVLayerFilter)::get().get_class_by_name("search-multiple");
	PV_VALID(selected(search_multiple->get_default_args_for_view(*view), "case"),
	         std::string("Does not match case"));
	PV_VALID(selected(search_multiple->get_preferred_args_for_view(*view), "case"),
	         std::string("Match case"));

	// Not a plugin, but the settings of libsquey are read the same way.
	PV_ASSERT_VALID(Squey::settings::undo_steps.set_value(5));
	PV_VALID(Squey::PVAnalysisHistory(env.root).max_steps(), size_t(5));

	QFile::remove(user_file);
	return 0;
}
