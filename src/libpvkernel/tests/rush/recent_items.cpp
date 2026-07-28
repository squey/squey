//
// MIT License
//
// © Squey, 2024
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

// Round-trip test for the recent sources:
//   add_source() -> get_list() -> get_string_from_entry() -> PVSourceDescription(ss)
//   -> get_format_from_inputs()
// It covers a CSV source (format stored on disk, the classic case) and a Parquet
// source (format generated from the file schema, not stored on disk), including the
// multi-file concatenation case.

#include "../../plugins/common/parquet/PVParquetAPI.h"
#include "../../plugins/common/parquet/PVParquetFileDescription.h"

#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVSourceDescription.h>
#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/PVRecentItemsManager.h>
#include <pvkernel/core/PVSerializedSource.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <QDir>
#include <QFile>
#include <QStringList>

#include <iostream>
#include <string>
#include <vector>

// Build a Parquet source (input description + generated format) as the import path does.
static PVRush::PVFormat build_parquet_source(const QStringList& paths,
                                             PVRush::PVInputType::list_inputs& inputs)
{
	auto* input_desc = new PVRush::PVParquetFileDescription(paths);
	inputs << PVRush::PVInputDescription_p(input_desc);
	PVRush::PVParquetAPI api(input_desc);
	return PVRush::PVFormat(api.get_format().documentElement());
}

UNICODE_MAIN()
{
	if (argc <= 5) {
		std::cerr << "Usage: <csv> <csv_format> <parquet1> <parquet2> <pvconfig_template>"
		          << std::endl;
		return 1;
	}

#ifdef _WIN32
	std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
	const std::string csv_file = conv.to_bytes(argv[1]);
	const std::string csv_format = conv.to_bytes(argv[2]);
	const std::string parquet_file1 = conv.to_bytes(argv[3]);
	const std::string parquet_file2 = conv.to_bytes(argv[4]);
	const std::string pvconfig_template = conv.to_bytes(argv[5]);
#else
	const std::string csv_file = argv[1];
	const std::string csv_format = argv[2];
	const std::string parquet_file1 = argv[3];
	const std::string parquet_file2 = argv[4];
	const std::string pvconfig_template = argv[5];
#endif

	// This test writes to recents.ini: it must run in an isolated config dir.
	// PVConfig's config dir is frozen before main(), so SQUEY_CONFIG_DIR has to be
	// provided through the environment (set by CTest).
	if (qEnvironmentVariableIsEmpty("SQUEY_CONFIG_DIR")) {
		std::cerr << "SQUEY_CONFIG_DIR must be set to isolate this test" << std::endl;
		return 1;
	}

	// Without a user config file, PVConfig tries to copy the (absent) system one
	// and throws: provide one in the isolated dir before anything reads PVConfig.
	// Report a failure here rather than letting it surface as PVConfig's opaque
	// "No config file found" three steps later.
	if (not QDir().mkpath(PVCore::PVConfig::user_dir())) {
		std::cerr << "Can't create config dir '" << PVCore::PVConfig::user_dir().toStdString()
		          << "'" << std::endl;
		return 1;
	}
	if (not QFile::exists(PVCore::PVConfig::user_path()) and
	    not QFile::copy(QString::fromStdString(pvconfig_template),
	                    PVCore::PVConfig::user_path())) {
		std::cerr << "Can't copy '" << pvconfig_template << "' to '"
		          << PVCore::PVConfig::user_path().toStdString() << "'" << std::endl;
		return 1;
	}

	// Start from a clean recent items file, before the manager (a singleton) reads
	// it for the first time.
	const QString recents_file = PVCore::PVConfig::user_dir() + "recents.ini";
	QFile::remove(recents_file);

	pvtest::init_ctxt();

	auto& recents = PVCore::PVRecentItemsManager::get();

	// -------------------------------------------------------------------------
	// CSV source (format stored on disk)
	// -------------------------------------------------------------------------
	{
		PVRush::PVSourceCreator_p sc =
		    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("text_file");
		PV_ASSERT_VALID(sc != nullptr);

		// A local file source doesn't require credentials.
		PV_VALID(sc->need_credential(), false);

		PVRush::PVFormat format("format", QString::fromStdString(csv_format));
		const qsizetype nb_axes = format.get_axes().size();
		PV_ASSERT_VALID(nb_axes >= 2);

		PVRush::PVInputType::list_inputs inputs;
		inputs << PVRush::PVInputDescription_p(
		    new PVRush::PVFileDescription(QString::fromStdString(csv_file)));

		recents.add_source(sc, inputs, format);

		std::vector<PVCore::PVSerializedSource> sources =
		    recents.get_list<PVCore::Category::SOURCES>();
		PV_VALID(sources.size(), size_t(1));

		const PVCore::PVSerializedSource& ss = sources.front();
		PV_VALID(ss.sc_name, std::string("text_file"));
		PV_ASSERT_VALID(not ss.format_path.empty()); // format stored on disk
		PV_VALID(ss.input_desc.size(), size_t(1));

		// The label shows the format name between brackets.
		auto [label, filenames] = recents.get_string_from_entry(ss);
		PV_ASSERT_VALID(label.contains("[format]"));
		PV_VALID(filenames.size(), qsizetype(1));

		// Rebuild: the format is loaded back from disk and the input is a plain
		// PVFileDescription (not a Parquet one).
		PVRush::PVSourceDescription sd(ss);
		PV_VALID(sd.get_format().get_axes().size(), nb_axes);
		PV_ASSERT_VALID(not sd.get_format().get_full_path().isEmpty());
		PV_VALID(sd.get_inputs().size(), qsizetype(1));
		PV_ASSERT_VALID(dynamic_cast<PVRush::PVFileDescription*>(sd.get_inputs()[0].get()) !=
		                nullptr);
		PV_ASSERT_VALID(dynamic_cast<PVRush::PVParquetFileDescription*>(
		                    sd.get_inputs()[0].get()) == nullptr);

		// clear_missing_files() (run at startup) keeps the valid source.
		recents.clear_missing_files();
		PV_VALID(recents.get_list<PVCore::Category::SOURCES>().size(), size_t(1));
	}

	// -------------------------------------------------------------------------
	// Parquet single-file source (format generated from the schema)
	// -------------------------------------------------------------------------
	{
		PVRush::PVSourceCreator_p sc =
		    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("parquet");
		PV_ASSERT_VALID(sc != nullptr);

		// A local file source doesn't require credentials.
		PV_VALID(sc->need_credential(), false);

		PVRush::PVInputType_p input_type = sc->supported_type_lib();

		recents.clear(PVCore::Category::SOURCES);

		QStringList paths;
		paths << QString::fromStdString(parquet_file1);
		PVRush::PVInputType::list_inputs inputs;
		PVRush::PVFormat format = build_parquet_source(paths, inputs);
		const qsizetype nb_axes = format.get_axes().size();
		PV_ASSERT_VALID(nb_axes >= 2);

		recents.add_source(sc, inputs, format);

		std::vector<PVCore::PVSerializedSource> sources =
		    recents.get_list<PVCore::Category::SOURCES>();
		PV_VALID(sources.size(), size_t(1));

		const PVCore::PVSerializedSource& ss = sources.front();
		PV_VALID(ss.sc_name, std::string("parquet"));
		PV_ASSERT_VALID(ss.format_path.empty()); // generated format => no file on disk
		PV_VALID(ss.input_desc.size(), size_t(1));
		PV_VALID(ss.input_desc.front().size(), size_t(1));
		PV_VALID(ss.input_desc.front().front(), parquet_file1);

		// The label must not contain empty "[]" brackets.
		auto [label, filenames] = recents.get_string_from_entry(ss);
		PV_ASSERT_VALID(not label.contains("[]"));
		PV_VALID(filenames.size(), qsizetype(1));

		// Rebuild must NOT throw despite the empty format (the crash fix) and must
		// rebuild a PVParquetFileDescription.
		PVRush::PVSourceDescription sd(ss);
		PV_VALID(sd.get_inputs().size(), qsizetype(1));
		auto* rebuilt =
		    dynamic_cast<PVRush::PVParquetFileDescription*>(sd.get_inputs()[0].get());
		PV_ASSERT_VALID(rebuilt != nullptr);
		PV_VALID(rebuilt->paths().size(), qsizetype(1));
		PV_VALID(rebuilt->paths().front().toStdString(), parquet_file1);
		PV_ASSERT_VALID(sd.get_format().get_full_path().isEmpty());

		// The format can be regenerated from the inputs and matches the import one.
		PVRush::PVFormat regenerated = input_type->get_format_from_inputs(sd.get_inputs());
		PV_VALID(regenerated.get_axes().size(), nb_axes);

		// clear_missing_files() (run at startup) must keep the valid source and
		// must not throw.
		recents.clear_missing_files();
		PV_VALID(recents.get_list<PVCore::Category::SOURCES>().size(), size_t(1));
	}

	// -------------------------------------------------------------------------
	// Parquet multi-file source (concatenation: a single input holding every path)
	// -------------------------------------------------------------------------
	{
		PVRush::PVSourceCreator_p sc =
		    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("parquet");
		recents.clear(PVCore::Category::SOURCES);

		QStringList paths;
		paths << QString::fromStdString(parquet_file1)
		      << QString::fromStdString(parquet_file2);
		PVRush::PVInputType::list_inputs inputs;
		PVRush::PVFormat format = build_parquet_source(paths, inputs);

		recents.add_source(sc, inputs, format);

		std::vector<PVCore::PVSerializedSource> sources =
		    recents.get_list<PVCore::Category::SOURCES>();
		PV_VALID(sources.size(), size_t(1));

		// A single input description holding the two paths.
		const PVCore::PVSerializedSource& ss = sources.front();
		PV_VALID(ss.input_desc.size(), size_t(1));
		PV_VALID(ss.input_desc.front().size(), size_t(2));

		PVRush::PVSourceDescription sd(ss);
		PV_VALID(sd.get_inputs().size(), qsizetype(1));
		auto* rebuilt =
		    dynamic_cast<PVRush::PVParquetFileDescription*>(sd.get_inputs()[0].get());
		PV_ASSERT_VALID(rebuilt != nullptr);
		PV_VALID(rebuilt->paths().size(), qsizetype(2));
		PV_VALID(rebuilt->paths()[0].toStdString(), parquet_file1);
		PV_VALID(rebuilt->paths()[1].toStdString(), parquet_file2);
	}

	// Cleanup.
	QFile::remove(recents_file);

	return 0;
}
