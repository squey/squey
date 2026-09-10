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

// What happens between a path handed over by a desktop -- or typed on the
// command line -- and the importer that reads it. Nothing else says what to do
// with such a path, and every way this can go wrong goes wrong silently, as a
// message box in front of a user rather than as a failing build.

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputType.h>

#include "common.h"

#include <QFile>
#include <QString>
#include <QTemporaryDir>

#include <string>

static PVRush::PVInputType_p importer_of(const QString& file_name)
{
	return LIB_CLASS(PVRush::PVInputType)::get().get_class_by_extension(file_name);
}

static void expect_importer(const QString& file_name, const std::string& expected)
{
	PV_VALID(importer_of(file_name)->name().toStdString(), expected,
	         "file name", file_name.toStdString());
}

static void expect_unsupported(const QString& file_name)
{
	bool refused = false;
	try {
		importer_of(file_name);
	} catch (const std::runtime_error&) {
		refused = true;
	}
	PV_ASSERT_VALID(refused, "a file no importer handles was accepted",
	                file_name.toStdString());
}

int main()
{
	// Loads the input type plugins, among the rest.
	pvtest::init_ctxt();

	expect_importer("/tmp/traffic.csv", "file");
	expect_importer("/tmp/traffic.tsv", "file");
	expect_importer("/tmp/capture.pcap", "pcap");
	expect_importer("/tmp/capture.pcapng", "pcap");
	expect_importer("/tmp/events.parquet", "parquet");

	// A compressed file belongs to the importer of what it holds, not to the
	// one of the container: its extension has two components, and only the
	// whole of it tells them apart.
	expect_importer("/tmp/traffic.csv.gz", "file");
	expect_importer("/tmp/traffic.csv.zst", "file");
	expect_importer("/tmp/traffic.tsv.bz2", "file");

	// Extensions come in whatever case the system that wrote them used.
	expect_importer("/tmp/TRAFFIC.CSV", "file");
	expect_importer("/tmp/Capture.PcapNG", "pcap");
	expect_importer("/tmp/traffic.CSV.GZ", "file");

	// Names with dots of their own are not extensions: only the components that
	// match are used, one at a time, from the longest.
	expect_importer("/tmp/report.2026.01.csv", "file");
	expect_importer("/tmp/report.2026.01.csv.gz", "file");

	// And what nothing handles is refused rather than sent to an importer that
	// would fail later, with less to say about why.
	expect_unsupported("/tmp/archive.gz");
	expect_unsupported("/tmp/notes.txt");
	expect_unsupported("/tmp/README");

	// A format named with --format is the user's answer to the question the
	// importer would otherwise go looking for one for. It has to reach the
	// import untouched: overwritten with what sits next to the file, the option
	// does nothing at all, and says nothing about it either.
	{
		// A directory of this test's own: the importer opens what it is given,
		// and a shared path is a collision waiting for the next parallel run.
		QTemporaryDir dir;
		PV_ASSERT_VALID(dir.isValid(), "temporary directory", dir.path().toStdString());
		const QString csv = dir.filePath("traffic.csv");
		QFile file(csv);
		PV_ASSERT_VALID(file.open(QIODevice::WriteOnly), "test file", csv.toStdString());
		file.write("a,b,c\n1,2,3\n");
		file.close();

		PVRush::PVInputType_p in_file =
		    LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file");
		const QString named_format = dir.filePath("a-format-of-my-own.format");
		QString format = named_format;
		PVRush::hash_formats formats;
		PVRush::PVInputType::list_inputs inputs;
		PVCore::PVArgumentList args;

		const bool accepted =
		    in_file->create_widget_with_input_files({csv}, formats, inputs, format, args, nullptr);

		PV_ASSERT_VALID(accepted, "named format", named_format.toStdString());
		PV_VALID(format.toStdString(), named_format.toStdString());
		PV_VALID(inputs.size(), qsizetype(1));
	}

	return 0;
}
