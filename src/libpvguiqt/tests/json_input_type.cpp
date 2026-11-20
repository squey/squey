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

// The import of files of one JSON object per line, from parameters: the type of input finds the
// columns of the files and makes their format, which it makes again for a source reopened from the
// recent items.

#include "common.h"
#include "test-env.h"

#include <squey/PVRoot.h>
#include <squey/PVScene.h>
#include <squey/PVSource.h>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVTaskProgress.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceCreator.h>

#include <QApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include <string>

namespace
{

void write(QString const& path, QByteArray const& content)
{
	QFile file(path);
	PV_ASSERT_VALID(file.open(QIODevice::WriteOnly), "file", path.toStdString());
	file.write(content);
}

/** The names and types of the axes of @p format. */
std::string axes_of(PVRush::PVFormat const& format)
{
	std::string axes;
	for (PVRush::PVAxisFormat const& axis : format.get_axes()) {
		axes += (axes.empty() ? "" : ",") + axis.get_name().toStdString() + ":" +
		        axis.get_type().toStdString();
	}
	return axes;
}

} // namespace

int main(int argc, char** argv)
{
	init_env();
	QApplication app(argc, argv); // argv carries "-platform offscreen"

	QTemporaryDir directory;
	PV_ASSERT_VALID(directory.isValid());
	const QString first = directory.filePath("first.jsonl");
	const QString second = directory.filePath("second.ndjson");
	write(first, R"({"x": 1, "y": {"z": "a"}, "t": "2024-01-02T03:04:05Z"})"
	             "\n");
	write(second, R"({"x": 2})"
	              "\n"
	              R"({"x": 3, "y": {"z": "b"}})"
	              "\n");

	const PVRush::PVInputType_p type =
	    LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("json");
	PV_ASSERT_VALID(type->has_generated_format());

	PVRush::PVInputType::list_inputs inputs;
	PVRush::PVFormat format;
	const QString params = QJsonDocument(QJsonObject{{"paths", QJsonArray{first, second}}})
	                           .toJson(QJsonDocument::Compact);
	PVCore::PVTaskProgress progress;
	PV_ASSERT_VALID(type->create_source_description_params(params, inputs, format, progress));
	const std::string axes = "filename:string,x:number_int64,y.z:string,t:time";
	PV_VALID(axes_of(format), axes);

	Squey::PVRoot root;
	Squey::PVSource& source = root.emplace_add_child("json").emplace_add_child(
	    inputs, LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("json"), format);
	source.wait_extract_end(source.extract(0));
	const PVRush::PVNraw& nraw = source.get_rushnraw();
	PV_VALID(size_t(nraw.row_count()), size_t(3));
	PV_VALID(nraw.at_string(0, PVCol(0)), std::string("first.jsonl"));
	PV_VALID(nraw.at_string(2, PVCol(0)), std::string("second.ndjson"));
	PV_VALID(nraw.at_string(2, PVCol(2)), std::string("b"));

	// As the recent items keep it.
	const PVRush::PVInputDescription_p reopened = type->load_input_from_string(
	    {first.toStdString(), second.toStdString()}, /*multi_inputs=*/false);
	PV_VALID(axes_of(type->get_format_from_inputs(PVRush::PVInputType::list_inputs{reopened})),
	         axes);

	return 0;
}
