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

// That an investigation opens again when its format is not plain ASCII.
//
// The format goes into the archive as a document, and its length used to be
// counted in characters while what was written was UTF-8: each accented letter
// cut one byte off the end of the document. The archive was written without a
// complaint and could not be read back -- "Premature end of document" -- which
// is where a session with its axes named in French ended up. The axes here are
// named the way that data named them, and the file is read back into a root of
// its own, as the application does when it opens one.

#include "common.h"

#include <squey/PVRoot.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/core/squey_assert.h>

#include <pvbase/general.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

int main()
{
	// Written for the occasion: what matters is the names, not the rows.
	const std::string csv = pvtest::get_tmp_filename() + ".csv";
	const std::string format = csv + ".format";
	std::ofstream(csv) << "750,4400\n748,1300\n";
	std::ofstream(format) << R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep=",">
  <field><axis name="Température bain" type="number_uint32"/></field>
  <field><axis name="Intensité compensée" type="number_uint32"/></field>
 </splitter>
</param>
)";

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);

	const std::string path = pvtest::get_tmp_filename() + ".pvi";
	{
		PVCore::PVSerializeArchiveZip archive(QString::fromStdString(path),
		                                      PVCore::PVSerializeArchive::write,
		                                      SQUEY_ARCHIVES_VERSION, true);
		env.root.save_to_file(archive);
	}

	Squey::PVRoot opened;
	{
		PVCore::PVSerializeArchiveZip archive(QString::fromStdString(path),
		                                      PVCore::PVSerializeArchive::read,
		                                      SQUEY_ARCHIVES_VERSION, true);
		opened.load_from_archive(archive);
	}

	const auto sources = opened.get_children<Squey::PVSource>();
	PV_ASSERT_VALID(sources.size() == 1, "sources found in the file", sources.size());
	const auto& axes = sources.front()->get_format().get_axes();
	PV_ASSERT_VALID(axes.size() == 2, "axes read back", axes.size());
	PV_VALID(axes.at(0).get_name().toStdString(), std::string("Température bain"));
	PV_VALID(axes.at(1).get_name().toStdString(), std::string("Intensité compensée"));
	std::cout << "opened again with its axes named '" << axes.at(0).get_name().toStdString()
	          << "' and '" << axes.at(1).get_name().toStdString() << "'" << std::endl;

	std::remove(path.c_str());
	std::remove(format.c_str());
	std::remove(csv.c_str());
	std::cout << "investigation_accents: ok" << std::endl;
	return 0;
}
