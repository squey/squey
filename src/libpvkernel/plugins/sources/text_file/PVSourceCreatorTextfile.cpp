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

#include "PVSourceCreatorTextfile.h"

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVInputFile.h>
#include <assert.h>
#include <qbytearray.h>
#include <qsettings.h>
#include <qvariant.h>
#include <memory>

#include "PVTextFileSource.h"
#include "pvkernel/core/PVLogger.h"
#include "pvkernel/rush/PVInput_types.h"

PVRush::PVSourceCreatorTextfile::source_p
PVRush::PVSourceCreatorTextfile::create_source_from_input(PVInputDescription_p input) const
{
	QSettings& pvconfig = PVCore::PVConfig::get().config();

	PVLOG_DEBUG("(text_file plugin) create source for %s\n", qPrintable(input->human_name()));
	auto* file = reinterpret_cast<PVFileDescription*>(input.get());
	PVRush::PVInput_p ifile(new PVRush::PVInputFile(file->path().toUtf8()));
	// FIXME: chunk size must be computed somewhere once and for all !
	int size_chunk = pvconfig.value("pvkernel/max_size_chunk").toInt();
	if (size_chunk <= 0) {
		size_chunk = 4096 * 100; // Aligned on a page boundary (4ko)
	}
	source_p src = source_p(new PVTextFileSource(ifile, file, size_chunk));

	return src;
}

QString PVRush::PVSourceCreatorTextfile::supported_type() const
{
	return {"file"};
}

QString PVRush::PVSourceCreatorTextfile::name() const
{
	return {"text"};
}
