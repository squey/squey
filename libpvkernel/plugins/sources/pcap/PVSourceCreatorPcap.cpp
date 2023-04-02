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

#include "PVSourceCreatorPcap.h"
#include "PVPcapSource.h"
#include "../../common/pcap/PVPcapDescription.h"

#include <pvkernel/core/PVConfig.h>

#include <QDir>
#include <QStringList>
#include <QFileInfo>

PVPcapsicum::PVSourceCreatorPcap::source_p
PVPcapsicum::PVSourceCreatorPcap::create_source_from_input(PVRush::PVInputDescription_p input) const
{
	QSettings& pvconfig = PVCore::PVConfig::get().config();

	PVLOG_DEBUG("(pcap plugin) create source for %s\n", qPrintable(input->human_name()));
	PVRush::PVPcapDescription* pcap_desc = dynamic_cast<PVRush::PVPcapDescription*>(input.get());
	assert(pcap_desc);
	PVRush::PVInput_p ifile(new PVRush::PVInputFile(pcap_desc->path().toLocal8Bit().constData()));
	// FIXME: chunk size must be computed somewhere once and for all !
	int size_chunk = pvconfig.value("pvkernel/max_size_chunk").toInt();
	if (size_chunk <= 0) {
		size_chunk = 4096 * 100; // Aligned on a page boundary (4ko)
	}
	source_p src(new PVRush::PVPcapSource(ifile, pcap_desc, size_chunk));

	return src;
}

QString PVPcapsicum::PVSourceCreatorPcap::supported_type() const
{
	return {"pcap"};
}

QString PVPcapsicum::PVSourceCreatorPcap::name() const
{
	return {"pcap"};
}
