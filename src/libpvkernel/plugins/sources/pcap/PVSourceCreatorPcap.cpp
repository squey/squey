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

#include <pvkernel/core/PVKernelSettings.h>
#include <assert.h>
#include <qbytearray.h>
#include <memory>

#include "PVPcapSource.h"
#include "../../common/pcap/PVPcapDescription.h"
#include "pvkernel/core/PVLogger.h"
#include "pvkernel/rush/PVInput.h"
#include "pvkernel/rush/PVInput_types.h"

PVPcapsicum::PVSourceCreatorPcap::source_p
PVPcapsicum::PVSourceCreatorPcap::create_source_from_input(PVRush::PVInputDescription_p input) const
{
	PVLOG_DEBUG("(pcap plugin) create source for %s\n", qPrintable(input->human_name()));
	auto* pcap_desc = dynamic_cast<PVRush::PVPcapDescription*>(input.get());
	if (pcap_desc == nullptr) {
		throw PVRush::PVInputException("'" + input->human_name().toStdString() +
		                               "' is not a capture");
	}
	const int size_chunk = PVCore::settings::import_chunk_size.value();
	return source_p(new PVRush::PVPcapSource(*pcap_desc, size_chunk));
}

QString PVPcapsicum::PVSourceCreatorPcap::supported_type() const
{
	return {"pcap"};
}

QString PVPcapsicum::PVSourceCreatorPcap::name() const
{
	return {"pcap"};
}
