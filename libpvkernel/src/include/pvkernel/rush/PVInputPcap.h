/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVINPUTPCAP_FILE_H
#define PVINPUTPCAP_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInput.h>
#include <string>

// PCAP
#include <pcap/pcap.h>

namespace PVRush {

class PVInputPcap : public PVInput {
public:
	PVInputPcap(pcap_t* pcap);
	PVInputPcap(const char* path);
	virtual ~PVInputPcap();
public:
	size_t operator()(char* buffer, size_t n);
	int datalink() const;
	virtual input_offset current_input_offset();
	virtual void seek_begin();
	bool seek(input_offset off);
	virtual QString human_name();
private:
	void post_init();
protected:
	pcap_t* _pcap;
	std::string _path;
	int _datalink;
	input_offset _next_packet;

	CLASS_INPUT(PVRush::PVInputPcap)
};

}


#endif
