/**
 * \file PVInputPcap.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVINPUTPCAP_FILE_H
#define PVINPUTPCAP_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInput.h>
#include <string>

// PCAP
#ifdef WIN32
#include <winsock2.h>
#include <pcap.h>
#else
#include <pcap/pcap.h>
#endif

namespace PVRush {

class LibKernelDecl PVInputPcap : public PVInput {
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
