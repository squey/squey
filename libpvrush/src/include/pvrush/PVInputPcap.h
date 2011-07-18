#ifndef PVINPUTPCAP_FILE_H
#define PVINPUTPCAP_FILE_H

#include <pvcore/general.h>
#include <pvrush/PVInput.h>
#include <string>

// PCAP
#ifdef WIN32
#include <winsock2.h>
#include <pcap.h>
#else
#include <pcap/pcap.h>
#endif

namespace PVRush {

class LibRushDecl PVInputPcap : public PVInput {
public:
	PVInputPcap(pcap_t* pcap);
	PVInputPcap(const char* path);
	virtual ~PVInputPcap();
public:
	size_t operator()(char* buffer, size_t n);
	int datalink() const;
	virtual input_offset current_input_offset();
	virtual void seek_begin();
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
