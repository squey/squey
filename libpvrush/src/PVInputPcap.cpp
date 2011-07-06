#include <pvrush/PVInputPcap.h>

PVRush::PVInputPcap::PVInputPcap(pcap_t* pcap):
	PVInput(),
	_pcap(pcap)
{
	post_init();
}

PVRush::PVInputPcap::~PVInputPcap()
{
	if (_pcap)
		pcap_close(_pcap);
}

PVRush::PVInputPcap::PVInputPcap(const char* path)
{
	_path = path;
	char errbuf[PCAP_ERRBUF_SIZE];
	_pcap = pcap_open_offline(path, errbuf);
	if (!_pcap) {
		PVLOG_WARN("Cannot open PCAP file %s: %s\n", path, errbuf);
		_pcap = NULL;
		return;
	}
	post_init();
}

void PVRush::PVInputPcap::post_init()
{
	_datalink = pcap_datalink(_pcap);
	_next_packet = 0;
}

size_t PVRush::PVInputPcap::operator()(char* buffer, size_t n)
{
	if (!_pcap)
		return 0;
	// TODO: put more than one packet in a chunk, and align them !
	struct pcap_pkthdr pheader;
	u_char *packet;
	packet = (u_char*) pcap_next(_pcap, &pheader);
	if (packet == NULL)
		return 0;
	size_t ret = sizeof(int)+sizeof(struct pcap_pkthdr)+pheader.caplen;
	if (ret > n) {
		PVLOG_WARN("(PVInputPcap) Packet discared because do not fit in %d bytes !\n", n);
		return 0;
	}

	// TOFIX: we should be able to recover datalink from a field !
	*((int*) buffer) = _datalink;
	buffer += sizeof(int);
	memcpy(buffer, &pheader, sizeof(struct pcap_pkthdr));
	memcpy(buffer+sizeof(struct pcap_pkthdr), packet, pheader.caplen);

	_next_packet++;

	return ret;
}

int PVRush::PVInputPcap::datalink() const
{
	return _datalink;
}

PVRush::PVInputPcap::input_offset PVRush::PVInputPcap::current_input_offset()
{
	return _next_packet;
}

void PVRush::PVInputPcap::seek_begin()
{
	if (_path.empty()) {
		PVLOG_ERROR("FIXME: unable to seek from beggining from a live capture !\n");
		assert(false);
	}

	if (_pcap) {
		pcap_close(_pcap);
	}

	char errbuf[PCAP_ERRBUF_SIZE];
	_pcap = pcap_open_offline(_path.c_str(), errbuf);
	if (!_pcap) {
		PVLOG_WARN("Cannot open PCAP file %s: %s\n", _path.c_str(), errbuf);
		return;
	}
	post_init();
}

QString PVRush::PVInputPcap::human_name()
{
	if (_path.empty())
		return QString("PCAP live capture");
	return QString(_path.c_str());
}

IMPL_INPUT(PVRush::PVInputPcap)
