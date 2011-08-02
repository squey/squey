#include "PVFieldSplitterPcapPacket.h"
#include <pvkernel/core/network.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVInputPcap.h>

#include <QStringList>
#include <QHash>
#include <QDateTime>

// PCAP
#ifdef WIN32
#include <winsock2.h>
#include <pcap.h>
#include <pcap/sll.h>
#else
#include <pcap/pcap.h>
#include <pcap/sll.h>
#include <arpa/inet.h>
#endif

#include <dnet.h>

#include <tbb/scalable_allocator.h>

#define TCP_SESSIONS_MAX 4096
#define TCP_DATA_KEEP_MAX 524288	// 512k should be enough to keep for modern internet :)
QHash<uint32_t, QString> tcp_data;		// Key = ack number; string = data

struct pcap_decode_buf {
	PVCore::list_fields *l;
	PVCore::list_fields::iterator it_ins;
	PVCore::PVElement* parent;
	char* data;
	size_t rem_len;
	size_t nelts;
};

typedef int (*packet_decoder_function)(pcap_decode_buf *buf, struct pcap_pkthdr *pheader, u_char *packet);
static int pcap_decode_layer2_sll(pcap_decode_buf *buf, struct pcap_pkthdr *pheader, u_char *packet);
static int pcap_decode_layer2_10MB(pcap_decode_buf *buf, struct pcap_pkthdr *pheader, u_char *packet);
static int pcap_decode_layer3_IP(pcap_decode_buf *buf, struct pcap_pkthdr *pheader, u_char *packet, uint32_t len);
static int pcap_decode_layer4_TCP(pcap_decode_buf *buf, struct pcap_pkthdr *pheader, u_char *packet, uint32_t len);
static int pcap_decode_layer4_UDP(pcap_decode_buf *buf, struct pcap_pkthdr *pheader, u_char *packet, uint32_t len);
static int pcap_decode_layer4_ICMP(pcap_decode_buf *buf, struct pcap_pkthdr *pheader, u_char *packet, uint32_t len);


PVFilter::PVFieldSplitterPcapPacket::PVFieldSplitterPcapPacket(PVCore::PVArgumentList const& args)
{
	_datalink_type = -1;
	INIT_FILTER(PVFilter::PVFieldSplitterPcapPacket, args);
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterPcapPacket)
{
	PVCore::PVArgumentList args;
	args["datalink"] = QVariant((int) -1);
	PVLOG_DEBUG("test: %d\n", args["datalink"].toInt());
	return args;
}

void PVFilter::PVFieldSplitterPcapPacket::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_datalink_type = args["datalink"].toInt();
	PVLOG_DEBUG("(PVFieldSplitterPcapPacket) datalink set to %d\n", _datalink_type);
}

PVCore::list_fields::size_type PVFilter::PVFieldSplitterPcapPacket::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	if (field.size() < sizeof(struct pcap_pkthdr)) {
		PVLOG_WARN("(PVFieldSplitterPcapPacket) field is too short to hold a pcap_pkthdr struct. Ignoring it !\n");
		return 0;
	}

	if (_datalink_type == -1) {
		PVRush::PVInputPcap* pcap = dynamic_cast<PVRush::PVInputPcap*>(field.elt_parent()->chunk_parent()->source()->get_input().get());
		if (pcap) {
			_datalink_type = pcap->datalink();
		}
		else {
			PVLOG_WARN("(PVFieldSplitterPcapPacket) datalink hasn't been set and the input source isn't supported by libpcap. Datalink is set to Ethernet by default !\n");
			_datalink_type = DLT_EN10MB;
		}
	}

	// Into the field, the packet has first a pcap_pkthdr struct, and then the payload
	struct pcap_pkthdr* pheader = (struct pcap_pkthdr*) (field.begin());
	size_t len_packet = pheader->caplen;

	if (field.size() < sizeof(struct pcap_pkthdr) + len_packet) {
		PVLOG_WARN("(PVFieldSplitterPcapPacket) field is too short to hold a pcap_pkthdr struct with its payload. Ignoring it !\n");
		field.set_invalid();
		return 0;
	}

	// Copy everything has this is going to be erased !
	u_char* payload = (u_char*) (pheader + 1);

	static tbb::scalable_allocator<u_char> alloc;
	u_char* payload_copy = alloc.allocate(len_packet);

	struct pcap_pkthdr header = *pheader;
	memcpy(payload_copy, payload, len_packet);

	// We grow the field buffer by as musch as it can
	field.grow_by(0);

	pcap_decode_buf buf;
	buf.l = &l;
	buf.it_ins = it_ins;
	buf.data = field.begin();
	buf.parent = field.elt_parent();
	buf.rem_len = field.size();
	buf.nelts = 0;

	packet_decoder_function packet_decode;
	switch (_datalink_type) {
		case DLT_LINUX_SLL:
			packet_decode = pcap_decode_layer2_sll;
			break;
		case DLT_EN10MB: // We are almost always in this case
			packet_decode = pcap_decode_layer2_10MB;
			break;
			// FIXME: We still need to handle other datalinks, should figure it out when we cannot get an IP packet that exists :)
		default:
			{
				PVLOG_ERROR("Unsupported datalink: %s (%d)\n", pcap_datalink_val_to_name(_datalink_type), _datalink_type);
				field.set_invalid();
				alloc.deallocate(payload_copy, len_packet);
				return 0;
			}
	}

	int ret = packet_decode(&buf, &header, payload_copy);
	if (ret <= 0) {
		// Field is invalid !
		field.set_invalid();
		field.elt_parent()->set_invalid();
	}

	alloc.deallocate(payload_copy, len_packet);
	return buf.nelts;
}

static void pcap_decode_add_field(pcap_decode_buf* buf, QString const& field)
{
	size_t bufsize = field.size() * sizeof(QChar);
	if (bufsize > buf->rem_len) {
		PVLOG_WARN("(PVFieldSplitterPcapPacket) buffer is too small to handle packet decoding. Ignoring field !\n");
		return;
	}
	memcpy(buf->data, field.constData(), bufsize);
	PVCore::PVField f(*buf->parent, buf->data, buf->data+bufsize);
	buf->data += bufsize;
	buf->rem_len -= bufsize;
	buf->l->insert(buf->it_ins, f);
	buf->nelts++;
}

static QString pcap_decode_get_time_as_string(struct pcap_pkthdr *pheader)
{
	struct timeval packet_tv;
	time_t t;
	QDateTime qt_datetime;

	packet_tv = pheader->ts;
	t = packet_tv.tv_sec;
	qt_datetime = qt_datetime.fromTime_t(t);
	QString time_str = qt_datetime.toString("dd-MM-yyyy hh:mm:ss");
	// PVLOG_INFO("Packet time string='%s'\n", time_str.toUtf8().data());

	return time_str;
}

static int pcap_decode_layer2_sll(pcap_decode_buf *buf, struct pcap_pkthdr *pheader, u_char *packet)
{
	struct sll_header *SLL;

	if (pheader->caplen < SLL_HDR_LEN) {
		PVLOG_ERROR("Capture length < SLL header length! Cannot process!\n");
		return 0;
	}
	PVLOG_HEAVYDEBUG("We decode SLL packet of length: %d\n", pheader->caplen);

	//pcap_decode_get_time_as_string(pheader);

	SLL = (struct sll_header *)packet;
	switch(ntohs(SLL->sll_protocol)) {
	case ETH_TYPE_IP:
		PVLOG_HEAVYDEBUG("Packet type = IP\n");
		return pcap_decode_layer3_IP(buf, pheader, packet + SLL_HDR_LEN, pheader->caplen - SLL_HDR_LEN);
	case ETH_TYPE_ARP:
		PVLOG_HEAVYDEBUG("Packet type = ARP\n");
		return 0;	// No supported yet, we count as not decoded
	case ETH_TYPE_IPV6:
		PVLOG_HEAVYDEBUG("Packet type = IP6\n");
		return 0;	// No supported yet, we count as not decoded
	default:
		PVLOG_ERROR("Unknown protocol=%d\n", SLL->sll_protocol);
	}

	return 0;
}

static int pcap_decode_layer2_10MB(pcap_decode_buf *buf, struct pcap_pkthdr *pheader, u_char *packet)
{
	PVLOG_HEAVYDEBUG("We decode 10MB packet\n");
	struct eth_hdr *ether;

	ether = (struct eth_hdr *)packet;
	switch(ntohs(ether->eth_type)) {
	case ETH_TYPE_IP:
		return pcap_decode_layer3_IP(buf, pheader, packet + ETH_HDR_LEN, pheader->caplen - ETH_HDR_LEN);
	default:
		// We cannot decode the IP packet
		return 0;
	}

	return 0;
}

static int pcap_decode_layer3_IP(pcap_decode_buf *buf, struct pcap_pkthdr *pheader, u_char *packet, uint32_t len)
{
	struct ip_hdr *ip;

	ip = (struct ip_hdr *)packet;
	QString ipsource(PVCore::Network::ipv4_ntoa(ip->ip_src));
	QString ipdest(PVCore::Network::ipv4_ntoa(ip->ip_dst));


	pcap_decode_add_field(buf, pcap_decode_get_time_as_string(pheader));
	pcap_decode_add_field(buf, ipsource);
	pcap_decode_add_field(buf, ipdest);
	pcap_decode_add_field(buf, QString("%1").arg(ip->ip_len));

	switch(ip->ip_p) {
	case IP_PROTO_TCP:
		// pcap_decode_layer4_TCP(nraw, pheader, pcaph, packet + IP_HDR_LEN, len - IP_HDR_LEN);
		// printf("IP HDR LEN=%d\n", IP_HDR_LEN);
		return pcap_decode_layer4_TCP(buf, pheader, packet + IP_HDR_LEN, len - IP_HDR_LEN);
	case IP_PROTO_UDP:
		return pcap_decode_layer4_UDP(buf, pheader, packet + IP_HDR_LEN, len - IP_HDR_LEN);
	case IP_PROTO_ICMP:
		return pcap_decode_layer4_ICMP(buf, pheader, packet + IP_HDR_LEN, len - IP_HDR_LEN);
	default:
		PVLOG_ERROR("Unknown Layer 4 protocol!\n");
	}

	return 0;
}

static int pcap_decode_layer4_TCP(pcap_decode_buf *buf, struct pcap_pkthdr* /*pheader*/, u_char *packet, uint32_t /*len*/)
{
	QByteArray netflow_env = qgetenv("PICVIZ_NONETFLOW"); // FIXME: Evil, should avoid calling this for *each* packet!
	struct tcp_hdr *tcp;

	tcp = (struct tcp_hdr *)(packet);
		
	if ((tcp->th_flags == TH_SYN) || (netflow_env.isEmpty())) {
		pcap_decode_add_field(buf, "tcp");
		
		pcap_decode_add_field(buf, QString("%1").arg(ntohs(tcp->th_win)));
		pcap_decode_add_field(buf, QString("%1").arg(ntohs(tcp->th_sport)));
		pcap_decode_add_field(buf, QString("%1").arg(ntohs(tcp->th_dport)));
		return 4;
	}
	return 0;
}


static int pcap_decode_layer4_UDP(pcap_decode_buf *buf, struct pcap_pkthdr* /*pheader*/, u_char* packet, uint32_t /*len*/)
{
	struct udp_hdr *udp;

	// PVLOG_INFO("We have a UDP packet!\n");
	pcap_decode_add_field(buf, "udp");
	udp = (struct udp_hdr *)(packet);
	
	pcap_decode_add_field(buf, "0"); // No window for UDP
	pcap_decode_add_field(buf, QString("%1").arg(ntohs(udp->uh_sport)));
	pcap_decode_add_field(buf, QString("%1").arg(ntohs(udp->uh_dport)));
	return 4;
}

static int pcap_decode_layer4_ICMP(pcap_decode_buf* /*buf*/, struct pcap_pkthdr* /*pheader*/, u_char* /*packet*/, uint32_t /*len*/)
{
	PVLOG_INFO("ICMP not supported yet!\n");
	return 0;
}

IMPL_FILTER(PVFilter::PVFieldSplitterPcapPacket)
