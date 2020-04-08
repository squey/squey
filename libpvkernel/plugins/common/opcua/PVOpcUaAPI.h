/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#ifndef PVOPCUAAPI_FILE_H
#define PVOPCUAAPI_FILE_H

#include <open62541.h>

#include <QString>
#include <functional>

#include <boost/date_time/posix_time/posix_time.hpp>

namespace PVRush
{

class PVOpcUaInfos;

class PVOpcUaAPI
{
  public:
	struct NodeId
	{
		NodeId(QString id);
		NodeId(NodeId&&) = delete;
		~NodeId();
		UA_NodeId open62541() const { return _node_id; }

	  private:
		UA_NodeId _node_id;
	};

	static const char* pvcop_type(int opcua_type_index);
	static std::string to_json_string(UA_Variant const& value);
	static void print_datetime(UA_DateTime date);
	static boost::posix_time::ptime to_ptime(UA_DateTime date);

	PVOpcUaAPI(PVOpcUaInfos const& infos);
	PVOpcUaAPI(PVOpcUaAPI&&) = delete;
	~PVOpcUaAPI();

	void read_node_history(NodeId node_id,
	                       UA_DateTime start_time,
	                       UA_DateTime end_time,
	                       uint32_t values_per_node,
	                       std::function<bool(UA_HistoryData*, bool)> callback);

	UA_DateTime first_historical_datetime(NodeId node_id);

  private:
	void connect_to_server();
	void disconnect();
	void setup_encryption();

  public:
	bool read_history_data(const UA_NodeId* nodeId,
	                       UA_Boolean moreDataAvailable,
	                       const UA_ExtensionObject* data);

  private:
	PVOpcUaInfos const& _infos;
	UA_Client* _client = nullptr;
	std::function<bool(UA_HistoryData*, bool)> _read_callback;
};
}

#endif
