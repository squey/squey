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

namespace PVRush
{

class PVOpcUaInfos;

class PVOpcUaAPI
{
	struct NodeId
	{
		NodeId(QString id);
		NodeId(NodeId&&) = delete;
		~NodeId();
		UA_NodeId open62541() const { return _node_id; }

	  private:
		UA_NodeId _node_id;
	};

  public:
	PVOpcUaAPI(PVOpcUaInfos const& infos);
	PVOpcUaAPI(PVOpcUaAPI&&) = delete;
	~PVOpcUaAPI();

	void read_node_history(NodeId node_id, std::function<bool(UA_HistoryData*)> callback);

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
	std::function<bool(UA_HistoryData*)> _read_callback;
};
}

#endif
