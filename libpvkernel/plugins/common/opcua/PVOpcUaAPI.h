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

// extern "C"
// {
// size_t UA_calcSizeJson(const void* src,
//                        const UA_DataType* type,
//                        UA_String* namespaces,
//                        size_t namespaceSize,
//                        UA_String* serverUris,
//                        size_t serverUriSize,
//                        UA_Boolean useReversible) __attribute__((warn_unused_result));

// UA_StatusCode UA_encodeJson(const void* src,
//                             const UA_DataType* type,
//                             uint8_t** bufPos,
//                             const uint8_t** bufEnd,
//                             UA_String* namespaces,
//                             size_t namespaceSize,
//                             UA_String* serverUris,
//                             size_t serverUriSize,
//                             UA_Boolean useReversible) __attribute__((warn_unused_result));

// UA_StatusCode UA_decodeJson(const UA_ByteString* src, void* dst, const UA_DataType* type)
//     __attribute__((warn_unused_result));
// }

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

	PVOpcUaAPI(PVOpcUaInfos const& infos);
	PVOpcUaAPI(PVOpcUaAPI&&) = delete;
	~PVOpcUaAPI();

	void read_node_history(NodeId node_id,
	                       UA_DateTime start_time,
	                       UA_DateTime end_time,
	                       std::function<bool(UA_HistoryData*)> callback);

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
	std::function<bool(UA_HistoryData*)> _read_callback;
};
}

#endif
