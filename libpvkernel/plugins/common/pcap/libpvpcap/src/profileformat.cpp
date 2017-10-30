/**
 * \file
 *
 * Copyright (C) Philippe Saade - Justin Teliet 2014
 */

#include "../include/libpvpcap.h"
#include "../include/libpvpcap/profileformat.h"
#include "../include/libpvpcap/ws.h"

#include <pvkernel/core/PVUtils.h>

#include <iostream>

namespace pvpcap
{

/*******************************************************************************
 *
 * create_csv_spliter_format_root
 *
 ******************************************************************************/

PVRush::PVXmlTreeNodeDom* create_csv_spliter_format_root(QDomDocument& format_doc)
{
	// initialize the format root node
	PVRush::PVXmlTreeNodeDom* format_root(PVRush::PVXmlTreeNodeDom::new_format(format_doc));

	// create a DOM element for spliter node
	QDomElement csv_splitter_dom = format_doc.createElement(PVFORMAT_XML_TAG_SPLITTER_STR);
	csv_splitter_dom.setAttribute("type", "csv");

	// Set csv quote and separator
	csv_splitter_dom.setAttribute("quote", QString::fromStdString(ws_map_quote.at(pvpcap::QUOTE)));
	csv_splitter_dom.setAttribute("sep", pvpcap::SEPARATOR);
	format_root->getDom().appendChild(csv_splitter_dom);

	// create and attach splitter node
	PVRush::PVXmlTreeNodeDom* csv_splitter_node;
	csv_splitter_node = new PVRush::PVXmlTreeNodeDom(csv_splitter_dom);
	csv_splitter_node->setParent(format_root);
	format_root->addChild(csv_splitter_node);

	return csv_splitter_node;
}

/*******************************************************************************
 *
 * get_profile_format
 *
 ******************************************************************************/
QDomDocument get_format(const rapidjson::Document& json_data, size_t input_pcap_count)
{
	const rapidjson::Document& selected_fields = ws_get_selected_fields(json_data);
	assert(selected_fields.IsObject() && "selected_field_list is not a json object!");

	// Generate format file for this profile file
	// PVPcapTypeMap_p type_map = PVPcapTypeMap::get_map("ws");
	QDomDocument profile_format_doc;
	std::unique_ptr<PVRush::PVXmlTreeNodeDom> csv_format(
	    create_csv_spliter_format_root(profile_format_doc));

	// create "frame.global_number",  "stream_id" and "pcap_path" columns
	csv_format->addOneField("frame.global_number", "number_uint32");
	csv_format->addOneField("stream_id", "number_uint32");

	if (input_pcap_count > 1) {
		csv_format->addOneField("pcap_path", "string");
	}

	for (const auto& field : selected_fields.GetObject()) {
		const QString& type = QString::fromStdString(ws_map_type(field.value["type"].GetString()));
		const QString& name = field.name.GetString();
		PVRush::PVXmlTreeNodeDom* node = csv_format->addOneField(name, type);
		if (type == "time") {
			node->setAttribute(QString(PVFORMAT_AXIS_TYPE_FORMAT_STR),
			                   "MMM d, yyyy HH:m:ss.S 'UTC'");

			/* Use a converter substitution to workaround a parsing bug in
			 * pvcop::types::formater_datetime_us (boost::posix::date_time)
			 * when days of month do not have a leading zero ("  " -> " 0")
			 */
			PVFilter::PVFieldsConverterParamWidget_p in_t =
			    LIB_CLASS(PVFilter::PVFieldsConverterParamWidget)::get().get_class_by_name(
			        "substitution");

			QDomElement converter_elt = profile_format_doc.createElement(in_t->type_name());
			converter_elt.setAttribute("type", in_t->registered_name());
			PVRush::PVXmlTreeNodeDom child(converter_elt);

			PVCore::PVArgumentList args;
			args["modes"] = 2; // substring
			QStringList l;
			l << "  "
			  << " 0";
			args["substrings_map"] = PVCore::serialize_base64(l);

			child.setFromArgumentList(args);
			node->getParent()->getDom().appendChild(converter_elt);
		} else if (type == "mac_address") {
			node->setMappingProperties("mac_address-uni-lin", {}, {});
		}
	}

	// add wireshark specials fields
	if (json_data["options"]["source"].GetBool())
		csv_format->addOneField(QString::fromStdString(ws_map_special_fields.at("source")),
		                        "string");

	if (json_data["options"]["destination"].GetBool())
		csv_format->addOneField(QString::fromStdString(ws_map_special_fields.at("destination")),
		                        "string");

	if (json_data["options"]["protocol"].GetBool())
		csv_format->addOneField(QString::fromStdString(ws_map_special_fields.at("protocol")),
		                        "string");

	if (json_data["options"]["info"].GetBool())
		csv_format->addOneField(QString::fromStdString(ws_map_special_fields.at("info")), "string");

	// TODO: Wireshark option specials fields
	return profile_format_doc;
}

} // namespace pvpcap
