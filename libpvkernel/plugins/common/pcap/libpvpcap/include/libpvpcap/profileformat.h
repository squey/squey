/*!
 * \file
 * \brief Manage Exceptions for the tree of the protocols in pcap.
 *
 * \copyright (C) Philippe Saade - Justin Teliet 2014
 * \copyright (C) ESI Group INENDI 2017
 */

#ifndef LIBPVPCAP_PROFILEFORMAT_H
#define LIBPVPCAP_PROFILEFORMAT_H

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <pvkernel/rush/PVFormat_types.h>

#include <QDomDocument>

/*!
 * Namespace that groups together the components of managing
 * the tree of protocols in a pcap file.
 */
namespace pvpcap
{

PVRush::PVXmlTreeNodeDom* create_csv_spliter_format_root(QDomDocument& doc);

QDomDocument get_format(const rapidjson::Document& json_data, size_t input_pcap_count);

} /* namespace pvpcap */

#endif /* LIBPVPCAP_EXCEPTION_H */
