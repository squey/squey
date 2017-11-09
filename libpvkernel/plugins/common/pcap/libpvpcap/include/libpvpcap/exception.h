/*!
 * \file
 * \brief Manage Exceptions for the tree of the protocols in pcap.
 *
 * \copyright (C) Philippe Saade - Justin Teliet 2014
 * \copyright (C) ESI Group INENDI 2017
 */

#ifndef LIBPVPCAP_EXCEPTION_H
#define LIBPVPCAP_EXCEPTION_H

#include <stdexcept>

/*!
 * Namespace that groups together the components of managing
 * the tree of protocols in a pcap file.
 */
namespace pvpcap
{

/*!
 * Exception raised when there is a problem in the pcap processing.
 *
 */
class PcapTreeException : public std::runtime_error
{
	using std::runtime_error::runtime_error;
};

} /* namespace pvpcap */

#endif /* LIBPVPCAP_EXCEPTION_H */
