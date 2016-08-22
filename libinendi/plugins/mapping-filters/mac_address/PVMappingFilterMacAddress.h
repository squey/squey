/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */

#ifndef INENDI_PVMAPPINGFILTER_MAC_ADDRESS_H
#define INENDI_PVMAPPINGFILTER_MAC_ADDRESS_H

#include <inendi/PVMappingFilter.h>

namespace Inendi
{

/**
 * Base MAC address mappings class.
 *
 * MAC addresses (MAC for short) can be splitted into 2 parts:
 * - the vendor part (vendor for short);
 * - a vendor specific part (NIC for short).
 */

class PVMappingFilterMacAddressBase : public PVMappingFilter
{
  public:
	PVMappingFilterMacAddressBase() { INIT_FILTER_NOPARAM(PVMappingFilterMacAddressBase); }

	std::unordered_set<std::string> list_usable_type() const override { return {"mac_address"}; }
};

/**
 * Distribute MAC addresses linearly by removing the 16 LSB
 */
class PVMappingFilterMacAddressL : public PVMappingFilterMacAddressBase
{
  public:
	PVMappingFilterMacAddressL() { INIT_FILTER_NOPARAM(PVMappingFilterMacAddressL); }

	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override;

	QString get_human_name() const override { return QString("Linear MAC"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterMacAddressL)
};

/**
 * Distribute MAC addresses uniformly in [0;2^32)
 */
class PVMappingFilterMacAddressU : public PVMappingFilterMacAddressBase
{
  public:
	PVMappingFilterMacAddressU() { INIT_FILTER_NOPARAM(PVMappingFilterMacAddressU); }

	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override;

	QString get_human_name() const override { return QString("Uniform MAC"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterMacAddressU)
};

/**
 * Distribute MAC addresses linearly according to their vendor part and uniformly according
 * to the NIC part.
 */
class PVMappingFilterMacAddressLU : public PVMappingFilterMacAddressBase
{
  public:
	PVMappingFilterMacAddressLU() { INIT_FILTER_NOPARAM(PVMappingFilterMacAddressLU); }

	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override;

	QString get_human_name() const override { return QString("Linear vendor - Uniform NIC"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterMacAddressLU)
};

/**
 * Distribute MAC addresses uniformly according to their vendor part and linearly according
 * to the NIC part.
 */
class PVMappingFilterMacAddressUL : public PVMappingFilterMacAddressBase
{
  public:
	PVMappingFilterMacAddressUL() { INIT_FILTER_NOPARAM(PVMappingFilterMacAddressUL); }

	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override;

	QString get_human_name() const override { return QString("Uniform vendor - Linear NIC"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterMacAddressUL)
};

/**
 * Distribute MAC addresses uniformly according to their vendor part and uniformly according
 * to the NIC part.
 */
class PVMappingFilterMacAddressUU : public PVMappingFilterMacAddressBase
{
  public:
	PVMappingFilterMacAddressUU() { INIT_FILTER_NOPARAM(PVMappingFilterMacAddressUU); }

	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override;

	QString get_human_name() const override { return QString("Uniform vendor - Uniform NIC"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterMacAddressUU)
};
}

#endif // INENDI_PVMAPPINGFILTER_MAC_ADDRESS_H
