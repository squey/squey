/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVPUREMAPPINGFILTER_H
#define INENDI_PVPUREMAPPINGFILTER_H

#include <inendi/PVMappingFilter.h>

namespace Inendi {

/*! \brief Template-helper implementation of a pure mapping filter
 *
 *  This class implements basic iterations over chunks and an NRAW's column for
 *  a pure mapping filter.
 *  A pure mapping filter is a filter for which the result of the computation of an element only
 *  depends on that element, and nothing else. This provides room for parallelisation...
 */
template <typename MappingImpl>
class PVPureMappingFilter: public PVMappingFilter
{
	typedef MappingImpl mapping_impl_t;

public:
	virtual decimal_storage_type operator()(PVCore::PVField const& f) override
	{
		return mapping_impl_t::process_utf16((uint16_t const*) f.begin(), f.size()/sizeof(uint16_t), this);
	}

	virtual decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw) override
	{
		auto const& array =  nraw.collection().column(col);
		assert(array.size() <= _dest_size);

#pragma omp parallel for
		for(size_t row=0; row< array.size(); row++) {
			std::string content = array.at(row);
			this->_dest[row] = mapping_impl_t::process_utf8(content.c_str(), content.size(), this);
		}

		return this->_dest;
	}

	/**
	 * redefinition of PVMappingFilter::finalize(...)
	 *
	 * As a pure mapping filter does not need pre/post processing,
	 * ::finalize must not do anything and must not be overridden.
	 */
	decimal_storage_type* finalize(PVCol const, PVRush::PVNraw const&) override final { return nullptr; }

	virtual bool is_pure() const override { return true; }
};

}

#endif
