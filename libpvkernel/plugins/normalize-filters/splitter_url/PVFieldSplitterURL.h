/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVFIELDSPLITTERURL_H
#define PVFILTER_PVFIELDSPLITTERURL_H

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <QChar>

#include <furl/furl.h>

#include <tbb/combinable.h>

namespace PVFilter
{

class PVFieldSplitterURL : public PVFieldsFilter<one_to_many>
{
  public:
	PVFieldSplitterURL();

  protected:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields& l,
	                                           PVCore::list_fields::iterator it_ins,
	                                           PVCore::PVField& field) override;

  private:
	enum ColKind {
		_col_proto = 0,
		_col_subdomain = 1,
		_col_host = 2,
		_col_domain = 3,
		_col_tld = 4,
		_col_port = 5,
		_col_url = 6,
		_col_variable = 7,
		_col_fragment = 8,
		_col_credentials = 9
	};

	tbb::combinable<furl_handler_t> _furl_handler;

	CLASS_FILTER(PVFilter::PVFieldSplitterURL)
};
}

#endif
