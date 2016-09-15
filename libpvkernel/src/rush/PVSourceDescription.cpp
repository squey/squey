/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVInputDescription.h> // for PVInputDescription_p, etc
#include <pvkernel/rush/PVInputType.h>        // for PVInputType::list_inputs
#include <pvkernel/rush/PVSourceDescription.h>

#include <boost/iterator/indirect_iterator.hpp> // for indirect_iterator, etc
#include <boost/iterator/iterator_facade.hpp>   // for operator!=

#include <QFile>
#include <QList>
#include <QString>

#include <algorithm> // for equal
#include <memory>    // for __shared_ptr

bool PVRush::PVSourceDescription::operator==(const PVSourceDescription& other) const
{
	// FIXME: PVSourceCreator and PVFormat should have their own operator==

	return _inputs.size() == other._inputs.size() && //!\\ compare lists size before applying
	       // std::equal to avoid segfault !
	       std::equal(boost::make_indirect_iterator(_inputs.begin()),
	                  boost::make_indirect_iterator(_inputs.end()),
	                  boost::make_indirect_iterator(other._inputs.begin())) &&
	       _source_creator_p->registered_name() == other._source_creator_p->registered_name() &&
	       _format.get_full_path() == other._format.get_full_path();
}
