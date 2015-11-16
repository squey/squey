/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVPLOTTINGPROPERTIES_H
#define INENDI_PVPLOTTINGPROPERTIES_H

//#include <QList>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVFormat.h>

#include <inendi/PVAxis.h>
#include <inendi/PVRoot.h>
#include <inendi/PVPlottingFilter.h>

namespace Inendi {

class PVPlotting;
class PVMapping;

/**
* \class PVPlottingProperties
*
* \brief Stored functions and variables that can to be modified by those functions
*/
class PVPlottingProperties {
	friend class PVCore::PVSerializeObject;
	friend class PVPlotting;
	friend class PVPlotted;
public:
	PVPlottingProperties(PVMapping const& mapping, PVRush::PVFormat const& fmt, PVCol idx);
	PVPlottingProperties(PVMapping const& mapping, PVRush::PVAxisFormat const& axis, PVCol idx);

protected:
	// Serialization
	PVPlottingProperties() { _mapping = NULL; }
	void set_mapping(const PVMapping& mapping);

	// For PVPlotting
	inline void set_uptodate() { _is_uptodate = true; }
	inline void invalidate() { _is_uptodate = false; }

public:
	PVPlottingFilter::p_type get_plotting_filter();
	void set_from_axis(PVRush::PVAxisFormat const& axis);
	void set_from_axis(Inendi::PVAxis const& axis);
	void set_mode(QString const& mode);
	void set_args(PVCore::PVArgumentList const& args);
	inline PVCore::PVArgumentList const& get_args() const { return _args; }
	inline QString const& get_mode() const { return _mode; }
	QString get_type() const;
	inline bool is_uptodate() const { return _is_uptodate; }

public:
	bool operator==(PVPlottingProperties const& org);

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

private:
	QString _type;
	QString _mode;
	PVCol _index;
	PVPlottingFilter::p_type _plotting_filter;
	PVCore::PVArgumentList _args;
	bool _is_uptodate;
	const PVMapping* _mapping;
};

}

#endif	/* INENDI_PVPLOTTINGPROPERTIES_H */
