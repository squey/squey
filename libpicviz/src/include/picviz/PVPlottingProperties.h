//! \file PVPlottingProperties.h
//! $Id: PVPlottingProperties.h 3062 2011-06-07 08:33:36Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVPLOTTINGPROPERTIES_H
#define PICVIZ_PVPLOTTINGPROPERTIES_H

//#include <QList>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVFormat.h>

#include <picviz/PVRoot.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlotting;
class PVMapping;

/**
* \class PVPlottingProperties
*
* \brief Stored functions and variables that can to be modified by those functions
*/
class LibPicvizDecl PVPlottingProperties {
	friend class PVCore::PVSerializeObject;
	friend class PVPlotting;
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
	void set_mode(QString const& mode);
	void set_args(PVCore::PVArgumentList const& args);
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

#endif	/* PICVIZ_PVPLOTTINGPROPERTIES_H */
