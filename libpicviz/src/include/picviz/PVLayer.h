/**
 * \file PVLayer.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVLAYER_H
#define PICVIZ_PVLAYER_H

#include <QtCore>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <picviz/PVLinesProperties.h>
#include <picviz/PVSelection.h>
#include <picviz/PVLayer_types.h>

//#include <tbb/cache_aligned_allocator.h>

#include <vector>

#define PICVIZ_LAYER_NAME_MAXLEN 1000

#define PICVIZ_LAYER_ARCHIVE_EXT "pvl"
#define PICVIZ_LAYER_ARCHIVE_FILTER "Picviz layer-stack files (*." PICVIZ_LAYER_ARCHIVE_EXT ")"

namespace Picviz {

class PVPlotted;

/**
 * \class PVLayer
 */
class LibPicvizDecl PVLayer {
	friend class PVCore::PVSerializeObject;
public:
	typedef std::vector<PVRow> list_row_indexes_t;
private:
	int                index;
	PVLinesProperties  lines_properties;
	bool               locked;
	QString            name;
	PVSelection        selection;
	bool               visible;
	PVRow              selectable_count;
	list_row_indexes_t _row_mins;
	list_row_indexes_t _row_maxs;

public:

	/**
	 * Constructor
	 */
	PVLayer(const QString & name_, const PVSelection & sel_ = PVSelection(), const PVLinesProperties & lp_ = PVLinesProperties());

	void A2B_copy_restricted_by_selection_and_nelts(PVLayer &b, PVSelection const& selection, PVRow nelts);

	int get_index() const {return index;}
	const PVLinesProperties& get_lines_properties() const {return lines_properties;}
	PVLinesProperties& get_lines_properties() {return lines_properties;}
	bool get_locked() const {return locked;}
	const QString & get_name() const {return name;}
	const PVSelection & get_selection() const {return selection;}
	PVSelection& get_selection() {return selection;}
	bool get_visible() const {return visible;}

	void compute_selectable_count(PVRow const& nrows);
	PVRow get_selectable_count() const { return selectable_count; }

	void compute_min_max(PVPlotted const& plotted);
	bool get_min_for_col(PVCol col, PVRow& row) const;
	bool get_max_for_col(PVCol col, PVRow& row) const;
	inline list_row_indexes_t get_mins() const { return _row_mins; }
	inline list_row_indexes_t const& get_maxs() const { return _row_maxs; }

	void reset_to_empty_and_default_color();
	void reset_to_full_and_default_color();
	void reset_to_default_color();

	void set_index(int index_) {index = index_;}
	void set_locked(bool locked_) {locked = locked_;}
	void set_name(const QString & name_) {name = name_; name.truncate(PICVIZ_LAYER_NAME_MAXLEN);}
	void set_visible(bool visible_) {visible = visible_;}

public:
	void load_from_file(QString const& path);
	void save_to_file(QString const& path);

protected:
	// Default constructor is needed when recreating the object
	PVLayer() { PVLayer(""); }

	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
};

}

// This must be done outside of any namespace
// This metatype is used for PVLayer widget selection.
Q_DECLARE_METATYPE(Picviz::PVLayer*)

#endif /* PICVIZ_PVLAYER_H */
