/**
 * \file PVLinesProperties.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVLINESPROPERTIES_H
#define PICVIZ_PVLINESPROPERTIES_H

#include <QVector>

#include <picviz/general.h>

#include <pvkernel/core/PVHSVColor.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <picviz/PVSelection.h>

#define PICVIZ_LINESPROPS_CHUNK_SIZE sizeof(PVCore::PVHSVColor)
#define PICVIZ_LINESPROPS_NUMBER_OF_CHUNKS PICVIZ_LINES_MAX
#define PICVIZ_LINESPROPS_NUMBER_OF_BYTES  PICVIZ_LINESPROPS_NUMBER_OF_CHUNKS * PICVIZ_LINESPROPS_CHUNK_SIZE

namespace Picviz {

/**
 * \class PVLinesProperties
 */
class LibPicvizDecl PVLinesProperties {
	friend class PVCore::PVSerializeObject;

public:
	typedef std::allocator<PVCore::PVHSVColor> color_allocator_type;
	typedef PVCore::PVHSVColor::h_type h_type;
private:
	static color_allocator_type _color_allocator;

public:

	/**
	 * Constructor
	 */
	PVLinesProperties();
	PVLinesProperties(const PVLinesProperties & rhs);
	PVLinesProperties(PVLinesProperties&& rhs)
	{
		_table = rhs._table;
		rhs._table = NULL;
	}

	/**
	 * Destructor
	 */
	~PVLinesProperties();

	/**
	 * Gets the PVHSVColor of a given line
	 *
	 * @param r The index of the line (its row number)
	 */
	inline PVCore::PVHSVColor& get_line_properties(const PVRow r) { return _table[r]; }
	inline const PVCore::PVHSVColor get_line_properties(const PVRow r) const { return _table[r]; }

	inline void line_set_color(const PVRow r, const PVCore::PVHSVColor h) { get_line_properties(r) = h; }

	PVLinesProperties & operator=(const PVLinesProperties & rhs);

	void A2A_set_to_line_properties_restricted_by_selection_and_nelts(PVCore::PVHSVColor color, PVSelection const& selection, PVRow nelts);

	void A2B_copy_restricted_by_selection_and_nelts(PVLinesProperties &b, PVSelection const& selection, PVRow nelts);
	void A2B_copy_zombie_off_restricted_by_selection_and_nelts(PVLinesProperties &b,  PVSelection const& selection, PVRow nelts);
	void A2B_copy_zombie_on_restricted_by_selection_and_nelts(PVLinesProperties &b,  PVSelection const& selection, PVRow nelts);
	void reset_to_default_color();
	void selection_set_color(PVSelection const& selection, const PVRow nelts, const PVCore::PVHSVColor c);
	void set_random(const PVRow n);

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

private:
	PVCore::PVHSVColor* _table;
};

}

#endif /* PICVIZ_PVLINESPROPERTIES_H */
