//! \file PVLinesProperties.h
//! $Id: PVLinesProperties.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLINESPROPERTIES_H
#define PICVIZ_PVLINESPROPERTIES_H

#include <QVector>

#include <picviz/general.h>

#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <picviz/PVSelection.h>

#define PICVIZ_LINESPROPS_CHUNK_SIZE sizeof(PVCore::PVColor)
#define PICVIZ_LINESPROPS_NUMBER_OF_CHUNKS PICVIZ_LINES_MAX
#define PICVIZ_LINESPROPS_NUMBER_OF_BYTES  PICVIZ_LINESPROPS_NUMBER_OF_CHUNKS * PICVIZ_LINESPROPS_CHUNK_SIZE

namespace Picviz {

/**
 * \class PVLinesProperties
 */
class LibPicvizDecl PVLinesProperties {
	friend class PVCore::PVSerializeObject;

public:
	typedef std::allocator<PVCore::PVColor> color_allocator_type;
private:
	static color_allocator_type _color_allocator;

public:
	PVRow last_index; /*<! FIXME: Do we really need this?  */
	PVCore::PVColor *table;

	/**
	 * Constructor
	 */
	PVLinesProperties();

	PVLinesProperties(const PVLinesProperties & rhs);

	/**
	 * Destructor
	 */
	~PVLinesProperties();

	/**
	 * Gets the PVColor of a given line
	 *
	 * @param line The index of the line (its row number)
	 */
	PVCore::PVColor& get_line_properties(PVRow line);
	const PVCore::PVColor& get_line_properties(PVRow line) const;

	/**
	 * Gets the A value of the color of the specified line
	 *
	 * @param line The index of the line (its row number)
	 */
	unsigned char line_get_a(PVRow line);
	
	/**
	 * Gets the B value of the color of the specified line
	 *
	 * @param line The index of the line (its row number)
	 */
	unsigned char line_get_b(PVRow line);

	/**
	 * Gets the G value of the color of the specified line
	 *
	 * @param line The index of the line (its row number)
	 */
	unsigned char line_get_g(PVRow line);

	/**
	 * Gets the R value of the color of the specified line
	 *
	 * @param line The index of the line (its row number)
	 */
	unsigned char line_get_r(PVRow line);

	/**
	 * Sets the A value of the color of the specified line
	 *
	 * @param line The index of the line (its row number)
	 * @param a    The A value that should be set
	 */
	void line_set_a(PVRow line, unsigned char a);
	
	/**
	 * Sets the B value of the color of the specified line
	 *
	 * @param line The index of the line (its row number)
	 * @param b    The B value that should be set
	 */
	void line_set_b(PVRow line, unsigned char b);

	/**
	 * Sets the G value of the color of the specified line
	 *
	 * @param line The index of the line (its row number)
	 * @param g    The G value that should be set
	 */
	void line_set_g(PVRow line, unsigned char g);

	/**
	 * Sets the R value of the color of the specified line
	 *
	 * @param line The index of the line (its row number)
	 * @param r    The R value that should be set
	 */
	void line_set_r(PVRow line, unsigned char r);

	/**
	 * Sets the R,G,B values of the color of the specified line
	 *
	 * @param line The index of the line (its row number)
	 * @param r    The R value that should be set
	 * @param g    The G value that should be set
	 * @param b    The B value that should be set
	 */
	void line_set_rgb(PVRow line, unsigned char r, unsigned char g, unsigned char b);

	/**
	 * Sets the R,G,B,A values of the color of the specified line
	 *
	 * @param line The index of the line (its row number)
	 * @param r    The R value that should be set
	 * @param g    The G value that should be set
	 * @param b    The B value that should be set
	 * @param a    The A value that should be set
	 */
	void line_set_rgba(PVRow line, unsigned char r, unsigned char g, unsigned char b, unsigned char a);

	/**
	 * Sets the R,G,B values of the specified line from a PVCore::PVColor
	 *
	 * @param line  The index of the line (its row number)
	 * @param color The PVCore::PVColor used to set R,G,B
	 */
	void line_set_rgb_from_color(PVRow line, PVCore::PVColor color);

	/**
	 * Sets the R,G,B,A values of the specified line from a PVCore::PVColor
	 *
	 * @param line  The index of the line (its row number)
	 * @param color The PVCore::PVColor used to set R,G,B,A
	 */
	void line_set_rgba_from_color(PVRow line, const PVCore::PVColor &color);

	PVLinesProperties & operator=(const PVLinesProperties & rhs);

	void A2A_set_to_line_properties_restricted_by_selection_and_nelts(PVCore::PVColor color, PVSelection const& selection, PVRow nelts);

	/* void picviz_lines_properties_A2B_copy(picviz_lines_properties_t *b); */ //* It is replaced by =
	void A2B_copy_restricted_by_selection_and_nelts(PVLinesProperties &b, PVSelection const& selection, PVRow nelts);
	void A2B_copy_zombie_off_restricted_by_selection_and_nelts(PVLinesProperties &b,  PVSelection const& selection, PVRow nelts);
	void A2B_copy_zombie_on_restricted_by_selection_and_nelts(PVLinesProperties &b,  PVSelection const& selection, PVRow nelts);
	void reset_to_default_color();
	void selection_set_rgba(PVSelection const& selection, PVRow nelts, unsigned char r, unsigned char g, unsigned char b, unsigned char a);

	void debug();

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
};

}

#endif /* PICVIZ_PVLINESPROPERTIES_H */
