/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVLINESPROPERTIES_H
#define INENDI_PVLINESPROPERTIES_H

#include <QVector>

#include <inendi/general.h>

#include <pvkernel/core/PVHSVColor.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <inendi/PVSelection.h>

#define INENDI_LINESPROPS_CHUNK_SIZE sizeof(PVCore::PVHSVColor)
#define INENDI_LINESPROPS_NUMBER_OF_CHUNKS INENDI_LINES_MAX
#define INENDI_LINESPROPS_NUMBER_OF_BYTES  INENDI_LINESPROPS_NUMBER_OF_CHUNKS * INENDI_LINESPROPS_CHUNK_SIZE

namespace Inendi {

/**
 * \class PVLinesProperties
 */
class PVLinesProperties {
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
		if(_table) {
			free_table();
		}
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
	inline PVCore::PVHSVColor const* get_buffer() const { assert(_table); return _table; }
	inline PVCore::PVHSVColor& get_line_properties(const PVRow r) { assert(_table); return _table[r]; }
	inline const PVCore::PVHSVColor get_line_properties(const PVRow r) const { assert(_table); return _table[r]; }

	inline void line_set_color(const PVRow r, const PVCore::PVHSVColor h) { get_line_properties(r) = h; }

	PVLinesProperties & operator=(const PVLinesProperties & rhs);
	PVLinesProperties & operator=(PVLinesProperties && rhs)
	{
		_table = rhs._table;
		rhs._table = NULL;
		return *this;
	}

	void A2A_set_to_line_properties_restricted_by_selection_and_nelts(PVCore::PVHSVColor color, PVSelection const& selection, PVRow nelts);

	void A2B_copy_restricted_by_selection_and_nelts(PVLinesProperties &b, PVSelection const& selection, PVRow nelts);
	void A2B_copy_zombie_off_restricted_by_selection_and_nelts(PVLinesProperties &b,  PVSelection const& selection, PVRow nelts);
	void reset_to_default_color();
	void selection_set_color(PVSelection const& selection, const PVRow nelts, const PVCore::PVHSVColor c);
	void set_random(const PVRow n);
	void set_linear(const PVRow n);

private:
	inline void allocate_table()
	{
		assert(!_table);
		_table = _color_allocator.allocate(INENDI_LINESPROPS_NUMBER_OF_CHUNKS);
	}

	inline void free_table()
	{
		if (_table) {
			_color_allocator.deallocate(_table, INENDI_LINESPROPS_NUMBER_OF_CHUNKS);
		}
	}
protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

private:
	PVCore::PVHSVColor* _table;
};

}

#endif /* INENDI_PVLINESPROPERTIES_H */