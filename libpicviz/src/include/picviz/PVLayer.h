//! \file PVLayer.h
//! $Id: PVLayer.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLAYER_H
#define PICVIZ_PVLAYER_H

#include <QtCore>

#include <pvcore/general.h>

#include <picviz/PVLinesProperties.h>
#include <picviz/PVSelection.h>


#define PICVIZ_LAYER_NAME_MAXLEN 1000

namespace Picviz {

/**
 * \class PVLayer
 */
class LibPicvizDecl PVLayer {
private:
	int                index;
	PVLinesProperties  lines_properties;
	bool               locked;
	QString            name;
	PVSelection        selection;
	bool               visible;
	
public:

	/**
	 * Constructor
	 */
// 	PVLayer(const QString & name_);
	PVLayer(const QString & name_, const PVSelection & sel_ = PVSelection(), const PVLinesProperties & lp_ = PVLinesProperties());


	int get_index() const {return index;}
	const PVLinesProperties& get_lines_properties() const {return lines_properties;}
	PVLinesProperties& get_lines_properties() {return lines_properties;}
	bool get_locked() const {return locked;}
	const QString & get_name() const {return name;}
	const PVSelection & get_selection() const {return selection;}
	PVSelection& get_selection() {return selection;}
	bool get_visible() const {return visible;}

	PVLayer & operator=(const PVLayer & rhs);
	
	void reset_to_empty_and_default_color();
	void reset_to_full_and_default_color();

	void set_index(int index_) {index = index_;}
	void set_lines_properties(const PVLinesProperties & lines_properties_);
	void set_locked(bool locked_) {locked = locked_;}
	void set_name(const QString & name_) {name = name_; name.truncate(PICVIZ_LAYER_NAME_MAXLEN);}
	void set_selection(const PVSelection & selection_);
	void set_visible(bool visible_) {visible = visible_;}

};
}

#endif /* PICVIZ_PVLAYER_H */



/*



enum _picviz_layer_mode_t {
	PICVIZ_LAYER_NORMAL,
	PICVIZ_LAYER_DIFFERENCE,
	PICVIZ_LAYER_ADDITION,
	PICVIZ_LAYER_SUBSTRACT,
};
typedef enum _picviz_layer_mode_t picviz_layer_mode_t;

*/
