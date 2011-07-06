//! \file plugins.h
//! $Id: plugins.h 3011 2011-05-30 11:24:42Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef _PICVIZ_PLUGINS_H_
#define _PICVIZ_PLUGINS_H_

#include <picviz/general.h>

#define PICVIZ_PLUGINSLIST_MAXSIZE 32768

LibExport char *picviz_plugins_get_functions_dir(void);
LibExport char *picviz_plugins_get_filters_dir(void);
LibExport char *picviz_plugins_get_layer_filters_dir(void);
LibExport char *picviz_plugins_get_mapping_filters_dir(void);
LibExport char *picviz_plugins_get_plotting_filters_dir(void);

#endif /* _PICVIZ_PLUGINS_H_ */
