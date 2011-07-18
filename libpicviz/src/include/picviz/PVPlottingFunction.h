//! \file PVPlottingFunction.h
//! $Id: PVPlottingFunction.h 2875 2011-05-19 04:18:05Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVPLOTTINGFUNCTION_H
#define PICVIZ_PVPLOTTINGFUNCTION_H

#include <QLibrary>
#include <QString>

#include <pvcore/general.h>
#include <picviz/PVPtrObjects.h>

#ifndef picviz_plotting_init_func_string
    #define picviz_plotting_init_func_string "picviz_plotting_init"
#endif
#ifndef picviz_plotting_terminate_func_string
    #define picviz_plotting_terminate_func_string "picviz_plotting_terminate"
#endif
#ifndef picviz_plotting_exec_func_string
    #define picviz_plotting_exec_func_string "picviz_plotting_exec"
#endif
#ifndef picviz_plotting_test_func_string
    #define picviz_plotting_test_func_string "picviz_plotting_test"
#endif

namespace Picviz {
class PVPlotting;
typedef int (*plotting_init_func)();
typedef int (*plotting_terminate_func)();
typedef float (*plotting_exec_func)(const PVPlotting_p plotting, PVCol index, float value, void *userdata, bool is_first);
typedef int (*plotting_test_func)(void); /* This function is used to perform all the different tests in CTest (/tests) */


/**
 * \class PVPlottingFunction
 */
class LibPicvizDecl PVPlottingFunction {
public:
	PVPlottingFunction(QString filename);
	~PVPlottingFunction();

	QLibrary *lib;

	QString type;
	QString mode;

	void *userdata;

	plotting_init_func      init_func;
	plotting_terminate_func terminate_func;
	plotting_exec_func      exec_func;
	plotting_test_func      test_func;
};
}

#endif	/* PICVIZ_PVPLOTTINGFUNCTION_H */
