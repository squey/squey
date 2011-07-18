//! \file PVMappingFunction.h
//! $Id: PVMappingFunction.h 2875 2011-05-19 04:18:05Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVMAPPINGFUNCTION_H
#define PICVIZ_PVMAPPINGFUNCTION_H

#include <QLibrary>
#include <QString>

#include <pvcore/general.h>
#include <picviz/PVPtrObjects.h>

#ifndef picviz_mapping_init_func_string
    #define picviz_mapping_init_func_string "picviz_mapping_init"
#endif
#ifndef picviz_mapping_terminate_func_string
    #define picviz_mapping_terminate_func_string "picviz_mapping_terminate"
#endif
#ifndef picviz_mapping_exec_func_string
    #define picviz_mapping_exec_func_string "picviz_mapping_exec"
#endif
#ifndef picviz_mapping_test_func_string
    #define picviz_mapping_test_func_string "picviz_mapping_test"
#endif

namespace Picviz {

typedef int (*mapping_init_func)();
typedef int (*mapping_terminate_func)();
typedef float (*mapping_exec_func)(const PVMapping_p mapping, PVCol index, QString &value, void *userdata, bool is_first);
typedef int (*mapping_test_func)(void); /* This function is used to perform all the different tests in CTest (/tests) */


/**
 * \class PVMappingFunction
 */
class LibPicvizDecl PVMappingFunction {
public:
	PVMappingFunction(QString filename);
	~PVMappingFunction();

	QLibrary *lib;

	QString type;
	QString mode;

	void *userdata;

	mapping_init_func      init_func;
	mapping_terminate_func terminate_func;
	mapping_exec_func      exec_func;
	mapping_test_func      test_func;
};
}

#endif	/* PICVIZ_PVMAPPINGFUNCTION_H */
