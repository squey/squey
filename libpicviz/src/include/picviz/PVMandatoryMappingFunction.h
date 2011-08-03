//! \file PVMandatoryMappingFunction.h
//! $Id: PVMandatoryMappingFunction.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVMANDATORYMAPPINGFUNCTION_H
#define PICVIZ_PVMANDATORYMAPPINGFUNCTION_H

#include <QLibrary>
#include <QString>

#include <pvkernel/core/general.h>
#include <picviz/PVPtrObjects.h>

#ifndef picviz_mandatory_mapping_init_func_string
    #define picviz_mandatory_mapping_init_func_string "picviz_mandatory_mapping_init"
#endif
#ifndef picviz_mandatory_mapping_terminate_func_string
    #define picviz_mandatory_mapping_terminate_func_string "picviz_mandatory_mapping_terminate"
#endif
#ifndef picviz_mandatory_mapping_exec_func_string
    #define picviz_mandatory_mapping_exec_func_string "picviz_mandatory_mapping_exec"
#endif
#ifndef picviz_mandatory_mapping_test_func_string
    #define picviz_mandatory_mapping_test_func_string "picviz_mandatory_mapping_test"
#endif

namespace Picviz {

typedef int (*mandatory_mapping_init_func)();
typedef int (*mandatory_mapping_terminate_func)();
typedef void (*mandatory_mapping_exec_func)(PVMapping_p mapping, pvrow row, pvcol col, QString &value, float mapped_pos, void *userdata, bool is_first);
typedef int (*mandatory_mapping_test_func)(void); /* This function is used to perform all the different tests in CTest (/tests) */

/**
 * \class PVMandatoryMappingFunction
 */
class LibPicvizDecl PVMandatoryMappingFunction {
public:
	PVMandatoryMappingFunction(QString filename);
	~PVMandatoryMappingFunction();

	QLibrary *lib;

	QString name;

	void *userdata;

	mandatory_mapping_init_func      init_func;
	mandatory_mapping_terminate_func terminate_func;
	mandatory_mapping_exec_func      exec_func;
	mandatory_mapping_test_func      test_func;
};
}

#endif	/* PICVIZ_PVMANDATORYMAPPINGFUNCTION_H */
