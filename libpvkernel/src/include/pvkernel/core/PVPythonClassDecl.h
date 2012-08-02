/**
 * \file PVPythonClassDecl.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVPYTHONCLASSREGISTER
#define PVCORE_PVPYTHONCLASSREGISTER

#ifdef ENABLE_PYTHON_SUPPORT

namespace PVCore {

class PVPythonInitializer;

class LibKernelDecl PVPythonClassDecl
{
	friend class PVPythonInitializer;
protected:
	virtual void declare() = 0;
	virtual PVPythonClassDecl* clone() const = 0;
};

class LibKernelDecl PVPythonClassRegister
{
public:
	PVPythonClassRegister(PVPythonClassDecl const& c);
};

}

#define PYTHON_EXPOSE()\
	public:\
		struct __PythonDecl: public PVCore::PVPythonClassDecl\
		{\
			void declare();\
			PVCore::PVPythonClassDecl* clone() const { return new __PythonDecl(); }\
		};

#define PYTHON_EXPOSE_IMPL(T)\
	static PVCore::PVPythonClassRegister __python_register__ = PVCore::PVPythonClassRegister(T::__PythonDecl());\
	void T::__PythonDecl::declare()

#endif

#endif
