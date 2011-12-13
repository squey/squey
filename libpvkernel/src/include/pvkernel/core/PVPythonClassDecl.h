#ifndef PVCORE_PVPYTHONCLASSREGISTER
#define PVCORE_PVPYTHONCLASSREGISTER

namespace PVCore {

class PVPythonInitializer;

class LibKernelDecl PVPythonClassDecl
{
	friend class PVPythonInitializer;
protected:
	virtual void declare() = 0;
	virtual PVPythonClassDecl* clone() const = 0;
};

struct LibKernelDecl PVPythonClassRegister
{
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
	static PVCore::PVPythonClassRegister __python_register__(T::__PythonDecl());\
	void T::__PythonDecl::declare()

#endif
