/* Test the Python Binding */
#include <Python.h>

int main(void)
{
        PyObject *mainmod;
        PyObject *func;
        PyObject *result;

	Py_Initialize();

        PyRun_SimpleString(
                "from cpicviz.datatree import *\n"
                "def testDataTree():\n"
                "    dt = DataTree(\"test\")\n"
                );
        mainmod = PyImport_AddModule("__main__");
        func = PyObject_GetAttrString(mainmod, "testDataTree");
        if ( ! func ) {
                fprintf(stderr, "Cannot find 'testDataTree' function!\n");
                return 1;
        }

        result = PyObject_CallFunctionObjArgs(func, NULL);
        if ( ! result) {
                fprintf(stderr, "Cannot run function!\n");
                return 1;
        }

	Py_Finalize();

	return 0;
}
