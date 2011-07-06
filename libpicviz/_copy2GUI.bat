mkdir ..\picviz-gui\picvizNG_gui\parsers
mkdir ..\picviz-gui\picvizNG_gui\normalize
mkdir ..\picviz-gui\picvizNG_gui\functions

copy src\Debug\picviz.dll ..\picviz-gui\picvizNG_gui\
copy plugins\parsers\*.pcre ..\picviz-gui\picvizNG_gui\parsers\

copy bindings\python-ctypes\cpicviz\*.py C:\dev\Python26\Lib\cpicviz\
copy bindings\python-ctypes\cpicviz\core\*.py C:\dev\Python26\Lib\cpicviz\core\

mkdir bindings\python-ctypes\parsers
mkdir bindings\python-ctypes\normalize
mkdir bindings\python-ctypes\functions

copy src\Debug\picviz.dll bindings\python-ctypes\
copy plugins\parsers\*.pcre bindings\python-ctypes\parsers\

mkdir ..\picviz-inspector\src\Debug\parsers
mkdir ..\picviz-inspector\src\Debug\normalize
mkdir ..\picviz-inspector\src\Debug\functions

copy src\Debug\picviz.dll ..\picviz-inspector\src\Debug\
copy plugins\parsers\*.pcre ..\picviz-inspector\src\Debug\parsers\
