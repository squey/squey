@echo off

set argc=0
for %%a in (%*) do set /a argc+=1

if %argc% geq 1 goto run 
echo Syntax: %0 "Customer name"
goto :eof

:run
set customer=%1

xcopy /Y ..\obfuscate\%customer%.h libpicviz\src\include\picviz\api-obfuscate.h

rem substitute string
rem *****************
rem set orig_str=//#include <picviz/api-obfuscate.h>
rem set dest_str=#include <picviz/api-obfuscate.h>
set dest_file=.\libpicviz\src\include\picviz\general.h

sed "s/\/\/AUTO_CUSTOMER_RELEASE //" %dest_file% > local
xcopy /Y local %dest_file%
del local

devenv Picviz_Inspector.sln /Rebuild

cpack

:eof

