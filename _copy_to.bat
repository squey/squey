REM
REM \file _copy_to.bat
REM
REM Copyright (C) Picviz Labs 2010-2012

@echo off

set ReleaseMode=Release

set dest=%userprofile%\Desktop\%1
:set dest="C:\Program Files (x86)\Picviz Inspector 2.0"
set qtdir=C:\Qt\4.6.3\bin
set zlibdll=C:\dev\zlib\zlib1.dll
set GnuWinDir=C:\dev\GnuWin32\bin

echo Copy to %dest%
mkdir %dest%
mkdir %dest%\filters
mkdir %dest%\functions
mkdir %dest%\normalize
mkdir %dest%\normalize-helpers
mkdir %dest%\normalize-helpers\pcre

:echo "copying QT Libraries..."
:xcopy %qtdir%\phonon4.dll %dest%
:xcopy %qtdir%\QtCore4.dll %dest%
:xcopy %qtdir%\QtGui4.dll %dest%
:xcopy %qtdir%\QtMultimedia4.dll %dest%
:xcopy %qtdir%\QtNetwork4.dll %dest%
:xcopy %qtdir%\QtOpenGL4.dll %dest%
:xcopy %qtdir%\QtScript4.dll %dest%
:xcopy %qtdir%\QtSvg4.dll %dest%
:xcopy %qtdir%\QtWebEngine5.dll %dest%
:xcopy %qtdir%\QtXml4.dll %dest%
:xcopy %qtdir%\QtXmlPatterns4.dll %dest%

:echo "copying zlib..."
:xcopy %zlibdll% %dest%

:echo "copying GnuWin32..."
:xcopy %GnuWinDir%\pcre3.dll %dest%
:xcopy %GnuWinDir%\pcreposix3.dll %dest%

echo "copying pvcore..."
xcopy libpvcore\src\%ReleaseMode%\pvcore.dll %dest%

echo "copying pvfilter..."
xcopy libpvfilter\src\%ReleaseMode%\pvfilter.dll %dest%

echo "copying pvrush..."
xcopy libpvrush\src\%ReleaseMode%\pvrush.dll %dest%

echo "copying picviz..."
xcopy libpicviz\src\%ReleaseMode%\picviz.dll %dest%
:xcopy libpicviz\plugins\filters\dshield-ipascii.txt %dest%\filters
:xcopy libpicviz\plugins\filters\RelWithDebInfo\*.dll %dest%\filters
:xcopy libpicviz\plugins\functions\RelWithDebInfo\*.dll %dest%\functions
:xcopy libpicviz\plugins\normalize\%ReleaseMode%\*.dll %dest%\normalize
:xcopy libpicviz\plugins\normalize-helpers\pcre\*.pcre %dest%\normalize-helpers\pcre

echo "copying picviz inspector"
xcopy picviz-inspector\src\%ReleaseMode%\picviz-inspector.exe %dest%
:xcopy picviz-inspector\src\icons\inspector.ico %dest%

