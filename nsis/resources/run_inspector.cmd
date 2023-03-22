@echo off

setlocal EnableDelayedExpansion
setlocal EnableExtensions

@REM Check if WSLg is available
set wslg_cmd=powershell.exe -Command "$prev = [Console]::OutputEncoding; [Console]::OutputEncoding = [System.Text.Encoding]::Unicode;  wsl -v | Select-String 'WSLg : (.*)' | %% { $_.matches.groups[1].value }; [Console]::OutputEncoding = $prev"
for /F "tokens=*" %%i in ('%wslg_cmd%') do set WSLG_VERSION=%%i

set arg_count=0
for %%x in (%*) do (
   set /A arg_count+=1
   set "arg_vec[!arg_count!]=%%~x"
)
if not %arg_count% == 1 (
    echo "flatpak package name must be passed in parameter"
    exit
)

@REM configure paths
set inspector_path=%~dp0
set inspector_path=%inspector_path:~0,-1%
set path_cmd=wsl wslpath -a "%inspector_path%"
for /F "tokens=*" %%i in ('%path_cmd%') do set inspector_path_linux=%%i
set inspector_path_linux=%inspector_path_linux: =\ %
set appdata_cmd=wsl wslpath -a "%APPDATA%"
for /F "tokens=*" %%i in ('%appdata_cmd%') do set appdata_path_linux=%%i
set appdata_path_linux=%appdata_path_linux: =\ %
set userprofile_cmd=wsl wslpath -u "%userprofile%"
for /F "tokens=*" %%i in ('%userprofile_cmd%') do set userprofile_path=%%i
set userprofile_path=%userprofile_path: =\ %

set stop_vcxsrv=false
If [%WSLG_VERSION%]==[] (
    @REM Run VcXsrv if not already running
    reg add "HKCU\Software\Microsoft\Windows NT\CurrentVersion\AppCompatFlags\Layers" /v "%inspector_path%\VcXsrv\vcxsrv.exe" /t REG_SZ /d "~ HIGHDPIAWARE" /f > nul 2>&1
    tasklist /fi "imagename eq vcxsrv.exe" | find "vcxsrv.exe" > nul 2>&1
    if %errorlevel% == 1 (
        start "VcXsrv windows xserver.exe" "%inspector_path%\VcXsrv\vcxsrv.exe" :0 -ac -terminate -lesspointer -multiwindow -multimonitors -clipboard -dpi auto -notrayicon -swrastwgl 
        set stop_vcxsrv=true
    )
    set display_cmd=wsl -d inspector_linux --exec sh -c "echo -n `cat /etc/resolv.conf | grep nameserver | awk '{print $2; exit;}'`:0.0"
    for /F "tokens=*" %%i in ('%display_cmd%') do set display=%%i
    set display=%display: =\ %
    set DISPLAY_CONFIG=DISPLAY=%display%
)

@REM Run Inspector
wsl -d inspector_linux --user inendi --exec sh -c "%inspector_path_linux%/setup_config_dir.sh %appdata_path_linux%; flatpak run --user --device=shm --allow=devel --env='WSL_USERPROFILE=%userprofile_path%' --env='QTWEBENGINE_CHROMIUM_FLAGS=--disable-dev-shm-usage' --command=bash %1 -c '%DISPLAY_CONFIG% /app/bin/inendi-inspector'"

@REM Stop VcXsrv if needed
set instance_count_cmd="tasklist /FI "imagename eq inendi-inspector" 2>nul | find /I /C "inendi-inspector""
for /F "tokens=*" %%i in ('%instance_count_cmd%') do set instance_count=%%i
if [%WSLG_VERSION%]==[] if %stop_vcxsrv% == true if %instance_count% == 0 (
    taskkill /f /im "vcxsrv.exe" >nul 2>&1
    reg delete "HKCU\Software\Microsoft\Windows NT\CurrentVersion\AppCompatFlags\Layers" /v "%inspector_path%\VcXsrv\vcxsrv.exe" /f >nul 2>&1
)

@REM Update WSL and Inspector
wsl -d inspector_linux --user root --exec sh -c "%inspector_path_linux%/update.sh %1"
