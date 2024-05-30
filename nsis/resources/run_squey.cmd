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
set squey_path=%~dp0
set squey_path=%squey_path:~0,-1%
set path_cmd=wsl wslpath -a "%squey_path%"
for /F "tokens=*" %%i in ('%path_cmd%') do set squey_path_linux=%%i
set squey_path_linux=%squey_path_linux: =\ %
set appdata_cmd=wsl wslpath -a "%APPDATA%"
for /F "tokens=*" %%i in ('%appdata_cmd%') do set appdata_path_linux=%%i
set appdata_path_linux=%appdata_path_linux: =\ %
set userprofile_cmd=wsl wslpath -u "%userprofile%"
for /F "tokens=*" %%i in ('%userprofile_cmd%') do set userprofile_path=%%i
set userprofile_path=%userprofile_path: =\ %

set stop_vcxsrv=false
If [%WSLG_VERSION%]==[] (
    @REM Run VcXsrv if not already running
    reg add "HKCU\Software\Microsoft\Windows NT\CurrentVersion\AppCompatFlags\Layers" /v "%squey_path%\VcXsrv\vcxsrv.exe" /t REG_SZ /d "~ HIGHDPIAWARE" /f > nul 2>&1
    tasklist /fi "imagename eq vcxsrv.exe" | find "vcxsrv.exe" > nul 2>&1
    if %errorlevel% == 1 (
        start "VcXsrv windows xserver.exe" "%squey_path%\VcXsrv\vcxsrv.exe" :0 -ac -terminate -lesspointer -multiwindow -multimonitors -clipboard -dpi auto -notrayicon -swrastwgl 
        set stop_vcxsrv=true
    )
    set display_cmd=wsl -d squey_linux --exec sh -c "echo -n `cat /etc/resolv.conf | grep nameserver | awk '{print $2; exit;}'`:0.0"
    for /F "tokens=*" %%i in ('%display_cmd%') do set display=%%i
    set display=%display: =\ %
    set DISPLAY_CONFIG=DISPLAY=%display%
)

@REM configure monitors scaling factor
set index=0
for /f "usebackq" %%i in (`wmic path Win32_PnPEntity where "Service='monitor' and Status='OK'" get PNPDeviceID /format:value ^| findstr /r /v "^$"`) do (
	set pnp_device_ids[!index!]=%%i
	set /A index+=1
)
set /A end=index-1
set index=0
for /F "usebackq" %%i in (`wmic path Win32_DesktopMonitor get PNPDeviceID /format:value^| findstr /r /v "^$"`) do (
	call:iterate_pnp_device_ids "%%i" !index!
	set /A index+=1
)
goto:compute_index_mapping_end
:iterate_pnp_device_ids
	set pnp_device_id=%1
	set index=%2
	FOR /L %%G IN (0,1,%end%) DO (
		call:compute_index_mapping %pnp_device_id% %index% %%G
	)
	goto:eof
:iterate_pnp_device_ids_end
:compute_index_mapping
	set pdi1=%1
	set index=%2
	set current_index=%3
	set pdi2="!pnp_device_ids[%current_index%]!"
	if [%pdi1%] == [%pdi2%] (
		set index_mapping[%current_index%]=%index%
	)
	goto:eof
:compute_index_mapping_end
set index=0
for /F "usebackq" %%i in (`wmic path Win32_DesktopMonitor get ScreenWidth /value^| findstr /r /v "^$"`) do (
	for /f "tokens=1,2 delims==" %%a in ("%%i") do (
		if [%%b]==[] (
			set resolutions[!index!]=1920.0
		) else (
			set resolutions[!index!]=%%b.0
		)
		set /A index+=1
	)
)
FOR /L %%G IN (0,1,%end%) DO (
	call:divide %%G
)
goto:divide_end
:divide
	set decimals=1
	set /A one=1, decimalsP1=decimals+1
	for /L %%i in (1,1,%decimals%) do set "one=!one!0"
	set dividend=!resolutions[%1]!
	set divider=1920.0
	set "fpdividend=%dividend:.=%"
	set "fpdivider=%divider:.=%"
	set /A div=fpdividend*one/fpdivider
	set ratios[%1]=!div:~0,-%decimals%!.!div:~-%decimals%!
	goto:eof
:divide_end
FOR /L %%G IN (0,1,%end%) DO (
	call:resolve_index_mapping %%G
)
goto:resolve_index_mapping_end
:resolve_index_mapping
	set index=%1
	set mapped_index=!index_mapping[%index%]!
	call set "QT_SCREEN_SCALE_FACTORS=%%QT_SCREEN_SCALE_FACTORS%%!ratios[%mapped_index%]!;"
	goto:eof
:resolve_index_mapping_end

echo QT_SCREEN_SCALE_FACTORS=%QT_SCREEN_SCALE_FACTORS%

@REM Configure theme
set DARK_THEME=false
for /F "tokens=3" %%A in ('Reg Query "HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize" /v "AppsUseLightTheme"') DO (set APPS_USE_LIGHT_THEME= %%A)
If ["%APPS_USE_LIGHT_THEME%"]==[" 0x0"] (
    set DARK_THEME=true
)

@REM Run Squey
wsl -d squey_linux --user squey --exec sh -c "%squey_path_linux%/setup_config_dir.sh %appdata_path_linux%; flatpak run --user --device=shm --allow=devel --env='WSL_USERPROFILE=%userprofile_path%' --env='QTWEBENGINE_CHROMIUM_FLAGS=--disable-dev-shm-usage' --env='QT_SCREEN_SCALE_FACTORS=%QT_SCREEN_SCALE_FACTORS%' --env='DARK_THEME=%DARK_THEME%' --env='DISABLE_FOLLOW_SYSTEM_THEME=true' --command=bash %1 -c '%DISPLAY_CONFIG% /app/bin/squey'"

@REM Stop VcXsrv if needed
set instance_count_cmd="tasklist /FI "imagename eq squey" 2>nul | find /I /C "squey""
for /F "tokens=*" %%i in ('%instance_count_cmd%') do set instance_count=%%i
if [%WSLG_VERSION%]==[] if %stop_vcxsrv% == true if %instance_count% == 0 (
    taskkill /f /im "vcxsrv.exe" >nul 2>&1
    reg delete "HKCU\Software\Microsoft\Windows NT\CurrentVersion\AppCompatFlags\Layers" /v "%squey_path%\VcXsrv\vcxsrv.exe" /f >nul 2>&1
)

@REM Update WSL and Squey
wsl -d squey_linux --user root --exec sh -c "%squey_path_linux%/update.sh %1"
