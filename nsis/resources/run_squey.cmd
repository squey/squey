@echo off

setlocal EnableDelayedExpansion
setlocal EnableExtensions

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
set userprofile_cmd=wsl wslpath -u "%userprofile%"
for /F "tokens=*" %%i in ('%userprofile_cmd%') do set userprofile_path=%%i
set userprofile_path=%userprofile_path: =\ %

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

@REM Setup config dir
pushd %cd%
mkdir "%appdata%\Squey" 2> NUL
copy setup_config_dir.sh "%appdata%\Squey" > NUL
cd "%appdata%\Squey"
wsl -d squey_linux --user squey --exec sh -c './setup_config_dir.sh'
popd

@REM Run Squey
pushd %cd%
cd "%userprofile%"
wsl -d squey_linux --user squey --exec sh -c "flatpak run --user --nosocket=wayland --device=shm --allow=devel --env='WSL_USERPROFILE=%userprofile_path%' --env='QTWEBENGINE_CHROMIUM_FLAGS=--disable-dev-shm-usage' --env='QT_SCREEN_SCALE_FACTORS=%QT_SCREEN_SCALE_FACTORS%' --env='DARK_THEME=%DARK_THEME%' --env='DISABLE_FOLLOW_SYSTEM_THEME=true' --command=bash %1 -c '%DISPLAY_CONFIG% /app/bin/squey'"

@REM Update WSL and Squey
popd
wsl -d squey_linux --user root --exec sh -c "./update.sh %1"
