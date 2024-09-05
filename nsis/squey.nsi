;
; MIT License
;
; © ESI Group, 2015
;
; Permission is hereby granted, free of charge, to any person obtaining a copy of
; this software and associated documentation files (the "Software"), to deal in
; the Software without restriction, including without limitation the rights to
; use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
; the Software, and to permit persons to whom the Software is furnished to do so,
; subject to the following conditions:
;
; The above copyright notice and this permission notice shall be included in all
; copies or substantial portions of the Software.
;
; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
; IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
; FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
; COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
; IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
; CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

; http://nsis.sourceforge.net/ExecDos_plug-in
; http://nsis.sourceforge.net/AccessControl_plug-in
; http://nsis.sourceforge.net/KillProcDLL_plug-in

; Make the installer DPI Aware to properly render fonts
ManifestDPIAware true
Unicode true

;--------------------------------
;       Define constants
;--------------------------------
    ; Customizable values
	!define DISPLAY_NAME "Squey"
	!define PRODUCT_NAME "squey"
	!define FLATPAK_PACKAGE_NAME "org.squey.Squey"
	!define FLATPAKREF_URL "https://dl.flathub.org/repo/appstream/org.squey.Squey.flatpakref"

    !define INTERNAL_NAME "Squey"
    !define NAME "Squey"
    !define OUTFILE "${PRODUCT_NAME}_installer.exe"
	!define UNINSTALL_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
	
	; Linux
	!define ALPINE_LINUX_LATEST_STABLE_URL "https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64"
	!define ALPINE_LINUX_LATEST_RELEASES_FILENAME "latest-releases.yaml"
	!define WSL_DISTRO_NAME "squey_linux"
	
	!define /date FILE_VERSION "%Y.%m.%d.%H"
	!define /date PRODUCT_VERSION "%Y-%m-%d %H:%M:%S"
	!define /date YEAR "%Y"
    VIProductVersion "${FILE_VERSION}"
    VIAddVersionKey ProductName "${INTERNAL_NAME}"
    VIAddVersionKey FileVersion "${FILE_VERSION}"
    VIAddVersionKey ProductVersion "${PRODUCT_VERSION}"
    VIAddVersionKey LegalCopyright "(C) Squey <contact@squey.org> ${YEAR}"
	
;--------------------------------
;            General
;--------------------------------

	; Request application privileges
	RequestExecutionLevel user

	; Name and file
	Name "${DISPLAY_NAME}"
	OutFile ${OUTFILE}
	SetCompressor /SOLID lzma

	; Default installation folder
	InstallDir "$LocalAppData\Programs\${NAME}"

	; Override installation folder from registry if available
	InstallDirRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${INTERNAL_NAME}" ""
	
	; Include logic instructions
	!include LogicLib.nsh
	!include nsDialogs.nsh
    !include x64.nsh
	!include WinVer.nsh
	!include FileFunc.nsh
	
;--------------------------------
;       Interface Settings
;--------------------------------

	;Include Modern UI
	!include "MUI2.nsh"
	!define MUI_ABORTWARNING
	!define MUI_HEADERIMAGE
	!define MUI_HEADERIMAGE_BITMAP "resources\${PRODUCT_NAME}.png"
	  
	!define MUI_ICON "resources\${PRODUCT_NAME}.ico"
	!define MUI_UNICON "resources\${PRODUCT_NAME}.ico"
	
;--------------------------------
;              Pages
;--------------------------------
	!insertmacro MUI_PAGE_LICENSE "resources\LICENSE.txt"
	!insertmacro MUI_PAGE_COMPONENTS
	!insertmacro MUI_PAGE_DIRECTORY
	!insertmacro MUI_PAGE_INSTFILES

	# These indented statements modify settings for MUI_PAGE_FINISH
	!define MUI_FINISHPAGE_NOAUTOCLOSE
	!define MUI_FINISHPAGE_RUN
	!define MUI_FINISHPAGE_RUN_TEXT "Run ${DISPLAY_NAME}"
	!define MUI_FINISHPAGE_RUN_FUNCTION "RunSquey"
	!insertmacro MUI_PAGE_FINISH

	!insertmacro MUI_UNPAGE_CONFIRM
	UninstPage custom un.CustomUninstallerPage un.LeaveCustomPage
	!insertmacro MUI_UNPAGE_INSTFILES
	!insertmacro MUI_UNPAGE_FINISH
  
;--------------------------------
;            Languages
;--------------------------------
 
  !insertmacro MUI_LANGUAGE "English"
  
;--------------------------------
;        ConfigureUninstaller
;--------------------------------

Function ConfigureUninstaller
   	WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${INTERNAL_NAME}" "" "$INSTDIR"
	WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${INTERNAL_NAME}" "DisplayName" "${DISPLAY_NAME}"
	WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${INTERNAL_NAME}" "UninstallString" "$INSTDIR\uninstall.exe"
	WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${INTERNAL_NAME}" "DisplayIcon" "$INSTDIR\${PRODUCT_NAME}.ico"
	WriteUninstaller "$INSTDIR\uninstall.exe"
FunctionEnd

;--------------------------------
;        Squey
;--------------------------------

Function InstallSquey
	; Extract files
	File "resources\7z.exe"
	File "resources\7z.dll"
	File "resources\yq.exe"
	File "resources\install_squey.sh"
	File "resources\setup_config_dir.sh"
	File "resources\run_squey.cmd"
	File "resources\update.sh"

    ; Download Alpine Linux for WSL
	var /GLOBAL ALPINE_LINUX_FILENAME
	nsExec::ExecToStack `$WINDIR\SysNative\WindowsPowerShell\v1.0\powershell.exe -Command curl.exe -s -k ${ALPINE_LINUX_LATEST_STABLE_URL}/${ALPINE_LINUX_LATEST_RELEASES_FILENAME} | .\yq.exe -M '.[] | select(.title == \"\"\"Mini root filesystem\"\"\") .file'`
	Pop $0
	Pop $ALPINE_LINUX_FILENAME
	Strcpy $ALPINE_LINUX_FILENAME $ALPINE_LINUX_FILENAME -1 ; Remove LF
	nsExec::ExecToStack `curl.exe -s -k ${ALPINE_LINUX_LATEST_STABLE_URL}/$ALPINE_LINUX_FILENAME -w %{http_code} -o $ALPINE_LINUX_FILENAME`
	Pop $0
	Pop $1
	${If} $1 != "200"
		MessageBox MB_OK|MB_ICONSTOP "Download failed"
		Delete "$ALPINE_LINUX_FILENAME"
		Abort
	${EndIf}
	Delete yq.exe

	; Install Linux for WSL
    DetailPrint "Preparing WSL..."
	nsExec::ExecToLog '$WINDIR\SysNative\cmd.exe /C wsl --import ${WSL_DISTRO_NAME} linux $ALPINE_LINUX_FILENAME --version 2'
	Delete "$ALPINE_LINUX_FILENAME"
	
	; Install Squey
	DetailPrint "Installing Squey..."
	nsExec::ExecToLog '$WINDIR\SysNative\cmd.exe /C wsl --user root -d squey_linux --exec sh -c "./install_squey.sh ${FLATPAKREF_URL}"'
	Pop $0
	Delete "install_squey.sh"
	${If} $0 != "0"
		MessageBox MB_OK|MB_ICONSTOP "Installation failed.$\r$\n$\r$\nPlease click $\"Show details$\" button to display more information."
		Abort
	${EndIf}
	
FunctionEnd

;--------------------------------
;        Section Squey
;--------------------------------

Section
	SetRegView 64
	AddSize 2529285
	
	Call ConfigureUninstaller
	Call InstallSquey
SectionEnd

Section "Start Menu Shortcut" SecStartMenuShortcut
	; Create start menu shortcut
	CreateShortCut "$SMPROGRAMS\${DISPLAY_NAME}.lnk" "powershell.exe" '-WindowStyle Hidden -Command ".\run_squey.cmd" "${FLATPAK_PACKAGE_NAME}"' "$INSTDIR\${PRODUCT_NAME}.ico" 0 SW_SHOWMINIMIZED
SectionEnd

Section "Desktop Shortcut" SecDesktopShortcut
    ; Create desktop shortcut
	CreateShortCut "$DESKTOP\${DISPLAY_NAME}.lnk" "powershell.exe" '-WindowStyle Hidden -Command ".\run_squey.cmd" "${FLATPAK_PACKAGE_NAME}"' "$INSTDIR\${PRODUCT_NAME}.ico" 0 SW_SHOWMINIMIZED
SectionEnd

;--------------------------------
;         Initialization
;--------------------------------

Function .onInit
	SetRegView 64
	
	SetOutPath "$INSTDIR"
	
	; Check if program is already installed
	ReadRegStr $0 HKCU "${UNINSTALL_KEY}" "UninstallString"
	StrCmp $0 "" not_installed installed
installed:
    MessageBox MB_OKCANCEL|MB_ICONINFORMATION "${DISPLAY_NAME} is already installed and should be uninstalled first." IDOK uninst_prog
    Abort
not_installed:
    Return
uninst_prog:
    ExecWait '$0 _?=$INSTDIR'

	; Check windows version
	${WinVerGetBuild} $0
	${IfNot} ${AtLeastWin10}
	${OrIf} $0 < 19041
		IfSilent +2
		MessageBox MB_OK|MB_ICONSTOP "Your OS needs to be one of the following (or newer) to support WSL2 : $\r$\n > Windows 10 64 bits version 2004$\r$\n > Windows Server 2019"
		SetErrorlevel 1000
		Abort
	${EndIf}

	# Check if WSL needs to be enabled
	nsExec::ExecToStack `$WINDIR\SysNative\wsl.exe --status`
	Pop $0
	${If} $0 != 0
		IfSilent +3
		MessageBox MB_OK|MB_ICONSTOP "Microsoft WSL2 feature is required to run this software."
		ExecShell open "https://docs.microsoft.com/windows/wsl/install"
		SetErrorlevel 1001
		Abort
	${EndIf}

	; Check if WSLg is available
	var /GLOBAL WSLG_VERSION
	nsExec::ExecToStack `$WINDIR\SysNative\WindowsPowerShell\v1.0\powershell.exe -Command "[Console]::OutputEncoding = [System.Text.Encoding]::Unicode;  wsl -v | Select-String 'WSLg' | ForEach-Object { $_.Line } | Select-String ': (.*)' | % { $$($$_.matches.groups[1].value) }"`
	Pop $0
	Pop $WSLG_VERSION
	Strcpy $WSLG_VERSION $WSLG_VERSION -2 ; Remove CRLF
	${If} $WSLG_VERSION == ""
		IfSilent cancel
		MessageBox MB_YESNO|MB_ICONEXCLAMATION "Microsoft WSLg is not supported by your system.$\n$\nIt is recommended to install WSL2 from Microsoft Store to benefit from WSLg.$\n$\nDo you want to install WSL2 from Microsoft store ?" IDYES true IDNO false
		true:
			ExecShell open "https://aka.ms/wslstorepage"
		false:
			cancel:
			SetErrorlevel 1002
			Abort
	${Endif}
	
	File "resources\${PRODUCT_NAME}.ico"

FunctionEnd

;--------------------------------
;           Run application after installation
;--------------------------------

Function RunSquey
  SetOutPath "$INSTDIR"
  
  ExecDos::exec /ASYNC '$WINDIR\SysNative\cmd.exe /C ""$INSTDIR\run_squey.cmd" ${FLATPAK_PACKAGE_NAME}"'
  
FunctionEnd

;--------------------------------
;           Descriptions
;--------------------------------

  ;Language strings
  LangString DESC_StartMenuShortcut ${LANG_ENGLISH} "Start Menu Shortcut"
  LangString DESC_DesktopShortcut ${LANG_ENGLISH} "Desktop Shortcut"
  

  ;Assign language strings to sections
  !insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
	!insertmacro MUI_DESCRIPTION_TEXT ${SecStartMenuShortcut} $(DESC_StartMenuShortcut)
	!insertmacro MUI_DESCRIPTION_TEXT ${SecDesktopShortcut} $(DESC_DesktopShortcut)
  !insertmacro MUI_FUNCTION_DESCRIPTION_END

  
;--------------------------------
;       Uninstaller Section
;--------------------------------

Function un.onInit
	; Ask for confirmation
	IfSilent +3
	MessageBox MB_OKCANCEL|MB_ICONQUESTION "Permanantly remove ${DISPLAY_NAME}?" IDOK +2
		Abort
FunctionEnd

Section "un.Uninstaller Section"

	SetRegView 64

	; Kill running instances of squey
	KillProcDLL::KillProc "squey"
	
	; Unregister WSL distro
	nsExec::ExecToLog '$WINDIR\SysNative\cmd.exe /C wsl --unregister ${WSL_DISTRO_NAME}'
	
    ; Remove installation directory
	SetOutPath $TEMP ; Because we cannot remove current working directory
    RMDir /r "$INSTDIR"
	ExecDos::exec 'rmdir /S /Q $INSTDIR'
	
	; Remove start menu & desktop shortcuts
	
	Delete "$SMPROGRAMS\${DISPLAY_NAME}.lnk"
	Delete "$DESKTOP\${DISPLAY_NAME}.lnk"

    ; Delete registry key
	DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${INTERNAL_NAME}"

SectionEnd

Var CheckboxConfig

Function un.CustomUninstallerPage
	IfSilent skip_config_files_section
	nsDialogs::Create 1018
	Pop $0
	
	${NSD_CreateCheckbox} 35 35 100% 8u "Keep configuration files"
	Pop $CheckboxConfig
	${NSD_SetState} $CheckboxConfig ${BST_CHECKED}

	nsDialogs::Show
	skip_config_files_section:
FunctionEnd

Function un.LeaveCustomPage
	SetRegView 64
	IfSilent delete_config_files

	${NSD_GetState} $CheckboxConfig $0
	${If} $0 == ${BST_UNCHECKED}
		delete_config_files:
		nsExec::ExecToStack '$WINDIR\SysNative\cmd.exe /C echo %APPDATA%'
		Pop $0
		Pop $1
		Strcpy $1 $1 -2 ; Remove CRLF
		RMDir /r "$1\Squey"
	${EndIf}

FunctionEnd
