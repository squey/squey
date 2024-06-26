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

; http://nsis.sourceforge.net/Inetc_plug-in
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
	
	; VcXsrv
	!define VCXSRV_MAIN_REG_KEY "SOFTWARE\VcXsrv"
	!define VCXSRV_REG_KEY "Install_Dir_64"
	!define VCXSRV_HTTP_LINK "https://sourceforge.net/projects/vcxsrv/files/latest/download"
	!define VCXSRV_SETUP "vcxsrv_installer.exe"
	
	; Linux
	!define ALPINE_LINUX_LATEST_STABLE_URL "https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64"
	!define ALPINE_LINUX_LATEST_RELEASES_FILENAME "latest-releases.yaml"
	!define WSL_DISTRO_NAME "squey_linux"
	
	!define StrRep "!insertmacro StrRep"
	
	!define /date FILE_VERSION "%Y.%m.%d.%H"
	!define /date PRODUCT_VERSION "%Y-%m-%d %H:%M:%S"
	!define /date YEAR "%Y"
    VIProductVersion "${FILE_VERSION}"
    VIAddVersionKey ProductName "${INTERNAL_NAME}"
    VIAddVersionKey FileVersion "${FILE_VERSION}"
    VIAddVersionKey ProductVersion "${PRODUCT_VERSION}"
    VIAddVersionKey LegalCopyright "(C) Squey <contact@squey.org> ${YEAR}"

!macro StrRep output string old new
    Push `${string}`
    Push `${old}`
    Push `${new}`
    !ifdef __UNINSTALL__
        Call un.StrRep
    !else
        Call StrRep
    !endif
    Pop ${output}
!macroend
 
!macro Func_StrRep un
    Function ${un}StrRep
        Exch $R2 ;new
        Exch 1
        Exch $R1 ;old
        Exch 2
        Exch $R0 ;string
        Push $R3
        Push $R4
        Push $R5
        Push $R6
        Push $R7
        Push $R8
        Push $R9
 
        StrCpy $R3 0
        StrLen $R4 $R1
        StrLen $R6 $R0
        StrLen $R9 $R2
        loop:
            StrCpy $R5 $R0 $R4 $R3
            StrCmp $R5 $R1 found
            StrCmp $R3 $R6 done
            IntOp $R3 $R3 + 1 ;move offset by 1 to check the next character
            Goto loop
        found:
            StrCpy $R5 $R0 $R3
            IntOp $R8 $R3 + $R4
            StrCpy $R7 $R0 "" $R8
            StrCpy $R0 $R5$R2$R7
            StrLen $R6 $R0
            IntOp $R3 $R3 + $R9 ;move offset by length of the replacement string
            Goto loop
        done:
 
        Pop $R9
        Pop $R8
        Pop $R7
        Pop $R6
        Pop $R5
        Pop $R4
        Pop $R3
        Push $R0
        Push $R1
        Pop $R0
        Pop $R1
        Pop $R0
        Pop $R2
        Exch $R1
    FunctionEnd
!macroend
!insertmacro Func_StrRep ""
!insertmacro Func_StrRep "un."
	
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
	!include "FileFunc.nsh"
	
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
;        VcXsrv
;--------------------------------
Function InstallVcXsrv

	SetOutPath "$INSTDIR"

	; Download and install VcXSrv if WSLg is not available
	${If} $WSLG_VERSION == ""
		; Get final installer URL from public latest-version installer URL (as 'inetc::get' doesn't handle redirections)
		nsExec::ExecToStack "curl.exe --insecure ${VCXSRV_HTTP_LINK} -o NUL -s -L -I -w %{url_effective}"
		Pop $0
		Pop $1

		; Download VcXSrv installer
		inetc::get "$1" "${VCXSRV_SETUP}"
		Pop $R0
		${If} $R0 != "OK"
			MessageBox MB_OK "Download failed: $R0"
			Delete "${VCXSRV_SETUP}"
			Abort
		${EndIf}
		
		CreateDirectory "$INSTDIR\VcXsrv"
		Rename ${VCXSRV_SETUP} "VcXsrv\${VCXSRV_SETUP}"
		nsExec::ExecToLog '$WINDIR\SysNative\cmd.exe /C cd VcXsrv && ..\7z.exe -aoa x "${VCXSRV_SETUP}"'
		Delete 7z.exe
		Delete 7z.dll
		
		Delete "${VCXSRV_SETUP}"
	${EndIf}
		
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
	File "resources\hideexec.exe" ; http://code.kliu.org/misc/hideexec/

    ; Download Alpine Linux for WSL
	var /GLOBAL ALPINE_LINUX_FILENAME
	nsExec::ExecToStack `$WINDIR\SysNative\WindowsPowerShell\v1.0\powershell.exe -Command .\curl.exe -s -k ${ALPINE_LINUX_LATEST_STABLE_URL}/${ALPINE_LINUX_LATEST_RELEASES_FILENAME} | .\yq.exe -M '.[] | select(.title == \"\"\"Mini root filesystem\"\"\") .file'`
	Pop $0
	Pop $ALPINE_LINUX_FILENAME
	Strcpy $ALPINE_LINUX_FILENAME $ALPINE_LINUX_FILENAME -1 ; Remove LF
	inetc::get "${ALPINE_LINUX_LATEST_STABLE_URL}/$ALPINE_LINUX_FILENAME" "$ALPINE_LINUX_FILENAME"
	Pop $R0
	${If} $R0 != "OK"
		MessageBox MB_OK "Download failed: $R0"
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
	nsExec::ExecToStack '$WINDIR\SysNative\cmd.exe /C wsl wslpath -a "$INSTDIR"'
	Pop $0
	Pop $1
	Strcpy $1 $1 -1 ; Remove CRLF in the end
	${StrRep} "$1" "$1" " " "\ " ; Escape spaces
	nsExec::ExecToLog '$WINDIR\SysNative\cmd.exe /C wsl --user root -d squey_linux --exec sh -c "$1/install_squey.sh ${FLATPAKREF_URL}"'
	Delete "install_squey.sh"
FunctionEnd

;--------------------------------
;        Section Squey
;--------------------------------
Section
	SetRegView 64
	AddSize 2529285
	
	Call ConfigureUninstaller
	Call InstallVcXsrv
	Call InstallSquey
SectionEnd

Section "Start Menu Shortcut" SecStartMenuShortcut
	; Create start menu shortcut
	SetShellVarContext current
	CreateShortCut "$SMPROGRAMS\${DISPLAY_NAME}.lnk" "$INSTDIR\hideexec.exe" '"$INSTDIR\run_squey.cmd" "${FLATPAK_PACKAGE_NAME}"' "$INSTDIR\${PRODUCT_NAME}.ico"
SectionEnd

Section "Desktop Shortcut" SecDesktopShortcut
    ; Create desktop shortcut
	SetShellVarContext current
	CreateShortCut "$DESKTOP\${DISPLAY_NAME}.lnk" "$INSTDIR\hideexec.exe" '"$INSTDIR\run_squey.cmd" "${FLATPAK_PACKAGE_NAME}"' "$INSTDIR\${PRODUCT_NAME}.ico"
SectionEnd

;--------------------------------
;         Initialization
;--------------------------------
Function .onInit
	SetRegView 64
	
	SetOutPath "$INSTDIR"

	File "resources\${PRODUCT_NAME}.ico"

	; Copy installer and exit if installing from Microsoft Store
   	${GetParameters} $R0
	${If} $R0 == "/S /N"
		; Copy installer
		CopyFiles "$ExePath" "$InstDir\"

		; Create shortcut to installer
		SetShellVarContext current
		CreateShortCut "$SMPROGRAMS\${DISPLAY_NAME}.lnk" "$InstDir\$ExeFile" "$INSTDIR\${PRODUCT_NAME}.ico"
		Quit
	${EndIf}

	File "resources\curl.exe"

	; Check windows version
	${WinVerGetBuild} $0
	${IfNot} ${AtLeastWin10}
	${OrIf} $0 < 19041
		IfSilent +2
		MessageBox MB_OK|MB_ICONEXCLAMATION "Your OS needs to be one of the following (or newer) to support WSL2 : $\r$\n > Windows 10 64 bits version 2004$\r$\n > Windows Server 2019"
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
	nsExec::ExecToStack `$WINDIR\SysNative\WindowsPowerShell\v1.0\powershell.exe -Command "[Console]::OutputEncoding = [System.Text.Encoding]::Unicode;  wsl -v | Select-String 'WSLg : (.*)' | % { $$($$_.matches.groups[1].value) }"`
	Pop $0
	Pop $WSLG_VERSION
	Strcpy $WSLG_VERSION $WSLG_VERSION -2 ; Remove CRLF
	no_wslg:
	${If} $WSLG_VERSION == ""
		IfSilent cancel
		MessageBox MB_YESNO|MB_ICONEXCLAMATION "Microsoft WSLg is not supported by your system.$\n$\nIt is recommended to install WSL2 from Microsoft Store to benefit from WSLg.$\n$\nDo you want to install WSL2 from Microsoft store ?" IDYES true IDNO false
		true:
			ExecShell open "https://aka.ms/wslstorepage"
		false:
			MessageBox MB_YESNOCANCEL|MB_ICONINFORMATION "Do you want to use VcXSrv instead of WSLg ?" IDYES +2 IDNO no_wslg
			cancel:
			SetErrorlevel 1002
			Abort
	${Endif}

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
	SetShellVarContext all
 
	; Ask for confirmation
	IfSilent +3
	MessageBox MB_OKCANCEL "Permanantly remove ${DISPLAY_NAME}?" IDOK +2
		Abort
FunctionEnd

Section "un.Uninstaller Section"

	SetRegView 64
	SetShellVarContext all

	; Kill running instances of squey
	KillProcDLL::KillProc "squey"
	
	; Unregister WSL distro
	ExecDos::exec 'wsl --unregister ${WSL_DISTRO_NAME}'
	
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
