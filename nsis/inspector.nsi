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
	!define DISPLAY_NAME "INENDI Inspector"
	!define PRODUCT_NAME "inendi-inspector"
	!define FLATPAK_PACKAGE_NAME "org.inendi.Inspector"
	!define FLATPAKREF_URL "https://inendi.gitlab.io/inspector/install.flatpakref"

    !define INTERNAL_NAME "INENDI Inspector"
    !define NAME "Inspector"
    !define OUTFILE "${PRODUCT_NAME}_installer.exe"
	
	; VcXsrv
	!define VCXSRV_MAIN_REG_KEY "SOFTWARE\VcXsrv"
	!define VCXSRV_REG_KEY "Install_Dir_64"
	!define VCXSRV_HTTP_LINK "https://sourceforge.net/projects/vcxsrv/files/latest/download"
	!define VCXSRV_SETUP "vcxsrv_installer.exe"
	
	; Linux
	!define LINUX_HTTP_LINK "https://aka.ms/wsl-debian-gnulinux"
	!define LINUX_ARCHIVE "wsl.zip"
	!define WSL_DISTRO_NAME "inspector_linux"
	
	!define StrRep "!insertmacro StrRep"
	
	!define /date FILE_VERSION "%Y.%m.%d.%H"
	!define /date PRODUCT_VERSION "%Y-%m-%d %H:%M:%S"
	!define /date YEAR "%Y"
    VIProductVersion "${FILE_VERSION}"
    VIAddVersionKey ProductName "${INTERNAL_NAME}"
    VIAddVersionKey FileVersion "${FILE_VERSION}"
    VIAddVersionKey ProductVersion "${PRODUCT_VERSION}"
    VIAddVersionKey LegalCopyright "(C) Jean-Baptiste Leonesio <jean-baptiste+inspector@leonesio.fr> ${YEAR}"

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
	RequestExecutionLevel admin

	; Name and file
	Name "${DISPLAY_NAME}"
	OutFile ${OUTFILE}
	SetCompressor /SOLID lzma

	; Default installation folder
	InstallDir "$PROGRAMFILES64\${NAME}"

	; Override installation folder from registry if available
	InstallDirRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${INTERNAL_NAME}" ""
	
	; Include logic instructions
	!include LogicLib.nsh
	!include nsDialogs.nsh
    !include x64.nsh
	!include WinVer.nsh
	
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
  !define MUI_FINISHPAGE_RUN_FUNCTION "RunInspector"
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
;        VcXsrv
;--------------------------------
Function InstallVcXsrv

	SetOutPath "$INSTDIR"

	; Get final installer URL from public latest-version installer URL (as 'inetc::get' doesn't handle redirections)
	File "resources\curl.exe"
	nsExec::ExecToStack "curl.exe --insecure ${VCXSRV_HTTP_LINK} -o NUL -s -L -I -w %{url_effective}"
	Pop $0
	Pop $1
	Delete curl.exe

	; Download installer
	inetc::get "$1" "${VCXSRV_SETUP}"
	Pop $R0
	${If} $R0 != "OK"
		MessageBox MB_OK "Download failed: $R0"
		Delete "${VCXSRV_SETUP}"
		Quit
	${EndIf}
	
	CreateDirectory "$INSTDIR\VcXsrv"
	Rename ${VCXSRV_SETUP} "VcXsrv\${VCXSRV_SETUP}"
	nsExec::ExecToLog '$WINDIR\SysNative\cmd.exe /C cd VcXsrv && ..\7z.exe -aoa x "${VCXSRV_SETUP}"'
	
	Delete "${VCXSRV_SETUP}"
		
FunctionEnd

;--------------------------------
;        Inspector
;--------------------------------
Function InstallInspector
    ; Download Linux for WSL
	inetc::get "${LINUX_HTTP_LINK}" "${LINUX_ARCHIVE}"
	Pop $R0
	${If} $R0 != "OK"
		MessageBox MB_OK "Download failed: $R0"
		Delete "${LINUX_ARCHIVE}"
		Quit
	${EndIf}
	CreateDirectory "$INSTDIR\wsl"
	Rename ${LINUX_ARCHIVE} "wsl\${LINUX_ARCHIVE}"
	nsExec::ExecToLog '$WINDIR\SysNative\WindowsPowerShell\v1.0\powershell.exe -Command "cd wsl; Expand-Archive -F wsl.zip; cd wsl ; move DistroLauncher-Appx_*_x64.appx wsl.zip ; Expand-Archive -F wsl.zip'
	Delete "7z.exe"
	Delete "7z.dll"
	
	; Install Linux for WSL
    DetailPrint "Preparing WSL..."
	nsExec::ExecToLog '$WINDIR\SysNative\wsl.exe --import ${WSL_DISTRO_NAME} linux wsl\wsl\wsl\install.tar.gz --version 2'
	AccessControl::GrantOnFile "$INSTDIR\linux" "(BU)" "FullAccess" ; Give builtin users full access to modify linux files
	RMDir /r "$INSTDIR\wsl"
	Delete "${LINUX_ARCHIVE}"
	
	; Install Inspector
	DetailPrint "Installing Inspector..."
	nsExec::ExecToStack '$WINDIR\SysNative\cmd.exe /C wsl wslpath -a "$INSTDIR"'
	Pop $0
	Pop $1
	Strcpy $1 $1 -1 ; Remove carriage return in the end
	${StrRep} "$1" "$1" " " "\ " ; Escape spaces
	nsExec::ExecToLog 'wsl --user root -d inspector_linux --exec bash -c "$1/install_inspector.sh ${FLATPAKREF_URL}"'
	Delete "install_inspector.sh"
FunctionEnd

;--------------------------------
;        Section Inspector
;--------------------------------
Section
	SetRegView 64
	AddSize 2537554

    SetOutPath "$INSTDIR"
	File "resources\7z.exe"
	File "resources\7z.dll"
	File "resources\install_inspector.sh"
	File "resources\setup_config_dir.sh"
	File "resources\run_inspector.cmd"
	File "resources\update_wsl.sh"
	File "resources\update_inspector.sh"
	File "resources\hideexec.exe" ; http://code.kliu.org/misc/hideexec/
	File "resources\${PRODUCT_NAME}.ico"
	
	Call InstallVcXsrv
	Call InstallInspector
	
	; Configure uninstaller
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${INTERNAL_NAME}" "" "$INSTDIR"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${INTERNAL_NAME}" "DisplayName" "${DISPLAY_NAME}"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${INTERNAL_NAME}" "UninstallString" "$INSTDIR\uninstall.exe"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${INTERNAL_NAME}" "DisplayIcon" "$INSTDIR\${PRODUCT_NAME}.ico"
	WriteUninstaller "$INSTDIR\uninstall.exe"
	
SectionEnd

Section "Start Menu Shortcut" SecStartMenuShortcut
	; Create start menu shortcut
	SetShellVarContext all
	CreateShortCut "$SMPROGRAMS\${DISPLAY_NAME}.lnk" "$INSTDIR\hideexec.exe" '"$INSTDIR\run_inspector.cmd" "${FLATPAK_PACKAGE_NAME}"' "$INSTDIR\${PRODUCT_NAME}.ico"
SectionEnd

Section "Desktop Shortcut" SecDesktopShortcut
    ; Create desktop shortcut
	SetShellVarContext all
	CreateShortCut "$DESKTOP\${DISPLAY_NAME}.lnk" "$INSTDIR\hideexec.exe" '"$INSTDIR\run_inspector.cmd" "${FLATPAK_PACKAGE_NAME}"' "$INSTDIR\${PRODUCT_NAME}.ico"
SectionEnd

;--------------------------------
;         Initialization
;--------------------------------
Function .onInit
	SetRegView 64

	; Check windows version
	${WinVerGetBuild} $0
	${IfNot} ${AtLeastWin10}
	${OrIf} $0 < 19041
		MessageBox MB_OK|MB_ICONEXCLAMATION "Your OS needs to be one of the following (or newer) to support WSL2 : $\r$\n > Windows 10 64 bits version 2004$\r$\n > Windows Server 2019"
		Quit
	${EndIf}

	; Check if WSL needs to be enabled
	IfFileExists "$WINDIR\SysNative\wslconfig.exe" skip
	MessageBox MB_OKCANCEL|MB_ICONINFORMATION "Windows Subsystem for Linux (WSL2) and Microsoft Virtual Machine Platform needs to be enabled in order to continue." IDOK ok IDCANCEL cancel
	ok:
		ExecDos::exec '$WINDIR\SysNative\cmd.exe /C dism.exe /Online /Enable-Feature /All /FeatureName:Microsoft-Windows-Subsystem-Linux /NoRestart /Quiet'
		ExecDos::exec '$WINDIR\SysNative\cmd.exe /C dism.exe /Online /Enable-Feature /All /FeatureName:VirtualMachinePlatform            /NoRestart /Quiet'
		ExecDos::exec '$WINDIR\SysNative\powershell.exe New-NetFirewallRule -DisplayName "WSL" -Direction Inbound  -InterfaceAlias "vEthernet (WSL)" -Action Allow'
		Pop $0
		${If} $0 != 3010 ; 3010=ERROR_SUCCESS_REBOOT_REQUIRED (The requested operation is successful. Changes will not be effective until the system is rebooted.)
			MessageBox MB_OK|MB_ICONEXCLAMATION  "Enabling WSL2 failed. Aborting."
			Quit
		${EndIf}

		; Automatically setup installer to autostart once after reboot
		WriteRegStr "HKLM" "SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnce" "${NAME}" "$EXEPATH"

		; Ask user for a reboot
		MessageBox MB_YESNO|MB_ICONQUESTION "Rebooting the system is necessary to finish enabling WSL2.$\r$\nDo you wish to reboot the system now?" IDYES yes IDNO no
		yes:
			Reboot
		no:
			Quit
	cancel:
		Quit
	skip:

FunctionEnd

;--------------------------------
;           Run application after installation
;--------------------------------
Function RunInspector
  SetOutPath "$INSTDIR"
  
  ExecDos::exec /ASYNC '$WINDIR\SysNative\cmd.exe /C ""$INSTDIR\run_inspector.cmd" ${FLATPAK_PACKAGE_NAME}"'
  
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
	MessageBox MB_OKCANCEL "Permanantly remove ${DISPLAY_NAME}?" IDOK +2
		Abort
FunctionEnd

Section "un.Uninstaller Section"

	SetRegView 64
	SetShellVarContext all

	; Kill running instances of inendi-inspector
	KillProcDLL::KillProc "inendi-inspector"
	
	; Unregister WSL distro
	ExecDos::exec 'wsl --unregister "${WSL_DISTRO_NAME}"'
	
    ; Remove installation directory
    RMDir /r "$INSTDIR"
	ExecDos::exec 'rmdir /S /Q "$INSTDIR"'
	
	; Remove start menu & desktop shortcuts
	Delete "$SMPROGRAMS\${DISPLAY_NAME}.lnk"
	Delete "$DESKTOP\${DISPLAY_NAME}.lnk"

    ; Delete registry key
	DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${INTERNAL_NAME}"

SectionEnd

Var CheckboxConfig

Function un.CustomUninstallerPage
	nsDialogs::Create 1018
	Pop $0
	
	${NSD_CreateCheckbox} 35 35 100% 8u "Keep configuration files"
	Pop $CheckboxConfig
	${NSD_SetState} $CheckboxConfig ${BST_CHECKED}

	nsDialogs::Show
FunctionEnd

Function un.LeaveCustomPage
	SetRegView 64

	${NSD_GetState} $CheckboxConfig $0
	${If} $0 == ${BST_UNCHECKED}
		nsExec::ExecToStack '$WINDIR\SysNative\cmd.exe /C echo %APPDATA%'
		Pop $0
		Pop $1
		Strcpy $1 $1 -2 ; Remove carriage return in the end
		RMDir /r "$1\Inspector"
	${EndIf}

FunctionEnd
