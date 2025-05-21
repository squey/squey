try {
    $ErrorActionPreference = "Stop"
    $IsAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
    Write-Host "This script is $(If (-Not $IsAdmin) {'NOT '})running as Administrator"
    $powershellVersion = "{0}.{1}.{2}.{3}" -f $PSVersionTable.PSVersion.Major, $PSVersionTable.PSVersion.Minor, $PSVersionTable.PSVersion.Build, $PSVersionTable.PSVersion.Revision
    Write-Host "Powershell version : $powershellVersion"

    # Add makeappx to PATH
    $kitsPath = "C:\Program Files (x86)\Windows Kits\10\bin"
    $versions = Get-ChildItem -Path $kitsPath -Recurse | Where-Object { $_.PSIsContainer -and $_.Name -match '^10\.0\.\d{5,}' }
    $latestVersion = $versions | Sort-Object Name -Descending | Select-Object -First 1
    Write-Host "Windows SDK version : $latestVersion"
    $makeAppxPath = Join-Path -Path $latestVersion.FullName -ChildPath "x64\makeappx.exe"
    $makeAppxDir = Split-Path -Path $makeAppxPath -Parent
    if (Test-Path $makeAppxPath) {
        Write-Host "makeappx.exe found : $makeAppxPath"
        $env:Path += ";$makeAppxDir"
    }
    else {
        Write-Host "makeappx.exe not found"
        exit 1
    }

    # Setup environment variables
    $projdir = "$env:CI_PROJECT_DIR"
    $appdir = "$projdir\builds\x86_64-w64-mingw32\GCC\RelWithDebInfo"
    $env:PATH = "$appdir" + ";" + $env:PATH
    $env:PVKERNEL_PLUGIN_PATH = "$appdir\plugins"
    $env:SQUEY_PLUGIN_PATH = "$appdir\plugins"
    $env:SQUEY_PYTHONHOME="$appdir\python"
    $env:SQUEY_PYTHONPATH="$env:SQUEY_PYTHONHOME\site-packages"
    $env:LIBRARY_PATH="-L$appdir"

    # Extract app and testsuite
    Remove-Item -Recurse -Force "$appdir" -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path "$appdir" -Force
    Get-ChildItem -Path export/x86_64-w64-mingw32/*.msix | ForEach-Object { makeappx unpack /p $_.FullName /d "$appdir" > $null }
    Expand-Archive -Path "export\x86_64-w64-mingw32\testsuite.zip" -DestinationPath "$appdir" -Force

    # Setup Squey config file
    $squey_unicode = "Squëy" # Validate unicode support
    $nraw_tmp = "$env:TEMP\$squey_unicode"
    $configdir = "$env:APPDATA\$squey_unicode"
    $env:SQUEY_CONFIG_DIR = $configdir
    $inifile = "$configdir\squey\config.ini"
    New-Item -ItemType Directory -Path "$configdir\squey" -Force
    Copy-Item -Path "$projdir\src\pvconfig.ini" -Destination "$inifile"
    New-Item -ItemType Directory -Path "$nraw_tmp" -Force
    $content = Get-Content $iniFile
    $content = $content -replace "(?<=^nraw_tmp=).*", ($nraw_tmp -replace "\\", "/")
    $content | Set-Content $iniFile

    # Run testsuite
    ctest --test-dir "$appdir" -j $env:NUMBER_OF_PROCESSORS --output-junit "$env:CI_PROJECT_DIR\junit.xml" --output-on-failure -T test -R 'SQUEY_TEST'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} catch {
    exit 1
}
