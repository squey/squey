
set package_root="msix_package_root"
set tmp_dir="tmp"
mkdir %tmp_dir%
copy %package_root%\AppxManifest.xml %tmp_dir%
copy Square44x44Logo.png %tmp_dir%\Square44x44Logo.targetsize-44_altform-unplated.png
copy Square150x150Logo.png %tmp_dir%\Square150x150Logo.targetsize-150_altform-unplated.png
makepri createconfig /cf %tmp_dir%\priconfig.xml /dq en-US /o
makepri new /cf %tmp_dir%\priconfig.xml /pr %tmp_dir% /mn %tmp_dir%\AppxManifest.xml /of resources.pri /o
del /s /q %tmp_dir%