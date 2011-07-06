#ifndef __WINLICENSDK__
#define __WINLICENSDK__

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#ifdef __cplusplus
 extern "C" {
#endif


// ***********************************************
// WinLicense constants definition
// ***********************************************

#define wlNoTrialExt             		 0;
#define wlAppExtended            		 1;
#define wlInvalidTrialExt        		 2;
#define wlNoMoreExt              		 3;
#define wlTrialOk                		 0;
#define wlTrialDaysExpired       		 1;
#define wlTrialExecExpired      		 2;
#define wlTrialDateExpired       		 3;
#define wlTrialRuntimExpired     		 4;
#define wlTrialGlobalExpired     		 5;
#define wlTrialInvalidCountry    		 6;
#define wlTrialManipulated    			 7;
#define wlMarkStolenKey          		 0;
#define wlMarkInvalidKey        		 1;
#define wlLicenseDaysExpired    		 1;
#define wlLicenseExecExpired     		 2;
#define wlLicenseDateExpired     		 3;
#define wlLicenseGlobalExpired   		 4;
#define wlLicenseRuntimeExpired  		 5;
#define wlLicenseActivationExpired  		 6;
#define wlIsTrial                		 0;
#define wlIsRegistered           		 1;
#define wlInvalidLicense         		 2;
#define wlInvalidHardwareLicense 		 3;
#define wlNoMoreHwdChanges       		 4;
#define wlLicenseExpired         		 5;
#define wlInvalidCountryLicense  		 6;
#define wlLicenseStolen          		 7;
#define wlWrongLicenseExp        		 8;
#define wlWrongLicenseHardware   		 9;
#define wlIsRegisteredNotActivared 		 10;
//#define wlIsRegisteredAndActivated 	 wlIsRegistered;
#define wlIsRegisteredAndActivated 		 1;
#define wlNoMoreInstancesAllowed 		 12;
#define wlNetworkNoServerRunning 		 13;
#define wlInstallLicenseDateExpired		 14;
#define wlLicenseDisabledInstance		 15;

#define wlPermKey                		 -1;
#define wlNoKey                  		 -2;
#define wlNoTrialDate            		 -1;
#define wlInvalidCounter         		 -1;


// License restrictions 

#define wlRegRestrictionDays           1;
#define wlRegRestrictionExec           2;
#define wlRegRestrictionDate           4;
#define wlRegRestrictionRuntime        8;
#define wlRegRestrictionGlobalTime     16;
#define wlRegRestrictionCountry        32;
#define wlRegRestrictionHardwareId     64;
#define wlRegRestrictionNetwork        128;
#define wlRegRestrictionInstallDate    256;
#define wlRegRestrictionCreationDate   512;
#define wlRegRestrictionEmbedUserInfo  1024;



// ***********************************************
// WinLicense typedef definition
// ***********************************************

typedef struct _sLicenseFeatures 
{ 
	unsigned	cb;						// size of struct
	unsigned 	NumDays;				// expiration days
	unsigned	NumExec;				// expiration executions
	SYSTEMTIME  ExpDate;				// expiration date 
	unsigned	CountryId;				// country ID
	unsigned	Runtime;				// expiration runtime
	unsigned	GlobalMinutes;  		// global time expiration
	SYSTEMTIME	InstallDate;			// Date to install the license since it was created
	unsigned	NetInstances;			// Network instances via shared file
	unsigned	EmbedLicenseInfoInKey;	// for Dynamic SmartKeys, it embeds Name+Company+Custom inside generated SmartKey
	unsigned    EmbedCreationDate;  	// Embed the date that the key was created
} sLicenseFeatures;


// ***********************************************
// WinLicense functions prototype
// ***********************************************

 #ifdef __BORLANDC__

_stdcall __declspec(dllimport) int WLGenTrialExtensionFileKey(char* TrialHash, int Level,\
                                                int NumDays, int NumExec, SYSTEMTIME* NewDate, int NumMinutes,\
                                                int TimeRuntime, char* BufferOut);

_stdcall __declspec(dllimport) int WLGenTrialExtensionRegistryKey(char* TrialHash, int Level,\
                                                int NumDays, int NumExec, SYSTEMTIME* NewDate, int NumMinutes,\
                                                int TimeRuntime, char* pKeyName, char* pKeyValueName, char* BufferOut);

_stdcall __declspec(dllimport) int WLGenPassword(char* TrialHash, char* Name, char* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseFileKey(char* LicenseHash, char* UserName, char* Organization,\
                                                char* CustomData, char* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                                                int CountryId, int Runtime, int GlobalMinutes, char* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseFileKeyW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                                                const wchar_t* CustomData, const wchar_t* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                                                int CountryId, int Runtime, int GlobalMinutes, char* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseFileKeyEx(const char* LicenseHash, const char* UserName, const char* Organization,\
                                                const char* CustomData, const char* MachineID, sLicenseFeatures* LicenseFeatures, char* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseFileKeyExW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                                                const wchar_t* CustomData, const wchar_t* MachineID, sLicenseFeatures* LicenseFeatures, char* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseRegistryKey(char* LicenseHash, char* UserName, char* Organization,\
                                                char* CustomData, char* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                                                int CountryId, int Runtime, int GlobalMinutes, char* KeyName, char* KeyValueName, char* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseRegistryKeyW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                                                const wchar_t* CustomData, const wchar_t* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                                                int CountryId, int Runtime, int GlobalMinutes, const wchar_t* KeyName, const wchar_t* KeyValueName, char* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseRegistryKeyEx(char* LicenseHash, char* UserName, char* Organization,\
                                                char* CustomData, char* MachineID, sLicenseFeatures* LicenseFeatures, char* KeyName, char* KeyValueName, char* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseRegistryKeyExW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                                                const wchar_t* CustomData, const wchar_t* MachineID, sLicenseFeatures* LicenseFeatures, 
                                                const wchar_t* KeyName, const wchar_t* KeyValueName, wchar_t* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseTextKey(char* LicenseHash, char* UserName, char* Organization,\
                                                char* CustomData, char* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                                                int CountryId, int Runtime, int GlobalMinutes, char* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseTextKeyW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                                                const wchar_t* CustomData, const wchar_t* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                                                int CountryId, int Runtime, int GlobalMinutes, wchar_t* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseTextKeyEx(const char* LicenseHash, const char* UserName, const char* Organization,\
                                                const char* CustomData, const char* MachineID, sLicenseFeatures* LicenseFeatures, char* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseTextKeyExW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                                                const wchar_t* CustomData, const wchar_t* MachineID, sLicenseFeatures* LicenseFeatures, wchar_t* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseSmartKey(char* LicenseHash, char* UserName, char* Organization,\
                                                char* CustomData, char* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                                                char* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseSmartKeyW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                                                const wchar_t* CustomData, const wchar_t* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                                                wchar_t* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseDynSmartKey(const char* LicenseHash, const char* UserName, const char* Organization,\
                                                const char* CustomData, const char* MachineID, sLicenseFeatures* LicenseFeatures, char* BufferOut);

_stdcall __declspec(dllimport) int WLGenLicenseDynSmartKeyW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                                                const wchar_t* CustomData, const wchar_t* MachineID, sLicenseFeatures* LicenseFeatures, wchar_t* BufferOut);

_stdcall __declspec(dllimport) int WLTrialGetStatus(int* pExtendedInfo);

_stdcall __declspec(dllimport) int WLTrialExtGetStatus(void);
 
_stdcall __declspec(dllimport) int WLRegGetStatus(int* pExtendedInfo);

_stdcall __declspec(dllimport) int WLRegGetLicenseInfo(char* pName, char* pCompanyName, char* pCustomData);

_stdcall __declspec(dllimport) int WLRegGetLicenseInfoW(wchar_t* pName, wchar_t* pCompanyName, wchar_t* pCustomData);

_stdcall __declspec(dllimport) int WLTrialTotalDays(void);
 
_stdcall __declspec(dllimport) int WLTrialTotalExecutions(void);

_stdcall __declspec(dllimport) int WLTrialDaysLeft(void);

_stdcall __declspec(dllimport) int WLTrialExecutionsLeft(void);

_stdcall __declspec(dllimport) int WLTrialExpirationDate(SYSTEMTIME* pExpDate);

_stdcall __declspec(dllimport) int WLTrialGlobalTimeLeft(void);

_stdcall __declspec(dllimport) int WLTrialRuntimeLeft(void);

_stdcall __declspec(dllimport) int WLTrialLockedCountry(void);

_stdcall __declspec(dllimport) int WLRegDaysLeft(void);

_stdcall __declspec(dllimport) int WLRegExecutionsLeft(void);

_stdcall __declspec(dllimport) int WLRegExpirationDate(SYSTEMTIME* pExpDate);

_stdcall __declspec(dllimport) int WLRegLicenseCreationDate(SYSTEMTIME* pCreationDate);

_stdcall __declspec(dllimport) int WLRegTotalExecutions(void);

_stdcall __declspec(dllimport) int WLRegTotalDays(void);

_stdcall __declspec(dllimport) int WLHardwareGetID(char* pHardwareId);

_stdcall __declspec(dllimport) int WLHardwareCheckID(char* pHardwareId);

_stdcall __declspec(dllimport) int WLRegSmartKeyCheck(const char* UserName, const char* Organization, const char* Custom, const char* AsciiKey);

_stdcall __declspec(dllimport) int WLRegSmartKeyCheckW(const wchar_t* UserName, const wchar_t* Organization, const wchar_t* Custom, const wchar_t* AsciiKey);

_stdcall __declspec(dllimport) int WLRegNormalKeyCheck(const char* AsciiKey);

_stdcall __declspec(dllimport) int WLRegNormalKeyCheckW(const wchar_t* AsciiKey);

_stdcall __declspec(dllimport) int WLRegNormalKeyInstallToFile(const char* AsciiKey);

_stdcall __declspec(dllimport) int WLRegNormalKeyInstallToFileW(const wchar_t* AsciiKey);

_stdcall __declspec(dllimport) int WLRegNormalKeyInstallToRegistry(const char* AsciiKey);

_stdcall __declspec(dllimport) int WLRegNormalKeyInstallToRegistryW(const wchar_t* AsciiKey);

_stdcall __declspec(dllimport) int WLRegSmartKeyInstallToFile(const char* UserName, const char* Organization, const char* Custom, const char* AsciiKey);

_stdcall __declspec(dllimport) int WLRegSmartKeyInstallToFileW(const wchar_t* UserName, const wchar_t* Organization, const wchar_t* Custom, const wchar_t* AsciiKey);

_stdcall __declspec(dllimport) int WLRegSmartKeyInstallToRegistry(const char* UserName, const char* Organization, const char* Custom, const char* AsciiKey);

_stdcall __declspec(dllimport) int WLRegSmartKeyInstallToRegistryW(const wchar_t* UserName, const wchar_t* Organization, const wchar_t* Custom, const wchar_t* AsciiKey);
 
_stdcall __declspec(dllimport) int WLTrialCustomCounterInc(int Value, int CounterId);

_stdcall __declspec(dllimport) int WLTrialCustomCounterDec(int Value, int CounterId);

_stdcall __declspec(dllimport) int WLTrialCustomCounter(int CounterId);

_stdcall __declspec(dllimport) int WLTrialCustomCounterSet(int Value, int CounterId);

_stdcall __declspec(dllimport) int WLRestartApplication(void);

_stdcall __declspec(dllimport) int WLRegLockedCountry(void);

_stdcall __declspec(dllimport) int WLRegRuntimeLeft(void);

_stdcall __declspec(dllimport) int WLRegGlobalTimeLeft(void);

_stdcall __declspec(dllimport) int WLRegDisableCurrentKey(int DisableFlags);

_stdcall __declspec(dllimport) int WLRegRemoveCurrentKey(void);
 
_stdcall __declspec(dllimport) int WLHardwareGetFormattedID(int BlockCharSize, int Uppercase, char* Buffer);
 
_stdcall __declspec(dllimport) int WLPasswordCheck(char* UserName, char* Password);

_stdcall __declspec(dllimport) int WLTrialExpireTrial(void);

_stdcall __declspec(dllimport) LPSTR WLStringDecrypt(char* pString);

_stdcall __declspec(dllimport) LPWSTR WLStringDecryptW(wchar_t* pString);

_stdcall __declspec(dllimport) void WLRegLicenseName(char* FileKeyName, char* RegKeyName, char* RegKeyValueName);

_stdcall __declspec(dllimport) int WLRestartApplicationArgs(char* pArgs);

_stdcall __declspec(dllimport) int WLActGetInfo(int* Custom1, int* Custom2, int* Custom3);

_stdcall __declspec(dllimport) int WLActCheck(char* ActivationCode);

_stdcall __declspec(dllimport) int WLActInstall(char* ActivationCode);

_stdcall __declspec(dllimport) int WLActExpirationDate(SYSTEMTIME* pExpDate);

_stdcall __declspec(dllimport) int WLActDaysToActivate(void);

_stdcall __declspec(dllimport) int WLActUninstall(void);

// (CURRENTLY DISABLED)_stdcall __declspec(dllimport) int WLRegGetLicenseHardwareID(char* pHardwareId);

_stdcall __declspec(dllimport) int WLGetCurrentCountry(void);

_stdcall __declspec(dllimport) int WLTrialExtGetLevel(void);

_stdcall __declspec(dllimport) int WLProtectCheckDebugger(void);

_stdcall __declspec(dllimport) int WLTrialExtendExpiration(int NumDays, int NumExec, SYSTEMTIME* NewDate,
                                                int Runtime, int GlobalMinutes);
  
_stdcall __declspec(dllimport) int WLTrialFirstRun(void);

_stdcall __declspec(dllimport) int WLRegFirstRun(void);
 
_stdcall __declspec(dllimport) int WLRegCheckMachineLocked(void);

_stdcall __declspec(dllimport) void WLSplashHide(void);

_stdcall __declspec(dllimport) void WLBufferCrypt(void* Buffer, int BufferLength, char* Password);

_stdcall __declspec(dllimport) void WLBufferDecrypt(void* Buffer, int BufferLength, char* Password);

_stdcall __declspec(dllimport) int WLRegSmartKeyInstallToFileInFolder(const char* UserName, const char* Organization, const char* Custom, const char* AsciiKey, const char* FilePath);

_stdcall __declspec(dllimport) int WLTrialDateDaysLeft(void);

_stdcall __declspec(dllimport) int WLRegDateDaysLeft(void);

_stdcall __declspec(dllimport) void WLResetLicenseFeatures(sLicenseFeatures *LicenseFeatures, int SizeStructure);

_stdcall __declspec(dllimport) int WLRegGetDynSmartKey(char* SmartKey);

_stdcall __declspec(dllimport) int WLRegDisableKeyCurrentInstance(void);

_stdcall __declspec(dllimport) int WLHardwareRuntimeCheckU3(void);

_stdcall __declspec(dllimport) void WLGetVersion(char* Buffer);

_stdcall __declspec(dllimport) int WLRegNetInstancesGet(void);

_stdcall __declspec(dllimport) int WLRegNetInstancesMax(void);

_stdcall __declspec(dllimport) void WLGetProtectionDate(SYSTEMTIME* pProtectionDate);

_stdcall __declspec(dllimport) int WLProtectCheckMem(void);

_stdcall __declspec(dllimport) int WLHardwareGetIdType(void);

_stdcall __declspec(dllimport) int WLTrialStringRead(const char *StringName, char *StringValue);

_stdcall __declspec(dllimport) int WLTrialStringReadW(const wchar_t *StringName, wchar_t *StringValue);

_stdcall __declspec(dllimport) int WLTrialStringWrite(const char *StringName, const char *StringValue);

_stdcall __declspec(dllimport) int WLTrialStringWriteW(const wchar_t *StringName, const wchar_t *StringValue);

_stdcall __declspec(dllimport) int WLTrialDebugCheck(void);

_stdcall __declspec(dllimport) int WLRegExpirationTimestamp(LPFILETIME lpFileTime);

_stdcall __declspec(dllimport) int WLTrialExpirationTimestamp(LPFILETIME lpFileTime);

_stdcall __declspec(dllimport) int WLRegGetLicenseRestrictions(void);

_stdcall __declspec(dllimport) int WLRegGetLicenseType(void);

_stdcall __declspec(dllimport) int WLCheckVirtualPC(void);

_stdcall __declspec(dllimport) int WLHardwareGetIDW(wchar_t * pHardwareId);

  
 #else

int _stdcall GenerateTrialExtensionKey(char* TrialHash, int Level, int NumDays, int NumExec,\
                                                SYSTEMTIME* NewDate, int NumMinutes, int TimeRuntime,\
                                                char* BufferOut);

int _stdcall WLGenTrialExtensionFileKey(char* TrialHash, int Level,\
                                            int NumDays, int NumExec, SYSTEMTIME* NewDate, int NumMinutes,\
                                            int TimeRuntime, char* BufferOut);

int _stdcall WLGenTrialExtensionRegistryKey(char* TrialHash, int Level,\
                          int NumDays, int NumExec, SYSTEMTIME* NewDate, int NumMinutes,\
                          int TimeRuntime, char* pKeyName, char* pKeyValueName, char* BufferOut);

int _stdcall WLGenPassword(char* TrialHash, char* Name, char* BufferOut);

int _stdcall WLGenLicenseFileKey(char* LicenseHash, char* UserName, char* Organization,\
                          char* CustomData, char* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                          int CountryId, int Runtime, int GlobalMinutes, char* BufferOut);

int _stdcall WLGenLicenseFileKeyW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                          const wchar_t* CustomData, const wchar_t* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                          int CountryId, int Runtime, int GlobalMinutes, char* BufferOut);

int _stdcall WLGenLicenseFileKeyEx(const char* LicenseHash, const char* UserName, const char* Organization,\
                                               const char* CustomData, const char* MachineID, sLicenseFeatures* LicenseFeatures, char* BufferOut);

int _stdcall WLGenLicenseFileKeyExW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                                                const wchar_t* CustomData, const wchar_t* MachineID, sLicenseFeatures* LicenseFeatures, char* BufferOut);

int _stdcall WLGenLicenseRegistryKey(char* LicenseHash, char* UserName, char* Organization,\
                          char* CustomData, char* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                          int CountryId, int Runtime, int GlobalMinutes, char* KeyName, char* KeyValueName, char* BufferOut);

int _stdcall WLGenLicenseRegistryKeyW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                                                const wchar_t* CustomData, const wchar_t* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                                                int CountryId, int Runtime, int GlobalMinutes, const wchar_t* KeyName, const wchar_t* KeyValueName, char* BufferOut);

int _stdcall WLGenLicenseRegistryKeyEx(const char* LicenseHash, const char* UserName, const char* Organization,\
                          const char* CustomData, const char* MachineID, sLicenseFeatures* LicenseFeatures, char* KeyName, char* KeyValueName, char* BufferOut);

int _stdcall WLGenLicenseRegistryKeyExW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                          const wchar_t* CustomData, const wchar_t* MachineID, sLicenseFeatures* LicenseFeatures, const wchar_t* KeyName, const wchar_t* KeyValueName, wchar_t* BufferOut);

int _stdcall WLGenLicenseTextKey(char* LicenseHash, char* UserName, char* Organization,\
                          char* CustomData, char* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                          int CountryId, int Runtime, int GlobalMinutes, char* BufferOut);

int _stdcall WLGenLicenseTextKeyW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                          const wchar_t* CustomData, const wchar_t* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                          int CountryId, int Runtime, int GlobalMinutes, wchar_t* BufferOut);

int _stdcall WLGenLicenseTextKeyEx(const char* LicenseHash, const char* UserName, const char* Organization,\
                          const char* CustomData, const char* MachineID, sLicenseFeatures* LicenseFeatures, char* BufferOut);

int _stdcall WLGenLicenseTextKeyExW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                          const wchar_t* CustomData, const wchar_t* MachineID, sLicenseFeatures* LicenseFeatures, wchar_t* BufferOut);

int _stdcall WLGenLicenseSmartKey(char* LicenseHash, char* UserName, char* Organization,\
                          char* CustomData, char* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                          char* BufferOut);

int _stdcall WLGenLicenseSmartKeyW(wchar_t* LicenseHash, wchar_t* UserName, wchar_t* Organization,\
                          wchar_t* CustomData, wchar_t* MachineID, int NumDays, int NumExec, SYSTEMTIME* NewDate, \
                          wchar_t* BufferOut);

int _stdcall WLGenLicenseDynSmartKey(const char* LicenseHash, const char* UserName, const char* Organization,\
                                                const char* CustomData, const char* MachineID, sLicenseFeatures* LicenseFeatures, char* BufferOut);

int _stdcall WLGenLicenseDynSmartKeyW(const wchar_t* LicenseHash, const wchar_t* UserName, const wchar_t* Organization,\
                                                const wchar_t* CustomData, const wchar_t* MachineID, sLicenseFeatures* LicenseFeatures, wchar_t* BufferOut);

int _stdcall WLRegGetStatus(int* pExtendedInfo);

int _stdcall WLTrialGetStatus(int* pExtendedInfo);

int _stdcall WLTrialExtGetStatus(void);
 
int _stdcall WLRegGetLicenseInfo(char* pName, char* pCompanyName, char* pCustomData);

int _stdcall WLRegGetLicenseInfoW(wchar_t* pName, wchar_t* pCompanyName, wchar_t* pCustomData);

int _stdcall WLTrialTotalDays(void);
 
int _stdcall WLTrialTotalExecutions(void);

int _stdcall WLTrialDaysLeft(void);

int _stdcall WLTrialExecutionsLeft(void);

int _stdcall WLTrialExpirationDate(SYSTEMTIME* pExpDate);

int _stdcall WLTrialGlobalTimeLeft(void);

int _stdcall WLTrialRuntimeLeft(void);

int _stdcall WLTrialLockedCountry(void);

int _stdcall WLRegDaysLeft(void);

int _stdcall WLRegExecutionsLeft(void);

int _stdcall WLRegExpirationDate(SYSTEMTIME* pExpDate);

int _stdcall WLRegLicenseCreationDate(SYSTEMTIME* pCreationDate);
 
int _stdcall WLRegTotalExecutions(void);

int _stdcall WLRegTotalDays(void);

int _stdcall WLHardwareGetID(char* pHardwareId);

int _stdcall WLHardwareCheckID(char* pHardwareId);

int _stdcall WLRegSmartKeyCheck(const char* UserName, const char* Organization, const char* Custom, const char* AsciiKey);

int _stdcall WLRegSmartKeyCheckW(const wchar_t* UserName, const wchar_t* Organization, const wchar_t* Custom, const wchar_t* AsciiKey);

int _stdcall WLRegNormalKeyCheck(const char* AsciiKey);

int _stdcall WLRegNormalKeyCheckW(const wchar_t* AsciiKey);

int _stdcall WLRegNormalKeyInstallToFile(const char* AsciiKey);

int _stdcall WLRegNormalKeyInstallToFileW(const wchar_t* AsciiKey);

int _stdcall WLRegNormalKeyInstallToRegistry(const char* AsciiKey);

int _stdcall WLRegNormalKeyInstallToRegistryW(const wchar_t* AsciiKey);

int _stdcall WLRegSmartKeyInstallToFile(const char* UserName, const char* Organization, const char* Custom, const char* AsciiKey);

int _stdcall WLRegSmartKeyInstallToRegistry(const char* UserName, const char* Organization, const char* Custom, const char* AsciiKey);

int _stdcall WLRegSmartKeyInstallToFileW(const wchar_t* UserName, const wchar_t* Organization, const wchar_t* Custom, const wchar_t* AsciiKey);
 
int _stdcall WLRegSmartKeyInstallToRegistryW(const wchar_t* UserName, const wchar_t* Organization, const wchar_t* Custom, const wchar_t* AsciiKey);

int _stdcall WLTrialCustomCounterInc(int Value, int CounterId);

int _stdcall WLTrialCustomCounterDec(int Value, int CounterId);

int _stdcall WLTrialCustomCounter(int CounterId);

int _stdcall WLTrialCustomCounterSet(int Value, int CounterId);

int _stdcall WLRestartApplication(void);

int _stdcall WLRegLockedCountry(void);

int _stdcall WLRegRuntimeLeft(void);

int _stdcall WLRegGlobalTimeLeft(void);

int _stdcall WLRegDisableCurrentKey(int DisableFlags);

int _stdcall WLRegRemoveCurrentKey(void);
 
int _stdcall WLHardwareGetFormattedID(int BlockCharSize, int Uppercase, char* Buffer);
 
int _stdcall WLPasswordCheck(char* UserName, char* Password);

int _stdcall WLTrialExpireTrial(void);

char* _stdcall WLStringDecrypt(char* pString);

wchar_t* _stdcall WLStringDecryptW(wchar_t* pString);

void _stdcall WLRegLicenseName(char* FileKeyName, char* RegKeyName, char* RegKeyValueName);

int _stdcall WLRestartApplicationArgs(char* pArgs);

int _stdcall WLActGetInfo(int* Custom1, int* Custom2, int* Custom3);

int _stdcall WLActCheck(char* ActivationCode);

 int _stdcall WLActInstall(char* ActivationCode);

 int _stdcall WLActExpirationDate(SYSTEMTIME* pExpDate);

int _stdcall WLActDaysToActivate(void);

int _stdcall WLActUninstall(void);

// (CURRENTLY DISABLED)int _stdcall WLRegGetLicenseHardwareID(char* pHardwareId);

int _stdcall WLGetCurrentCountry(void);

int _stdcall WLTrialExtGetLevel(void);

int _stdcall WLProtectCheckDebugger(void);

int _stdcall  WLTrialExtendExpiration(int NumDays, int NumExec, SYSTEMTIME* NewDate, int Runtime, int GlobalMinutes);

int _stdcall WLTrialFirstRun(void);

int _stdcall WLRegFirstRun(void);

int _stdcall WLRegCheckMachineLocked(void);

void _stdcall WLSplashHide(void);

void _stdcall WLBufferCrypt(void* Buffer, int BufferLength, char* Password);

void _stdcall WLBufferDecrypt(void* Buffer, int BufferLength, char* Password);

int _stdcall WLRegSmartKeyInstallToFileInFolder(const char* UserName, const char* Organization, const char* Custom, const char* AsciiKey, const char* FilePath);

int _stdcall WLTrialDateDaysLeft(void);
 
int _stdcall WLRegDateDaysLeft(void);

void _stdcall WLResetLicenseFeatures(sLicenseFeatures *LicenseFeatures, int SizeStructure);

int _stdcall WLRegGetDynSmartKey(char* SmartKey);

int _stdcall WLRegDisableKeyCurrentInstance(void);

int _stdcall WLHardwareRuntimeCheckU3(void);

void _stdcall WLGetVersion(char* Buffer);

int _stdcall WLRegNetInstancesGet(void);

int _stdcall WLRegNetInstancesMax(void);

void _stdcall WLGetProtectionDate(SYSTEMTIME* pProtectionDate);
 
int _stdcall WLProtectCheckCodeIntegrity(void);

int _stdcall WLHardwareGetIdType(void);

int _stdcall WLTrialStringRead(const char *StringName, char *StringValue);

int _stdcall WLTrialStringReadW(const wchar_t *StringName, wchar_t *StringValue);

int _stdcall WLTrialStringWrite(const char *StringName, const char *StringValue);

int _stdcall WLTrialStringWriteW(const wchar_t *StringName, const wchar_t *StringValue);

int _stdcall WLTrialDebugCheck(void);

int _stdcall WLRegExpirationTimestamp(LPFILETIME lpFileTime);

int _stdcall WLTrialExpirationTimestamp(LPFILETIME lpFileTime);

int _stdcall WLRegGetLicenseRestrictions(void);

int _stdcall WLRegGetLicenseType(void);

int _stdcall WLCheckVirtualPC(void);

int _stdcall WLHardwareGetIDW(wchar_t * pHardwareId);

    
 #endif

// WinLicense macros definition
 
 #ifdef __BORLANDC__
 
 #define REMOVE_BLOCK_START     __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x00, 0x00, 0x00, 0x00, \
                                           0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);
 #define REMOVE_BLOCK_END       __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x01, 0x00, 0x00, 0x00, \
                                           0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);

 #define CODEREPLACE_START      __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x00, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);
 #define CODEREPLACE_END        __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x01, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);
 
 #define REGISTERED_START       __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x02, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);
 #define REGISTERED_END         __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x03, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);
 
 #define ENCODE_START           __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x04, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);
 #define ENCODE_END             __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x05, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);

 #define CLEAR_START            __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x06, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);
 #define CLEAR_END              __emit__ (0xEB, 0x15, 0x57, 0x4C, 0x20, 0x20, 0x07, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20, 0x00, 0x00, \
                                          0x00, 0x00, 0x00);

 #define UNREGISTERED_START     __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x08, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);
 #define UNREGISTERED_END       __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x09, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);

 #define VM_START               __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x0C, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);
 #define VM_END                 __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x0D, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);

 #define VM_REGISTEREDVM_START  __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x0E, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);
 #define VM_REGISTEREDVM_END    __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x0F, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);

 #define VM_START_WITHLEVEL(x)  __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x0C, 0x00, 0x00, 0x00, \
                                          0x00, x, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);

 #define UNPROTECTED_START      __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x20, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);
 #define UNPROTECTED_END        __emit__ (0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x21, 0x00, 0x00, 0x00, \
                                          0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20);

 #define CHECK_PROTECTION(var, val) \
 asm {  \
   dw    0x10EB; \
   dd    0x091ab3167; \
   dd    0x08a8b717a; \
   dd    0x0bc117abd; \
   dd    0x0; \
   push  val; \
   pop   var; \
   dw    0x0CEB; \
   dd    0x0bc117abd; \
   dd    0x08a8b717a; \
   dd    0x091ab3167; \
}

 #define CHECK_CODE_INTEGRITY(var, val) \
 asm {  \
   dw    0x10EB; \
   dd    0x091ab3167; \
   dd    0x08a8b717a; \
   dd    0x0bc117abd; \
   dd    0x1; \
   push  val; \
   pop   var; \
   dw    0x0CEB; \
   dd    0x0bc117abd; \
   dd    0x08a8b717a; \
   dd    0x091ab3167; \
}

 #define CHECK_REGISTRATION(var, val) \
 asm {  \
   dw    0x10EB; \
   dd    0x091ab3167; \
   dd    0x08a8b717a; \
   dd    0x0bc117abd; \
   dd    0x2; \
   push  val; \
   pop   var; \
   dw    0x0CEB; \
   dd    0x0bc117abd; \
   dd    0x08a8b717a; \
   dd    0x091ab3167; \
}

 #define CHECK_VIRTUAL_PC(var, val) \
 asm {  \
   dw    0x10EB; \
   dd    0x091ab3167; \
   dd    0x08a8b717a; \
   dd    0x0bc117abd; \
   dd    0x3; \
   push  val; \
   pop   var; \
   dw    0x0CEB; \
   dd    0x0bc117abd; \
   dd    0x08a8b717a; \
   dd    0x091ab3167; \
}
 
 #define __WL_MACROS__
 #endif
 
#endif
 
 /* intel cpp compiler */
 
#ifndef __WL_MACROS__
 
 #ifdef __ICL
 
 #define REMOVE_BLOCK_START \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
 
 #define REMOVE_BLOCK_END \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x01 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \

  #define CODEREPLACE_START \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
 
 #define CODEREPLACE_END \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x01 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \

 #define REGISTERED_START \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x02 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
 
 #define REGISTERED_END \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x03 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
 
 
 #define ENCODE_START \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x04 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
 
 
 #define ENCODE_END \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x05 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
 
 
 #define CLEAR_START \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x06 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \

 
 #define CLEAR_END \
  __asm __emit 0xEB \
  __asm __emit 0x15 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x07 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 
 
  #define UNREGISTERED_START \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x08 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
 
 #define UNREGISTERED_END \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x09 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \

  #define VM_START_WITHLEVEL(x) \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x0C \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit x \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \

  #define VM_START \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x0C \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  
 #define VM_END \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x0D \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \

  #define REGISTEREDVM_START \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x0E \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  
 #define REGISTEREDVM_END \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x0F \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \

  #define UNPROTECTED_START \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  
 #define UNPROTECTED_END \
  __asm __emit 0xEB \
  __asm __emit 0x10 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \
  __asm __emit 0x21 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x00 \
  __asm __emit 0x57 \
  __asm __emit 0x4C \
  __asm __emit 0x20 \
  __asm __emit 0x20 \


 #define __WL_MACROS__
 
 #endif
#endif
 
 
 /* LCC by Jacob Navia */
 
#ifndef __WL_MACROS__
 
 #ifdef __LCC__
 
 #define REMOVE_BLOCK_START     __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x00, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");
 #define REMOVE_BLOCK_END       __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x01, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");

 #define CODEREPLACE_START      __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x00, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");
 #define CODEREPLACE_END        __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x01, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");
 
 #define REGISTERED_START       __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x02, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");
 #define REGISTERED_END         __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x03, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");
 
 #define ENCODE_START           __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x04, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");
 #define ENCODE_END             __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x05, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");
 
 #define CLEAR_START            __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x06, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");
 
 #define CLEAR_END              __asm__ (" .byte\t0xEB, 0x15, 0x57, 0x4C, 0x20, 0x20, 0x07, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20, 0x00, 0x00, \
                                         0x00, 0x00, 0x00");

 #define UNREGISTERED_START     __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x08, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");
 #define UNREGISTERED_END       __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x09, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");

 #define VM_START_WITHLEVEL(x)  __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x0C, 0x00, 0x00, 0x00, \
                                         0x00, "x", 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");

 #define VM_START               __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x0C, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");

 #define VM_END                 __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x0D, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");

 #define REGISTEREDVM_START     __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x0E, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");

 #define REGISTEREDVM_END       __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x0F, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");

 #define UNPROTECTED_START      __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x20, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");

 #define UNPROTECTED_END        __asm__ (" .byte\t0xEB, 0x10, 0x57, 0x4C, 0x20, 0x20, 0x21, 0x00, 0x00, 0x00, \
                                         0x00, 0x00, 0x00, 0x00, 0x57, 0x4C, 0x20, 0x20");

 #define __WL_MACROS__
 #endif
 
#endif
 
 
#ifndef __WL_MACROS__
 
 #define REMOVE_BLOCK_START \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
 
 #define REMOVE_BLOCK_END \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x01 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \

 #define CODEREPLACE_START \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
 
 #define CODEREPLACE_END \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x01 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \

 #define REGISTERED_START \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x02 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
 
#define REGISTERED_END \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x03 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
 
 #define ENCODE_START \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x04 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
 
 #define ENCODE_END \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x05 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
 
 #define CLEAR_START \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x06 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
 
 #define CLEAR_END \
  __asm _emit 0xEB \
  __asm _emit 0x15 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x07 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 

 #define UNREGISTERED_START \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x08 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
 
#define UNREGISTERED_END \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x09 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \

 #define VM_START \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x0C \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \

 #define VM_START_WITHLEVEL(x) \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x0C \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit x \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
 
 #define VM_END \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x0D \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \

 #define REGISTEREDVM_START \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x0E \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \

 #define REGISTEREDVM_END \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x0F \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \


 #define UNPROTECTED_START \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \

 #define UNPROTECTED_END \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \
  __asm _emit 0x21 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x57 \
  __asm _emit 0x4C \
  __asm _emit 0x20 \
  __asm _emit 0x20 \

  #define CHECK_PROTECTION(var, val) \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x67 \
  __asm _emit 0x31 \
  __asm _emit 0xAB \
  __asm _emit 0x91 \
  __asm _emit 0x7A \
  __asm _emit 0x71 \
  __asm _emit 0x8B \
  __asm _emit 0x8A \
  __asm _emit 0xBD \
  __asm _emit 0x7A \
  __asm _emit 0x11 \
  __asm _emit 0xBC \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm push  val \
  __asm pop   var \
  __asm _emit 0xEB \
  __asm _emit 0x0C \
  __asm _emit 0xBD \
  __asm _emit 0x7A \
  __asm _emit 0x11 \
  __asm _emit 0xBC \
  __asm _emit 0x7A \
  __asm _emit 0x71 \
  __asm _emit 0x8B \
  __asm _emit 0x8A \
  __asm _emit 0x67 \
  __asm _emit 0x31 \
  __asm _emit 0xAB \
  __asm _emit 0x91 \

  #define CHECK_CODE_INTEGRITY(var, val) \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x67 \
  __asm _emit 0x31 \
  __asm _emit 0xAB \
  __asm _emit 0x91 \
  __asm _emit 0x7A \
  __asm _emit 0x71 \
  __asm _emit 0x8B \
  __asm _emit 0x8A \
  __asm _emit 0xBD \
  __asm _emit 0x7A \
  __asm _emit 0x11 \
  __asm _emit 0xBC \
  __asm _emit 0x01 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm push  val \
  __asm pop   var \
  __asm _emit 0xEB \
  __asm _emit 0x0C \
  __asm _emit 0xBD \
  __asm _emit 0x7A \
  __asm _emit 0x11 \
  __asm _emit 0xBC \
  __asm _emit 0x7A \
  __asm _emit 0x71 \
  __asm _emit 0x8B \
  __asm _emit 0x8A \
  __asm _emit 0x67 \
  __asm _emit 0x31 \
  __asm _emit 0xAB \
  __asm _emit 0x91 \

  #define CHECK_REGISTRATION(var, val) \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x67 \
  __asm _emit 0x31 \
  __asm _emit 0xAB \
  __asm _emit 0x91 \
  __asm _emit 0x7A \
  __asm _emit 0x71 \
  __asm _emit 0x8B \
  __asm _emit 0x8A \
  __asm _emit 0xBD \
  __asm _emit 0x7A \
  __asm _emit 0x11 \
  __asm _emit 0xBC \
  __asm _emit 0x02 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm push  val \
  __asm pop   var \
  __asm _emit 0xEB \
  __asm _emit 0x0C \
  __asm _emit 0xBD \
  __asm _emit 0x7A \
  __asm _emit 0x11 \
  __asm _emit 0xBC \
  __asm _emit 0x7A \
  __asm _emit 0x71 \
  __asm _emit 0x8B \
  __asm _emit 0x8A \
  __asm _emit 0x67 \
  __asm _emit 0x31 \
  __asm _emit 0xAB \
  __asm _emit 0x91 \

  #define CHECK_VIRTUAL_PC(var, val) \
  __asm _emit 0xEB \
  __asm _emit 0x10 \
  __asm _emit 0x67 \
  __asm _emit 0x31 \
  __asm _emit 0xAB \
  __asm _emit 0x91 \
  __asm _emit 0x7A \
  __asm _emit 0x71 \
  __asm _emit 0x8B \
  __asm _emit 0x8A \
  __asm _emit 0xBD \
  __asm _emit 0x7A \
  __asm _emit 0x11 \
  __asm _emit 0xBC \
  __asm _emit 0x03 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm _emit 0x00 \
  __asm push  val \
  __asm pop   var \
  __asm _emit 0xEB \
  __asm _emit 0x0C \
  __asm _emit 0xBD \
  __asm _emit 0x7A \
  __asm _emit 0x11 \
  __asm _emit 0xBC \
  __asm _emit 0x7A \
  __asm _emit 0x71 \
  __asm _emit 0x8B \
  __asm _emit 0x8A \
  __asm _emit 0x67 \
  __asm _emit 0x31 \
  __asm _emit 0xAB \
  __asm _emit 0x91 \

#ifdef __cplusplus
 }
#endif

#endif
