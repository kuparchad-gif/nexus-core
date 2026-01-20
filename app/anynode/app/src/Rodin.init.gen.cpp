// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeRodin_init() {}
	RODIN_API UFunction* Z_Construct_UDelegateFunction_Rodin_MultiListen__DelegateSignature();
	RODIN_API UFunction* Z_Construct_UDelegateFunction_Rodin_MultiPublish__DelegateSignature();
	RODIN_API UFunction* Z_Construct_UDelegateFunction_Rodin_MultiSubscibe__DelegateSignature();
	RODIN_API UFunction* Z_Construct_UDelegateFunction_Rodin_OnRodinWSClosed__DelegateSignature();
	RODIN_API UFunction* Z_Construct_UDelegateFunction_Rodin_OnRodinWSMessage__DelegateSignature();
	RODIN_API UFunction* Z_Construct_UDelegateFunction_Rodin_OnRodinWSOpened__DelegateSignature();
	RODIN_API UFunction* Z_Construct_UDelegateFunction_Rodin_OnRodinWSRawMessage__DelegateSignature();
	RODIN_API UFunction* Z_Construct_UDelegateFunction_Rodin_OnRodinWSServerClosed__DelegateSignature();
	RODIN_API UFunction* Z_Construct_UDelegateFunction_Rodin_WSServerPin__DelegateSignature();
	static FPackageRegistrationInfo Z_Registration_Info_UPackage__Script_Rodin;
	FORCENOINLINE UPackage* Z_Construct_UPackage__Script_Rodin()
	{
		if (!Z_Registration_Info_UPackage__Script_Rodin.OuterSingleton)
		{
			static UObject* (*const SingletonFuncArray[])() = {
				(UObject* (*)())Z_Construct_UDelegateFunction_Rodin_MultiListen__DelegateSignature,
				(UObject* (*)())Z_Construct_UDelegateFunction_Rodin_MultiPublish__DelegateSignature,
				(UObject* (*)())Z_Construct_UDelegateFunction_Rodin_MultiSubscibe__DelegateSignature,
				(UObject* (*)())Z_Construct_UDelegateFunction_Rodin_OnRodinWSClosed__DelegateSignature,
				(UObject* (*)())Z_Construct_UDelegateFunction_Rodin_OnRodinWSMessage__DelegateSignature,
				(UObject* (*)())Z_Construct_UDelegateFunction_Rodin_OnRodinWSOpened__DelegateSignature,
				(UObject* (*)())Z_Construct_UDelegateFunction_Rodin_OnRodinWSRawMessage__DelegateSignature,
				(UObject* (*)())Z_Construct_UDelegateFunction_Rodin_OnRodinWSServerClosed__DelegateSignature,
				(UObject* (*)())Z_Construct_UDelegateFunction_Rodin_WSServerPin__DelegateSignature,
			};
			static const UECodeGen_Private::FPackageParams PackageParams = {
				"/Script/Rodin",
				SingletonFuncArray,
				UE_ARRAY_COUNT(SingletonFuncArray),
				PKG_CompiledIn | 0x00000040,
				0xB7FE8C08,
				0x6C00E890,
				METADATA_PARAMS(nullptr, 0)
			};
			UECodeGen_Private::ConstructUPackage(Z_Registration_Info_UPackage__Script_Rodin.OuterSingleton, PackageParams);
		}
		return Z_Registration_Info_UPackage__Script_Rodin.OuterSingleton;
	}
	static FRegisterCompiledInInfo Z_CompiledInDeferPackage_UPackage__Script_Rodin(Z_Construct_UPackage__Script_Rodin, TEXT("/Script/Rodin"), Z_Registration_Info_UPackage__Script_Rodin, CONSTRUCT_RELOAD_VERSION_INFO(FPackageReloadVersionInfo, 0xB7FE8C08, 0x6C00E890));
PRAGMA_ENABLE_DEPRECATION_WARNINGS
