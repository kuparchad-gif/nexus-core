// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class Rodin : ModuleRules
{
	public Rodin(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
        string uWebSocketsRootDir = Path.Combine(PluginDirectory, "Source/ThirdParty/uWebSockets/");
        CppStandard = CppStandardVersion.Cpp17;
        
        PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
                "WebSockets",
                "zlib",
                "OpenSSL",
                "Json",
                "HTTP",
                "ImageWrapper",
                "ImageCore",
                "DesktopPlatform",
                "UnrealEd"
            }
			);
			
		
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore",
				"WebSockets",
				"zlib", 
				"OpenSSL",
				"Json",
                "ImageWrapper",
                "ImageCore"
				// ... add private dependencies that you statically link with here ...	
			}
			);

        if (Target.Type == TargetRules.TargetType.Editor)
        {
            PublicDependencyModuleNames.AddRange(new string[] {
                "Blutility",
                "Slate"
            });

            PrivateDependencyModuleNames.AddRange(
                new string[]
                {
                    "Projects",
                    "UnrealEd",
                    "ToolMenus",
                    "EditorScriptingUtilities",
                    "UMGEditor",
                    "SlateCore"
					// ... add private dependencies that you statically link with here ...	
				}
            );
        }
        if (Target.Platform == UnrealTargetPlatform.Mac)
        {
            PublicFrameworks.AddRange(new string[]
            {
                "CoreFoundation",
                "CoreServices"
            });
        }
        PublicIncludePaths.Add(Path.Combine(ModuleDirectory, "Public"));
        PrivateIncludePaths.Add(Path.Combine(ModuleDirectory, "Private"));

        PrivateIncludePaths.Add(Path.Combine(uWebSocketsRootDir, "includes"));

        PublicDefinitions.Add("WITH_WEBSOCKET_SERVER=1");
    }
}
