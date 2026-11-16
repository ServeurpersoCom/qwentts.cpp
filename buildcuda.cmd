@echo off
setlocal enabledelayedexpansion
rem Locate the newest Visual Studio with the C++ toolset (Community, Professional,
rem Enterprise or Build Tools alike) instead of hardcoding one edition's path.
rem Delayed expansion: the "(x86)" of the path would otherwise close the for's parenthesis.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`""!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath"`) do set "VSDIR=%%i"
if not defined VSDIR (
    echo No Visual Studio with the C++ toolset found. Install "Desktop development with C++" and retry.
    exit /b 1
)
call "!VSDIR!\VC\Auxiliary\Build\vcvars64.bat"
rem CMake and Ninja ship with Visual Studio; use them when none is on PATH.
set "PATH=!VSDIR!\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;!VSDIR!\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;!PATH!"

rd /s /q build 2>nul
mkdir build 2>nul
cd build

cmake .. -G "Ninja Multi-Config" -DGGML_CUDA=ON
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

cd ..
