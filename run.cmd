@echo off
set "PATH=D:\Qt\5.15.2\msvc2019_64\bin;D:\opencv4.8.0\opencv\build\x64\vc16\bin;%PATH%"
"%~dp0build-vs\Release\patmax_gui.exe" %*
