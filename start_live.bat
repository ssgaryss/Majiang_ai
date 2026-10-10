@echo off
rem ============================================================
rem  start_live.bat -- double-click entry (all logic lives in start_live.ps1)
rem
rem  ASCII-ONLY ON PURPOSE: cmd reads this file with the OEM code page,
rem  so non-ASCII comments turn into stray commands. (Same lesson the old
rem  repo recorded for watchdog.ps1 -- see its header.)
rem
rem  Uses %~dp0 only (this file's own directory) => no absolute path is
rem  hard-coded; move the folder anywhere and it still works.
rem  Self-check:  start_live.bat -Check
rem ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_live.ps1" %*
