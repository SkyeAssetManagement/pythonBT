@echo off
echo Running main.py with debug output capture...
python main.py ES time_window_strategy_vectorized --useDefaults --start "2020-01-01" > debug_output.txt 2>&1 &
echo Command started, output going to debug_output.txt
echo Wait 10 seconds then check the file...
timeout /t 10
echo Showing debug output:
type debug_output.txt