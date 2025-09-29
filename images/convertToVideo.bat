echo off
REM 2019-12-23, Johannes Gerstmayr
REM
REM helper file for EXUDYN to convert all frame00662.png, frame00663.png, ... files to a video
REM for higher quality use crf option (standard: -crf 23, range: 0-51, lower crf value means higher quality)

IF EXIST animation.mp4 (
    echo "animation.mp4 already exists! rename the file"
) ELSE (
   ffmpeg.exe -r 15 -start_number 662 -i frame%%05d.png -c:v libx264 -vf "fps=65,format=yuv420p" animation.mp4
)
