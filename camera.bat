@echo off
setlocal enabledelayedexpansion

REM 设置默认源图像路径
set "default_src_image=assets\examples\source\s12.jpg"
set "src_image=%default_src_image%"
set "animal_param="
set "paste_back="

REM 解析命名参数
:parse_args
if "%~1"=="" goto end_parse_args
if /i "%~1"=="--src_image" (
    set "src_image=%~2"
    shift
) else if /i "%~1"=="--animal" (
    set "animal_param=--animal"
) else if /i "%~1"=="--paste_back" (
    set "paste_back=--paste_back"
)
shift
goto parse_args
:end_parse_args

echo source image: [!src_image!]
echo use animal: [!animal_param!]
echo paste_back: [!paste_back!]

REM 执行Python命令
.\venv\python.exe .\run.py --cfg configs/trt_infer.yaml --realtime --dri_video 0 --src_image !src_image! !animal_param! !paste_back!

endlocal