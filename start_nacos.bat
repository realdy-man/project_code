@echo off

REM 检查Java环境
java -version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Java环境，请确保已安装Java 8或更高版本并配置JAVA_HOME环境变量
    pause
    exit /b 1
)

echo Java环境检查通过

REM 切换到Nacos目录
cd /d "%~dp0nacos-server-2.4.3\nacos\bin"

echo 正在启动Nacos服务...
echo 模式: standalone

echo 正在执行启动命令...
startup.cmd -m standalone

echo Nacos服务已启动
REM 设置环境变量
set MODE=standalone
set NACOS_AUTH_ENABLE=false

echo Nacos服务启动完成，访问地址: http://localhost:8848
pause
