# VRDemo
This is a VR demo, Only works on Windows now.

##使用说明
1. 程序依赖Tesseract，使用前修改CMakeLists.txt中Tesseract相关项，然后编译出VS2013的工程即可运行。
2. 程序默认的摄像头编号为1，根据个人情况进行修改。
3. 程序默认对摄像头画面进行了水平和垂直方向的镜像操作，如果不需要，可以在main函数开头处注释掉相关代码。

已知Bug：
	1. 程序运行前，将摄像头遮挡或者转向空白桌面，正常启动后，再移动到目标环境下。不然程序会在初始化时崩溃。