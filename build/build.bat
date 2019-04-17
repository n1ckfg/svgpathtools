@echo off

set BUILD_TARGET=svgpathtools.py
cd %cd%

del %BUILD_TARGET%

copy /b LICENSE.txt+..\svgpathtools\__init__.py+..\svgpathtools\bezier.py+..\svgpathtools\document.py+..\svgpathtools\misctools.py+..\svgpathtools\parser.py+..\svgpathtools\path.py+..\svgpathtools\paths2svg.py+..\svgpathtools\polytools.py+..\svgpathtools\smoothing.py+..\svgpathtools\svg_io_sax.py+..\svgpathtools\svg_to_paths.py %BUILD_TARGET%

@pause