@RD /S /Q %~dp0source\generated

sphinx-apidoc -o source\generated %par%simplepyml -e -t source\_templates\ -d -1 -T

./make.bat html