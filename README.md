# BottomUpHPE
a bottom-up human pose estimator

this repository is a python implementation of [4D Association](https://github.com/zhangyux15/4d_association)

![image](./resources/show.jpg)

The data directory tree should look like this:

```
${ROOT}
|-- data
    |-- seq_1
    |   |-- calibrations.json
    |   |-- skel.txt
    |   |-- detection
    |	|	|-- 18181923.txt
    |	|	|-- ...
    |	|	|-- 18307870.txt
    |   |-- video
    |	|	|-- 18181923.avi
    |	|	|-- ...
    |	|	|-- 18307870.avi
    |-- seq_2
    |   |-- calibrations.json
    |   |-- skel.txt
    |   |-- detection
    |	|	|-- 18181923.txt
    |	|	|-- ...
    |	|	|-- 18307870.txt
    |   |-- video
    |	|	|-- 18181923.avi
    |	|	|-- ...
    |	|	|-- 18307870.avi
```

prepare openpose model weight
1. download weight from [google drive](https://drive.google.com/file/d/1ghXakEXhBMCdV78K6tCFTPp_vjJDWmcE/view?usp=drive_link)
2. place this file into ./openpose/weigth/

demo:

```
python main.py
```

