# CV-Lab-1.-Chromakey
1. Реализовать программу согласно варианту задания: Chroma key. Задано некоторое изображение того же размера, что кадр видео и некоторый базовый цвет (область). На вход поступает изображение, программа отрисовывает окно, в которое выводится либо исходное изображение, либо применения замены всех пикселей 
заданного цвета на пиксели заданного изображения (переключение по нажатию клавиши). 
Базовый алгоритм, используемый в программе, необходимо реализовать в 3 вариантах: 
с использованием встроенных функций какой-либо библиотеки (OpenCV, PIL и др.) и нативно на Python + |с использованием Numba или C++|.
2. Сравнить быстродействие реализованных вариантов.
3. Сделать отчёт в виде readme на GitHub, там же должен быть выложен исходный код.

Хромакей — технология совмещения двух и более изображений или кадров в одной композиции

## Основной алгоритм 
На вход получаем обрабатываемое изображение с зелёным или синим фоном и изображения заднего фона.
1. Создаём черно-белую маску для изначального изображения. Черный - для самого изображения, белый для фона.
2. С помощью маски вырезаем исходное изображения, добавляя к нему черный фон.
3. С помощью маски вырезаем черный силуэт изображения на фоне.
4. Складываем два получившихся изображения
5. Выводим результат

В случае с видео, алгоритм повторяется последовательно для каждого кадра
## Структура программы

Программа состоит из двух модулей,в модуле chrom_script.py содержится реализация функции замены фона в трёх вариантах:
с использованием библиотеки cv, нативно на Python, с использованием Numba 
В модуле main происходит основная обработка видео, а также замеры скорости работы функция для одного кадра.

## Результат тестирования скорости работы.
![скорость работы](https://user-images.githubusercontent.com/114875779/214222771-a90733d5-8db5-41eb-a7a7-638def97cb95.PNG)

Как можно видеть, numba ускоряет работу алгоритма почти в 10 раз, библиотечная функция выполняется быстрее всех
