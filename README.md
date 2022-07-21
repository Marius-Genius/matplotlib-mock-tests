# matplotlib-mock-tests
<p align="justify">
Файл regression_tests.py содержит тесты для проверки правильности построения графиков согласно заданию.
Задание находится в файлах regression_exercise.ipynb и regression_exercise_2.ipynb в разделе Visualization.</br></br></p>
Для тестирования кода необходимо cконвертировать файл .ipynb в файл .py. с помощью команды

    jupyter nbconvert --to script <имя_файла>.ipynb

<p align="justify">
Файлы regression_exercise.ipynb и regression_exercise_2.ipynb были сконвертированы в regression_exercise.py и regression_exercise_2.py.
Код, строящий графики, в файлах regression_exercise.py и regression_exercise_2.py был оформлен в функцию plot_model(), не возвращающую значений,
после этого был добавлен вызов данной функции.
Название тестируемого файла следует указать в 3 строке файла regression_tests.py после import, а также в 6 строке после @patch.</br></p>
Пример:

    import unittest
    from unittest.mock import patch
    import <имя_файла> as ex
    
    
    @patch('<имя_файла>.plt')

После этого необходимо сохранить файл и запустить тесты с помощью команды

    python3 regression_tests.py
