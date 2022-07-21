# matplotlib-mock-tests
Файл regression_tests.py содержит тесты для проверки правильности построения графиков согласно заданию.
Задание находится в файлах regression_exercise.ipynb и regression_exercise_2.ipynb в разделе Visualization.</br></br>
Для тестирования кода необходимо cконвертировать файл .ipynb в файл .py. С помощью команды nbconvert файлы
regression_exercise.ipynb и regression_exercise_2.ipynb были сконвертированы в regression_exercise.py и regression_exercise_2.py.
Название тестируемого файла следует указать в 3 строке файла regression_tests.py после import, а также в 6 строке после @patch.
Пример: 
    import unittest
    from unittest.mock import patch
    import <filename> as ex
    
    
    @patch('<filename>.plt')
