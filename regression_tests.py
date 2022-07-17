import unittest
from unittest.mock import patch
import regression_exercise as ex


@patch('regression_exercise.plt')
class TestPlotModel(unittest.TestCase):

    def test_scatter_plot(self, mock_plt):
        ex.plot_model()
        x = ex.df.keys().tolist()[0]
        y = ex.df.keys().tolist()[1]
        passed = True
        try:
            mock_plt.scatter.assert_called_once_with(ex.df[x].values, ex.df[y].values)
        except:
            passed = False
        finally:
            assert passed, 'Check scatter plot'

    def test_first_model_plot(self, mock_plt):
        ex.plot_model()
        arg = mock_plt.plot.call_args_list
        is_in_calls = False
        is_smooth = False
        for i in range (len(arg)):
            x = arg[i][0][0]
            y = arg[i][0][1]
            func = ex.first_model_coeffs[0] + ex.first_model_coeffs[1] * x + ex.first_model_coeffs[2] * x ** 2
            if (len(y) == len(func)):
                is_in_calls = True
                for j in range (len(y)):
                    if (y[j] != func[j]):
                        is_in_calls = False
                        break
            if (is_in_calls):
                is_smooth = max(x) - min(x) + 1 < len(x)
                break

        assert is_in_calls, 'Check first model plot'
        assert is_smooth, 'First model curve should be smoother'

    def test_second_model_plot(self, mock_plt):
        ex.plot_model()
        arg = mock_plt.plot.call_args_list
        correct_coeffs = True
        is_in_calls = False
        is_smooth = False
        for i in range (len(arg)):
            x = arg[i][0][0]
            y = arg[i][0][1]
            func = [0] * len(x)
            try:
                for j in range (len(x)):
                    for p in range (ex._model_power + 1):
                        func[j] = func[j] + ex.second_model_coeffs[p] * x[j] ** p
            except:
                correct_coeffs = False
                break
            if (len(y) == len(func)):
                is_in_calls = True
                for k in range (len(y)):
                    if (y[k] != func[k]):
                        is_in_calls = False
                        break
            if (is_in_calls):
                is_smooth = max(x) - min(x) + 1 < len(x)
                break

        assert correct_coeffs, 'Check second_model_coeffs'
        assert is_in_calls, 'Check second model plot'
        assert is_smooth, 'Second model curve should be smoother'

    @unittest.skipUnless(len(ex.third_model_coeffs) > 0, 'No third_model_coeffs')
    def test_third_model_plot(self, mock_plt):
        ex.plot_model()
        arg = mock_plt.plot.call_args_list
        is_in_calls = False
        is_smooth = False
        for i in range (len(arg)):
            x = arg[i][0][0]
            y = arg[i][0][1]
            func = ex.third_model_coeffs[1] * x + ex.third_model_coeffs[0] 
            if (len(y) == len(func)):
                is_in_calls = True
                for j in range (len(y)):
                    if (y[j] != func[j]):
                        is_in_calls = False
                        break
            if (is_in_calls):
                is_smooth = max(x) - min(x) + 1 < len(x)
                break

        assert is_in_calls, 'Check third model plot'
        assert is_smooth, 'Third model curve should be smoother'
        
            
if __name__ == '__main__':
    unittest.main()