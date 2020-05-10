

#test dataset
# nosetests -v --nocapture tests/dataset_tests.py:dataset_test

#test layers
# nosetests -v --nocapture tests/neuron_tests.py:neuron_test
# nosetests -v --nocapture tests/layers_tests.py:layer_test

#test feedforward
# nosetests -v --nocapture tests/feedforward_tests.py:feedforward_test

#test backpropagation
nosetests -v --nocapture tests/feedforward_tests.py:backpropagation_test
