import {GradientLinearRegression} from './LinearRegression';
import NeuralNetwork from './NeuralNetwork';
import {RegressionTree} from './Tree';
import KNN from './KNN';

const jsmachine = {
    NeuralNetwork,
    KNN,
    Tree: {
        RegressionTree
    },
    LinearRegression: {
        GradientLinearRegression
    },
}

if (typeof window !== 'undefined') {
    window.jsmachine = jsmachine //eslint-disable-line
  }
  
  if (typeof module !== 'undefined') {
    module.exports = jsmachine;
  }

