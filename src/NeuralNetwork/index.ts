import * as math from 'mathjs';
import * as _ from 'lodash';

export default class NeuralNetwork {
    public name: string = 'NeuralNetwork';
    private momentum!: number;
    private learningRate!: number;
    private epochs!: number;
    private inputSize!: number;
    private hiddenSize!: number;
    private outputSize!: number;
    private activation!: (x: number, derivative: boolean) => number;
    private w1!: any;
    private w2!: any;
    constructor(config: any) {
        this.momentum = config.momentum || 0.9;
        this.learningRate = config.learningRate || 0.01;
        this.epochs = config.epochs || 10;
        this.hiddenSize = config.hiddenSize || 10;
        this.outputSize = config.outputSize || 1;

        switch (config.activation) {
            case 'sigmoid':
                this.activation = sigmoid;
                break;
            case 'relu':
                this.activation = ReLU;
                break;
            default:
                this.activation = ReLU;
                break;
        }
    }

    public fit(X: any, y: any) {
        this.inputSize = X.toArray()[0].length;
        // this.w1 = math.random([this.inputSize, this.hiddenSize], -1.0, 1.0);
        // this.w2 = math.random([this.hiddenSize, this.outputSize], -1.0, 1.0);
        this.w1 = math.zeros([this.inputSize, this.hiddenSize]);
        this.w2 = math.zeros([this.hiddenSize, this.outputSize]);
        for (let i = 0; i < this.epochs; i++) {
            const inputLayer = X;
            const hiddenLayer = math.multiply(inputLayer, this.w1).map((d: number) => sigmoid(d, false));
            const outputLayer = math.multiply(hiddenLayer, this.w2).map((d: number) => this.activation(d, false));

            const outputError = math.subtract(y, outputLayer);
            const deltaOutput = math.dotMultiply(outputError, outputLayer.map((d: number) => sigmoid(d, true)));
            const hiddenError = math.multiply(deltaOutput, math.transpose(this.w2));
            const deltaHidden = math.dotMultiply(hiddenError, hiddenLayer.map((d: number) => this.activation(d, true)));

            this.w2 = math.add(math.multiply(this.w2, this.momentum), math.multiply(math.transpose(hiddenLayer), math.multiply(deltaOutput, this.learningRate)));
            this.w1 = math.add(math.multiply(this.w1, this.momentum), math.multiply(math.transpose(inputLayer), math.multiply(deltaHidden, this.learningRate)));
        }
    }

    public predict(X: any) {
        const inputLayer = X;
        const hiddenLayer = math.multiply(inputLayer, this.w1).map((d: number) => sigmoid(d, false));
        const outputLayer = math.multiply(hiddenLayer, this.w2).map((d: number) => this.activation(d, false));

        return _.flatten(outputLayer.toArray());
    }
}

function sigmoid(x: number, derivative: boolean): number {
    const fx = 1 / (1 + math.exp(-x));
    if (derivative) {
       return fx * (1 - fx);
    }
    return fx;
}

function ReLU(x: number, derivative: boolean): number {
    if (x < 0) {
        return 0;
    }
    if (derivative) {
        return 1;
    }
    return x;
}

function tanh(x: number, derivative: boolean): number {
    if (derivative) {
        return 1 - math.pow(math.tanh(x), 2);
    }
    return math.tanh(x);
}