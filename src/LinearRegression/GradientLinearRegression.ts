import * as math from 'mathjs';
import * as _ from 'lodash';

export default class GradientLinearRegression {
    private batchSize!: number;
    private epochs!: number;
    private learningRate!: number;
    private features!: number;
    private theta!: number[];
    constructor(config: any) {
        this.batchSize = config.batchSize || 8;
        this.epochs = config.epochs || 10;
        this.learningRate = config.learningRate || 0.15;
    }

    public fit(X: any, y: any) {
        // * X(m*n): m-># samples ; n->features;
        // * y(m*1): m-># samples;
        this.features = X.toArray()[0].length;
        this.theta = _.fill(Array(this.features), 0);
        for (let i = 0; i < this.epochs; i++) {
            this.BGD(this.theta, X.toArray(), y.toArray());
        }
    }
    public predict(X: any) {
        return math.multiply(this.theta, math.transpose(X)).toArray();
    }

    private hypothesis(theta: any, X: any) {
        // * X(m * n): m->number of features; n->batch size
        const h = math.multiply(theta, X);
        return h;
    }
    private BGD(theta: number[], X: number[][], y: number[]) {
        const batchs = Math.ceil(X.length / this.batchSize);
        for (let j = 0; j < batchs; j++) {
            const Xbatch = X.slice(j * this.batchSize, (j + 1) * this.batchSize);
            const Ybatch = _.flatten(y).slice(j * this.batchSize, (j + 1) * this.batchSize);
            const h = this.hypothesis(theta, math.transpose(Xbatch));
            for (let k = 0; k < this.features; k++) {
                const b = math.sum(math.dotMultiply(math.subtract(h, Ybatch), math.transpose(math.matrix(Xbatch)).toArray()[k]));
                theta[k] = theta[k] - (this.learningRate / this.features) * b;
            }
        }
        this.theta = theta;
    }
}