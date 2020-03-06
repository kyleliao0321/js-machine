import * as math from 'mathjs';
import * as _ from 'lodash';

export default class RegressionTree {
    public name: string = 'RegressionTree';
    private options: any;
    private maxDepth: number;
    private minSamples: number;
    private splitColumn: number;
    private isLeaf: boolean;
    private minCV: number;
    private predictValue?: number;
    private splitFeature: number[];
    private childs: RegressionTree[];
    constructor(options: any) {
        this.options = options;
        this.maxDepth = options.maxDepth || 3;
        this.minSamples = options.minSamples || 3;
        this.minCV = options.minCV || 5;
        this.isLeaf = options.isLeaf || false;
        this.childs = [];
        this.splitFeature = [];
        this.splitColumn = 0;
    }

    /**
     * Recursivly build the tree to partition the dataset by features.
     * When the partition satisfy the terminate condition, make the current node as leaf.
     * Then get the mean of remaining target value array as prediction traffic.
     * @param X             2D matrix with dimension (# samples, features).
     * @param y             2D matrix with dimension (# samples, 1).
     * @param currentDepth  Current depth.
     */
    public fit(X: any, y: any, currentDepth: number= 0) {
        const XTranspose = math.transpose(X);
        const yTranspose = math.transpose(y);
        const XtransArr: any = XTranspose.valueOf();
        const yTransArr: any = yTranspose.valueOf()[0];

        // Terminate conditions.
        if (this.CoefficientOfVarience(yTransArr) < this.minCV) {
            this.isLeaf = true;
            this.predictValue = this.getPredicValue(yTransArr);
            return;
        }
        if (XtransArr.length <= this.minSamples) {
            this.isLeaf = true;
            this.predictValue = this.getPredicValue(yTransArr);
            return;
        }
        if (currentDepth > this.maxDepth) {
            this.isLeaf = true;
            this.predictValue = this.getPredicValue(yTransArr);
            return;
        }
        // Find the best feature (lowest sum of square error) to split the dataset.
        const {splitColumn, minSSE} = this.bestSplit(XtransArr, yTransArr);
        this.splitColumn = splitColumn;

        // Split the dataset into multiple pairs, each with different split-feature-value.
        const splitPairs = this.splitData(X, y, splitColumn, _.uniq(XtransArr[splitColumn]));
        Object.keys(splitPairs).forEach((f: string) => {
            this.splitFeature.push(Number(f));
            const newTree = new RegressionTree(this.options);
            const xMatrix: any = math.matrix(splitPairs[f].X);
            const yMatrix: any = math.transpose(math.matrix([splitPairs[f].y]));
            newTree.fit(xMatrix, yMatrix, currentDepth + 1);
            this.childs.push(newTree);
        });
    }
    public predict(X: any): any {
        const Xarray = X.toArray();
        const result: number[] = [];
        Xarray.forEach((feature: number[]) => {
            result.push(this._predict(feature));
        });
        return result;
    }
    /**
     * Traveling down the tree to find the belonging leaf.
     */
    private _predict(feature: number[]): any {
        if (!this.isLeaf) {
            const f = feature[this.splitColumn];
            const randomSelect: any = _.sample(this.splitFeature);
            // If the feature is new in Tree, randomly select a feature to travle.
            const childIndex = this.splitFeature.indexOf(f) === -1 ? this.splitFeature.indexOf(randomSelect) : this.splitFeature.indexOf(f);
            return this.childs[childIndex]._predict(feature);
        }
        return this.predictValue;
    }
    private getPredicValue(yTransArr: number[]) {
        return math.mean(yTransArr);
    }
    private bestSplit(XtransArr: number[][], yTransArr: number[]): {splitColumn: number, minSSE: number} {
        const rootSTD = this.StandardDeviation(yTransArr);
        const eachSSE = XtransArr.map((feature: number[]) => {
            // return rootSTD - this.branchSTD(feature, yTransArr);
            return this.SumOfSquareError(feature, yTransArr);
        });
        let splitColumn = 0;
        // let maxSDR = -Infinity;
        // eachSDR.forEach((SDR: number, index: number) => {
        //     if (SDR > maxSDR) {
        //         maxSDR = SDR;
        //         splitColumn = index;
        //     }
        // });
        let minSSE = Infinity;
        eachSSE.forEach((SSE: number, index: number) => {
            if (SSE < minSSE) {
                minSSE = SSE;
                splitColumn = index;
            }
        });
        return {
            splitColumn,
            minSSE
        };
    }
    private branchSTD(feature: number[], target: number[]): number {
        const eachInfo: any = {};
        feature.forEach((f: number, index: number) => {
            if (!eachInfo[f]) {
                eachInfo[f] = {
                    target: [],
                    count: 0
                };
            }
            eachInfo[f].target.push(target[index]);
            eachInfo[f].count++;
        });
        let STD = 0;
        Object.keys(eachInfo).forEach((f: string) => {
            const p = (eachInfo[f].count / feature.length);
            const std = this.StandardDeviation(eachInfo[f].target);
            STD += p * std;
        });
        return STD;
    }
    private splitData(X: any, y: any, splitColumn: number, classes: number[]): any {
        const splitPairs: any = {};
        for (let i = 0; i < X.size()[0]; i++) {
            const sample: any = math.row(X, i).valueOf()[0];
            const f = sample[splitColumn];
            if (!splitPairs[f]) {
                splitPairs[f] = {
                    X: [],
                    y: []
                };
            }
            sample.splice(splitColumn, 1);
            splitPairs[f].X.push(sample);
            splitPairs[f].y.push(y.get([i, 0]));
        }
        return splitPairs;
    }
    private StandardDeviation(arr: number[]): number {
        return math.std(arr);
    }
    private CoefficientOfVarience(arr: number[]): number {
        const mean = math.mean(arr);
        const std = this.StandardDeviation(arr);
        return (std / mean) * 100;
    }
    private SumOfSquareError(feature: number[], target: number[]): number {
        const eachInfo: any = {};
        feature.forEach((f: number, index: number) => {
            if (!eachInfo[f]) {
                eachInfo[f] = {
                    target: [],
                };
            }
            eachInfo[f].target.push(target[index]);
        });
        let SSE = 0;
        Object.keys(eachInfo).forEach((f: string) => {
            const mean = math.mean(eachInfo[f].target);
            const tmp = target.map((d: number) => Math.pow(d - mean, 2));
            SSE += math.sum(tmp);
        });
        return SSE;
    }
}
