import * as math from 'mathjs';
import * as _ from 'lodash';
import KDTreeNode from './KDTreeNode';

export default class KNN {
    public name: string = 'KNN';
    private K!: number;
    private root!: KDTreeNode|null;
    constructor(config: any) {
        // Number of nearest neighbor to calculate the prediction traffic.
        this.K = config.K || 5;
        this.root = null;
    }

    /**
     * By Building KD Tree data strcuture, it can reduce time complexity from O(n^3) -> O(D*K*log(n)) in searching
     * @param X 2D matrix with dimension (# samples, features).
     * @param y 2D matrix with dimension (# samples, 1).
     */
    public fit(X: any, y: any) {
        const zip = _.zip(X.toArray(), _.flatten(y.toArray()));
        this.root = this.buildTree(zip, 0);
    }

    /**
     * Obtained the nearest neighbor list, and using distance as weight of each neighbors to calculate the traffics.
     * @param X 2D matrix with dimension (# samples, features).
     */
    public predict(X: any) {
        function softmax(array: number[]) {
            return math.divide(math.exp(array), math.sum(math.exp(array))).map((d: number) => d ? d : 0);
        }
        return X.toArray().map((x: number[]) => {
            const closeList = this.searchNearestNeighbors(this.root, x, 1, []);
            const dArray = closeList.map((o: any) => o.d);
            const weights = softmax(dArray);
            let tmp = 0;
            closeList.forEach((o: any, idx: number) => tmp += weights[idx] * o.y);
            return tmp;
        });
    }

    /**
     * Resursivly build the KD Tree for later searching.
     * Inorder to make prediction stable and the KD Tree be balanced, the function first sort the nodeList
     * based on given compareDimension. Then, using the mid-node as new node, and partition the nodeList to
     * left and right.
     * @param nodeList  Currently remaining nodes which are not included in tree yet.
     * @param depth     Current Node depth, for determining which dimension to compare.
     */
    private buildTree(nodeList: any[], depth: number): null|KDTreeNode {
        if (nodeList.length === 0) {
            return null;
        }

        const features = nodeList[0][0].length;
        const compareDimension = depth % features;
        // ! ES2019/ES10 already make sorting time complexity stable by using TimSort
        // ! But FireFox is still unstalbe in this situation, time complexity is roughly 2 times longer than Chrome.
        nodeList.sort((a, b) => a[0][compareDimension] - b[0][compareDimension]);

        const median = Math.floor(nodeList.length / 2);

        return new KDTreeNode(
            nodeList[median][0],
            nodeList[median][1],
            this.buildTree(nodeList.slice(0, median), depth + 1),
            this.buildTree(nodeList.slice(median + 1), depth + 1)
        );
    }
    /**
     * Recursivly update the neighbors list by traveling down the KD Tree.
     * The detail of searching algorithm can be found at here : https://en.wikipedia.org/wiki/K-d_tree
     * @param node      Currently visted node.
     * @param x         Query node.
     * @param depth     Currently visted depth.
     * @param neighbor  Current neighbors.
     */
    private searchNearestNeighbors(node: KDTreeNode|null, x: number[], depth: number, neighbor: any[]): any[] {
        if (node === null) {
            return neighbor;
        }
        const max = this.maxInNeighbors(neighbor);
        const curDistance = node.distanceWith(x);
        if (curDistance < max) {
            neighbor = this.updateNeighbors(node.y, curDistance, neighbor);
        }

        const compareDimension = depth % x.length;
        const direction = node.compareBoundary(x[compareDimension], compareDimension);
        const withinSideNode = direction === 'left' ? node.left : node.right;
        const outsideNode = direction === 'left' ? node.right : node.left;
        const boundaryDistance = node.boundaryDistance(x[compareDimension], compareDimension);

        neighbor = this.searchNearestNeighbors(withinSideNode, x, depth + 1, neighbor);
        if (boundaryDistance < curDistance) {
            // When the distance btw query node and current node is longer than the distance with boundary.
            // It is possible that the much closer node is on the other side of boundary.
            // Otherwise, skip the other side.
            return this.searchNearestNeighbors(outsideNode, x, depth + 1, neighbor);
        }
        return neighbor;
    }
    private maxInNeighbors(neighbor: any[]): number {
        if (neighbor.length === 0) {
            return Infinity;
        }
        let max = -Infinity;
        neighbor.forEach((o: any) => max = o.d > max ? o.d : max);
        return max;
    }
    private argMaxOfNeighbors(neighbor: any[]): number {
        const max = this.maxInNeighbors(neighbor);
        for (let i = 0; i < neighbor.length; i++) {
            if (max === neighbor[i].d) {
                return i;
            }
        }
        return -1;
    }
    private updateNeighbors(y: number, d: number, neighbor: any[]) {
        if (neighbor.length < this.K) {
            neighbor.push({
                y,
                d
            });
            return neighbor;
        }
        const argMax = this.argMaxOfNeighbors(neighbor);
        return neighbor.map((o: any, idx: number) => {
            if (idx === argMax) {
                return {
                    y,
                    d
                };
            }
            return o;
        });
    }
}