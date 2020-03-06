import * as math from 'mathjs';
/**
 * Helper class for KD Tree data structure.
 */
export default class KDTreeNode {
    x!: number[];
    public y!: number;
    public left!: KDTreeNode|null;
    public right!: KDTreeNode|null;
    constructor(...args: any) {
        this.x = args[0];
        this.y = args[1];
        this.left = args[2];
        this.right = args[3];
    }

    public asParent(node: KDTreeNode, direction: string) {
        if (direction === 'left') {
            this.left = node;
        } else {
            this.right = node;
        }
    }
    public compareBoundary(x: number, idx: number) {
        if (x <= this.x[idx]) {
            return 'left';
        }
        return 'right';
    }
    /**
     * Using Euclidean Distance to calculate the distance btw two nodes.
     * This is the actual distance btw two nodes.
     * @param x compare node's position.
     */
    public distanceWith(x: number[]): any {
        const tmp: any = math.subtract(x, this.x);
        return math.pow(math.sum(tmp.map((d: number) => math.pow(d, 2))), 0.5);
    }
    /**
     * The distance between query node and boundary axis.
     * This might be smaller than Euclidean Distance.
     * @param x
     * @param idx
     */
    public boundaryDistance(x: number, idx: number) {
        return Math.abs(x - this.x[idx]);
    }
}