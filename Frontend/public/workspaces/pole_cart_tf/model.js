
const ACTIONS = new Array();
const OPTIONS = [-1, -0.5, 0, 0.5, 1];

var index = 0;
var i;
for(i = 0; i < OPTIONS.length; i++) {
	let joint_0 = OPTIONS[i];
	
  var j;
  for (j = 0; j < OPTIONS.length; j++) {
  	let joint_1 = OPTIONS[j];
    
    var k;
    for (k = 0; k < OPTIONS.length; k++) {
   		let joint_2 = OPTIONS[k];
      let action = {
      	joint_0: joint_0,
        joint_1: joint_1,
        joint_2: joint_2,
        index: index++
      }
      console.log(JSON.stringify(action, null, 3));
      ACTIONS.push(action);
    }
  }
}

const NUM_ACTIONS = index;
console.log(`num_actions = ${NUM_ACTIONS}`);

function indexToAction(index) {
    var i;
    for (i = 0; i < ACTIONS.length; i++) {
        if (ACTIONS[i].index == index) {
            return [
                ACTIONS[i].joint_0,
                ACTIONS[i].joint_1,
                ACTIONS[i].joint_2
            ]
        }
    }
    return undefined;
}

function actionToIndex(action) {
    var i;
    for (i = 0; i < ACTIONS.length; i++) {
        if (ACTIONS[i].joint_0 == action.joint_0
         && ACTIONS[i].joint_1 == action.joint_1
         && ACTIONS[i].joint_2 == action.joint_2) {
             return ACTIONS[i].index;
        }
    }
    return undefined;
}


class Model {
    constructor(hiddenLayerSizesOrModel, numStates, numActions, batchSize) {
        this.numStates = numStates;
        this.numActions = numActions;
        this.batchSize = batchSize;
        
        if (hiddenLayerSizesOrModel instanceof tf.LayersModel) {
            this.network = hiddenLayerSizesOrModel;
            this.network.summary();
            this.network.compile({optimizer: 'adam', loss: 'meanSquaredError'});
        } else {
            this.defineModel(hiddenLayerSizesOrModel);
        }
    }
    
    defineModel(hiddenLayerSizes) {
        if (!Array.isArray(hiddenLayerSizes)) {
            hiddenLayerSizes = [hiddenLayerSizes];
        }
        this.network = tf.sequential();
        hiddenLayerSizes.forEach((hiddenLayerSize, i) => {
            this.network.add(tf.layers.dense({
                units: hiddenLayerSize,
                activiation: 'relu',
                inputShape: i == 0 ? [this.numStates] : undefined
            }));
        });
        this.network.add(tf.layers.dense({units: this.numActions}));
        
        this.network.summary();
        this.network.compile({optimizer: 'adam', loss: 'meanSquaredError'});
    }
    
    predict(states) {
        return tf.tidy(() => this.network.predict(states));
    }
    
    async train(xBatch, yBatch) {
        await this.network.fit(xBatch, yBatch);
    }
    
    // Returns action -1, 0, 1
    chooseAction(state, eps) {
        if (Math.random() < eps) {
            return indexToAction(Math.floor(Math.random() * this.numActions));
        } else {
            return tf.tidy(() => {
                return indexToAction(this.network.predict(state).argMax(1).dataSync()[0]);
                //const logits = this.network.predict(state);
                //const sigmoid = tf.sigmoid(logits);
                //const probs = tf.div(sigmoid, tf.sum(sigmoid));
                //return indexToAction(tf.multinomial(probs, 1).dataSync()[0]);
            });
        }
    }
}