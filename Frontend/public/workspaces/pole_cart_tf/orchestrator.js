const MIN_EPSILON = 0.01;
const MAX_EPSILON = 0.2;
const LAMBDA = 0.01;

class Orchestrator {
    
    constructor(maxStepsPerGame, model, memory, discountRate) {
        this.maxStepsPerGame = maxStepsPerGame;
        this.model = model;
        this.memory = memory;
        this.eps = 0;
        this.steps = 0;
        this.discountRate = discountRate;
        
        this.rewardStore = new Array();
        this.maxPositionStore = new Array();
    }
    
    
    async run() {
        // reset the simulation. And get a new scene_data to create env.
        await api.doReset();
        api.latest_scene_data = null;
        let scene_data = await api.getSceneData();
        let env = new Environment(scene_data);
        
        let state = this.env.getStateTensor();
        let totalReward = 0;
        let step = 0;
        
        while (step < this.maxStepsPerGame) {
            const action = this.model.chooseAction(state, this.eps)
            console.log("step " + step + ": action = " + action);
            const done = this.env.update(action);
            const reward = this.env.upright_millis;
            
            let nextState = this.env.getStateTensor();
            
            if (done) nextState = null;
            
            this.memory.addSample([state, action, reward, nextState]);
            
            this.steps += 1;
            this.eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * Math.exp(-LAMBDA * this.steps);
            
            state = nextState;
            totalReward = reward;
            step += 1;
            
            if (done || step == this.maxStepsPerGame) {
                this.rewardStore.push(totalReward);
                break;
            }
        }
        await this.replay();
    }
    
    async replay() {
        const batch = this.memory.sample(this.model.batchSize);
        const states = batch.map(([state, , , ]) => state);
        const nextStates = batch.map(
            ([, , , nextState]) => nextState ? nextState : tf.zeros([this.model.numStates])
        );
        
        // Predict the values of each action at each state
        const qsa = states.map((state) => this.model.predict(state));
        // Predict the values of each action at each next state
        const qsad = nextStates.map((nextState) => this.model.predict(nextState));

        let x = new Array();
        let y = new Array();
        
        // Update the states rewards with the discounted next states rewards
        batch.forEach(
            ([state, action, reward, nextState], index) => {
                const currentQ = qsa[index];
                
                
                currentQ[action] = nextState ? reward + this.discountRate * qsad[index].max().dataSync() : reward;
                x.push(state.dataSync());
                y.push(currentQ.dataSync());
            }
        );
    }
}